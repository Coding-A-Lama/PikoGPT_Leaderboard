"""Direct Preference Optimization (DPO) training.

Loads a frozen reference model and a trainable policy model, both initialized
from the same SFT checkpoint. The DPO loss compares the per-sequence log-prob
margins of (chosen, rejected) under policy vs reference:

    margin_policy    = logp_pi(chosen)  - logp_pi(rejected)
    margin_reference = logp_ref(chosen) - logp_ref(rejected)
    loss             = -log_sigmoid(beta * (margin_policy - margin_reference))

Per-sequence log probability is summed only over response tokens (labels != -100),
matching the SFT response-only convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain.dpo.dpo_dataset import DPODataset
from domain.inference.checkpoint_loader import load_model_from_checkpoint
from domain.logging.experiment_logger import StageExperimentLogger


@dataclass(slots=True)
class DPOResult:
    final_train_loss: float
    final_val_loss: float
    final_val_pref_accuracy: float
    optimizer_steps: int
    train_rows: int
    validation_rows: int


class DPOTrainingService:
    def train(
        self,
        *,
        policy_checkpoint_path: str,
        reference_checkpoint_path: str,
        train_dataset_path: str,
        validation_dataset_path: str,
        experiment_logger: StageExperimentLogger,
        device_name: str,
        sequence_length: int,
        eos_token_id: int,
        fallback_model_settings: dict[str, Any],
        batch_size: int,
        gradient_accumulation_steps: int,
        epochs: int,
        learning_rate: float,
        beta: float,
        max_grad_norm: float,
        weight_decay: float,
        seed: int,
        log_every: int,
        validation_interval_steps: int,
        validation_steps_per_epoch: int | None,
        checkpoint_dir: str,
        use_mixed_precision: bool,
        warmup_steps: int = 0,
        min_lr_ratio: float = 1.0,
        best_checkpoint_filename: str = "best_checkpoint.pt",
    ) -> DPOResult:
        torch.manual_seed(seed)
        device = torch.device(device_name)

        train_rows = self._load_rows(train_dataset_path)
        validation_rows = self._load_rows(validation_dataset_path)
        if not train_rows:
            raise ValueError(f"DPO train dataset at {train_dataset_path} is empty")

        train_dataset = DPODataset(
            rows=train_rows,
            sequence_length=sequence_length,
            pad_token_id=eos_token_id,
        )
        validation_dataset = DPODataset(
            rows=validation_rows,
            sequence_length=sequence_length,
            pad_token_id=eos_token_id,
        )

        experiment_logger.info(
            f"Loaded DPO datasets: train={len(train_rows)} val={len(validation_rows)}"
        )

        policy = load_model_from_checkpoint(
            checkpoint_path=policy_checkpoint_path,
            fallback_model_settings=fallback_model_settings,
            device=device,
        )
        reference = load_model_from_checkpoint(
            checkpoint_path=reference_checkpoint_path,
            fallback_model_settings=fallback_model_settings,
            device=device,
        )

        for parameter in reference.model.parameters():
            parameter.requires_grad_(False)
        reference.model.eval()
        policy.model.train()

        optimizer = torch.optim.AdamW(
            policy.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        amp_enabled = use_mixed_precision and device.type == "cuda"
        amp_dtype = torch.float16 if amp_enabled else None
        scaler = torch.amp.GradScaler(enabled=amp_enabled)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        # Cosine-with-warmup LR schedule. With min_lr_ratio=1.0 (the default
        # for back-compat) the schedule is constant after warmup.
        micro_batches_per_epoch = max(1, len(train_loader))
        total_optimizer_steps = max(
            math.ceil(micro_batches_per_epoch * epochs / max(1, gradient_accumulation_steps)),
            1,
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=_cosine_warmup_schedule(
                total_steps=total_optimizer_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=min_lr_ratio,
            ),
        )

        best_val_loss = float("inf")
        best_checkpoint_path = Path(checkpoint_dir) / best_checkpoint_filename
        completed_steps = 0
        running_loss = 0.0
        running_count = 0
        last_val_metrics: dict[str, float] | None = None

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            micro_batches_in_window = 0
            for step_in_epoch, batch in enumerate(
                tqdm(train_loader, desc=f"dpo epoch {epoch}", unit="batch"), start=1
            ):
                loss = self._compute_dpo_loss(
                    policy_model=policy.model,
                    reference_model=reference.model,
                    batch=batch,
                    device=device,
                    beta=beta,
                    amp_dtype=amp_dtype,
                )

                if scaler.is_enabled():
                    scaler.scale(loss / gradient_accumulation_steps).backward()
                else:
                    (loss / gradient_accumulation_steps).backward()

                running_loss += float(loss.detach().item())
                running_count += 1
                micro_batches_in_window += 1

                is_step = (
                    step_in_epoch % gradient_accumulation_steps == 0
                    or step_in_epoch == len(train_loader)
                )
                if not is_step:
                    continue

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), max_norm=max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                completed_steps += 1
                micro_batches_in_window = 0

                if completed_steps % log_every == 0:
                    avg_loss = running_loss / max(running_count, 1)
                    experiment_logger.info(
                        f"epoch={epoch}/{epochs} step={completed_steps} avg_dpo_loss={avg_loss:.6f}"
                    )
                    experiment_logger.write_metrics(
                        step=str(completed_steps),
                        values={
                            "epoch": epoch,
                            "step": completed_steps,
                            "train_loss": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                    )

                if completed_steps % validation_interval_steps == 0:
                    last_val_metrics = self._evaluate(
                        policy_model=policy.model,
                        reference_model=reference.model,
                        loader=validation_loader,
                        device=device,
                        beta=beta,
                        amp_dtype=amp_dtype,
                        max_steps=validation_steps_per_epoch,
                    )
                    experiment_logger.info(
                        f"validation step={completed_steps} loss={last_val_metrics['loss']:.6f} "
                        f"pref_acc={last_val_metrics['pref_accuracy']:.4f} "
                        f"chosen_logp={last_val_metrics['avg_chosen_logp']:.3f} "
                        f"rejected_logp={last_val_metrics['avg_rejected_logp']:.3f}"
                    )
                    experiment_logger.write_metrics(
                        step="validation",
                        values={
                            "step": completed_steps,
                            **last_val_metrics,
                        },
                    )
                    if last_val_metrics["loss"] < best_val_loss:
                        best_val_loss = last_val_metrics["loss"]
                        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "step": completed_steps,
                                "validation_loss": last_val_metrics["loss"],
                                "pref_accuracy": last_val_metrics["pref_accuracy"],
                                "architecture": policy.architecture,
                                "model_config": _saved_model_config(
                                    policy=policy,
                                    fallback_model_settings=fallback_model_settings,
                                ),
                                "model_state_dict": policy.model.state_dict(),
                            },
                            best_checkpoint_path,
                        )
                        experiment_logger.info(
                            f"Saved best DPO checkpoint to {best_checkpoint_path} (val_loss={best_val_loss:.6f})"
                        )
                    policy.model.train()

        if last_val_metrics is None:
            last_val_metrics = self._evaluate(
                policy_model=policy.model,
                reference_model=reference.model,
                loader=validation_loader,
                device=device,
                beta=beta,
                amp_dtype=amp_dtype,
                max_steps=validation_steps_per_epoch,
            )

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        final_path = experiment_logger.model_path
        torch.save(
            {
                "architecture": policy.architecture,
                "model_config": _saved_model_config(
                    policy=policy,
                    fallback_model_settings=fallback_model_settings,
                ),
                "model_state_dict": policy.model.state_dict(),
            },
            final_path,
        )
        experiment_logger.info(f"Saved final DPO model to {final_path}")

        return DPOResult(
            final_train_loss=running_loss / max(running_count, 1) if running_count else 0.0,
            final_val_loss=last_val_metrics["loss"],
            final_val_pref_accuracy=last_val_metrics["pref_accuracy"],
            optimizer_steps=completed_steps,
            train_rows=len(train_rows),
            validation_rows=len(validation_rows),
        )

    def _load_rows(self, path: str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"DPO dataset path does not exist: {path}")
        ds = load_from_disk(str(p))
        if not isinstance(ds, Dataset):
            raise TypeError(f"DPO dataset at {path} is not a single-split Dataset")
        rows: list[dict[str, Any]] = []
        for row in ds:
            rows.append(
                {
                    "chosen_input_ids": list(row["chosen_input_ids"]),
                    "chosen_labels": list(row["chosen_labels"]),
                    "rejected_input_ids": list(row["rejected_input_ids"]),
                    "rejected_labels": list(row["rejected_labels"]),
                }
            )
        return rows

    def _compute_dpo_loss(
        self,
        *,
        policy_model: nn.Module,
        reference_model: nn.Module,
        batch: dict[str, Tensor],
        device: torch.device,
        beta: float,
        amp_dtype: torch.dtype | None,
    ) -> Tensor:
        chosen_input = batch["chosen_input_ids"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_input = batch["rejected_input_ids"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            policy_chosen_logits = policy_model(chosen_input)
            policy_rejected_logits = policy_model(rejected_input)
            with torch.no_grad():
                ref_chosen_logits = reference_model(chosen_input)
                ref_rejected_logits = reference_model(rejected_input)

        logp_pi_chosen = sequence_log_prob(policy_chosen_logits, chosen_labels)
        logp_pi_rejected = sequence_log_prob(policy_rejected_logits, rejected_labels)
        logp_ref_chosen = sequence_log_prob(ref_chosen_logits, chosen_labels)
        logp_ref_rejected = sequence_log_prob(ref_rejected_logits, rejected_labels)

        margin_policy = logp_pi_chosen - logp_pi_rejected
        margin_reference = logp_ref_chosen - logp_ref_rejected
        diff = margin_policy - margin_reference
        loss = -F.logsigmoid(beta * diff).mean()
        return loss

    def _evaluate(
        self,
        *,
        policy_model: nn.Module,
        reference_model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        beta: float,
        amp_dtype: torch.dtype | None,
        max_steps: int | None,
    ) -> dict[str, float]:
        policy_model.eval()
        total_loss = 0.0
        total_pref_correct = 0
        total_pairs = 0
        total_chosen_logp = 0.0
        total_rejected_logp = 0.0
        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(loader, start=1):
                    if max_steps is not None and batch_index > max_steps:
                        break
                    chosen_input = batch["chosen_input_ids"].to(device)
                    chosen_labels = batch["chosen_labels"].to(device)
                    rejected_input = batch["rejected_input_ids"].to(device)
                    rejected_labels = batch["rejected_labels"].to(device)
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                        policy_chosen_logits = policy_model(chosen_input)
                        policy_rejected_logits = policy_model(rejected_input)
                        ref_chosen_logits = reference_model(chosen_input)
                        ref_rejected_logits = reference_model(rejected_input)
                    logp_pi_chosen = sequence_log_prob(policy_chosen_logits, chosen_labels)
                    logp_pi_rejected = sequence_log_prob(policy_rejected_logits, rejected_labels)
                    logp_ref_chosen = sequence_log_prob(ref_chosen_logits, chosen_labels)
                    logp_ref_rejected = sequence_log_prob(ref_rejected_logits, rejected_labels)
                    margin_policy = logp_pi_chosen - logp_pi_rejected
                    margin_reference = logp_ref_chosen - logp_ref_rejected
                    diff = margin_policy - margin_reference
                    loss = -F.logsigmoid(beta * diff).mean()
                    total_loss += float(loss.item()) * chosen_input.size(0)
                    total_pref_correct += int((margin_policy > 0).sum().item())
                    total_pairs += int(chosen_input.size(0))
                    total_chosen_logp += float(logp_pi_chosen.sum().item())
                    total_rejected_logp += float(logp_pi_rejected.sum().item())
        finally:
            policy_model.train()
        if total_pairs == 0:
            return {"loss": float("nan"), "pref_accuracy": 0.0, "avg_chosen_logp": 0.0, "avg_rejected_logp": 0.0}
        return {
            "loss": total_loss / total_pairs,
            "pref_accuracy": total_pref_correct / total_pairs,
            "avg_chosen_logp": total_chosen_logp / total_pairs,
            "avg_rejected_logp": total_rejected_logp / total_pairs,
        }


def _saved_model_config(*, policy, fallback_model_settings: dict[str, Any]) -> dict[str, Any]:
    """Compose the model_config dict written into DPO checkpoints.

    Putting fallback_model_settings first and the actually-built policy values
    last guarantees the saved config reflects what the model in memory really
    is, even if the config-file defaults differ.
    """
    config = {**fallback_model_settings}
    config["vocab_size"] = policy.vocab_size
    config["max_position_embeddings"] = policy.max_position_embeddings
    return config


def _cosine_warmup_schedule(*, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return lr_lambda


def sequence_log_prob(logits: Tensor, labels: Tensor) -> Tensor:
    """Sum of log P(token_t) over positions where labels[t] != -100, per sequence.

    `logits` is `(B, T, V)` aligned with `labels` `(B, T)` — the SFT/DPO datasets
    already perform the next-token shift, so position t of `labels` is the gold
    target produced from `logits[:, t, :]`.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    valid_mask = labels != -100
    safe_labels = labels.masked_fill(~valid_mask, 0)
    selected = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    selected = selected * valid_mask.float()
    return selected.sum(dim=-1)
