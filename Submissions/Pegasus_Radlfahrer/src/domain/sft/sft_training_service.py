"""SFT training loop for Alpaca-style instruction fine-tuning.

Reuses the pretraining cross-entropy formulation (sum loss / valid tokens) and the
cosine-with-warmup LR schedule, but replaces the windowed pretraining dataset
with a per-example SFTDataset whose labels already mask the prompt and padding
with -100. The base pretrained checkpoint is loaded once at startup; periodic
checkpoints, the best validation checkpoint, and the final model.pt are written
in the same dict format used by the pretraining stage so inference can load them.
"""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain.logging.experiment_logger import StageExperimentLogger
from domain.model.config_training_model import TrainingModelConfig
from domain.model.model_factory import build_model_from_config
from domain.sft.sft_dataset import SFTDataset
from domain.training.checkpoint_manager import TrainingCheckpointManager


@dataclass(slots=True)
class SFTTrainingResult:
    final_train_loss: float
    final_val_loss: float
    final_val_accuracy: float
    optimizer_steps_completed: int
    micro_batches_completed: int
    train_rows: int
    validation_rows: int


class SFTTrainingService:
    """Train a decoder-only model on prepared Alpaca-style SFT examples."""

    def train(
        self,
        *,
        train_dataset_path: str,
        validation_dataset_path: str,
        base_checkpoint_path: str,
        experiment_logger: StageExperimentLogger,
        device_name: str,
        vocab_size: int,
        eos_token_id: int,
        model_config: TrainingModelConfig,
        batch_size: int,
        gradient_accumulation_steps: int,
        sequence_length: int,
        epochs: int,
        steps_per_epoch: int | None,
        validation_steps_per_epoch: int | None,
        validation_interval_steps: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        min_lr_ratio: float,
        seed: int,
        log_every: int,
        checkpoint_interval_seconds: float,
        checkpoint_dir: str,
        checkpoint_keep_latest_count: int,
        best_checkpoint_filename: str,
        use_mixed_precision: bool,
        num_workers: int | None,
    ) -> SFTTrainingResult:
        torch.manual_seed(seed)
        device = torch.device(device_name)

        train_rows = self._load_rows(train_dataset_path)
        validation_rows = self._load_rows(validation_dataset_path)
        if not train_rows:
            raise ValueError(f"SFT training dataset at {train_dataset_path} is empty")

        experiment_logger.info(
            f"Loaded SFT datasets: train_rows={len(train_rows)} val_rows={len(validation_rows)}"
        )

        train_dataset = SFTDataset(
            rows=train_rows,
            sequence_length=sequence_length,
            pad_token_id=eos_token_id,
            vocab_size=vocab_size,
        )
        validation_dataset = SFTDataset(
            rows=validation_rows,
            sequence_length=sequence_length,
            pad_token_id=eos_token_id,
            vocab_size=vocab_size,
        )

        architecture = getattr(model_config, "architecture", "gpt2")
        model, built_model_config = build_model_from_config(
            architecture=architecture,
            vocab_size=vocab_size,
            max_position_embeddings=model_config.max_position_embeddings,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            num_attention_heads=model_config.num_attention_heads,
            tie_word_embeddings=model_config.tie_word_embeddings,
            mlp_hidden_size=model_config.mlp_hidden_size,
            qkv_bias=model_config.qkv_bias,
            dropout=model_config.dropout,
            n_kv_heads=model_config.n_kv_heads,
            intermediate_size=model_config.intermediate_size,
            rope_theta=model_config.rope_theta,
            norm_type=getattr(model_config, "norm_type", "layernorm"),
            norm_placement=getattr(model_config, "norm_placement", "pre_norm"),
            positional_encoding_type=getattr(model_config, "positional_encoding_type", "learned_absolute"),
            attention_type=getattr(model_config, "attention_type", "mha"),
            ffn_type=getattr(model_config, "ffn_type", "gelu"),
        )

        self._load_base_checkpoint(
            model=model,
            base_checkpoint_path=base_checkpoint_path,
            logger=experiment_logger,
        )

        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        amp_enabled = use_mixed_precision and device.type == "cuda"
        scaler = GradScaler(enabled=amp_enabled)
        amp_dtype = torch.float16 if amp_enabled else None

        loader_options = self._loader_options(
            device_name=device_name,
            num_workers_override=num_workers,
        )
        train_loader_steps = max(1, math.ceil(len(train_dataset) / batch_size))
        effective_steps_per_epoch = train_loader_steps
        if steps_per_epoch is not None:
            effective_steps_per_epoch = min(steps_per_epoch, train_loader_steps)
        total_micro_batches = max(effective_steps_per_epoch * epochs, 1)
        total_optimizer_steps = max(math.ceil(total_micro_batches / gradient_accumulation_steps), 1)

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=self._cosine_warmup_schedule(
                total_steps=total_optimizer_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=min_lr_ratio,
            ),
        )

        checkpoint_manager = TrainingCheckpointManager(
            checkpoint_dir=Path(checkpoint_dir),
            keep_latest_count=checkpoint_keep_latest_count,
            best_checkpoint_filename=best_checkpoint_filename,
            logger=experiment_logger,
            enabled=True,
        )

        experiment_logger.info(
            (
                f"Initialized SFT training with batch_size={batch_size}, "
                f"gradient_accumulation_steps={gradient_accumulation_steps}, "
                f"sequence_length={sequence_length}, epochs={epochs}, "
                f"effective_steps_per_epoch={effective_steps_per_epoch}, "
                f"total_optimizer_steps={total_optimizer_steps}, lr={learning_rate}, device={device_name}"
            )
        )

        completed_optimizer_steps = 0
        completed_micro_batches = 0
        running_loss_sum = 0.0
        running_valid_tokens = 0
        last_checkpoint_time = time.monotonic()
        final_validation_metrics: dict[str, float] | None = None

        model.train()
        for epoch in range(1, epochs + 1):
            train_generator = torch.Generator()
            train_generator.manual_seed(seed + epoch - 1)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=train_generator,
                **loader_options,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_options,
            )

            optimizer.zero_grad(set_to_none=True)
            window_valid_tokens = 0
            for step_in_epoch, batch in enumerate(
                tqdm(
                    train_loader,
                    desc=f"sft epoch {epoch}",
                    unit="batch",
                    total=effective_steps_per_epoch,
                ),
                start=1,
            ):
                if step_in_epoch > effective_steps_per_epoch:
                    break

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device).contiguous()
                is_optimizer_step = (
                    step_in_epoch % gradient_accumulation_steps == 0
                    or step_in_epoch == effective_steps_per_epoch
                )

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(input_ids).contiguous()
                    assert logits.shape[:2] == labels.shape, "logits/labels mismatch"
                    vocabulary_size = logits.size(-1)
                    loss_sum = F.cross_entropy(
                        logits.view(-1, vocabulary_size),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )

                if scaler.is_enabled():
                    scaler.scale(loss_sum).backward()
                else:
                    loss_sum.backward()

                completed_micro_batches += 1
                valid_tokens_local = int((labels != -100).sum().item())
                window_valid_tokens += valid_tokens_local
                if math.isfinite(float(loss_sum.detach().item())):
                    running_loss_sum += float(loss_sum.detach().item())
                    running_valid_tokens += valid_tokens_local
                else:
                    experiment_logger.warning(
                        f"Non-finite SFT loss at epoch={epoch} step_in_epoch={step_in_epoch}"
                    )

                if not is_optimizer_step:
                    continue

                if window_valid_tokens <= 0:
                    optimizer.zero_grad(set_to_none=True)
                    window_valid_tokens = 0
                    continue

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    self._scale_gradients(model, 1.0 / window_valid_tokens)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    self._scale_gradients(model, 1.0 / window_valid_tokens)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                window_valid_tokens = 0
                scheduler.step()
                completed_optimizer_steps += 1

                average_loss = running_loss_sum / max(running_valid_tokens, 1)
                if completed_optimizer_steps % log_every == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    experiment_logger.info(
                        (
                            f"epoch={epoch}/{epochs} optimizer_step={completed_optimizer_steps}/"
                            f"{total_optimizer_steps} micro_step={completed_micro_batches} "
                            f"avg_train_loss={average_loss:.6f} lr={current_lr:.8f} "
                            f"valid_tokens={running_valid_tokens}"
                        )
                    )
                    experiment_logger.write_metrics(
                        step=str(completed_optimizer_steps),
                        values={
                            "epoch": epoch,
                            "optimizer_step": completed_optimizer_steps,
                            "micro_step": completed_micro_batches,
                            "train_loss": average_loss,
                            "learning_rate": current_lr,
                            "valid_tokens": running_valid_tokens,
                        },
                    )

                now = time.monotonic()
                if now - last_checkpoint_time >= checkpoint_interval_seconds:
                    self._save_periodic_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_manager=checkpoint_manager,
                        step=completed_optimizer_steps,
                        model=model,
                        architecture=architecture,
                        built_model_config=built_model_config,
                        average_train_loss=average_loss,
                        logger=experiment_logger,
                    )
                    last_checkpoint_time = now

                if completed_optimizer_steps % validation_interval_steps == 0:
                    metrics = self._evaluate(
                        model=model,
                        loader=validation_loader,
                        device=device,
                        amp_dtype=amp_dtype,
                        max_steps=validation_steps_per_epoch,
                    )
                    final_validation_metrics = metrics
                    experiment_logger.info(
                        (
                            f"validation epoch={epoch} optimizer_step={completed_optimizer_steps} "
                            f"val_loss={metrics['loss']:.6f} accuracy={metrics['accuracy']:.6f}"
                        )
                    )
                    experiment_logger.write_metrics(
                        step="validation",
                        values={
                            "epoch": epoch,
                            "optimizer_step": completed_optimizer_steps,
                            "train_loss": average_loss,
                            "val_loss": metrics["loss"],
                            "accuracy": metrics["accuracy"],
                        },
                    )
                    if checkpoint_manager.should_update_best(metrics["loss"]):
                        self._save_best_checkpoint(
                            checkpoint_manager=checkpoint_manager,
                            model=model,
                            architecture=architecture,
                            built_model_config=built_model_config,
                            average_train_loss=average_loss,
                            validation_loss=metrics["loss"],
                            logger=experiment_logger,
                        )
                        checkpoint_manager.record_best_checkpoint(metrics["loss"])
                    model.train()

        if final_validation_metrics is None:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                **loader_options,
            )
            final_validation_metrics = self._evaluate(
                model=model,
                loader=validation_loader,
                device=device,
                amp_dtype=amp_dtype,
                max_steps=validation_steps_per_epoch,
            )

        final_checkpoint = {
            "architecture": architecture,
            "model_config": built_model_config.model_dump(mode="json"),
            "model_state_dict": model.state_dict(),
        }
        torch.save(final_checkpoint, experiment_logger.model_path)
        experiment_logger.info(f"Saved final SFT model to {experiment_logger.model_path}")

        average_loss = running_loss_sum / max(running_valid_tokens, 1)
        return SFTTrainingResult(
            final_train_loss=average_loss,
            final_val_loss=final_validation_metrics["loss"],
            final_val_accuracy=final_validation_metrics["accuracy"],
            optimizer_steps_completed=completed_optimizer_steps,
            micro_batches_completed=completed_micro_batches,
            train_rows=len(train_rows),
            validation_rows=len(validation_rows),
        )

    def _load_rows(self, dataset_path: str) -> list[dict[str, Any]]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"SFT dataset path does not exist: {dataset_path}")
        dataset = load_from_disk(str(path))
        if not isinstance(dataset, Dataset):
            raise TypeError(f"SFT dataset at {dataset_path} is not a single-split Dataset")
        return [
            {"input_ids": list(row["input_ids"]), "labels": list(row["labels"])}
            for row in dataset
        ]

    def _load_base_checkpoint(
        self,
        *,
        model: nn.Module,
        base_checkpoint_path: str,
        logger: StageExperimentLogger,
    ) -> None:
        path = Path(base_checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"base_checkpoint_path does not exist: {base_checkpoint_path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            detected_format = "model_state_dict"
        elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
            detected_format = "model"
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            detected_format = "bare_state_dict"
        else:
            raise TypeError(
                f"Unsupported base checkpoint format at {base_checkpoint_path}: {type(checkpoint).__name__}"
            )

        prefix = "_orig_mod."
        if state_dict and all(key.startswith(prefix) for key in state_dict.keys()):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
            detected_format = f"{detected_format}+stripped_orig_mod_prefix"

        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load base checkpoint from {base_checkpoint_path}: {error}"
            ) from error

        # Tied word embeddings legitimately cause the LM head weight to be
        # absent from the saved state dict, so we ignore that one well-known
        # missing key. Anything else means the SFT model architecture does not
        # match the base checkpoint and we must fail loudly instead of
        # silently training a half-loaded model.
        known_safe_missing = {"lm_head.weight", "output_projection.weight"}
        unsafe_missing = [key for key in missing if key not in known_safe_missing]
        if unsafe_missing or unexpected:
            raise RuntimeError(
                (
                    "SFT base checkpoint does not match the SFT model architecture: "
                    f"missing={unsafe_missing[:8]} unexpected={list(unexpected)[:8]} "
                    f"(checkpoint={base_checkpoint_path}, format={detected_format}). "
                    "Verify that the [model] block in the SFT config matches the base "
                    "training config (architecture, hidden_size, num_layers, "
                    "num_attention_heads, n_kv_heads, intermediate_size, "
                    "max_position_embeddings, tie_word_embeddings, vocab_size)."
                )
            )

        logger.info(
            (
                f"Loaded base SFT checkpoint from {base_checkpoint_path} "
                f"(format={detected_format}, missing={len(missing)}, unexpected={len(unexpected)})"
            )
        )
        if missing:
            logger.info(
                f"SFT base checkpoint missing keys treated as tied/known-safe: {missing}"
            )

    def _evaluate(
        self,
        *,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        amp_dtype: torch.dtype | None,
        max_steps: int | None,
    ) -> dict[str, float]:
        model.eval()
        total_loss_sum = 0.0
        total_valid_tokens = 0
        correct_predictions = 0
        total_predictions = 0
        processed_batches = 0
        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(loader, start=1):
                    if max_steps is not None and batch_index > max_steps:
                        break
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device).contiguous()
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                        logits = model(input_ids).contiguous()
                        vocabulary_size = logits.size(-1)
                        loss_sum = F.cross_entropy(
                            logits.view(-1, vocabulary_size),
                            labels.view(-1),
                            ignore_index=-100,
                            reduction="sum",
                        )
                    valid_tokens = int((labels != -100).sum().item())
                    total_loss_sum += float(loss_sum.item())
                    total_valid_tokens += valid_tokens

                    predictions = logits.argmax(dim=-1)
                    valid_positions = labels != -100
                    correct_predictions += int(((predictions == labels) & valid_positions).sum().item())
                    total_predictions += int(valid_positions.sum().item())
                    processed_batches += 1
        finally:
            model.train()

        if total_valid_tokens == 0 or processed_batches == 0:
            return {"loss": float("nan"), "accuracy": 0.0}
        average_loss = total_loss_sum / total_valid_tokens
        accuracy = correct_predictions / total_predictions if total_predictions else 0.0
        return {"loss": average_loss, "accuracy": accuracy}

    def _save_periodic_checkpoint(
        self,
        *,
        checkpoint_dir: str,
        checkpoint_manager: TrainingCheckpointManager,
        step: int,
        model: nn.Module,
        architecture: str,
        built_model_config,
        average_train_loss: float,
        logger: StageExperimentLogger,
    ) -> None:
        checkpoint_dir_path = Path(checkpoint_dir)
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir_path / f"checkpoint_step_{step:08d}.pt"
        torch.save(
            {
                "step": step,
                "average_train_loss": average_train_loss,
                "architecture": architecture,
                "model_config": built_model_config.model_dump(mode="json"),
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        logger.info(f"Saved SFT checkpoint to {checkpoint_path}")
        checkpoint_manager.record_latest_checkpoint(checkpoint_path)

    def _save_best_checkpoint(
        self,
        *,
        checkpoint_manager: TrainingCheckpointManager,
        model: nn.Module,
        architecture: str,
        built_model_config,
        average_train_loss: float,
        validation_loss: float,
        logger: StageExperimentLogger,
    ) -> None:
        path = checkpoint_manager.best_checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "validation_loss": validation_loss,
                "average_train_loss": average_train_loss,
                "architecture": architecture,
                "model_config": built_model_config.model_dump(mode="json"),
                "model_state_dict": model.state_dict(),
            },
            path,
        )
        logger.info(f"Saved best SFT checkpoint to {path} (val_loss={validation_loss:.6f})")

    def _loader_options(self, device_name: str, num_workers_override: int | None) -> dict[str, object]:
        if num_workers_override is not None:
            num_workers = max(0, num_workers_override)
        else:
            num_workers = 0
        options: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": device_name.startswith("cuda"),
        }
        if num_workers > 0:
            options["persistent_workers"] = True
        return options

    @staticmethod
    def _scale_gradients(model: nn.Module, scale: float) -> None:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.detach().mul_(scale)

    @staticmethod
    def _cosine_warmup_schedule(total_steps: int, warmup_steps: int, min_lr_ratio: float):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return lr_lambda
