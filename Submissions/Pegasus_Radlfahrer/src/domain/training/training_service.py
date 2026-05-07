from __future__ import annotations

import io
import math
import os
import shutil
import sys
import tempfile
import time
import __main__
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torchinfo import summary as torchinfo_summary
from tqdm import tqdm

from domain.inference.inference_service import GPTInferenceService
from domain.logging.experiment_logger import StageExperimentLogger
from domain.logging.training_run_logger import TrainingRunLogger
from domain.model.config_gpt_decoder import GPTDecoderConfig
from domain.model.config_hybrid_decoder import HybridDecoderConfig
from domain.model.config_llama_decoder import LlamaDecoderConfig
from domain.model.config_training_model import TrainingModelConfig
from domain.model.gpt_decoder_model import GPT2StyleDecoder
from domain.model.llama_decoder_model import LlamaDecoder
from domain.model.model_factory import build_model_from_config
from domain.runtime.distributed_runtime_service import DistributedRuntimeContext
from domain.training.checkpoint_manager import TrainingCheckpointManager
from domain.training.training_data_preparation_service import PreparedTrainingDataset
from domain.training.window_dataset_factory import DocumentWindowDatasetFactory, WindowDatasetFactory


@dataclass
class TrainingResult:
    final_train_loss: float
    final_val_loss: float
    final_val_accuracy: float
    vocab_size: int
    epochs_completed: int
    optimizer_steps_completed: int
    micro_batches_completed: int


@dataclass
class TrainingCheckpointState:
    step: int
    average_train_loss: float
    epoch: int = 1
    step_in_epoch: int = 0
    micro_step: int = 0
    cumulative_loss_sum: float = 0.0
    cumulative_valid_tokens: int = 0


@dataclass
class GPTTrainingService:
    """Core training loop for decoder-only language models."""

    window_dataset_factory: WindowDatasetFactory = field(default_factory=DocumentWindowDatasetFactory)
    training_run_logger: TrainingRunLogger | None = None
    inference_service: GPTInferenceService = field(default_factory=GPTInferenceService)

    def train(
        self,
        prepared_training_dataset: PreparedTrainingDataset,
        prepared_validation_dataset: PreparedTrainingDataset,
        vocab_size: int,
        model_config: TrainingModelConfig,
        batch_size: int,
        gradient_accumulation_steps: int,
        sequence_length: int,
        epochs: int,
        steps_per_epoch: int | None,
        validation_steps_per_epoch: int | None,
        validation_interval_steps: int,
        inference_interval_steps: int,
        inference_prompt: str,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        min_lr_ratio: float,
        seed: int,
        log_every: int,
        checkpoint_interval_seconds: float,
        f_param: int,
        checkpoint_dir: str,
        checkpoint_keep_latest_count: int,
        best_checkpoint_filename: str,
        device_name: str,
        distributed_runtime_context: DistributedRuntimeContext,
        experiment_logger: StageExperimentLogger,
        on_validation_result=None,
        use_mixed_precision: bool = False,
        use_torch_compile: bool = False,
        num_workers: int | None = None,
        subset_fraction: float = 1.0,
        enable_profiler: bool = False,
        profiler_wait_steps: int = 1,
        profiler_warmup_steps: int = 1,
        profiler_active_steps: int = 3,
        profiler_repeat: int = 1,
        profiler_output_dir: str | None = None,
    ) -> TrainingResult:
        torch.manual_seed(seed)
        device = torch.device(device_name)

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

        model_summary = self._build_model_summary(
            model=model,
            model_config=built_model_config,
            device_name=device_name,
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        world_size = max(distributed_runtime_context.world_size, 1)
        tokens_per_micro_batch = batch_size * sequence_length if (batch_size and sequence_length) else 0
        global_tokens_per_micro_batch = tokens_per_micro_batch * world_size
        effective_batch_size = batch_size * gradient_accumulation_steps * world_size
        effective_tokens_per_optimizer_step = global_tokens_per_micro_batch * gradient_accumulation_steps
        token_budget = f_param * total_parameters
        budget_limited_total_micro_batches = None
        if global_tokens_per_micro_batch > 0:
            budget_limited_total_micro_batches = max(1, token_budget // global_tokens_per_micro_batch)
        torchinfo_text = self._build_torchinfo_summary(
            model=model,
            sequence_length=sequence_length,
            batch_size=batch_size,
            device_name=device_name,
        )

        if self.training_run_logger is not None:
            self.training_run_logger.write_torchinfo_summary(torchinfo_text)

        if use_torch_compile and hasattr(torch, "compile"):
            model = torch.compile(model)
            experiment_logger.info("Applied torch.compile to model")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        amp_enabled = use_mixed_precision and device.type == "cuda"
        scaler = GradScaler(enabled=amp_enabled)
        amp_dtype = torch.float16 if amp_enabled else None

        experiment_logger.info(
            (
                f"Loaded training dataset='{prepared_training_dataset.dataset_path}' "
                f"with rows={prepared_training_dataset.source_rows}, "
                f"tokenized_column='{prepared_training_dataset.token_column}', vocab_size={vocab_size}, "
                f"usable_sequences={prepared_training_dataset.usable_rows}, "
                f"total_tokens={prepared_training_dataset.total_tokens}, "
                f"memmap='{prepared_training_dataset.tokenized_examples.token_file_path}'"
            )
        )
        experiment_logger.write_event(
            "dataset_loaded",
            level="INFO",
            split="train",
            dataset_path=prepared_training_dataset.dataset_path,
            source_rows=prepared_training_dataset.source_rows,
            usable_rows=prepared_training_dataset.usable_rows,
            total_tokens=prepared_training_dataset.total_tokens,
            sequence_length_stats=self._sequence_length_stats(prepared_training_dataset),
        )
        experiment_logger.info(
            (
                f"Loaded validation dataset='{prepared_validation_dataset.dataset_path}' "
                f"with rows={prepared_validation_dataset.source_rows}, "
                f"tokenized_column='{prepared_validation_dataset.token_column}', "
                f"usable_sequences={prepared_validation_dataset.usable_rows}, "
                f"total_tokens={prepared_validation_dataset.total_tokens}, "
                f"memmap='{prepared_validation_dataset.tokenized_examples.token_file_path}'"
            )
        )
        experiment_logger.write_event(
            "dataset_loaded",
            level="INFO",
            split="validation",
            dataset_path=prepared_validation_dataset.dataset_path,
            source_rows=prepared_validation_dataset.source_rows,
            usable_rows=prepared_validation_dataset.usable_rows,
            total_tokens=prepared_validation_dataset.total_tokens,
            sequence_length_stats=self._sequence_length_stats(prepared_validation_dataset),
        )
        experiment_logger.info("Model summary:\n" + model_summary)
        experiment_logger.info("Torchinfo summary:\n" + torchinfo_text)
        training_dataset = self.window_dataset_factory.create(
            prepared_training_dataset,
            sequence_length,
            vocab_size,
        )
        validation_dataset = self.window_dataset_factory.create(
            prepared_validation_dataset,
            sequence_length,
            vocab_size,
        )

        if subset_fraction < 1.0 and len(training_dataset) > 0:
            from torch.utils.data import Subset
            full_size = len(training_dataset)
            subset_size = max(1, int(full_size * subset_fraction))
            training_dataset = Subset(training_dataset, list(range(subset_size)))
            experiment_logger.info(
                f"Applied subset_fraction={subset_fraction}: using {subset_size} of {full_size} training samples"
            )

        loader_options = self._build_dataloader_options(device_name=device_name, num_workers_override=num_workers)
        train_sampler = self._build_sampler(
            dataset=training_dataset,
            distributed_runtime_context=distributed_runtime_context,
            shuffle=True,
            seed=seed,
        )
        validation_sampler = self._build_sampler(
            dataset=validation_dataset,
            distributed_runtime_context=distributed_runtime_context,
            shuffle=False,
            seed=seed,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=validation_sampler,
            **loader_options,
        )

        dataset_steps_per_epoch = max(len(train_sampler), 0) if train_sampler is not None else (len(training_dataset) + batch_size - 1) // batch_size
        effective_steps_per_epoch = dataset_steps_per_epoch
        if steps_per_epoch is not None:
            effective_steps_per_epoch = min(steps_per_epoch, dataset_steps_per_epoch)
        total_micro_batches = max(effective_steps_per_epoch * epochs, 1)
        if budget_limited_total_micro_batches is not None:
            total_micro_batches = min(total_micro_batches, budget_limited_total_micro_batches)
        total_optimizer_steps = max(math.ceil(total_micro_batches / gradient_accumulation_steps), 1)

        experiment_logger.info(
            (
                "Initialized training with "
                f"per_device_batch_size={batch_size}, "
                f"gradient_accumulation_steps={gradient_accumulation_steps}, "
                f"world_size={world_size}, "
                f"effective_batch_size={effective_batch_size}, "
                f"effective_batch_tokens={effective_tokens_per_optimizer_step}, "
                f"sequence_length={sequence_length}, epochs={epochs}, steps_per_epoch={steps_per_epoch}, "
                f"validation_steps_per_epoch={validation_steps_per_epoch}, "
                f"validation_interval_steps={validation_interval_steps}, "
                f"inference_interval_steps={inference_interval_steps}, "
                f"effective_steps_per_epoch={effective_steps_per_epoch}, lr={learning_rate}, device={device_name}, "
                f"checkpoint_interval_seconds={checkpoint_interval_seconds}, checkpoint_dir={checkpoint_dir}, "
                f"checkpoint_keep_latest_count={checkpoint_keep_latest_count}, "
                f"best_checkpoint_filename={best_checkpoint_filename}, "
                f"f_param={f_param}, token_budget={token_budget}, "
                f"budget_limited_total_micro_batches={budget_limited_total_micro_batches}, "
                f"total_optimizer_steps={total_optimizer_steps}"
            )
        )
        experiment_logger.write_event(
            "training_initialized",
            level="INFO",
            total_parameters=total_parameters,
            independent_learnable_parameters=self._count_independent_learnable_parameters(model),
            tie_word_embeddings=built_model_config.tie_word_embeddings,
            per_device_batch_size=batch_size,
            effective_batch_size=effective_batch_size,
            effective_batch_tokens=effective_tokens_per_optimizer_step,
            total_optimizer_steps=total_optimizer_steps,
            total_micro_batches=total_micro_batches,
            world_size=world_size,
            rank=distributed_runtime_context.rank,
            local_rank=distributed_runtime_context.local_rank,
            metrics_scope="global_aggregates",
        )

        profiler = self._build_profiler(
            enable_profiler=enable_profiler,
            device=device,
            experiment_logger=experiment_logger,
            profiler_wait_steps=profiler_wait_steps,
            profiler_warmup_steps=profiler_warmup_steps,
            profiler_active_steps=profiler_active_steps,
            profiler_repeat=profiler_repeat,
            profiler_output_dir=profiler_output_dir,
        )

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
            enabled=distributed_runtime_context.is_primary,
        )

        checkpoint_state = self._load_latest_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=experiment_logger,
        )
        self._move_training_state_to_device(model=model, optimizer=optimizer, device=device)
        training_model = self._wrap_model_for_distributed_training(
            model=model,
            device=device,
            distributed_runtime_context=distributed_runtime_context,
        )

        final_train_loss = checkpoint_state.average_train_loss if checkpoint_state is not None else 0.0
        completed_steps = checkpoint_state.step if checkpoint_state is not None else 0
        completed_micro_batches = checkpoint_state.micro_step if checkpoint_state is not None else 0

        final_validation_metrics = None
        profiler_context = profiler if profiler is not None else nullcontext()
        tqdm_disabled = distributed_runtime_context.is_distributed and not distributed_runtime_context.is_primary
        with profiler_context:
            for epoch in tqdm(
                range(1, epochs + 1),
                desc="total training progress",
                unit="epoch",
                disable=tqdm_disabled,
            ):
                remaining_budget_micro_batches = total_micro_batches - completed_micro_batches
                if remaining_budget_micro_batches <= 0:
                    experiment_logger.info(
                        "Stopping training because token budget was reached at "
                        f"micro_step={completed_micro_batches}/{total_micro_batches} "
                        f"(optimizer_step={completed_steps}/{total_optimizer_steps})"
                    )
                    break

                epoch_checkpoint_state = None
                if checkpoint_state is not None and checkpoint_state.epoch == epoch:
                    epoch_checkpoint_state = checkpoint_state

                train_generator = torch.Generator()
                train_generator.manual_seed(seed + epoch - 1)
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch - 1)
                train_loader = DataLoader(
                    training_dataset,
                    batch_size=batch_size,
                    shuffle=train_sampler is None,
                    sampler=train_sampler,
                    generator=train_generator if train_sampler is None else None,
                    **loader_options,
                )

                if checkpoint_state is not None and checkpoint_state.epoch > epoch:
                    experiment_logger.info(
                        f"Skipping epoch {epoch}/{epochs} because checkpoint has already advanced to epoch {checkpoint_state.epoch}."
                    )
                    continue

                if (
                    epoch_checkpoint_state is not None
                    and effective_steps_per_epoch > 0
                    and epoch_checkpoint_state.step_in_epoch >= effective_steps_per_epoch
                ):
                    experiment_logger.info(
                        f"Latest checkpoint already covers epoch {epoch}/{epochs}; advancing to next epoch"
                    )
                    checkpoint_state = None
                    continue

                (
                    final_train_loss,
                    completed_steps,
                    completed_micro_batches,
                    validation_metrics,
                ) = self.train_epoch(
                    model=training_model,
                    loader=train_loader,
                    validation_loader=validation_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    logger=experiment_logger,
                    device=device,
                    distributed_runtime_context=distributed_runtime_context,
                    checkpoint_interval_seconds=checkpoint_interval_seconds,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_manager=checkpoint_manager,
                    epoch_number=epoch,
                    total_epochs=epochs,
                    completed_steps=completed_steps,
                    completed_micro_batches=completed_micro_batches,
                    checkpoint_state=epoch_checkpoint_state,
                    total_training_steps=total_optimizer_steps,
                    total_micro_batches=total_micro_batches,
                    log_every=log_every,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    sequence_length=sequence_length,
                    batch_size=batch_size,
                    max_steps_in_epoch=min(
                        effective_steps_per_epoch,
                        remaining_budget_micro_batches + (epoch_checkpoint_state.step_in_epoch if epoch_checkpoint_state else 0),
                    ),
                    validation_steps_per_epoch=validation_steps_per_epoch,
                    validation_interval_steps=validation_interval_steps,
                    inference_interval_steps=inference_interval_steps,
                    inference_prompt=inference_prompt,
                    on_validation_result=on_validation_result,
                    scaler=scaler,
                    amp_dtype=amp_dtype,
                    profiler=profiler,
                )
                checkpoint_state = None
                if validation_metrics is not None:
                    final_validation_metrics = validation_metrics

        if profiler is not None:
            try:
                written_files = self._finalize_profiler_trace_files_with_retry(
                    experiment_logger.experiment_dir / "profiler"
                    if profiler_output_dir is None
                    else Path(profiler_output_dir)
                )
                if written_files:
                    experiment_logger.info(
                        f"PyTorch profiling finished. Wrote {len(written_files)} trace file(s)."
                    )
                else:
                    experiment_logger.info("PyTorch profiling finished, but no trace files were written.")
            except Exception as error:
                experiment_logger.info(
                    f"Profiler trace finalization failed ({error}). Continuing training without profiler artifacts."
                )

        # if caching skips all epochs, validation will not run, thus re-run here
        if final_validation_metrics is None:
            final_validation_metrics = self.evaluate(
                model=training_model,
                loader=validation_loader,
                logger=experiment_logger,
                device=device,
                distributed_runtime_context=distributed_runtime_context,
                max_steps=validation_steps_per_epoch,
                amp_dtype=amp_dtype,
            )

        if distributed_runtime_context.is_distributed:
            torch.distributed.barrier()
        if distributed_runtime_context.is_primary:
            final_checkpoint = {
                "architecture": architecture,
                "model_config": built_model_config.model_dump(mode="json"),
                "model_state_dict": self._state_dict_model(model).state_dict(),
            }
            torch.save(final_checkpoint, experiment_logger.model_path)
            experiment_logger.info(f"Saved model checkpoint to {experiment_logger.model_path}")
            experiment_logger.write_event(
                "checkpoint_saved",
                level="INFO",
                reason="final",
                path=str(experiment_logger.model_path),
                optimizer_step=completed_steps,
                micro_step=completed_micro_batches,
            )

        return TrainingResult(
            final_train_loss=final_train_loss,
            final_val_loss=final_validation_metrics["loss"],
            final_val_accuracy=final_validation_metrics["accuracy"],
            vocab_size=vocab_size,
            epochs_completed=epochs,
            optimizer_steps_completed=completed_steps,
            micro_batches_completed=completed_micro_batches,
        )

    def _build_dataloader_options(self, device_name: str, num_workers_override: int | None = None) -> dict[str, object]:
        if num_workers_override is not None:
            num_workers = num_workers_override
        else:
            cpu_count = os.cpu_count() or 1
            main_file = getattr(__main__, "__file__", None)
            has_main_file = bool(main_file) and not str(main_file).startswith("<")
            num_workers = min(4, max(cpu_count - 1, 0)) if has_main_file else 0
        options: dict[str, object] = {
            "num_workers": num_workers,
            "pin_memory": device_name.startswith("cuda"),
        }
        if num_workers > 0:
            options["persistent_workers"] = True
            options["prefetch_factor"] = 4
        return options

    def _sequence_length_stats(self, prepared_dataset: PreparedTrainingDataset) -> dict[str, float | int | None]:
        lengths = prepared_dataset.tokenized_examples.document_lengths
        if lengths.size == 0:
            return {"mean": None, "p95": None, "max": None}
        return {
            "mean": float(lengths.mean()),
            "p95": float(np.percentile(lengths, 95)),
            "max": int(lengths.max()),
        }

    def _system_health(self, device: torch.device) -> dict[str, float | None]:
        health: dict[str, float | None] = {
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
            "gpu_memory_peak_mb": None,
            "ram_used_mb": None,
            "ram_available_mb": None,
        }
        if device.type == "cuda":
            health["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2)
            health["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2)
            health["gpu_memory_peak_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
        try:
            import psutil

            memory = psutil.virtual_memory()
            health["ram_used_mb"] = round(memory.used / (1024 * 1024), 2)
            health["ram_available_mb"] = round(memory.available / (1024 * 1024), 2)
        except Exception:
            pass
        return health

    def _is_near_cuda_oom(self, system_health: dict[str, float | None]) -> bool:
        reserved = system_health.get("gpu_memory_reserved_mb")
        if reserved is None or not torch.cuda.is_available():
            return False
        try:
            total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        except Exception:
            return False
        return reserved / max(total_mb, 1.0) >= 0.9

    def train_epoch(
        self,
        model,
        loader,
        validation_loader,
        optimizer,
        scheduler,
        logger,
        device,
        checkpoint_interval_seconds,
        distributed_runtime_context: DistributedRuntimeContext,
        checkpoint_dir,
        epoch_number,
        total_epochs,
        completed_steps,
        completed_micro_batches,
        checkpoint_state: TrainingCheckpointState | None,
        total_training_steps,
        total_micro_batches,
        log_every=10,
        gradient_accumulation_steps: int = 1,
        sequence_length: int = 0,
        batch_size: int = 0,
        max_steps_in_epoch: int | None = None,
        validation_steps_per_epoch: int | None = None,
        validation_interval_steps: int = 100,
        inference_interval_steps: int = 0,
        inference_prompt: str = "",
        on_validation_result=None,
        scaler: GradScaler | None = None,
        amp_dtype: torch.dtype | None = None,
        profiler=None,
        checkpoint_manager: TrainingCheckpointManager | None = None,
    ):
        model.train()
        if checkpoint_manager is None:
            checkpoint_manager = TrainingCheckpointManager(
                checkpoint_dir=Path(checkpoint_dir),
                keep_latest_count=5,
                best_checkpoint_filename="best_checkpoint.pt",
                logger=logger,
                enabled=False,
            )

        running_loss_sum = 0.0
        running_grad_norm = 0.0
        running_valid_tokens = 0
        processed_batches = 0
        processed_optimizer_steps = 0
        clipped_optimizer_steps = 0
        processed_examples = 0
        observed_token_slots = 0
        num_batches = len(loader)
        epoch_steps = num_batches if max_steps_in_epoch is None else min(max_steps_in_epoch, num_batches)
        start_step = checkpoint_state.step_in_epoch if checkpoint_state else 0
        last_checkpoint_time = time.monotonic()
        epoch_start_time = time.monotonic()
        world_size = max(distributed_runtime_context.world_size, 1)
        tokens_per_micro_batch = batch_size * sequence_length if (batch_size and sequence_length) else 0
        effective_batch_size = batch_size * gradient_accumulation_steps * world_size
        effective_batch_tokens = tokens_per_micro_batch * gradient_accumulation_steps * world_size
        clip_threshold = 0.3

        if checkpoint_state is not None:
            running_loss_sum = checkpoint_state.cumulative_loss_sum
            running_valid_tokens = checkpoint_state.cumulative_valid_tokens
            logger.info(
                (
                    f"Resuming training from checkpoint at epoch={epoch_number}/{total_epochs} "
                    f"micro_step={checkpoint_state.step_in_epoch}/{epoch_steps} "
                    f"optimizer_step={checkpoint_state.step}/{total_training_steps} "
                    f"global_micro_step={checkpoint_state.micro_step}/{total_micro_batches} "
                    f"with avg_train_loss={checkpoint_state.average_train_loss:.6f}"
                )
            )
        else:
            logger.info(f"Starting epoch {epoch_number}/{total_epochs}")

        if start_step >= epoch_steps:
            logger.info(
                f"Latest checkpoint already covers all training batches for epoch {epoch_number}/{total_epochs}; skipping epoch"
            )
            return (
                checkpoint_state.average_train_loss if checkpoint_state is not None else 0.0,
                completed_steps,
                completed_micro_batches,
                None,
            )

        optimizer.zero_grad(set_to_none=True)
        current_global_step = completed_steps
        current_micro_step = completed_micro_batches
        final_validation_metrics = None
        window_valid_tokens_local = 0
        for step_in_epoch, batch in enumerate(
            tqdm(
                loader,
                desc=f"training epoch {epoch_number}",
                unit="batch",
                disable=distributed_runtime_context.is_distributed and not distributed_runtime_context.is_primary,
            ),
            start=1,
        ):
            if step_in_epoch > epoch_steps:
                break
            if step_in_epoch <= start_step:
                continue

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device).contiguous()
            is_optimizer_step = (
                step_in_epoch % gradient_accumulation_steps == 0 or step_in_epoch == epoch_steps
            )

            sync_context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not is_optimizer_step
                else nullcontext()
            )
            with sync_context:
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(input_ids).contiguous()

                    assert logits.shape[:2] == labels.shape, "input, label mismatch"

                    vocabulary_size = logits.size(-1)
                    loss_sum = F.cross_entropy(
                        logits.view(-1, vocabulary_size),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    loss_for_backward = loss_sum

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            processed_batches += 1
            current_micro_step += 1
            valid_tokens_local = int((labels != -100).sum().item())
            window_valid_tokens_local += valid_tokens_local
            loss_sum_global = self._distributed_sum(loss_sum.detach(), device, distributed_runtime_context)
            valid_tokens_global = self._distributed_sum(valid_tokens_local, device, distributed_runtime_context)
            examples_global = self._distributed_sum(int(input_ids.size(0)), device, distributed_runtime_context)
            token_slots_global = self._distributed_sum(int(labels.numel()), device, distributed_runtime_context)
            processed_examples += int(examples_global)
            observed_token_slots += int(token_slots_global)
            if math.isfinite(loss_sum_global):
                running_loss_sum += loss_sum_global
            else:
                logger.warning(
                    (
                        f"Non-finite training loss detected at epoch={epoch_number}/{total_epochs} "
                        f"micro_step={step_in_epoch}/{epoch_steps}"
                    )
                )
                logger.write_event(
                    "non_finite_train_loss",
                    level="WARNING",
                    epoch=epoch_number,
                    micro_step=step_in_epoch,
                    global_micro_step=current_micro_step,
                )
            running_valid_tokens += valid_tokens_global
            average_loss = running_loss_sum / max(running_valid_tokens, 1)
            now = time.monotonic()

            if not is_optimizer_step:
                if profiler is not None:
                    profiler.step()
                continue

            global_window_valid_tokens = self._distributed_sum(
                window_valid_tokens_local,
                device,
                distributed_runtime_context,
            )
            if global_window_valid_tokens <= 0:
                logger.info(
                    (
                        f"Skipping optimizer step at epoch={epoch_number}/{total_epochs} "
                        f"micro_step={step_in_epoch}/{epoch_steps} because the accumulation window "
                        "contains no valid target tokens."
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                window_valid_tokens_local = 0
                if profiler is not None:
                    profiler.step()
                continue

            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                gradient_scale = world_size / global_window_valid_tokens
                self._scale_gradients(model, gradient_scale)
                grad_norm = self.grad_l2_norm(model)
                if math.isfinite(grad_norm) and grad_norm > clip_threshold:
                    clipped_optimizer_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_threshold)
                scaler.step(optimizer)
                scaler.update()
            else:
                gradient_scale = world_size / global_window_valid_tokens
                self._scale_gradients(model, gradient_scale)
                grad_norm = self.grad_l2_norm(model)
                if math.isfinite(grad_norm) and grad_norm > clip_threshold:
                    clipped_optimizer_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_threshold)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_valid_tokens_local = 0
            scheduler.step()

            processed_optimizer_steps += 1
            current_global_step += 1
            grad_norm = self._distributed_mean(grad_norm, device, distributed_runtime_context)
            if math.isfinite(grad_norm):
                running_grad_norm += grad_norm
            else:
                logger.warning(
                    (
                        f"Non-finite gradient norm detected at epoch={epoch_number}/{total_epochs} "
                        f"optimizer_step={current_global_step}/{total_training_steps}"
                    )
                )
                logger.write_event(
                    "non_finite_grad_norm",
                    level="WARNING",
                    epoch=epoch_number,
                    optimizer_step=current_global_step,
                    micro_step=current_micro_step,
                )

            if now - last_checkpoint_time >= checkpoint_interval_seconds or step_in_epoch == epoch_steps:
                checkpoint_start_time = time.monotonic()
                checkpoint_path = self._save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=current_global_step,
                    micro_step=current_micro_step,
                    epoch=epoch_number,
                    step_in_epoch=step_in_epoch,
                    average_train_loss=average_loss,
                    cumulative_loss_sum=running_loss_sum,
                    cumulative_valid_tokens=running_valid_tokens,
                    model=self._state_dict_model(model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    logger=logger,
                    distributed_runtime_context=distributed_runtime_context,
                )
                checkpoint_duration_sec = time.monotonic() - checkpoint_start_time
                checkpoint_manager.record_latest_checkpoint(checkpoint_path)
                logger.write_event(
                    "checkpoint_saved",
                    level="INFO",
                    reason="periodic" if step_in_epoch != epoch_steps else "epoch_end",
                    path=str(checkpoint_path),
                    optimizer_step=current_global_step,
                    micro_step=current_micro_step,
                    duration_sec=checkpoint_duration_sec,
                )
                last_checkpoint_time = now

            if current_global_step % log_every == 0 or step_in_epoch == epoch_steps:
                avg_grad_norm = running_grad_norm / max(processed_optimizer_steps, 1)
                current_lr = optimizer.param_groups[0]["lr"]
                wall_time = time.monotonic() - epoch_start_time
                tokens_per_sec = (
                    running_valid_tokens / max(wall_time, 1e-9)
                    if running_valid_tokens > 0
                    else 0.0
                )
                steps_remaining = max(total_training_steps - current_global_step, 0)
                steps_per_sec = current_global_step / max(wall_time, 1e-9)
                eta_seconds = steps_remaining / steps_per_sec if steps_per_sec > 0 else None
                clip_ratio = clipped_optimizer_steps / max(processed_optimizer_steps, 1)
                padding_rate = (
                    1.0 - (running_valid_tokens / max(observed_token_slots, 1))
                    if observed_token_slots > 0
                    else None
                )
                system_health = self._system_health(device)

                logger.info(
                    (
                        f"epoch={epoch_number}/{total_epochs} "
                        f"micro_step={step_in_epoch}/{epoch_steps} "
                        f"global_micro_step={current_micro_step}/{total_micro_batches} "
                        f"optimizer_step={current_global_step}/{total_training_steps} "
                        f"avg_train_loss={average_loss:.6f} "
                        f"avg_grad_norm={avg_grad_norm:.6f} "
                        f"learning_rate={current_lr:.8f} "
                        f"tokens_per_sec={tokens_per_sec:.2f} "
                        f"clip_ratio={clip_ratio:.4f}"
                    )
                )
                logger.write_metrics(
                    step=str(current_global_step),
                    values={
                        "epoch": epoch_number,
                        "micro_step": current_micro_step,
                        "train_loss": average_loss,
                        "train_perplexity": self._perplexity(average_loss),
                        "val_loss": None,
                        "learning_rate": current_lr,
                        "grad_norm": avg_grad_norm,
                        "tokens_per_sec": tokens_per_sec,
                        "steps_per_sec": steps_per_sec,
                        "eta_seconds": eta_seconds,
                        "tokens_processed": int(running_valid_tokens),
                        "examples_processed": processed_examples,
                        "batches_processed": processed_batches,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "effective_batch_size": effective_batch_size,
                        "effective_batch_tokens": effective_batch_tokens,
                        "clip_ratio": clip_ratio,
                        "padding_rate": padding_rate,
                        **system_health,
                    },
                )
                if self._is_near_cuda_oom(system_health):
                    logger.warning(
                        (
                            f"Near-OOM CUDA memory usage detected at optimizer_step={current_global_step}: "
                            f"gpu_memory_allocated_mb={system_health.get('gpu_memory_allocated_mb')} "
                            f"gpu_memory_reserved_mb={system_health.get('gpu_memory_reserved_mb')}"
                        )
                    )
                    logger.write_event(
                        "near_oom",
                        level="WARNING",
                        optimizer_step=current_global_step,
                        **system_health,
                    )

                if self.training_run_logger is not None:
                    self.training_run_logger.log_train_step(
                        step=current_global_step,
                        micro_step=current_micro_step,
                        train_loss=average_loss,
                        learning_rate=current_lr,
                        grad_norm=avg_grad_norm,
                        tokens_per_sec=tokens_per_sec,
                        wall_time_sec=wall_time,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        effective_batch_size=effective_batch_size,
                        effective_batch_tokens=effective_batch_tokens,
                        tokens_processed=int(running_valid_tokens),
                        examples_processed=processed_examples,
                        batches_processed=processed_batches,
                        clip_ratio=clip_ratio,
                        gpu_memory_allocated_mb=system_health.get("gpu_memory_allocated_mb"),
                        gpu_memory_peak_mb=system_health.get("gpu_memory_peak_mb"),
                        ram_used_mb=system_health.get("ram_used_mb"),
                        ram_available_mb=system_health.get("ram_available_mb"),
                        eta_seconds=eta_seconds,
                        padding_rate=padding_rate,
                    )
                    self.training_run_logger.update_plots()

            if current_global_step % validation_interval_steps == 0:
                validation_start_time = time.monotonic()
                validation_metrics = self.evaluate(
                    model=model,
                    loader=validation_loader,
                    logger=logger,
                    device=device,
                    distributed_runtime_context=distributed_runtime_context,
                    max_steps=validation_steps_per_epoch,
                    amp_dtype=amp_dtype,
                )
                validation_runtime_sec = time.monotonic() - validation_start_time
                final_validation_metrics = validation_metrics
                val_loss = validation_metrics["loss"]
                previous_best_val_loss = checkpoint_manager.best_validation_loss
                delta_val_loss = (
                    val_loss - previous_best_val_loss
                    if previous_best_val_loss is not None
                    else None
                )
                logger.write_metrics(
                    step="validation",
                    values={
                        "epoch": epoch_number,
                        "optimizer_step": current_global_step,
                        "micro_step": current_micro_step,
                        "train_loss": average_loss,
                        "val_loss": val_loss,
                        "accuracy": validation_metrics["accuracy"],
                        "val_perplexity": self._perplexity(val_loss),
                        "validation_runtime_sec": validation_runtime_sec,
                        "best_val_loss": previous_best_val_loss,
                        "delta_val_loss": delta_val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                )

                if self.training_run_logger is not None:
                    self.training_run_logger.log_eval_step(
                        step=current_global_step,
                        val_loss=val_loss,
                        accuracy=validation_metrics["accuracy"],
                        runtime_sec=validation_runtime_sec,
                        best_val_loss=previous_best_val_loss,
                        delta_val_loss=delta_val_loss,
                    )
                    self.training_run_logger.update_plots()

                if on_validation_result is not None:
                    on_validation_result(current_global_step, val_loss)

                if checkpoint_manager.should_update_best(val_loss):
                    best_checkpoint_path = self._save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=current_global_step,
                        micro_step=current_micro_step,
                        epoch=epoch_number,
                        step_in_epoch=step_in_epoch,
                        average_train_loss=average_loss,
                        cumulative_loss_sum=running_loss_sum,
                        cumulative_valid_tokens=running_valid_tokens,
                        model=self._state_dict_model(model),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        logger=logger,
                        distributed_runtime_context=distributed_runtime_context,
                        checkpoint_path=checkpoint_manager.best_checkpoint_path,
                        validation_loss=val_loss,
                        checkpoint_label="best training checkpoint",
                    )
                    checkpoint_manager.record_best_checkpoint(val_loss)
                    logger.write_event(
                        "checkpoint_saved",
                        level="INFO",
                        reason="best",
                        path=str(best_checkpoint_path),
                        optimizer_step=current_global_step,
                        micro_step=current_micro_step,
                        val_loss=val_loss,
                    )

                model.train()

            if inference_interval_steps > 0 and current_global_step % inference_interval_steps == 0:
                self._run_inference_sanity_check(
                    model=model,
                    logger=logger,
                    device=device,
                    distributed_runtime_context=distributed_runtime_context,
                    optimizer_step=current_global_step,
                    micro_step=current_micro_step,
                    prompt=inference_prompt,
                    sequence_length=sequence_length,
                    vocabulary_size=vocabulary_size,
                )
                model.train()

            if profiler is not None:
                profiler.step()

        final_average_loss = running_loss_sum / max(running_valid_tokens, 1)
        return final_average_loss, current_global_step, current_micro_step, final_validation_metrics

    def _run_inference_sanity_check(
        self,
        model: nn.Module,
        logger: StageExperimentLogger,
        device: torch.device,
        distributed_runtime_context: DistributedRuntimeContext,
        optimizer_step: int,
        micro_step: int,
        prompt: str,
        sequence_length: int,
        vocabulary_size: int,
    ) -> None:
        if not distributed_runtime_context.is_primary:
            return

        generation_model = self._state_dict_model(model)
        max_new_tokens = max(1, min(16, sequence_length // 4 or 1))
        result = self.inference_service.run_with_model(
            model=generation_model,
            input_text=prompt,
            max_new_tokens=max_new_tokens,
            device_name=str(device),
            max_position_embeddings=sequence_length,
            vocab_size=vocabulary_size,
            temperature=0.0,
            top_k=min(50, vocabulary_size),
        )
        generated_only_token_ids = result.generated_token_ids[len(result.input_token_ids) :]

        logger.info(
            (
                f"Inference sanity check at optimizer_step={optimizer_step}: "
                f"prompt={prompt!r} generated_text={result.generated_text!r}"
            )
        )
        logger.write_metrics(
            step="inference",
            values={
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "input_text": prompt,
                "input_token_ids": result.input_token_ids,
                "input_length": len(result.input_token_ids),
                "max_new_tokens": max_new_tokens,
                "generated_token_ids": result.generated_token_ids,
                "generated_length": len(result.generated_token_ids),
                "generated_new_token_ids": generated_only_token_ids,
                "generated_new_tokens": len(generated_only_token_ids),
                "generated_text": result.generated_text,
                "device": str(device),
            },
        )

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        logger: StageExperimentLogger,
        device: torch.device,
        distributed_runtime_context: DistributedRuntimeContext,
        max_steps: int | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_batches = len(loader)
        effective_total_batches = total_batches if max_steps is None else min(max_steps, total_batches)
        processed_batches = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(
                tqdm(
                    loader,
                    desc="validation batch",
                    unit="batch",
                    disable=distributed_runtime_context.is_distributed and not distributed_runtime_context.is_primary,
                ),
                start=1,
            ):
                if batch_index > effective_total_batches:
                    break

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device).contiguous()

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(input_ids).contiguous()

                    vocabulary_size = logits.size(-1)
                    loss = F.cross_entropy(
                        logits.view(-1, vocabulary_size),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                total_loss += loss.item()
                processed_batches += 1

                predictions = logits.argmax(dim=-1)
                valid_positions = labels != -100
                correct_predictions += ((predictions == labels) & valid_positions).sum().item()
                total_predictions += valid_positions.sum().item()

        if distributed_runtime_context.is_distributed:
            reduced_metrics = torch.tensor(
                [total_loss, float(processed_batches), float(correct_predictions), float(total_predictions)],
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(reduced_metrics, op=torch.distributed.ReduceOp.SUM)
            total_loss = float(reduced_metrics[0].item())
            processed_batches = int(reduced_metrics[1].item())
            correct_predictions = int(reduced_metrics[2].item())
            total_predictions = int(reduced_metrics[3].item())

        average_loss = total_loss / max(processed_batches, 1)
        accuracy = correct_predictions / total_predictions if total_predictions else 0.0
        logger.info(
            (
                f"Validation complete: avg_val_loss={average_loss:.6f} accuracy={accuracy:.6f} "
                f"batches={min(processed_batches, effective_total_batches)}/{total_batches}"
            )
        )
        return {"loss": average_loss, "accuracy": accuracy}

    def _build_sampler(
        self,
        dataset,
        distributed_runtime_context: DistributedRuntimeContext,
        shuffle: bool,
        seed: int,
    ) -> DistributedSampler | None:
        if not distributed_runtime_context.is_distributed:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=distributed_runtime_context.world_size,
            rank=distributed_runtime_context.rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    def _wrap_model_for_distributed_training(
        self,
        model: nn.Module,
        device: torch.device,
        distributed_runtime_context: DistributedRuntimeContext,
    ) -> nn.Module:
        if not distributed_runtime_context.is_distributed:
            return model

        if device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[distributed_runtime_context.local_rank],
                output_device=distributed_runtime_context.local_rank,
            )

        return DistributedDataParallel(model)

    def _state_dict_model(self, model: nn.Module) -> nn.Module:
        if isinstance(model, DistributedDataParallel):
            return model.module
        return model

    def _distributed_mean(
        self,
        value: torch.Tensor | float,
        device: torch.device,
        distributed_runtime_context: DistributedRuntimeContext,
    ) -> float:
        if not distributed_runtime_context.is_distributed:
            if isinstance(value, torch.Tensor):
                return float(value.detach().item())
            return float(value)

        if isinstance(value, torch.Tensor):
            reduced_value = value.detach().to(device=device, dtype=torch.float64)
        else:
            reduced_value = torch.tensor(float(value), device=device, dtype=torch.float64)

        torch.distributed.all_reduce(reduced_value, op=torch.distributed.ReduceOp.SUM)
        reduced_value /= distributed_runtime_context.world_size

        return float(reduced_value.item())

    def _distributed_sum(
        self,
        value: torch.Tensor | float | int,
        device: torch.device,
        distributed_runtime_context: DistributedRuntimeContext,
    ) -> float:
        if not distributed_runtime_context.is_distributed:
            if isinstance(value, torch.Tensor):
                return float(value.detach().item())
            return float(value)

        if isinstance(value, torch.Tensor):
            reduced_value = value.detach().to(device=device, dtype=torch.float64)
        else:
            reduced_value = torch.tensor(float(value), device=device, dtype=torch.float64)

        torch.distributed.all_reduce(reduced_value, op=torch.distributed.ReduceOp.SUM)
        return float(reduced_value.item())

    def _scale_gradients(self, model: nn.Module, scale: float) -> None:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.detach().mul_(scale)

    def grad_l2_norm(self, model):
        total = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_sq = parameter.grad.detach().pow(2).sum().item()
                if not math.isfinite(grad_sq):
                    return float("nan")
                total += grad_sq
        return total ** 0.5

    def _build_model_summary(
        self,
        model: nn.Module,
        model_config: GPTDecoderConfig | LlamaDecoderConfig | HybridDecoderConfig,
        device_name: str,
    ) -> str:
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        independent_learnable_parameters = self._count_independent_learnable_parameters(model)

        summary_lines = [
            f"device={device_name}",
            f"architecture={type(model).__name__}",
            f"vocab_size={model_config.vocab_size}",
            f"max_position_embeddings={model_config.max_position_embeddings}",
            f"hidden_size={model_config.hidden_size}",
            f"num_layers={model_config.num_layers}",
            f"num_attention_heads={model_config.num_attention_heads}",
            f"tie_word_embeddings={model_config.tie_word_embeddings}",
        ]
        if isinstance(model_config, GPTDecoderConfig):
            summary_lines.append(f"mlp_hidden_size={model_config.mlp_hidden_size}")
            summary_lines.append(f"dropout={model_config.dropout}")
        elif isinstance(model_config, LlamaDecoderConfig):
            summary_lines.append(f"n_kv_heads={model_config.n_kv_heads}")
            summary_lines.append(f"intermediate_size={model_config.intermediate_size}")
        elif isinstance(model_config, HybridDecoderConfig):
            summary_lines.append(f"norm_type={model_config.norm_type}")
            summary_lines.append(f"norm_placement={model_config.norm_placement}")
            summary_lines.append(f"positional_encoding_type={model_config.positional_encoding_type}")
            summary_lines.append(f"attention_type={model_config.attention_type}")
            summary_lines.append(f"ffn_type={model_config.ffn_type}")
            summary_lines.append(f"n_kv_heads={model_config.n_kv_heads}")
            summary_lines.append(f"mlp_hidden_size={model_config.mlp_hidden_size}")
            summary_lines.append(f"intermediate_size={model_config.intermediate_size}")
            summary_lines.append(f"tie_word_embeddings={model_config.tie_word_embeddings}")
        summary_lines.extend([
            f"total_parameters={total_parameters}",
            f"trainable_parameters={trainable_parameters}",
            f"independent_learnable_parameters={independent_learnable_parameters}",
            "architecture:",
            str(model),
        ])
        return "\n".join(summary_lines)

    @staticmethod
    def _count_independent_learnable_parameters(model: nn.Module) -> int:
        seen_parameter_ids: set[int] = set()
        independent_parameters = 0
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen_parameter_ids:
                continue
            seen_parameter_ids.add(parameter_id)
            independent_parameters += parameter.numel()
        return independent_parameters

    @staticmethod
    def _perplexity(loss: float) -> float:
        if not math.isfinite(loss):
            return float("inf")
        return math.exp(min(loss, math.log(sys.float_info.max)))

    @staticmethod
    def _cosine_warmup_schedule(
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float,
    ):
        """Warmup + Cosine Decay + Floor schedule (VL04 recipe).

        1. Linear warmup from 0 to peak LR over warmup_steps.
        2. Cosine decay from peak LR down toward floor.
        3. Floor at min_lr_ratio * peak LR (never reaches zero).
        """
        import math

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return lr_lambda

    def _build_torchinfo_summary(
        self,
        model: nn.Module,
        sequence_length: int,
        batch_size: int,
        device_name: str,
    ) -> str:
        try:
            stats = torchinfo_summary(
                model,
                input_data=torch.randint(0, model.config.vocab_size, (batch_size, sequence_length)),
                col_names=("input_size", "output_size", "num_params", "trainable"),
                depth=3,
                verbose=0,
            )
            return str(stats)
        except Exception as error:
            return f"torchinfo summary unavailable: {error}"

    def _build_profiler(
        self,
        *,
        enable_profiler: bool,
        device: torch.device,
        experiment_logger: StageExperimentLogger,
        profiler_wait_steps: int,
        profiler_warmup_steps: int,
        profiler_active_steps: int,
        profiler_repeat: int,
        profiler_output_dir: str | None,
    ):
        if not enable_profiler:
            return None

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        trace_dir = Path(profiler_output_dir) if profiler_output_dir else experiment_logger.experiment_dir / "profiler"
        trace_dir.mkdir(parents=True, exist_ok=True)

        try:
            on_trace_ready = self._build_profiler_trace_handler(
                trace_dir=trace_dir,
                experiment_logger=experiment_logger,
            )
            experiment_logger.info(f"PyTorch profiler enabled. Writing traces to {trace_dir}")
            return torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=profiler_wait_steps,
                    warmup=profiler_warmup_steps,
                    active=profiler_active_steps,
                    repeat=profiler_repeat,
                ),
                on_trace_ready=on_trace_ready,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
        except Exception as error:
            experiment_logger.info(
                f"Failed to enable PyTorch profiler ({error}). Continuing without profiler."
            )
            return None

    def _build_profiler_trace_handler(
        self,
        *,
        trace_dir: Path,
        experiment_logger: StageExperimentLogger,
    ):
        base_handler = torch.profiler.tensorboard_trace_handler(str(trace_dir))

        def on_trace_ready(profiler) -> None:
            try:
                base_handler(profiler)
            except Exception as error:
                experiment_logger.info(
                    f"Profiler trace export failed ({error}). Continuing training without profiler artifacts."
                )

        return on_trace_ready

    def _finalize_profiler_trace_files_with_retry(self, trace_dir: Path) -> list[Path]:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                return self._finalize_profiler_trace_files(trace_dir)
            except PermissionError as error:
                last_error = error
                time.sleep(0.2 * (attempt + 1))

        if last_error is not None:
            raise last_error
        return []

    def _finalize_profiler_trace_files(self, trace_dir: Path) -> list[Path]:
        for temp_trace_path in trace_dir.glob("*.tmp"):
            final_trace_path = temp_trace_path.with_name(temp_trace_path.name.removesuffix(".tmp"))
            if final_trace_path.exists():
                final_trace_path.unlink()
            try:
                temp_trace_path.replace(final_trace_path)
            except PermissionError:
                shutil.copy2(temp_trace_path, final_trace_path)

        return sorted(
            path for path in trace_dir.iterdir()
            if path.is_file() and path.suffix != ".tmp"
        )

    def _logger_warning(self, logger: StageExperimentLogger, message: str) -> None:
        warning = getattr(logger, "warning", None)
        if callable(warning):
            warning(message)
            return
        logger.info(message)

    def _write_logger_event(self, logger: StageExperimentLogger, event_type: str, **values) -> None:
        write_event = getattr(logger, "write_event", None)
        if callable(write_event):
            write_event(event_type, **values)

    def _load_latest_checkpoint(
        self,
        checkpoint_dir: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        device: torch.device,
        logger: StageExperimentLogger,
    ) -> TrainingCheckpointState | None:
        checkpoint_dir_path = Path(checkpoint_dir)
        for checkpoint_path in self._checkpoint_paths_descending(checkpoint_dir_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                cpu_rng_state = checkpoint.get("cpu_rng_state")
                if cpu_rng_state is None:
                    cpu_rng_state = checkpoint.get("torch_rng_state")
                if cpu_rng_state is not None:
                    torch.set_rng_state(cpu_rng_state)

                if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
                    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

                if device.type == "mps" and checkpoint.get("mps_rng_state") is not None:
                    torch.mps.set_rng_state(checkpoint["mps_rng_state"])

                logger.info(f"Loaded training checkpoint from {checkpoint_path}")
                self._write_logger_event(
                    logger,
                    "checkpoint_loaded",
                    level="INFO",
                    reason="resume",
                    path=str(checkpoint_path),
                    step=int(checkpoint["step"]),
                )
                step = int(checkpoint["step"])
                return TrainingCheckpointState(
                    step=step,
                    average_train_loss=float(checkpoint["average_train_loss"]),
                    epoch=int(checkpoint.get("epoch", 1)),
                    step_in_epoch=int(checkpoint.get("step_in_epoch", step)),
                    micro_step=int(checkpoint.get("micro_step", step)),
                    cumulative_loss_sum=float(
                        checkpoint.get(
                            "cumulative_loss_sum",
                            float(checkpoint["average_train_loss"])
                            * float(checkpoint.get("cumulative_valid_tokens", checkpoint.get("step_in_epoch", step))),
                        )
                    ),
                    cumulative_valid_tokens=int(
                        checkpoint.get("cumulative_valid_tokens", checkpoint.get("step_in_epoch", step))
                    ),
                )
            except Exception as error:
                self._logger_warning(
                    logger,
                    f"Failed to load training checkpoint from {checkpoint_path}: {error}. Trying an older checkpoint.",
                )
                self._write_logger_event(
                    logger,
                    "checkpoint_load_failed",
                    level="WARNING",
                    path=str(checkpoint_path),
                    error=str(error),
                )

        logger.info(f"No valid training checkpoint found in {checkpoint_dir}; starting from scratch")
        self._write_logger_event(
            logger,
            "checkpoint_loaded",
            level="INFO",
            reason="fresh_start",
            path=None,
        )
        return None

    def _save_checkpoint(
        self,
        *,
        checkpoint_dir: str,
        step: int,
        micro_step: int | None = None,
        average_train_loss: float,
        cumulative_loss_sum: float | None = None,
        cumulative_valid_tokens: int | None = None,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        logger: StageExperimentLogger,
        distributed_runtime_context: DistributedRuntimeContext | None = None,
        epoch: int = 1,
        step_in_epoch: int | None = None,
        checkpoint_path: Path | None = None,
        validation_loss: float | None = None,
        checkpoint_label: str = "training checkpoint",
    ) -> Path:
        resolved_checkpoint_path = checkpoint_path or self._checkpoint_path(Path(checkpoint_dir), step)
        if distributed_runtime_context is not None and not distributed_runtime_context.is_primary:
            return resolved_checkpoint_path
        checkpoint_path = resolved_checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "step": step,
            "micro_step": step if micro_step is None else micro_step,
            "epoch": epoch,
            "step_in_epoch": step if step_in_epoch is None else step_in_epoch,
            "average_train_loss": average_train_loss,
            "cumulative_loss_sum": average_train_loss if cumulative_loss_sum is None else cumulative_loss_sum,
            "cumulative_valid_tokens": step if cumulative_valid_tokens is None else cumulative_valid_tokens,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps_rng_state": torch.mps.get_rng_state() if self._is_mps_available() else None,
        }
        if validation_loss is not None:
            checkpoint["validation_loss"] = validation_loss
        self._atomic_torch_save(checkpoint, checkpoint_path)
        logger.info(f"Saved {checkpoint_label} to {checkpoint_path}")
        return checkpoint_path

    def _move_training_state_to_device(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        model.to(device)
        self._move_optimizer_state_to_device(optimizer=optimizer, device=device)

    def _move_optimizer_state_to_device(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def _is_mps_available(self) -> bool:
        return bool(
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "get_rng_state")
            and hasattr(torch.mps, "set_rng_state")
        )

    def _checkpoint_paths_descending(self, checkpoint_dir: Path) -> list[Path]:
        if not checkpoint_dir.exists():
            return []

        return sorted(
            checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=self._checkpoint_step_from_path,
            reverse=True,
        )

    def _checkpoint_path(self, checkpoint_dir: Path, step: int) -> Path:
        return checkpoint_dir / f"checkpoint_step_{step:08d}.pt"

    def _checkpoint_step_from_path(self, checkpoint_path: Path) -> int:
        return int(checkpoint_path.stem.split("_")[-1])

    def _atomic_torch_save(self, checkpoint: dict, checkpoint_path: Path) -> None:
        if os.name == "nt":
            torch.save(checkpoint, checkpoint_path)
            return

        fd, temp_name = tempfile.mkstemp(dir=checkpoint_path.parent)
        os.close(fd)
        temp_path = Path(temp_name)

        try:
            torch.save(checkpoint, temp_path)
            self._replace_with_retry(temp_path=temp_path, checkpoint_path=checkpoint_path)
        finally:
            if temp_path.exists():
                self._unlink_with_retry(temp_path)

    def _replace_with_retry(self, temp_path: Path, checkpoint_path: Path) -> None:
        for attempt in range(5):
            try:
                temp_path.replace(checkpoint_path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))

    def _unlink_with_retry(self, temp_path: Path) -> None:
        for attempt in range(5):
            try:
                temp_path.unlink()
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))
