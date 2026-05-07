"""Live training run logger that maintains plots and a torchinfo summary in a timestamped log folder."""

from __future__ import annotations

import io
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class TrainingRunLogger:
    """Creates a timestamped folder under ``log/`` and continuously updates training plots."""

    def __init__(self, log_root: str = "log", enabled: bool = True) -> None:
        self.enabled = enabled
        self.run_id = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-UTC")
        self.run_dir = self._unique_run_dir(Path(log_root), timestamp)
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._events_path = self.run_dir / "events.jsonl"
        self._metadata_path = self.run_dir / "run_metadata.json"

        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self._steps: list[int] = []
        self._train_loss: list[float] = []
        self._train_ppl: list[float] = []
        self._learning_rate: list[float] = []
        self._grad_norm: list[float] = []
        self._tokens_per_sec: list[float] = []
        self._wall_time: list[float] = []

        self._eval_steps: list[int] = []
        self._val_loss: list[float] = []
        self._val_ppl: list[float] = []

    def _unique_run_dir(self, log_root: Path, timestamp: str) -> Path:
        candidate = log_root / timestamp
        if not candidate.exists():
            return candidate
        return log_root / f"{timestamp}-{self.run_id[:8]}"

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record.setdefault("run_id", self.run_id)
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_run_metadata(self, metadata: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        self._metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def log_event(self, event_type: str, level: str = "INFO", **values: Any) -> None:
        if not self.enabled:
            return
        record = {
            "run_id": self.run_id,
            "type": event_type,
            "level": level.upper(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_torchinfo_summary(self, summary_text: str) -> None:
        if not self.enabled:
            return
        (self.run_dir / "model_summary.txt").write_text(summary_text, encoding="utf-8")

    def log_train_step(
        self,
        step: int,
        micro_step: int,
        train_loss: float,
        learning_rate: float,
        grad_norm: float,
        tokens_per_sec: float,
        wall_time_sec: float,
        gradient_accumulation_steps: int,
        effective_batch_size: int,
        effective_batch_tokens: int,
        tokens_processed: int,
        examples_processed: int,
        batches_processed: int,
        clip_ratio: float,
        gpu_memory_allocated_mb: float | None = None,
        gpu_memory_peak_mb: float | None = None,
        ram_used_mb: float | None = None,
        ram_available_mb: float | None = None,
        eta_seconds: float | None = None,
        padding_rate: float | None = None,
    ) -> None:
        self._steps.append(step)
        self._train_loss.append(train_loss)
        train_ppl = self._perplexity(train_loss)
        self._train_ppl.append(train_ppl)
        self._learning_rate.append(learning_rate)
        self._grad_norm.append(grad_norm)
        self._tokens_per_sec.append(tokens_per_sec)
        self._wall_time.append(wall_time_sec)

        self._append_jsonl({
            "type": "train",
            "step": step,
            "micro_step": micro_step,
            "train_loss": train_loss,
            "train_perplexity": train_ppl,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "tokens_per_sec": tokens_per_sec,
            "wall_time_sec": wall_time_sec,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "effective_batch_tokens": effective_batch_tokens,
            "tokens_processed": tokens_processed,
            "examples_processed": examples_processed,
            "batches_processed": batches_processed,
            "clip_ratio": clip_ratio,
            "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
            "gpu_memory_peak_mb": gpu_memory_peak_mb,
            "ram_used_mb": ram_used_mb,
            "ram_available_mb": ram_available_mb,
            "eta_seconds": eta_seconds,
            "padding_rate": padding_rate,
        })

    def log_eval_step(
        self,
        step: int,
        val_loss: float,
        accuracy: float | None = None,
        runtime_sec: float | None = None,
        best_val_loss: float | None = None,
        delta_val_loss: float | None = None,
    ) -> None:
        self._eval_steps.append(step)
        self._val_loss.append(val_loss)
        val_ppl = self._perplexity(val_loss)
        self._val_ppl.append(val_ppl)

        self._append_jsonl({
            "type": "eval",
            "step": step,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "accuracy": accuracy,
            "validation_runtime_sec": runtime_sec,
            "best_val_loss": best_val_loss,
            "delta_val_loss": delta_val_loss,
        })

    def update_plots(self) -> None:
        if not self.enabled:
            return
        if len(self._steps) < 2:
            return

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))

        self._plot_train_loss(axes[0, 0])
        self._plot_train_perplexity(axes[0, 1])
        self._plot_lr_schedule(axes[0, 2])
        self._plot_grad_norm(axes[1, 0])
        self._plot_val_loss(axes[1, 1])
        self._plot_training_speed(axes[1, 2])
        self._plot_generalization_gap(axes[2, 0])
        self._plot_val_perplexity(axes[2, 1])
        self._plot_cumulative_tokens(axes[2, 2])

        fig.suptitle("PegasusGPT Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(self.run_dir / "training_dashboard.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        self._save_individual_plots()

    def _plot_train_loss(self, ax: plt.Axes) -> None:
        ax.plot(self._steps, self._train_loss, color="#2471a3", linewidth=1.5, alpha=0.7, label="Train Loss")
        smoothed = self._ema(self._train_loss)
        ax.plot(self._steps, smoothed, color="#1a5276", linewidth=2, label="Smoothed")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Loss")
        ax.set_title("Train Loss over Steps", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def _plot_train_perplexity(self, ax: plt.Axes) -> None:
        ax.plot(self._steps, self._train_ppl, color="#2471a3", linewidth=1.5, alpha=0.7, label="Train PPL")
        smoothed = self._ema(self._train_ppl)
        ax.plot(self._steps, smoothed, color="#1a5276", linewidth=2, label="Smoothed")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Training Perplexity over Steps", fontweight="bold")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def _plot_lr_schedule(self, ax: plt.Axes) -> None:
        ax.plot(self._steps, self._learning_rate, color="#27ae60", linewidth=2)
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule", fontweight="bold")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, -3))
        ax.grid(alpha=0.3)

    def _plot_grad_norm(self, ax: plt.Axes) -> None:
        ax.plot(self._steps, self._grad_norm, color="#e67e22", linewidth=1, alpha=0.6, label="Grad Norm")
        smoothed = self._ema(self._grad_norm)
        ax.plot(self._steps, smoothed, color="#d35400", linewidth=2, label="Smoothed")
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, label="Clip threshold")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_title("Gradient Norm over Steps", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def _plot_val_loss(self, ax: plt.Axes) -> None:
        if self._eval_steps:
            ax.plot(self._eval_steps, self._val_loss, color="#c0392b", linewidth=2, marker="o", markersize=4, label="Val Loss")
            ax.set_xlabel("Optimizer Step")
            ax.set_ylabel("Validation Loss")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No validation data yet", ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("Validation Loss per Eval Step", fontweight="bold")
        ax.grid(alpha=0.3)

    def _plot_training_speed(self, ax: plt.Axes) -> None:
        ax.plot(self._steps, self._tokens_per_sec, color="#8e44ad", linewidth=1, alpha=0.6, label="tok/s")
        smoothed = self._ema(self._tokens_per_sec)
        ax.plot(self._steps, smoothed, color="#6c3483", linewidth=2, label="Smoothed")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Tokens / second")
        ax.set_title("Training Speed", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def _plot_generalization_gap(self, ax: plt.Axes) -> None:
        if self._eval_steps:
            gap = []
            for eval_step, val_loss in zip(self._eval_steps, self._val_loss):
                closest_idx = self._find_closest_step_idx(eval_step)
                if closest_idx is not None:
                    gap.append(self._train_loss[closest_idx] - val_loss)
                else:
                    gap.append(0.0)
            ax.axhspan(-0.15, 0.15, alpha=0.1, color="green", label="Ideal zone", zorder=0)
            ax.plot(self._eval_steps, gap, color="#27ae60", linewidth=2, marker="o", markersize=4, label="Train - Val")
            ax.axhline(0, color="black", linestyle="--", alpha=0.4)
            ax.set_xlabel("Optimizer Step")
            ax.set_ylabel("Train Loss − Val Loss")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No validation data yet", ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("Generalization Gap", fontweight="bold")
        ax.grid(alpha=0.3)

    def _plot_val_perplexity(self, ax: plt.Axes) -> None:
        if self._eval_steps:
            ax.plot(self._eval_steps, self._val_ppl, color="#c0392b", linewidth=2, marker="o", markersize=4, label="Val PPL")
            ax.set_xlabel("Optimizer Step")
            ax.set_ylabel("Perplexity")
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No validation data yet", ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("Validation Perplexity", fontweight="bold")
        ax.grid(alpha=0.3)

    def _plot_cumulative_tokens(self, ax: plt.Axes) -> None:
        if self._wall_time and self._tokens_per_sec:
            cumulative = []
            total = 0.0
            prev_time = 0.0
            for wt, tps in zip(self._wall_time, self._tokens_per_sec):
                dt = wt - prev_time
                total += tps * dt
                cumulative.append(total / 1e6)
                prev_time = wt
            ax.plot(self._steps, cumulative, color="#2980b9", linewidth=2)
            ax.set_xlabel("Step")
            ax.set_ylabel("Cumulative Tokens (M)")
            ax.set_title("Total Tokens Processed", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        ax.grid(alpha=0.3)

    def _save_individual_plots(self) -> None:
        plots = [
            ("train_loss.png", self._plot_train_loss),
            ("train_perplexity.png", self._plot_train_perplexity),
            ("lr_schedule.png", self._plot_lr_schedule),
            ("grad_norm.png", self._plot_grad_norm),
            ("val_loss.png", self._plot_val_loss),
            ("training_speed.png", self._plot_training_speed),
            ("generalization_gap.png", self._plot_generalization_gap),
            ("val_perplexity.png", self._plot_val_perplexity),
        ]
        for filename, plot_fn in plots:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_fn(ax)
            plt.tight_layout()
            fig.savefig(self.run_dir / filename, dpi=130, bbox_inches="tight")
            plt.close(fig)

    def _find_closest_step_idx(self, target_step: int) -> int | None:
        if not self._steps:
            return None
        best_idx = 0
        best_dist = abs(self._steps[0] - target_step)
        for i, s in enumerate(self._steps):
            dist = abs(s - target_step)
            if dist <= best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    @staticmethod
    def _ema(values: list[float], alpha: float = 0.15) -> list[float]:
        if not values:
            return []
        smoothed = [values[0]]
        for v in values[1:]:
            if math.isnan(v):
                smoothed.append(smoothed[-1])
            else:
                smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
        return smoothed

    @staticmethod
    def _perplexity(loss: float) -> float:
        if not math.isfinite(loss):
            return float("inf")
        return math.exp(min(loss, math.log(sys.float_info.max)))
