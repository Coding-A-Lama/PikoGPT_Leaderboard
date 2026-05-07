from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricsFileSummary:
    path: Path
    label: str
    modified_at: float
    size_bytes: int


class TrainingLogReader:
    """Reads training JSONL metrics without owning or mutating trainer state."""

    _RUN_DIRECTORY_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-UTC")
    _LIVE_RUN_SELECTION_WINDOW_SECONDS = 60.0

    def __init__(self, root: str | Path = "data/training") -> None:
        self.root = Path(root)

    def discover_metrics_files(self) -> list[MetricsFileSummary]:
        if not self.root.exists():
            return []

        summaries: list[MetricsFileSummary] = []
        for path in self.root.rglob("metrics.jsonl"):
            if not path.is_file():
                continue
            stat = path.stat()
            summaries.append(
                MetricsFileSummary(
                    path=path,
                    label=self._label_for(path),
                    modified_at=stat.st_mtime,
                    size_bytes=stat.st_size,
                )
            )
        return sorted(summaries, key=lambda item: item.modified_at, reverse=True)

    def select_metrics_file(self, requested_path: str | Path | None = None) -> Path | None:
        if requested_path is not None:
            path = Path(requested_path)
            return path if path.exists() else None

        discovered = self.discover_metrics_files()
        if not discovered:
            return None

        newest_mtime = discovered[0].modified_at
        recent_live_runs = [
            summary
            for summary in discovered
            if self._is_timestamped_run_metrics(summary.path)
            and newest_mtime - summary.modified_at <= self._LIVE_RUN_SELECTION_WINDOW_SECONDS
        ]
        if recent_live_runs:
            return recent_live_runs[0].path
        return discovered[0].path

    def read_dashboard_payload(self, metrics_path: str | Path) -> dict[str, Any]:
        path = Path(metrics_path)
        records, skipped_lines = self._read_jsonl(path)
        normalized = [
            record
            for record in (
                self._normalize_record(item, index=index)
                for index, item in enumerate(records, start=1)
            )
            if record is not None
        ]

        return {
            "source": {
                "path": str(path),
                "label": self._label_for(path),
                "exists": path.exists(),
                "modified_at": path.stat().st_mtime if path.exists() else None,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "record_count": len(normalized),
                "skipped_lines": skipped_lines,
            },
            "latest": self._latest_values(normalized),
            "series": {
                "train_loss": self._series(normalized, "train_loss"),
                "train_perplexity": self._series(normalized, "train_perplexity"),
                "val_loss": self._series(normalized, "val_loss"),
                "val_perplexity": self._series(normalized, "val_perplexity"),
                "learning_rate": self._series(normalized, "learning_rate"),
                "grad_norm": self._series(normalized, "grad_norm"),
                "tokens_per_sec": self._series(normalized, "tokens_per_sec"),
                "clip_ratio": self._series(normalized, "clip_ratio"),
                "gpu_memory_allocated_mb": self._series(normalized, "gpu_memory_allocated_mb"),
                "gpu_memory_peak_mb": self._series(normalized, "gpu_memory_peak_mb"),
                "ram_used_mb": self._series(normalized, "ram_used_mb"),
            },
            "records": normalized[-1000:],
        }

    def _read_jsonl(self, path: Path) -> tuple[list[dict[str, Any]], int]:
        if not path.exists():
            return [], 0

        records: list[dict[str, Any]] = []
        skipped_lines = 0
        with path.open("r", encoding="utf-8") as metrics_file:
            for line in metrics_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError:
                    skipped_lines += 1
                    continue
                if isinstance(decoded, dict):
                    records.append(decoded)
                else:
                    skipped_lines += 1
        return records, skipped_lines

    def _normalize_record(self, record: dict[str, Any], index: int) -> dict[str, Any] | None:
        x_value = self._x_value(record)
        if x_value is None:
            return None

        normalized: dict[str, Any] = {
            "x": x_value,
            "plot_x": index,
            "step": record.get("step"),
            "epoch": record.get("epoch"),
            "type": record.get("type") or record.get("stage") or "training",
            "timestamp": record.get("timestamp"),
        }
        for key in (
            "train_loss",
            "train_perplexity",
            "val_loss",
            "val_perplexity",
            "learning_rate",
            "grad_norm",
            "tokens_per_sec",
            "accuracy",
            "micro_step",
            "wall_time_sec",
            "steps_per_sec",
            "eta_seconds",
            "tokens_processed",
            "examples_processed",
            "batches_processed",
            "clip_ratio",
            "padding_rate",
            "gpu_memory_allocated_mb",
            "gpu_memory_reserved_mb",
            "gpu_memory_peak_mb",
            "ram_used_mb",
            "ram_available_mb",
        ):
            value = self._finite_float(record.get(key))
            if value is not None:
                normalized[key] = value

        if "train_loss" in normalized and "train_perplexity" not in normalized:
            normalized["train_perplexity"] = self._perplexity(normalized["train_loss"])
        if "val_loss" in normalized and "val_perplexity" not in normalized:
            normalized["val_perplexity"] = self._perplexity(normalized["val_loss"])

        return normalized

    def _x_value(self, record: dict[str, Any]) -> int | float | None:
        step = record.get("step")
        if isinstance(step, int | float):
            return step
        if isinstance(step, str) and step.isdigit():
            return int(step)

        micro_step = self._finite_float(record.get("micro_step"))
        if micro_step is not None:
            return micro_step

        epoch = self._finite_float(record.get("epoch"))
        if epoch is not None:
            return epoch

        return None

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    @staticmethod
    def _perplexity(loss: float) -> float:
        return math.exp(min(loss, math.log(sys.float_info.max)))

    @staticmethod
    def _series(records: list[dict[str, Any]], key: str) -> list[dict[str, float]]:
        return [
            {
                "x": record["plot_x"],
                "step": record["x"],
                "y": record[key],
            }
            for record in records
            if key in record
        ]

    @staticmethod
    def _latest_values(records: list[dict[str, Any]]) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        for record in records:
            for key, value in record.items():
                if key not in {"type", "step", "epoch", "plot_x"} and value is not None:
                    latest[key] = value
        return latest

    def _label_for(self, path: Path) -> str:
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            relative = path
        if relative.name == "metrics.jsonl":
            return str(relative.parent) if str(relative.parent) != "." else str(path.parent)
        return str(relative)

    def _is_timestamped_run_metrics(self, path: Path) -> bool:
        return path.name == "metrics.jsonl" and bool(self._RUN_DIRECTORY_PATTERN.fullmatch(path.parent.name))
