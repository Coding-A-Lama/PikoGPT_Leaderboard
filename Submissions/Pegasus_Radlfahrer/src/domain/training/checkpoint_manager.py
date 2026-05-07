from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch


class CheckpointLogger(Protocol):
    def info(self, message: str) -> None:
        ...


@dataclass(slots=True)
class TrainingCheckpointManager:
    checkpoint_dir: Path
    keep_latest_count: int
    best_checkpoint_filename: str
    logger: CheckpointLogger
    enabled: bool = True
    best_validation_loss: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.enabled:
            self.best_validation_loss = self._load_best_validation_loss()

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.best_checkpoint_filename

    def record_latest_checkpoint(self, checkpoint_path: Path) -> None:
        if not self.enabled:
            return

        self._prune_latest_checkpoints()

    def should_update_best(self, validation_loss: float) -> bool:
        if not self.enabled:
            return False

        return self.best_validation_loss is None or validation_loss < self.best_validation_loss

    def record_best_checkpoint(self, validation_loss: float) -> None:
        if not self.enabled:
            return

        self.best_validation_loss = validation_loss
        self.logger.info(
            f"Updated best training checkpoint at {self.best_checkpoint_path} with val_loss={validation_loss:.6f}"
        )

    def _prune_latest_checkpoints(self) -> None:
        checkpoint_paths = self._checkpoint_paths_descending()
        stale_paths = checkpoint_paths[self.keep_latest_count:]
        for checkpoint_path in stale_paths:
            self._delete_checkpoint(checkpoint_path)

    def _delete_checkpoint(self, checkpoint_path: Path) -> None:
        try:
            checkpoint_path.unlink()
            self.logger.info(f"Deleted stale training checkpoint {checkpoint_path}")
        except FileNotFoundError:
            return

    def _checkpoint_paths_descending(self) -> list[Path]:
        if not self.checkpoint_dir.exists():
            return []

        return sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=self._checkpoint_step_from_path,
            reverse=True,
        )

    def _checkpoint_step_from_path(self, checkpoint_path: Path) -> int:
        return int(checkpoint_path.stem.split("_")[-1])

    def _load_best_validation_loss(self) -> float | None:
        best_checkpoint_path = self.best_checkpoint_path
        if not best_checkpoint_path.exists():
            return None

        try:
            checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        except Exception as error:
            self.logger.info(
                f"Failed to inspect best training checkpoint at {best_checkpoint_path}: {error}. "
                "It will be replaced by the next validation improvement."
            )
            return None

        validation_loss = checkpoint.get("validation_loss")
        if validation_loss is None:
            return None

        return float(validation_loss)
