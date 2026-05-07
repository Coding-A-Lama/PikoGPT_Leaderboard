import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageExperimentLogger:
    """Persists stage-level experiment artifacts to a local directory."""

    def __init__(
        self,
        experiment_dir: str,
        config: dict[str, Any],
        stage_name: str,
        enabled: bool = True,
        log_level: str = "INFO",
        local_run_dir: str | Path | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.enabled = enabled
        self.experiment_dir = Path(experiment_dir)
        self.local_run_dir = Path(local_run_dir) if local_run_dir is not None else None
        self.log_level = self._parse_log_level(log_level)

        self.config_path = self.experiment_dir / "config.json"
        self.log_path = self.experiment_dir / "train.log"
        self.metrics_path = self.experiment_dir / "metrics.jsonl"
        self.model_path = self.experiment_dir / "model.pt"

        if self.enabled:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            if self.local_run_dir is not None:
                self.local_run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config(config)
            self._setup_logger()
            self._ensure_model_artifact()
        else:
            self.logger = None

    def _write_config(self, config: dict[str, Any]) -> None:
        with self.config_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)
        if self.local_run_dir is not None:
            with (self.local_run_dir / "config.json").open("w", encoding="utf-8") as config_file:
                json.dump(config, config_file, indent=2, ensure_ascii=False)

    def _setup_logger(self) -> None:
        logger_name = f"stage.{self.stage_name}.{self.experiment_dir.name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self.log_level)
        self.logger.handlers = []

        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(file_handler)

        if self.local_run_dir is not None:
            run_file_handler = logging.FileHandler(self.local_run_dir / "train.log", encoding="utf-8")
            run_file_handler.setLevel(self.log_level)
            run_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(run_file_handler)

        self.logger.propagate = False

    def _ensure_model_artifact(self) -> None:
        self.model_path.touch(exist_ok=True)

    @staticmethod
    def _parse_log_level(log_level: str) -> int:
        normalized = log_level.upper()
        level = getattr(logging, normalized, None)
        if not isinstance(level, int):
            raise ValueError(f"Unsupported log_level='{log_level}'")
        return level

    def debug(self, message: str) -> None:
        if self.logger is not None:
            self.logger.debug(message)

    def info(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def warning(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)

    def error(self, message: str) -> None:
        if self.logger is not None:
            self.logger.error(message)

    def critical(self, message: str) -> None:
        if self.logger is not None:
            self.logger.critical(message)

    def write_metrics(self, step: str, values: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = {
            "step": step,
            "stage": self.stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        with self.metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(f"{json.dumps(record, ensure_ascii=False)}\n")

    def write_event(self, event_type: str, level: str = "INFO", **values: Any) -> None:
        if not self.enabled or self.local_run_dir is None:
            return
        record = {
            "type": event_type,
            "level": level.upper(),
            "stage": self.stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        with (self.local_run_dir / "events.jsonl").open("a", encoding="utf-8") as events_file:
            events_file.write(f"{json.dumps(record, ensure_ascii=False)}\n")
