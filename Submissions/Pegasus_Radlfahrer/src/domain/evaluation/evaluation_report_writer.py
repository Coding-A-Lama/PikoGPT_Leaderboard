"""Persists per-benchmark JSON results plus an overview JSON for comparison."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _summary_to_dict(summary: Any) -> dict[str, Any]:
    return {
        key: (
            [asdict(item) for item in value]
            if isinstance(value, list) and value and hasattr(value[0], "__dataclass_fields__")
            else value
        )
        for key, value in asdict(summary).items()
    }


def write_benchmark_result(*, output_dir: Path, benchmark: str, mode: str, summary: Any) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{benchmark}_{mode}.json"
    payload = _summary_to_dict(summary)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return file_path


def write_overview(*, output_dir: Path, overview: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / "overview.json"
    with overview_path.open("w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)
    return overview_path
