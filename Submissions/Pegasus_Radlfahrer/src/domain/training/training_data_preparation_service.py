from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from domain.training.memmap_token_dataset import MemmapTokenDataset
from domain.training.training_data_preparation_policy import (
    DocumentTrainingDataPreparationPolicy,
    TrainingDataPreparationPolicy,
)


@dataclass(frozen=True, slots=True)
class PreparedTrainingDataset:
    dataset_path: str
    token_column: str
    source_rows: int
    usable_rows: int
    total_tokens: int
    tokenized_examples: MemmapTokenDataset


@dataclass(slots=True)
class TrainingDataPreparationService:
    """Loads tokenized datasets and prepares a disk-backed token cache for training."""

    policy: TrainingDataPreparationPolicy = field(default_factory=DocumentTrainingDataPreparationPolicy)
    cache_root: str = "data/training/memmap_cache"
    cache_version: int = 1

    def prepare_dataset(
        self,
        tokenized_dataset_path: str,
        token_column: str,
        sequence_length: int,
    ) -> PreparedTrainingDataset:
        dataset = load_from_disk(tokenized_dataset_path)
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Dataset at {tokenized_dataset_path} is not a Dataset")
        if token_column not in dataset.column_names:
            raise ValueError(
                f"Token column '{token_column}' not found in dataset. Available columns: {dataset.column_names}"
            )
        if dataset.num_rows == 0:
            raise ValueError("Training dataset is empty")

        cache_dir = self._cache_dir_for_dataset(
            tokenized_dataset_path=tokenized_dataset_path,
            token_column=token_column,
            sequence_length=sequence_length,
        )
        expected_metadata = self._build_cache_metadata(
            dataset=dataset,
            tokenized_dataset_path=tokenized_dataset_path,
            token_column=token_column,
            sequence_length=sequence_length,
        )
        cache_dir, metadata = self._ensure_memmap_cache(
            dataset=dataset,
            cache_dir=cache_dir,
            expected_metadata=expected_metadata,
        )

        usable_rows = int(metadata["usable_rows"])
        total_tokens = int(metadata["total_tokens"])
        self.policy.validate_prepared_dataset(
            usable_rows=usable_rows,
            total_tokens=total_tokens,
            sequence_length=sequence_length,
        )

        document_offsets = np.load(cache_dir / "document_offsets.npy")
        document_lengths = np.load(cache_dir / "document_lengths.npy")
        token_dataset = MemmapTokenDataset(
            token_file_path=str(cache_dir / "tokens.bin"),
            document_offsets=document_offsets,
            document_lengths=document_lengths,
        )

        return PreparedTrainingDataset(
            dataset_path=tokenized_dataset_path,
            token_column=token_column,
            source_rows=dataset.num_rows,
            usable_rows=usable_rows,
            total_tokens=total_tokens,
            tokenized_examples=token_dataset,
        )

    def _cache_dir_for_dataset(
        self,
        tokenized_dataset_path: str,
        token_column: str,
        sequence_length: int,
    ) -> Path:
        dataset_key = self._dataset_cache_key(tokenized_dataset_path)
        config_key = self._config_cache_key(
            token_column=token_column,
            sequence_length=sequence_length,
        )
        return Path(self.cache_root) / dataset_key / config_key

    def _dataset_cache_key(self, tokenized_dataset_path: str) -> str:
        return hashlib.sha256(
            str(Path(tokenized_dataset_path).resolve()).encode("utf-8")
        ).hexdigest()[:16]

    def _config_cache_key(
        self,
        token_column: str,
        sequence_length: int,
    ) -> str:
        config_key_source = json.dumps(
            {
                "cache_version": self.cache_version,
                "token_column": token_column,
                "sequence_length": int(sequence_length),
                "policy_name": type(self.policy).__name__,
                "policy_config": self._stable_policy_config(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(config_key_source.encode("utf-8")).hexdigest()[:16]

    def _stable_policy_config(self) -> dict[str, object] | str:
        if is_dataclass(self.policy):
            return asdict(self.policy)
        return type(self.policy).__name__

    def _ensure_memmap_cache(
        self,
        dataset: Dataset,
        cache_dir: Path,
        expected_metadata: dict[str, object],
    ) -> tuple[Path, dict[str, object]]:
        for candidate_cache_dir in self._candidate_cache_dirs(
            primary_cache_dir=cache_dir,
            expected_metadata=expected_metadata,
        ):
            metadata_path = candidate_cache_dir / "metadata.json"
            tokens_path = candidate_cache_dir / "tokens.bin"
            offsets_path = candidate_cache_dir / "document_offsets.npy"
            lengths_path = candidate_cache_dir / "document_lengths.npy"

            existing_metadata = self._load_valid_cache_metadata(
                metadata_path=metadata_path,
                tokens_path=tokens_path,
                offsets_path=offsets_path,
                lengths_path=lengths_path,
                expected_metadata=expected_metadata,
            )
            if existing_metadata is not None:
                return candidate_cache_dir, existing_metadata

        cache_dir.mkdir(parents=True, exist_ok=True)

        usable_rows = 0
        total_tokens = 0
        document_offsets: list[int] = []
        document_lengths: list[int] = []
        temp_tokens_path = cache_dir / "tokens.bin.tmp"
        temp_offsets_path = cache_dir / "document_offsets.tmp.npy"
        temp_lengths_path = cache_dir / "document_lengths.tmp.npy"
        temp_metadata_path = cache_dir / "metadata.json.tmp"
        token_column = str(expected_metadata["token_column"])
        sequence_length = int(expected_metadata["sequence_length"])

        with temp_tokens_path.open("wb") as token_file:
            progress = tqdm(
                dataset,
                total=dataset.num_rows,
                desc="building training memmap cache",
                unit="sample",
            )
            for row in progress:
                token_ids = row[token_column]
                if not isinstance(token_ids, list):
                    continue
                if any(not isinstance(token_id, int) for token_id in token_ids):
                    continue
                if not self.policy.should_keep(token_ids, sequence_length):
                    continue

                token_array = np.asarray(token_ids, dtype=np.int32)
                document_offsets.append(total_tokens)
                document_lengths.append(int(token_array.shape[0]))
                token_array.tofile(token_file)

                usable_rows += 1
                total_tokens += int(token_array.shape[0])

        np.save(temp_offsets_path, np.asarray(document_offsets, dtype=np.int64))
        np.save(temp_lengths_path, np.asarray(document_lengths, dtype=np.int32))

        metadata: dict[str, object] = {
            **expected_metadata,
            "usable_rows": usable_rows,
            "total_tokens": total_tokens,
        }
        temp_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        temp_tokens_path.replace(tokens_path)
        temp_offsets_path.replace(offsets_path)
        temp_lengths_path.replace(lengths_path)
        temp_metadata_path.replace(metadata_path)
        return cache_dir, metadata

    def _candidate_cache_dirs(
        self,
        primary_cache_dir: Path,
        expected_metadata: dict[str, object],
    ) -> list[Path]:
        candidates = [primary_cache_dir]
        cache_root = Path(self.cache_root)
        config_key = primary_cache_dir.name
        if not cache_root.exists():
            return candidates

        for candidate in cache_root.glob(f"*/{config_key}"):
            if candidate == primary_cache_dir:
                continue
            metadata_path = candidate / "metadata.json"
            if not self._is_same_dataset_cache_candidate(metadata_path, expected_metadata):
                continue
            candidates.append(candidate)
        return candidates

    def _is_same_dataset_cache_candidate(
        self,
        metadata_path: Path,
        expected_metadata: dict[str, object],
    ) -> bool:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        existing_path = metadata.get("dataset_path")
        expected_path = expected_metadata.get("dataset_path")
        if isinstance(existing_path, str) and isinstance(expected_path, str):
            try:
                if Path(existing_path).samefile(Path(expected_path)):
                    return True
            except OSError:
                pass
            if str(Path(existing_path).resolve()) == str(Path(expected_path).resolve()):
                return True

        existing_fingerprint = metadata.get("dataset_fingerprint")
        expected_fingerprint = expected_metadata.get("dataset_fingerprint")
        return (
            isinstance(existing_fingerprint, str)
            and isinstance(expected_fingerprint, str)
            and existing_fingerprint != ""
            and existing_fingerprint == expected_fingerprint
        )

    def _build_cache_metadata(
        self,
        dataset: Dataset,
        tokenized_dataset_path: str,
        token_column: str,
        sequence_length: int,
    ) -> dict[str, object]:
        return {
            "cache_version": self.cache_version,
            "dataset_path": str(Path(tokenized_dataset_path).resolve()),
            "dataset_fingerprint": str(getattr(dataset, "_fingerprint", "")),
            "token_column": token_column,
            "sequence_length": int(sequence_length),
            "policy_name": type(self.policy).__name__,
            "policy_config": self._stable_policy_config(),
            "source_rows": int(dataset.num_rows),
            "token_dtype": "int32",
        }

    def _load_valid_cache_metadata(
        self,
        metadata_path: Path,
        tokens_path: Path,
        offsets_path: Path,
        lengths_path: Path,
        expected_metadata: dict[str, object],
    ) -> dict[str, object] | None:
        required_paths = (metadata_path, tokens_path, offsets_path, lengths_path)
        if any(not path.exists() for path in required_paths):
            return None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        strict_keys = (
            "cache_version",
            "token_column",
            "sequence_length",
            "policy_name",
            "policy_config",
            "token_dtype",
        )
        for key in strict_keys:
            expected_value = expected_metadata[key]
            if metadata.get(key) != expected_value:
                return None

        usable_rows = metadata.get("usable_rows")
        total_tokens = metadata.get("total_tokens")
        if not isinstance(usable_rows, int) or usable_rows < 0:
            return None
        if not isinstance(total_tokens, int) or total_tokens < 0:
            return None

        try:
            document_offsets = np.load(offsets_path)
            document_lengths = np.load(lengths_path)
        except (OSError, ValueError):
            return None

        if document_offsets.shape != (usable_rows,):
            return None
        if document_lengths.shape != (usable_rows,):
            return None
        if int(document_lengths.sum(dtype=np.int64)) != total_tokens:
            return None

        expected_token_bytes = total_tokens * np.dtype(np.int32).itemsize
        try:
            actual_token_bytes = tokens_path.stat().st_size
        except OSError:
            return None
        if actual_token_bytes != expected_token_bytes:
            return None

        return metadata
