import hashlib
import json
from pathlib import Path

from datasets import Dataset, load_from_disk


class TestSplitRemovalService:
    """Removes rows from a train split that are present in a test split."""

    def remove_test_split(self, train_dataset: Dataset, test_split_path: str) -> Dataset:
        test_path = Path(test_split_path)
        if not test_path.exists():
            raise FileNotFoundError(f"Test split path not found: {test_split_path}")

        test_split = load_from_disk(str(test_path))
        if not isinstance(test_split, Dataset):
            raise TypeError(f"Dataset at {test_split_path} is not a Dataset")

        comparable_columns = [
            column for column in train_dataset.column_names if column in test_split.column_names
        ]
        if not comparable_columns:
            raise ValueError(
                "No comparable columns between training dataset and test split; "
                "cannot remove test split entries"
            )

        test_row_hashes = {
            self._row_hash(row, comparable_columns)
            for row in test_split.select_columns(comparable_columns)
        }

        return train_dataset.filter(
            lambda row: self._row_hash(row, comparable_columns) not in test_row_hashes
        )

    @staticmethod
    def _row_hash(row: dict, comparable_columns: list[str]) -> str:
        normalized_row = {column: row[column] for column in comparable_columns}
        serialized_row = json.dumps(normalized_row, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(serialized_row.encode("utf-8")).hexdigest()
