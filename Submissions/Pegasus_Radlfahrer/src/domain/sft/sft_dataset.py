"""Dataset over pre-tokenized SFT examples with response-only labels.

The prepared Hugging Face Dataset is expected to contain, per row:
  - input_ids: list[int]   (already prompt + response, possibly shifted)
  - labels:    list[int]   (same length as input_ids, with -100 on prompt/padding)

Each example is padded up to `sequence_length`. Prompt tokens and padding tokens
are masked via `-100`; response tokens remain supervised.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        sequence_length: int,
        pad_token_id: int,
        vocab_size: int,
        ignore_index: int = -100,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")

        self._rows = rows
        self._sequence_length = sequence_length
        self._pad_token_id = pad_token_id
        self._vocab_size = vocab_size
        self._ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        row = self._rows[idx]
        input_ids_raw = list(row["input_ids"])
        labels_raw = list(row["labels"])

        if len(input_ids_raw) != len(labels_raw):
            raise ValueError(
                f"SFT row {idx}: input_ids length ({len(input_ids_raw)}) != labels length ({len(labels_raw)})"
            )
        if len(input_ids_raw) == 0:
            raise ValueError(f"SFT row {idx}: empty input_ids")
        if len(input_ids_raw) > self._sequence_length:
            raise ValueError(
                f"SFT row {idx}: length {len(input_ids_raw)} exceeds sequence_length {self._sequence_length}"
            )

        for token_id in input_ids_raw:
            if token_id < 0 or token_id >= self._vocab_size:
                raise ValueError(
                    f"SFT row {idx}: input_ids contains out-of-range token id {token_id} (vocab_size={self._vocab_size})"
                )
        for label_id in labels_raw:
            if label_id == self._ignore_index:
                continue
            if label_id < 0 or label_id >= self._vocab_size:
                raise ValueError(
                    f"SFT row {idx}: labels contains out-of-range token id {label_id} (vocab_size={self._vocab_size})"
                )

        real_length = len(input_ids_raw)
        pad_amount = self._sequence_length - real_length

        input_ids = input_ids_raw + [self._pad_token_id] * pad_amount
        labels = labels_raw + [self._ignore_index] * pad_amount

        valid_label_count = sum(1 for label in labels if label != self._ignore_index)
        if valid_label_count == 0:
            raise ValueError(f"SFT row {idx}: no valid response labels (all {self._ignore_index})")

        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        padding_mask = torch.zeros(self._sequence_length, dtype=torch.bool)
        padding_mask[:real_length] = True

        return {
            "input_ids": input_ids_t,
            "labels": labels_t,
            "padding_mask": padding_mask,
        }
