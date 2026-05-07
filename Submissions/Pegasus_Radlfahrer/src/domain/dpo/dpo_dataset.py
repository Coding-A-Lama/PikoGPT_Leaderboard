"""Dataset over prepared DPO preference pairs.

Each row contains tokenized full sequences (prompt + chosen / prompt + rejected)
together with label tensors that mask prompt tokens so only response tokens
contribute to the per-sequence log-probability used by the DPO loss.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


class DPODataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        sequence_length: int,
        pad_token_id: int,
        ignore_index: int = -100,
    ) -> None:
        self._rows = rows
        self._sequence_length = sequence_length
        self._pad_token_id = pad_token_id
        self._ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        row = self._rows[idx]
        chosen = self._pad_pair(
            input_ids=list(row["chosen_input_ids"]),
            labels=list(row["chosen_labels"]),
        )
        rejected = self._pad_pair(
            input_ids=list(row["rejected_input_ids"]),
            labels=list(row["rejected_labels"]),
        )
        return {
            "chosen_input_ids": chosen[0],
            "chosen_labels": chosen[1],
            "rejected_input_ids": rejected[0],
            "rejected_labels": rejected[1],
        }

    def _pad_pair(self, *, input_ids: list[int], labels: list[int]) -> tuple[Tensor, Tensor]:
        if len(input_ids) != len(labels):
            raise ValueError(f"DPO row mismatched input_ids/labels lengths: {len(input_ids)} vs {len(labels)}")
        if len(input_ids) > self._sequence_length:
            raise ValueError(
                f"DPO row exceeds sequence_length {self._sequence_length}: got {len(input_ids)}"
            )
        pad = self._sequence_length - len(input_ids)
        ids = input_ids + [self._pad_token_id] * pad
        labs = labels + [self._ignore_index] * pad
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labs, dtype=torch.long),
        )
