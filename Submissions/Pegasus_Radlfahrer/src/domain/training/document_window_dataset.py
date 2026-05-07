from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from domain.training.memmap_token_dataset import MemmapTokenDataset


class DocumentWindowDataset(Dataset):
    """Dataset over tokenized documents with optional memmap-backed storage."""

    def __init__(
        self,
        documents: list[list[int]] | MemmapTokenDataset,
        seq_len: int,
        stride: int | None = None,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        drop_last_document_window: bool = True,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        self.seq_len = seq_len
        self.window_len = seq_len + 1
        self.stride = stride if stride is not None else seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.drop_last = drop_last_document_window
        self._memmap_documents = documents if isinstance(documents, MemmapTokenDataset) else None
        self._list_documents = documents if isinstance(documents, list) else None

        if self.stride <= 0:
            raise ValueError("stride must be > 0")

        if self._memmap_documents is not None:
            document_lengths = self._memmap_documents.document_lengths.astype(np.int64, copy=False)
        else:
            document_lengths = np.asarray([len(doc) for doc in self._list_documents or []], dtype=np.int64)

        self.document_lengths = document_lengths
        self._full_window_counts = np.asarray(
            [self._full_window_count(int(doc_len)) for doc_len in self.document_lengths],
            dtype=np.int64,
        )
        self._sample_counts = np.asarray(
            [self._sample_count(int(doc_len), int(full_count)) for doc_len, full_count in zip(self.document_lengths, self._full_window_counts)],
            dtype=np.int64,
        )
        self._cumulative_sample_counts = np.cumsum(self._sample_counts, dtype=np.int64)

    def __len__(self) -> int:
        if self._cumulative_sample_counts.size == 0:
            return 0
        return int(self._cumulative_sample_counts[-1])

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        doc_idx = self._find_doc_index(idx)
        doc_start_index = 0 if doc_idx == 0 else int(self._cumulative_sample_counts[doc_idx - 1])
        sample_idx_within_doc = idx - doc_start_index

        doc_len = int(self.document_lengths[doc_idx])
        full_window_count = int(self._full_window_counts[doc_idx])
        start = self._start_for_sample(doc_len=doc_len, full_window_count=full_window_count, sample_idx=sample_idx_within_doc)
        window = self._document_slice(doc_idx=doc_idx, start=start, length=self.window_len)
        window_content_len = len(window)

        if window_content_len < 2:
            raise ValueError(f"Window at idx={idx} is too short for next-token prediction.")

        if window_content_len < self.window_len:
            pad_amount = self.window_len - window_content_len
            window = window + [self.pad_token_id] * pad_amount

        window_t = torch.tensor(window, dtype=torch.long)
        input_ids = window_t[:-1].clone()
        labels = window_t[1:].clone()

        real_input_len = min(window_content_len - 1, self.seq_len)
        padding_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        padding_mask[:real_input_len] = True
        labels[~padding_mask] = self.ignore_index

        return {
            "input_ids": input_ids,
            "labels": labels,
            "padding_mask": padding_mask,
        }

    def _document_slice(self, doc_idx: int, start: int, length: int) -> list[int]:
        if self._memmap_documents is not None:
            return self._memmap_documents.document_slice(doc_idx=doc_idx, start=start, length=length).tolist()
        document = self._list_documents[doc_idx]
        return document[start: start + length]

    def _find_doc_index(self, sample_idx: int) -> int:
        if sample_idx < 0 or sample_idx >= len(self):
            raise IndexError("sample index is out of bounds")
        return int(np.searchsorted(self._cumulative_sample_counts, sample_idx, side="right"))

    def _full_window_count(self, doc_len: int) -> int:
        if doc_len < self.window_len:
            return 0
        return 1 + (doc_len - self.window_len) // self.stride

    def _sample_count(self, doc_len: int, full_window_count: int) -> int:
        if doc_len < self.window_len:
            return 1 if (not self.drop_last and doc_len >= 2) else 0

        sample_count = full_window_count
        if not self.drop_last:
            next_start = full_window_count * self.stride
            if next_start < doc_len - 1:
                sample_count += 1
        return sample_count

    def _start_for_sample(self, doc_len: int, full_window_count: int, sample_idx: int) -> int:
        if doc_len < self.window_len:
            return 0
        if sample_idx < full_window_count:
            return sample_idx * self.stride
        return full_window_count * self.stride
