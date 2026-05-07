from __future__ import annotations

from bisect import bisect_right

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from domain.training.memmap_token_dataset import MemmapTokenDataset


class ContinuousWindowDataset(Dataset):
    """Dataset over tokenized documents using a virtual continuous token stream."""

    def __init__(
        self,
        documents: list[list[int]] | MemmapTokenDataset,
        seq_len: int,
        stride: int | None = None,
        pad_token_id: int = 50256,
        ignore_index: int = -100,
        drop_last_document_window: bool = True,
        eos_token_id: int = 50256,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")

        self.seq_len = seq_len
        self.window_len = seq_len + 1
        self.stride = stride if stride is not None else seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.drop_last = drop_last_document_window
        self.eos_token_id = eos_token_id
        self._memmap_documents = documents if isinstance(documents, MemmapTokenDataset) else None
        self._list_documents = documents if isinstance(documents, list) else None

        if self.stride <= 0:
            raise ValueError("stride must be > 0")

        self.document_lengths = self._document_lengths()
        self._stream_starts = self._build_stream_starts()
        self.total_stream_tokens = self._total_stream_tokens()
        self._full_window_count = self._compute_full_window_count()
        self._sample_count = self._compute_sample_count()

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("sample index is out of bounds")

        start = self._sample_start(idx)
        window = self._slice_virtual_stream(start, self.window_len)
        window_content_len = len(window)

        if window_content_len < 2:
            raise ValueError(f"Window at idx={idx} is too short for next-token prediction.")

        if window_content_len < self.window_len:
            window = window + [self.pad_token_id] * (self.window_len - window_content_len)

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

    def _document_lengths(self) -> list[int]:
        if self._memmap_documents is not None:
            return [int(length) for length in self._memmap_documents.document_lengths.tolist()]
        return [len(doc) for doc in self._list_documents or []]

    def _build_stream_starts(self) -> list[int]:
        stream_starts: list[int] = []
        stream_position = 0
        for doc_index, doc_len in enumerate(self.document_lengths):
            stream_starts.append(stream_position)
            stream_position += doc_len
            if doc_index < len(self.document_lengths) - 1:
                stream_position += 1
        return stream_starts

    def _total_stream_tokens(self) -> int:
        if not self.document_lengths:
            return 0
        return sum(self.document_lengths) + max(len(self.document_lengths) - 1, 0)

    def _compute_full_window_count(self) -> int:
        if self.total_stream_tokens < self.window_len:
            return 0
        return 1 + (self.total_stream_tokens - self.window_len) // self.stride

    def _compute_sample_count(self) -> int:
        if self.total_stream_tokens < 2:
            return 0
        if self.total_stream_tokens < self.window_len:
            return 0 if self.drop_last else 1

        sample_count = self._full_window_count
        if not self.drop_last:
            next_start = self._full_window_count * self.stride
            if next_start < self.total_stream_tokens - 1:
                sample_count += 1
        return sample_count

    def _sample_start(self, idx: int) -> int:
        if idx < self._full_window_count:
            return idx * self.stride
        return self._full_window_count * self.stride

    def _slice_virtual_stream(self, start: int, length: int) -> list[int]:
        tokens: list[int] = []
        position = start
        remaining = length

        while remaining > 0 and position < self.total_stream_tokens:
            doc_index = self._find_doc_index(position)
            doc_start = self._stream_starts[doc_index]
            doc_len = self.document_lengths[doc_index]
            doc_end = doc_start + doc_len
            has_separator = doc_index < len(self.document_lengths) - 1
            separator_end = doc_end + (1 if has_separator else 0)

            if position < doc_end:
                take_count = min(doc_end - position, remaining)
                doc_offset = position - doc_start
                tokens.extend(self._document_slice(doc_idx=doc_index, start=doc_offset, length=take_count))
                position += take_count
                remaining -= take_count
                continue

            if has_separator and position < separator_end:
                tokens.append(self.eos_token_id)
                position += 1
                remaining -= 1
                continue

            position = separator_end

        return tokens

    def _document_slice(self, doc_idx: int, start: int, length: int) -> list[int]:
        if self._memmap_documents is not None:
            return self._memmap_documents.document_slice(doc_idx=doc_idx, start=start, length=length).tolist()
        document = self._list_documents[doc_idx]
        return document[start: start + length]

    def _find_doc_index(self, position: int) -> int:
        doc_index = bisect_right(self._stream_starts, position) - 1
        return max(doc_index, 0)
