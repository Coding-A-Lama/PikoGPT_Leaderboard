from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class MemmapTokenDataset:
    token_file_path: str
    document_offsets: np.ndarray
    document_lengths: np.ndarray
    _tokens: np.memmap | None = field(default=None, init=False, repr=False)

    @property
    def document_count(self) -> int:
        return int(self.document_lengths.shape[0])

    @property
    def total_tokens(self) -> int:
        return int(self.document_lengths.sum(dtype=np.int64))

    def document_slice(self, doc_idx: int, start: int = 0, length: int | None = None) -> np.ndarray:
        doc_offset = int(self.document_offsets[doc_idx])
        doc_length = int(self.document_lengths[doc_idx])
        if start < 0 or start > doc_length:
            raise IndexError("document start offset is out of bounds")

        if length is None:
            end = doc_length
        else:
            end = min(start + length, doc_length)

        return self.tokens()[doc_offset + start: doc_offset + end]

    def tokens(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(Path(self.token_file_path), dtype=np.int32, mode="r")
        return self._tokens
