from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch.utils.data import Dataset

from domain.training.continuous_window_dataset import ContinuousWindowDataset
from domain.training.document_window_dataset import DocumentWindowDataset
from domain.training.training_data_preparation_service import PreparedTrainingDataset


class WindowDatasetFactory(Protocol):
    def create(
        self,
        prepared_dataset: PreparedTrainingDataset,
        sequence_length: int,
        vocab_size: int,
    ) -> Dataset:
        ...


@dataclass(frozen=True, slots=True)
class DocumentWindowDatasetFactory:
    drop_last_document_window: bool = True
    pad_token_id: int = 50256

    def create(
        self,
        prepared_dataset: PreparedTrainingDataset,
        sequence_length: int,
        vocab_size: int,
    ) -> Dataset:
        if self.pad_token_id >= vocab_size:
            raise ValueError("pad_token_id must be smaller than vocab_size")

        return DocumentWindowDataset(
            prepared_dataset.tokenized_examples,
            sequence_length,
            pad_token_id=self.pad_token_id,
            drop_last_document_window=self.drop_last_document_window,
        )


@dataclass(frozen=True, slots=True)
class ContinuousWindowDatasetFactory:
    drop_last_document_window: bool = True
    eos_token_id: int = 50256

    def create(
        self,
        prepared_dataset: PreparedTrainingDataset,
        sequence_length: int,
        vocab_size: int,
    ) -> Dataset:
        if self.eos_token_id >= vocab_size:
            raise ValueError("eos_token_id must be smaller than vocab_size")

        return ContinuousWindowDataset(
            prepared_dataset.tokenized_examples,
            sequence_length,
            pad_token_id=self.eos_token_id,
            drop_last_document_window=self.drop_last_document_window,
            eos_token_id=self.eos_token_id,
        )
