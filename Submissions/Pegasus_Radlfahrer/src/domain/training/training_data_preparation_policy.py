from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TrainingDataPreparationPolicy(Protocol):
    def should_keep(self, token_ids: list[int], sequence_length: int) -> bool:
        ...

    def validate_prepared_dataset(
        self,
        usable_rows: int,
        total_tokens: int,
        sequence_length: int,
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class DocumentTrainingDataPreparationPolicy:
    drop_last_document_window: bool = True

    def should_keep(self, token_ids: list[int], sequence_length: int) -> bool:
        minimum_length = sequence_length + 1 if self.drop_last_document_window else 2
        return len(token_ids) >= minimum_length

    def validate_prepared_dataset(
        self,
        usable_rows: int,
        total_tokens: int,
        sequence_length: int,
    ) -> None:
        if usable_rows == 0:
            raise ValueError(
                "No tokenized examples with enough length were found for configured sequence_length"
            )


@dataclass(frozen=True, slots=True)
class ContinuousTrainingDataPreparationPolicy:
    def should_keep(self, token_ids: list[int], sequence_length: int) -> bool:
        return bool(token_ids)

    def validate_prepared_dataset(
        self,
        usable_rows: int,
        total_tokens: int,
        sequence_length: int,
    ) -> None:
        total_stream_tokens = total_tokens + max(usable_rows - 1, 0)
        if total_stream_tokens < sequence_length + 1:
            raise ValueError(
                "No tokenized examples with enough total length were found for configured sequence_length"
            )
