"""Tokenizes Alpaca instruction examples into SFT tensors with masked labels."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from domain.sft.alpaca_template import format_alpaca_example


IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class AlpacaSFTPreparationResult:
    train_dataset: Dataset
    validation_dataset: Dataset
    train_rows: int
    validation_rows: int
    vocab_size: int
    skipped_missing_fields: int
    skipped_empty_output: int
    skipped_too_short: int
    truncated_examples: int
    total_response_tokens_train: int
    total_response_tokens_validation: int


class AlpacaSFTPreparationService:
    """Builds SFT train/validation datasets from Alpaca-style instruction data."""

    def prepare(
        self,
        *,
        dataset_name: str | None,
        local_json_path: str | None,
        validation_fraction: float,
        seed: int,
        max_examples: int | None,
        sequence_length: int,
        min_response_tokens: int,
        tokenizer_name: str,
        eos_token_id: int,
        template_name: str,
    ) -> AlpacaSFTPreparationResult:
        if template_name != "alpaca":
            raise ValueError(f"Unsupported template_name='{template_name}' (only 'alpaca' is supported)")
        if not (0.0 <= validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in [0, 1)")
        if sequence_length <= 1:
            raise ValueError("sequence_length must be > 1")
        if min_response_tokens < 1:
            raise ValueError("min_response_tokens must be >= 1")

        raw_examples = self._load_raw_examples(
            dataset_name=dataset_name,
            local_json_path=local_json_path,
        )
        rng = random.Random(seed)
        raw_examples = list(raw_examples)
        rng.shuffle(raw_examples)
        if max_examples is not None:
            raw_examples = raw_examples[:max_examples]

        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = int(1e30)
        vocab_size = int(tokenizer.vocab_size)
        eos_token_text = tokenizer.decode([eos_token_id])

        prepared_rows: list[dict[str, Any]] = []
        skipped_missing_fields = 0
        skipped_empty_output = 0
        skipped_too_short = 0
        truncated_examples = 0
        max_input_length = sequence_length

        for example in tqdm(raw_examples, desc="preparing alpaca sft", unit="example"):
            instruction = example.get("instruction")
            input_text = example.get("input")
            output = example.get("output")

            if not isinstance(instruction, str) or not isinstance(output, str):
                skipped_missing_fields += 1
                continue
            if output.strip() == "":
                skipped_empty_output += 1
                continue

            prompt_text, response_text = format_alpaca_example(
                instruction=instruction,
                input_text=input_text if isinstance(input_text, str) else None,
                output=output,
                eos_token=eos_token_text,
            )

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            response_ids = tokenizer.encode(response_text, add_special_tokens=False)

            if len(prompt_ids) < 1 or len(response_ids) < min_response_tokens:
                skipped_too_short += 1
                continue

            full_ids = prompt_ids + response_ids

            was_truncated = False
            if len(full_ids) > sequence_length + 1:
                full_ids = full_ids[: sequence_length + 1]
                was_truncated = True

            input_ids = full_ids[:-1]
            labels = list(full_ids[1:])

            mask_until = min(len(prompt_ids) - 1, len(labels))
            for i in range(mask_until):
                labels[i] = IGNORE_INDEX

            valid_label_count = sum(1 for label in labels if label != IGNORE_INDEX)
            if valid_label_count < min_response_tokens:
                skipped_too_short += 1
                continue

            if was_truncated:
                truncated_examples += 1

            if len(input_ids) > max_input_length:
                raise AssertionError(
                    f"Internal: input_ids length {len(input_ids)} exceeds sequence_length {sequence_length}"
                )

            prepared_rows.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "prompt_text": prompt_text,
                    "response_text": response_text,
                    "valid_label_count": valid_label_count,
                }
            )

        if not prepared_rows:
            raise RuntimeError("No valid SFT examples were produced from the input data.")

        rng.shuffle(prepared_rows)
        validation_count = int(len(prepared_rows) * validation_fraction)
        if validation_fraction > 0.0 and validation_count == 0 and len(prepared_rows) >= 2:
            validation_count = 1
        validation_rows = prepared_rows[:validation_count]
        train_rows = prepared_rows[validation_count:]

        if not train_rows:
            raise RuntimeError("Validation fraction consumed all examples; no training rows left.")

        train_dataset = Dataset.from_list(train_rows)
        validation_dataset = Dataset.from_list(validation_rows) if validation_rows else Dataset.from_dict(
            {
                "input_ids": [],
                "labels": [],
                "prompt_text": [],
                "response_text": [],
                "valid_label_count": [],
            }
        )

        total_response_tokens_train = sum(int(row["valid_label_count"]) for row in train_rows)
        total_response_tokens_validation = sum(int(row["valid_label_count"]) for row in validation_rows)

        return AlpacaSFTPreparationResult(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_rows=len(train_rows),
            validation_rows=len(validation_rows),
            vocab_size=vocab_size,
            skipped_missing_fields=skipped_missing_fields,
            skipped_empty_output=skipped_empty_output,
            skipped_too_short=skipped_too_short,
            truncated_examples=truncated_examples,
            total_response_tokens_train=total_response_tokens_train,
            total_response_tokens_validation=total_response_tokens_validation,
        )

    def _load_raw_examples(
        self,
        *,
        dataset_name: str | None,
        local_json_path: str | None,
    ) -> Iterable[dict[str, Any]]:
        if local_json_path:
            path = Path(local_json_path)
            if not path.exists():
                raise FileNotFoundError(f"local_json_path does not exist: {local_json_path}")
            with path.open("r", encoding="utf-8") as json_file:
                data = json.load(json_file)
            if not isinstance(data, list):
                raise ValueError(f"Local Alpaca JSON must be a list of examples (got {type(data).__name__})")
            return data

        if not dataset_name:
            raise ValueError("Either dataset_name or local_json_path must be provided")

        dataset = load_dataset(dataset_name, split="train")
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Loaded Alpaca dataset from '{dataset_name}' is not a Dataset")
        return (dict(row) for row in dataset)
