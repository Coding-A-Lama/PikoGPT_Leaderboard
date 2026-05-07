"""Builds DPO preference pairs from multiple-choice training data.

For each MC example we produce (prompt, chosen=correct_letter, rejected=wrong_letter)
where prompt uses the same Alpaca + benchmark template that evaluation/SFT use.
This way DPO directly optimizes the model's preference for the correct letter
over a wrong letter under the exact same prompt format.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import Dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from domain.evaluation.benchmark_types import letter_for_index
from domain.sft.alpaca_template import format_alpaca_response
from domain.sft.mc_templates import normalize_rows
from domain.evaluation.mc_prompt import build_mc_prompt, build_mc_prompt_bare


IGNORE_INDEX = -100


@dataclass(slots=True)
class DPOPreparationResult:
    train_dataset: Dataset
    validation_dataset: Dataset
    train_rows: int
    validation_rows: int
    counts_by_source: dict[str, int]
    skipped: int


class DPOPreparationService:
    def prepare(
        self,
        *,
        output_train_path: str,
        output_validation_path: str,
        sequence_length: int,
        validation_fraction: float,
        seed: int,
        tokenizer_name: str,
        eos_token_id: int,
        max_total_pairs: int,
        sources: list[tuple[str, str, str | None, Callable[[], Iterable[dict[str, Any]]]]],
        logger,
        bare_mc_fraction: float = 0.0,
    ) -> DPOPreparationResult:
        if not (0.0 <= validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in [0, 1)")
        if not (0.0 <= bare_mc_fraction <= 1.0):
            raise ValueError("bare_mc_fraction must be in [0, 1]")

        rng = random.Random(seed)
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = int(1e30)
        eos_token_text = tokenizer.decode([eos_token_id])

        prepared: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        skipped = 0
        per_source_target = max(1, max_total_pairs // max(1, len(sources)))

        for source_name, split_name, source_path, loader in sources:
            try:
                raw_rows = list(loader())
            except Exception as error:
                logger.warning(f"DPO source '{source_name}' failed to load: {error}; skipping")
                continue
            logger.info(
                f"Source audit: name={source_name} split={split_name} "
                f"rows={len(raw_rows)} path={source_path or 'hf'}"
            )
            normalized = normalize_rows(source_name, raw_rows)
            logger.info(
                f"Source audit: name={source_name} normalized_rows={len(normalized)}"
            )
            rng.shuffle(normalized)
            taken = 0
            for record in tqdm(normalized, desc=f"dpo:{source_name}", unit="ex"):
                if taken >= per_source_target:
                    break
                use_bare = bare_mc_fraction > 0.0 and rng.random() < bare_mc_fraction
                pair = self._build_pair(
                    record=record,
                    tokenizer=tokenizer,
                    sequence_length=sequence_length,
                    eos_token_text=eos_token_text,
                    rng=rng,
                    use_bare=use_bare,
                )
                if pair is None:
                    skipped += 1
                    continue
                pair["source"] = source_name
                pair["template_name"] = "bare" if use_bare else "alpaca"
                prepared.append(pair)
                counts[source_name] = counts.get(source_name, 0) + 1
                taken += 1

        if not prepared:
            raise RuntimeError("No DPO pairs were produced from the configured sources.")

        rng.shuffle(prepared)
        validation_count = int(len(prepared) * validation_fraction)
        if validation_fraction > 0.0 and validation_count == 0 and len(prepared) >= 2:
            validation_count = 1
        validation_rows = prepared[:validation_count]
        train_rows = prepared[validation_count:]
        if not train_rows:
            raise RuntimeError("Validation fraction consumed all DPO pairs; no training rows left.")

        train_dataset = Dataset.from_list(train_rows)
        validation_dataset = (
            Dataset.from_list(validation_rows)
            if validation_rows
            else Dataset.from_dict(self._empty_columns())
        )

        Path(output_train_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_validation_path).parent.mkdir(parents=True, exist_ok=True)
        train_dataset.save_to_disk(output_train_path)
        validation_dataset.save_to_disk(output_validation_path)

        return DPOPreparationResult(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_rows=len(train_rows),
            validation_rows=len(validation_rows),
            counts_by_source=counts,
            skipped=skipped,
        )

    @staticmethod
    def _empty_columns() -> dict[str, list]:
        return {
            "prompt_text": [],
            "chosen_text": [],
            "rejected_text": [],
            "chosen_input_ids": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_labels": [],
            "source": [],
            "template_name": [],
        }

    def _build_pair(
        self,
        *,
        record: dict[str, str],
        tokenizer: GPT2TokenizerFast,
        sequence_length: int,
        eos_token_text: str,
        rng: random.Random,
        use_bare: bool = False,
    ) -> dict[str, Any] | None:
        if use_bare:
            prompt_text = build_mc_prompt_bare(record["question"], record["choices"])
            response_prefix = " "  # leading space for clean BPE tokenization
        else:
            prompt_text = build_mc_prompt(record["question"], record["choices"])
            response_prefix = ""
        chosen_letter = record["answer_letter"]
        wrong_letters = [
            letter_for_index(i)
            for i in range(len(record["choices"]))
            if letter_for_index(i) != chosen_letter
        ]
        if not wrong_letters:
            return None
        rejected_letter = rng.choice(wrong_letters)

        chosen_text = format_alpaca_response(
            output=f"{response_prefix}{chosen_letter}", eos_token=eos_token_text
        )
        rejected_text = format_alpaca_response(
            output=f"{response_prefix}{rejected_letter}", eos_token=eos_token_text
        )

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        chosen_resp_ids = tokenizer.encode(chosen_text, add_special_tokens=False)
        rejected_resp_ids = tokenizer.encode(rejected_text, add_special_tokens=False)
        if not prompt_ids or not chosen_resp_ids or not rejected_resp_ids:
            return None

        chosen_pair = self._build_seq(prompt_ids, chosen_resp_ids, sequence_length)
        rejected_pair = self._build_seq(prompt_ids, rejected_resp_ids, sequence_length)
        if chosen_pair is None or rejected_pair is None:
            return None

        return {
            "prompt_text": prompt_text,
            "chosen_text": chosen_text,
            "rejected_text": rejected_text,
            "chosen_input_ids": chosen_pair[0],
            "chosen_labels": chosen_pair[1],
            "rejected_input_ids": rejected_pair[0],
            "rejected_labels": rejected_pair[1],
        }

    @staticmethod
    def _build_seq(
        prompt_ids: list[int], response_ids: list[int], sequence_length: int
    ) -> tuple[list[int], list[int]] | None:
        full_ids = prompt_ids + response_ids
        if len(full_ids) > sequence_length + 1:
            full_ids = full_ids[: sequence_length + 1]
        input_ids = full_ids[:-1]
        labels = list(full_ids[1:])
        mask_until = min(len(prompt_ids) - 1, len(labels))
        for i in range(mask_until):
            labels[i] = IGNORE_INDEX
        valid = sum(1 for label in labels if label != IGNORE_INDEX)
        if valid < 1:
            return None
        return input_ids, labels
