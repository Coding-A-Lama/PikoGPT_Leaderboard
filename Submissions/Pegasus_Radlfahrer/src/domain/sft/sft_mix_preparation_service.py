"""Builds a mixed SFT dataset: Alpaca + benchmark-style multiple-choice.

Output schema matches `prepare-alpaca-sft` exactly so the existing `sft` stage
can train on this dataset without modification.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import Dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from domain.sft.alpaca_template import format_alpaca_example
from domain.sft.mc_templates import (
    mc_record_to_alpaca_pair,
    mc_record_to_bare_pair,
    normalize_rows,
)


IGNORE_INDEX = -100


@dataclass(slots=True)
class SFTMixPreparationResult:
    train_dataset: Dataset
    validation_dataset: Dataset
    train_rows: int
    validation_rows: int
    vocab_size: int
    counts_by_source: dict[str, int]
    skipped: int
    truncated: int


@dataclass(slots=True)
class MixSourceSpec:
    name: str
    enabled: bool
    loader: Callable[[], Iterable[dict[str, Any]]]
    split: str = "train"
    path: str | None = None


class SFTMixPreparationService:
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
        max_total_examples: int,
        alpaca_fraction: float,
        multiple_choice_fraction: float,
        cloze_fraction: float,
        alpaca_loader: Callable[[], Iterable[dict[str, Any]]] | None,
        mc_sources: list[MixSourceSpec],
        cloze_loader: Callable[[], Iterable[dict[str, Any]]] | None,
        logger,
        bare_mc_fraction: float = 0.0,
    ) -> SFTMixPreparationResult:
        if not (0.0 <= validation_fraction < 1.0):
            raise ValueError("validation_fraction must be in [0, 1)")
        total_fraction = alpaca_fraction + multiple_choice_fraction + cloze_fraction
        if total_fraction <= 0:
            raise ValueError("At least one of alpaca/mc/cloze fractions must be > 0")
        if total_fraction > 1.001:
            raise ValueError(
                f"Sum of fractions ({total_fraction:.3f}) must be <= 1.0"
            )
        if not (0.0 <= bare_mc_fraction <= 1.0):
            raise ValueError("bare_mc_fraction must be in [0, 1]")

        rng = random.Random(seed)
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = int(1e30)
        eos_token_text = tokenizer.decode([eos_token_id])
        vocab_size = int(tokenizer.vocab_size)

        target_alpaca = int(max_total_examples * alpaca_fraction)
        target_mc = int(max_total_examples * multiple_choice_fraction)
        target_cloze = int(max_total_examples * cloze_fraction)

        prepared_rows: list[dict[str, Any]] = []
        counts_by_source: dict[str, int] = {}
        skipped = 0
        truncated = 0

        # 1) Alpaca slice
        if alpaca_loader is not None and target_alpaca > 0:
            logger.info(f"Loading Alpaca slice (target={target_alpaca})")
            alpaca_rows = list(alpaca_loader())
            logger.info(
                f"Source audit: name=alpaca split=train rows={len(alpaca_rows)} path=configured_alpaca_loader"
            )
            rng.shuffle(alpaca_rows)
            for raw in tqdm(alpaca_rows[:target_alpaca], desc="alpaca", unit="ex"):
                row, was_truncated = self._process_alpaca_example(
                    raw=raw,
                    tokenizer=tokenizer,
                    sequence_length=sequence_length,
                    eos_token_text=eos_token_text,
                )
                if row is None:
                    skipped += 1
                    continue
                if was_truncated:
                    truncated += 1
                row["source"] = "alpaca"
                row["task_type"] = "instruction"
                row["template_name"] = "alpaca"
                prepared_rows.append(row)
                counts_by_source["alpaca"] = counts_by_source.get("alpaca", 0) + 1

        # 2) MC slice (split equally across enabled sources)
        enabled_mc = [spec for spec in mc_sources if spec.enabled]
        if enabled_mc and target_mc > 0:
            per_source_target = max(1, target_mc // len(enabled_mc))
            for spec in enabled_mc:
                logger.info(f"Loading MC source '{spec.name}' (target={per_source_target})")
                try:
                    raw_rows = list(spec.loader())
                except Exception as error:
                    logger.warning(f"Failed to load MC source '{spec.name}': {error}; skipping")
                    continue
                logger.info(
                    f"Source audit: name={spec.name} split={spec.split} "
                    f"rows={len(raw_rows)} path={spec.path or 'hf'}"
                )
                normalized = normalize_rows(spec.name, raw_rows)
                logger.info(
                    f"Source audit: name={spec.name} normalized_rows={len(normalized)}"
                )
                rng.shuffle(normalized)
                taken = 0
                for record in tqdm(normalized, desc=f"mc:{spec.name}", unit="ex"):
                    if taken >= per_source_target:
                        break
                    use_bare = bare_mc_fraction > 0.0 and rng.random() < bare_mc_fraction
                    row, was_truncated = self._process_mc_example(
                        record=record,
                        tokenizer=tokenizer,
                        sequence_length=sequence_length,
                        eos_token_text=eos_token_text,
                        use_bare=use_bare,
                    )
                    if row is None:
                        skipped += 1
                        continue
                    if was_truncated:
                        truncated += 1
                    row["source"] = spec.name
                    row["task_type"] = "multiple_choice"
                    row["template_name"] = "bare" if use_bare else "alpaca"
                    prepared_rows.append(row)
                    counts_by_source[spec.name] = counts_by_source.get(spec.name, 0) + 1
                    taken += 1

        # 3) Cloze slice
        if cloze_loader is not None and target_cloze > 0:
            logger.info(f"Loading cloze slice (target={target_cloze})")
            try:
                cloze_rows = list(cloze_loader())
            except Exception as error:
                logger.warning(f"Failed to load cloze data: {error}; skipping")
                cloze_rows = []
            logger.info(f"Source audit: name=cloze split=train rows={len(cloze_rows)} path=local")
            rng.shuffle(cloze_rows)
            for raw in tqdm(cloze_rows[:target_cloze], desc="cloze", unit="ex"):
                row, was_truncated = self._process_cloze_example(
                    raw=raw,
                    tokenizer=tokenizer,
                    sequence_length=sequence_length,
                    eos_token_text=eos_token_text,
                )
                if row is None:
                    skipped += 1
                    continue
                if was_truncated:
                    truncated += 1
                row["source"] = "cloze"
                row["task_type"] = "cloze"
                row["template_name"] = "alpaca"
                prepared_rows.append(row)
                counts_by_source["cloze"] = counts_by_source.get("cloze", 0) + 1

        if not prepared_rows:
            raise RuntimeError("No SFT mix examples were produced from the configured sources")

        rng.shuffle(prepared_rows)
        validation_count = int(len(prepared_rows) * validation_fraction)
        if validation_fraction > 0.0 and validation_count == 0 and len(prepared_rows) >= 2:
            validation_count = 1
        validation_rows = prepared_rows[:validation_count]
        train_rows = prepared_rows[validation_count:]

        if not train_rows:
            raise RuntimeError("Validation fraction consumed all examples; no training rows left.")

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

        return SFTMixPreparationResult(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_rows=len(train_rows),
            validation_rows=len(validation_rows),
            vocab_size=vocab_size,
            counts_by_source=counts_by_source,
            skipped=skipped,
            truncated=truncated,
        )

    @staticmethod
    def _empty_columns() -> dict[str, list]:
        return {
            "input_ids": [],
            "labels": [],
            "prompt_text": [],
            "response_text": [],
            "valid_label_count": [],
            "source": [],
            "task_type": [],
            "template_name": [],
        }

    def _process_alpaca_example(
        self,
        *,
        raw: dict[str, Any],
        tokenizer: GPT2TokenizerFast,
        sequence_length: int,
        eos_token_text: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        instruction = raw.get("instruction")
        input_text = raw.get("input")
        output = raw.get("output")
        if not isinstance(instruction, str) or not isinstance(output, str) or output.strip() == "":
            return None, False
        prompt_text, response_text = format_alpaca_example(
            instruction=instruction,
            input_text=input_text if isinstance(input_text, str) else None,
            output=output,
            eos_token=eos_token_text,
        )
        return self._tokenize_pair(prompt_text, response_text, tokenizer, sequence_length)

    def _process_mc_example(
        self,
        *,
        record: dict[str, str],
        tokenizer: GPT2TokenizerFast,
        sequence_length: int,
        eos_token_text: str,
        use_bare: bool = False,
    ) -> tuple[dict[str, Any] | None, bool]:
        if use_bare:
            prompt_text, response_text = mc_record_to_bare_pair(record, eos_token_text)
        else:
            prompt_text, response_text = mc_record_to_alpaca_pair(record, eos_token_text)
        return self._tokenize_pair(prompt_text, response_text, tokenizer, sequence_length)

    def _process_cloze_example(
        self,
        *,
        raw: dict[str, Any],
        tokenizer: GPT2TokenizerFast,
        sequence_length: int,
        eos_token_text: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        prefix = raw.get("prefix")
        target = raw.get("target") or raw.get("answer") or raw.get("output")
        if not isinstance(prefix, str) or not isinstance(target, str):
            return None, False
        instruction = (
            "Predict the most likely next word.\n\n"
            f"Passage: {prefix.strip()}\n\nNext word:"
        )
        from domain.sft.alpaca_template import format_alpaca_prompt
        prompt_text = format_alpaca_prompt(instruction=instruction)
        response_text = f" {target.strip()}{eos_token_text}"
        return self._tokenize_pair(prompt_text, response_text, tokenizer, sequence_length)

    def _tokenize_pair(
        self,
        prompt_text: str,
        response_text: str,
        tokenizer: GPT2TokenizerFast,
        sequence_length: int,
    ) -> tuple[dict[str, Any] | None, bool]:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        if len(prompt_ids) < 1 or len(response_ids) < 1:
            return None, False

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
        if valid_label_count < 1:
            return None, was_truncated

        return (
            {
                "input_ids": input_ids,
                "labels": labels,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "valid_label_count": valid_label_count,
            },
            was_truncated,
        )


# ---------------------------------------------------------------------------
# Source loaders. They return iterables of raw dicts that the normalizers know
# how to handle. They support either an HF dataset name OR a local JSONL path.


def alpaca_loader(*, dataset_name: str | None, local_json_path: str | None) -> Iterable[dict[str, Any]]:
    if local_json_path:
        path = Path(local_json_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Local Alpaca JSON must be a list (got {type(data).__name__})")
        return data
    if not dataset_name:
        return []
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    return [dict(row) for row in ds]


def hf_or_local_loader(*, hf_loader: Callable[[], Iterable[dict[str, Any]]], local_path: str | None) -> Iterable[dict[str, Any]]:
    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Local source not found: {local_path}")
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return hf_loader()
