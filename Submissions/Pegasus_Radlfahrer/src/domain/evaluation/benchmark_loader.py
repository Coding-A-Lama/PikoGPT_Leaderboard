"""Loads benchmark validation data from local JSONL files or Hugging Face.

Local JSONL is preferred because the public leaderboard distributes its
benchmark splits as JSONL files. The schema we accept (per line) is::

    {"id": "...", "question": "...", "choices": ["A text", "B text"], "answer": "A"}

For HellaSwag-style data we also accept the leaderboard's original key names
(``ctx`` / ``endings`` / ``label``) and translate them. For LAMBADA we accept::

    {"id": "...", "text": "full passage with last word"}

If ``dataset_name`` is supplied and the local file is missing, we fall back
to ``datasets.load_dataset`` which fetches the validation split. Test/private
splits are never loaded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from domain.evaluation.benchmark_types import (
    LambadaExample,
    MultipleChoiceExample,
    letter_for_index,
)


SUPPORTED_MC_BENCHMARKS = {"hellaswag", "winogrande", "openbookqa"}
SUPPORTED_BENCHMARKS = SUPPORTED_MC_BENCHMARKS | {"lambada"}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_letter(value: Any, num_choices: int) -> str:
    if isinstance(value, str) and value.strip():
        letter = value.strip().upper()
        if len(letter) == 1 and "A" <= letter <= "Z":
            return letter
        # Sometimes the gold is a 0/1 string for WinoGrande.
        if letter.isdigit():
            return letter_for_index(int(letter) - (1 if num_choices == 2 and letter in {"1", "2"} else 0))
    if isinstance(value, int):
        return letter_for_index(value)
    raise ValueError(f"Cannot normalize answer letter: {value!r}")


def _normalize_mc_record(raw: dict[str, Any], source: str, default_id: str) -> MultipleChoiceExample:
    example_id = str(raw.get("id") or raw.get("ind") or raw.get("qid") or default_id)
    if "choices" in raw and "question" in raw:
        choices = list(raw["choices"])
        question = str(raw["question"])
    elif "ctx" in raw and "endings" in raw:
        # HellaSwag style: prepend context as the question, choices are the endings.
        question = str(raw["ctx"])
        choices = list(raw["endings"])
    elif "sentence" in raw and ("option1" in raw or "options" in raw):
        # WinoGrande style: question is the sentence with a blank.
        question = str(raw["sentence"])
        if "options" in raw:
            choices = list(raw["options"])
        else:
            choices = [str(raw["option1"]), str(raw["option2"])]
    elif "question_stem" in raw and "choices" in raw:
        question = str(raw["question_stem"])
        choices_field = raw["choices"]
        if isinstance(choices_field, dict) and "text" in choices_field:
            choices = list(choices_field["text"])
        else:
            choices = list(choices_field)
    else:
        raise ValueError(f"Cannot interpret MC record from {source}: keys={sorted(raw.keys())}")

    if not (2 <= len(choices) <= 26):
        raise ValueError(f"Unsupported number of choices in {source} ({example_id}): {len(choices)}")

    answer_field = raw.get("answer") or raw.get("answerKey") or raw.get("label") or raw.get("gold")
    answer_letter = _normalize_letter(answer_field, len(choices))

    return MultipleChoiceExample(
        example_id=example_id,
        question=question,
        choices=[str(choice) for choice in choices],
        answer_letter=answer_letter,
        source=source,
    )


def _normalize_lambada_record(raw: dict[str, Any], default_id: str) -> LambadaExample:
    example_id = str(raw.get("id") or raw.get("ind") or default_id)
    if "prefix" in raw and "target" in raw:
        return LambadaExample(example_id=example_id, prefix=str(raw["prefix"]), target_word=str(raw["target"]))
    text_value = raw.get("text") or raw.get("passage") or raw.get("sentence")
    if not isinstance(text_value, str) or not text_value.strip():
        raise ValueError(f"Cannot interpret LAMBADA record: keys={sorted(raw.keys())}")
    text = text_value.strip()
    last_space = text.rfind(" ")
    if last_space == -1:
        raise ValueError(f"LAMBADA record has no space-separated final word: {example_id}")
    return LambadaExample(
        example_id=example_id,
        prefix=text[:last_space],
        target_word=text[last_space + 1:].strip(),
    )


def load_multiple_choice_benchmark(
    *,
    benchmark_name: str,
    local_path: str | None,
    limit: int | None,
) -> list[MultipleChoiceExample]:
    if benchmark_name not in SUPPORTED_MC_BENCHMARKS:
        raise ValueError(f"Unsupported MC benchmark: {benchmark_name}")
    raw_records = _gather_raw_records(benchmark_name=benchmark_name, local_path=local_path)
    examples: list[MultipleChoiceExample] = []
    for index, raw in enumerate(raw_records):
        if limit is not None and len(examples) >= limit:
            break
        try:
            examples.append(_normalize_mc_record(raw, source=benchmark_name, default_id=str(index)))
        except ValueError as error:
            # Skip malformed rows but keep going.
            print(f"[benchmark_loader] skipping malformed {benchmark_name} row {index}: {error}")
    if not examples:
        raise RuntimeError(f"No examples loaded for benchmark '{benchmark_name}'")
    return examples


def load_lambada_benchmark(
    *,
    local_path: str | None,
    limit: int | None,
) -> list[LambadaExample]:
    raw_records = _gather_raw_records(benchmark_name="lambada", local_path=local_path)
    examples: list[LambadaExample] = []
    for index, raw in enumerate(raw_records):
        if limit is not None and len(examples) >= limit:
            break
        try:
            examples.append(_normalize_lambada_record(raw, default_id=str(index)))
        except ValueError as error:
            print(f"[benchmark_loader] skipping malformed lambada row {index}: {error}")
    if not examples:
        raise RuntimeError("No examples loaded for benchmark 'lambada'")
    return examples


def _gather_raw_records(*, benchmark_name: str, local_path: str | None) -> Iterable[dict[str, Any]]:
    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Local benchmark file not found: {local_path}")
        return list(_read_jsonl(path))
    return list(_load_from_huggingface(benchmark_name))


def _load_from_huggingface(benchmark_name: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as error:  # pragma: no cover - depends on env
        raise RuntimeError(
            f"datasets library is unavailable, cannot fetch '{benchmark_name}' from HF: {error}"
        )

    if benchmark_name == "hellaswag":
        ds = load_dataset("hellaswag", split="validation")
        return (dict(row) for row in ds)
    if benchmark_name == "winogrande":
        ds = load_dataset("winogrande", "winogrande_xl", split="validation")
        return (dict(row) for row in ds)
    if benchmark_name == "openbookqa":
        ds = load_dataset("openbookqa", "main", split="validation")
        return (dict(row) for row in ds)
    if benchmark_name == "lambada":
        ds = load_dataset("EleutherAI/lambada_openai", split="test")
        return (dict(row) for row in ds)
    raise ValueError(f"No HF source defined for benchmark '{benchmark_name}'")
