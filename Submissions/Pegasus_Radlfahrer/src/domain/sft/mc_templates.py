"""Normalize multiple-choice rows from common HF datasets into a uniform schema.

Each loader returns a list of dicts with keys:
    {"question", "choices", "answer_letter", "source", "task_type"}

These are then converted to Alpaca-style SFT examples by `sft_mix_preparation_service`
using the same MC instruction header used at evaluation time, so format matches.
"""
from __future__ import annotations

from typing import Any, Iterable

from domain.evaluation.benchmark_types import letter_for_index
from domain.evaluation.mc_prompt import build_mc_prompt, build_mc_prompt_bare
from domain.sft.alpaca_template import format_alpaca_response


def _to_letter(value: Any, num_choices: int) -> str | None:
    if isinstance(value, str) and value.strip():
        letter = value.strip().upper()
        if len(letter) == 1 and "A" <= letter < chr(ord("A") + num_choices):
            return letter
        if letter.isdigit():
            idx = int(letter)
            if 0 <= idx < num_choices:
                return letter_for_index(idx)
            if 1 <= idx <= num_choices:
                return letter_for_index(idx - 1)
    if isinstance(value, int):
        if 0 <= value < num_choices:
            return letter_for_index(value)
    return None


def _yes_no_letter(value: bool) -> str:
    return "A" if value else "B"


def _mc_record(question: str, choices: list[str], answer_letter: str, source: str) -> dict[str, str]:
    return {
        "question": question.strip(),
        "choices": [c.strip() for c in choices],
        "answer_letter": answer_letter,
        "source": source,
        "task_type": "multiple_choice",
    }


def _normalize_arc_row(row: dict[str, Any], source: str) -> dict[str, str] | None:
    question = row.get("question")
    choices_field = row.get("choices")
    answer = row.get("answerKey") or row.get("answer")
    if not isinstance(question, str) or not isinstance(choices_field, dict):
        return None
    texts = list(choices_field.get("text") or [])
    labels = list(choices_field.get("label") or [])
    if not texts or not answer:
        return None
    label_to_index = {str(label).strip().upper(): i for i, label in enumerate(labels)}
    answer_str = str(answer).strip().upper()
    if answer_str.isdigit() and 1 <= int(answer_str) <= len(texts):
        index = int(answer_str) - 1
    elif answer_str in label_to_index:
        index = label_to_index[answer_str]
    else:
        return None
    if not (2 <= len(texts) <= 5):
        return None
    answer_letter = letter_for_index(index)
    return _mc_record(question, texts, answer_letter, source)


def _normalize_piqa_row(row: dict[str, Any]) -> dict[str, str] | None:
    goal = row.get("goal")
    sol1 = row.get("sol1")
    sol2 = row.get("sol2")
    label = row.get("label")
    if not isinstance(goal, str) or not isinstance(sol1, str) or not isinstance(sol2, str):
        return None
    if label is None:
        return None
    answer_letter = "A" if int(label) == 0 else "B"
    return _mc_record(
        question=f"Which approach is more appropriate to achieve the following goal?\n\nGoal: {goal}",
        choices=[sol1, sol2],
        answer_letter=answer_letter,
        source="piqa",
    )


def _normalize_boolq_row(row: dict[str, Any]) -> dict[str, str] | None:
    question = row.get("question")
    passage = row.get("passage")
    answer = row.get("answer")
    if not isinstance(question, str) or not isinstance(passage, str) or answer is None:
        return None
    if not isinstance(answer, bool):
        if isinstance(answer, int):
            answer = bool(answer)
        elif isinstance(answer, str):
            answer = answer.strip().lower() in {"true", "yes", "1"}
        else:
            return None
    answer_letter = _yes_no_letter(answer)
    composed = f"Passage:\n{passage}\n\nQuestion:\n{question}"
    return _mc_record(
        question=composed,
        choices=["Yes", "No"],
        answer_letter=answer_letter,
        source="boolq",
    )


def _normalize_race_row(row: dict[str, Any]) -> dict[str, str] | None:
    article = row.get("article")
    question = row.get("question")
    options = row.get("options")
    answer = row.get("answer")
    if not isinstance(article, str) or not isinstance(question, str) or not isinstance(options, list) or not answer:
        return None
    answer_str = str(answer).strip().upper()
    if answer_str not in {"A", "B", "C", "D"}:
        return None
    composed = f"Passage:\n{article}\n\nQuestion:\n{question}"
    return _mc_record(composed, [str(o) for o in options], answer_str, "race")


def _normalize_openbookqa_row(row: dict[str, Any]) -> dict[str, str] | None:
    question = row.get("question_stem")
    choices_field = row.get("choices")
    answer = row.get("answerKey")
    if not isinstance(question, str) or not isinstance(choices_field, dict) or not answer:
        return None
    texts = list(choices_field.get("text") or [])
    labels = list(choices_field.get("label") or [])
    if not texts:
        return None
    label_to_index = {str(label).strip().upper(): i for i, label in enumerate(labels)}
    answer_str = str(answer).strip().upper()
    if answer_str not in label_to_index:
        return None
    return _mc_record(
        question=question,
        choices=texts,
        answer_letter=letter_for_index(label_to_index[answer_str]),
        source="openbookqa_train",
    )


def _normalize_hellaswag_row(row: dict[str, Any]) -> dict[str, str] | None:
    ctx = row.get("ctx") or row.get("ctx_a")
    endings = row.get("endings")
    label = row.get("label")
    if not isinstance(ctx, str) or not isinstance(endings, list) or label is None:
        return None
    letter = _to_letter(label, len(endings))
    if letter is None:
        return None
    return _mc_record(
        question=f"Choose the most plausible continuation:\n\n{ctx}",
        choices=[str(e) for e in endings],
        answer_letter=letter,
        source="hellaswag_train",
    )


def _normalize_winogrande_row(row: dict[str, Any]) -> dict[str, str] | None:
    sentence = row.get("sentence")
    option1 = row.get("option1")
    option2 = row.get("option2")
    answer = row.get("answer")
    if not isinstance(sentence, str) or not isinstance(option1, str) or not isinstance(option2, str) or not answer:
        return None
    if str(answer).strip() == "1":
        letter = "A"
    elif str(answer).strip() == "2":
        letter = "B"
    else:
        return None
    return _mc_record(
        question=f"Fill in the blank with the most plausible option:\n\n{sentence}",
        choices=[option1, option2],
        answer_letter=letter,
        source="winogrande_train",
    )


NORMALIZERS = {
    "arc_easy": lambda row: _normalize_arc_row(row, "arc_easy"),
    "arc_challenge": lambda row: _normalize_arc_row(row, "arc_challenge"),
    "piqa": _normalize_piqa_row,
    "boolq": _normalize_boolq_row,
    "race": _normalize_race_row,
    "openbookqa_train": _normalize_openbookqa_row,
    "hellaswag_train": _normalize_hellaswag_row,
    "winogrande_train": _normalize_winogrande_row,
}


def normalize_rows(source: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalizer = NORMALIZERS.get(source)
    if normalizer is None:
        raise ValueError(f"No normalizer for MC source '{source}'")
    out: list[dict[str, str]] = []
    for row in rows:
        record = normalizer(row)
        if record is not None:
            out.append(record)
    return out


def mc_record_to_alpaca_pair(record: dict[str, str], eos_token: str) -> tuple[str, str]:
    """Build the wrapped (Alpaca-style) (prompt_text, response_text) MC pair."""
    prompt = build_mc_prompt(record["question"], record["choices"])
    response = format_alpaca_response(output=record["answer_letter"], eos_token=eos_token)
    return prompt, response


def mc_record_to_bare_pair(record: dict[str, str], eos_token: str) -> tuple[str, str]:
    """Build the bare (leaderboard-style) (prompt_text, response_text) MC pair."""
    prompt = build_mc_prompt_bare(record["question"], record["choices"])
    # Match the bare-prompt convention: include a leading space so the answer
    # letter tokenizes consistently as a single ``▁A`` BPE token (e.g. " A").
    response = format_alpaca_response(output=f" {record['answer_letter']}", eos_token=eos_token)
    return prompt, response
