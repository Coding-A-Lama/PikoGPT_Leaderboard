"""Plain dataclasses describing a normalized benchmark example."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MultipleChoiceExample:
    example_id: str
    question: str
    choices: list[str]      # 2 or 4 choices, indexed A, B, ...
    answer_letter: str      # one of A/B/C/D
    source: str             # benchmark name


@dataclass(frozen=True, slots=True)
class LambadaExample:
    example_id: str
    prefix: str             # passage with the final word removed
    target_word: str        # gold last word
    source: str = "lambada"


def letter_for_index(index: int) -> str:
    if index < 0 or index > 25:
        raise ValueError(f"Choice index {index} out of supported range A-Z")
    return chr(ord("A") + index)


def index_for_letter(letter: str) -> int:
    letter = letter.strip().upper()
    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        raise ValueError(f"Invalid choice letter: {letter!r}")
    return ord(letter) - ord("A")
