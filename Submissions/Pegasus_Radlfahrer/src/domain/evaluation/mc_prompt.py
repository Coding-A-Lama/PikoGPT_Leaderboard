"""Benchmark-style prompt builder shared by evaluation and MC SFT/DPO data prep.

Two formats are supported:

- ``build_mc_prompt`` wraps the MC question in the Alpaca instruction template.
  This is what the local ``evaluate-benchmarks`` stage uses; SFT examples
  trained in this format match the local benchmark format exactly.

- ``build_mc_prompt_bare`` returns the MC question without the
  ``### Instruction:`` wrapper. The public PikoGPT leaderboard sends prompts
  in this bare style, so training a fraction of MC examples here helps the
  model handle the leaderboard's format directly.
"""
from __future__ import annotations

from domain.evaluation.benchmark_types import letter_for_index
from domain.sft.alpaca_template import format_alpaca_prompt


MC_INSTRUCTION_HEADER = "Question: {question}\n\n{enumerated_choices}\n\nAnswer with only the correct letter."
MC_BARE_FOOTER = "\n\nAnswer:"


def render_choices(choices: list[str]) -> str:
    return "\n".join(f"{letter_for_index(i)}. {choice}" for i, choice in enumerate(choices))


def build_mc_prompt(question: str, choices: list[str]) -> str:
    instruction = MC_INSTRUCTION_HEADER.format(
        question=question.strip(),
        enumerated_choices=render_choices(choices),
    )
    return format_alpaca_prompt(instruction=instruction)


def build_mc_prompt_bare(question: str, choices: list[str]) -> str:
    """Bare leaderboard-style prompt: ``Question: ...\\n\\nA. ...\\n...\\n\\nAnswer:``."""
    return (
        f"Question: {question.strip()}\n\n"
        f"{render_choices(choices)}"
        f"{MC_BARE_FOOTER}"
    )
