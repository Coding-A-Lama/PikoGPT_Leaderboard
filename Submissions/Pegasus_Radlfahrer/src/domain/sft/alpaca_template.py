"""Alpaca instruction template shared by SFT preparation and inference.

Keep the exact format here so training and inference never drift.
"""
from __future__ import annotations


ALPACA_PROMPT_WITHOUT_INPUT = (
    "### Instruction:\n"
    "{instruction}\n\n"
    "### Response:\n"
)

ALPACA_PROMPT_WITH_INPUT = (
    "### Instruction:\n"
    "{instruction}\n\n"
    "### Input:\n"
    "{input}\n\n"
    "### Response:\n"
)


def _has_input(input_text: str | None) -> bool:
    return input_text is not None and input_text.strip() != ""


def format_alpaca_prompt(instruction: str, input_text: str | None = None) -> str:
    """Return the prompt portion of an Alpaca example (ends with '### Response:\\n')."""
    if _has_input(input_text):
        return ALPACA_PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
    return ALPACA_PROMPT_WITHOUT_INPUT.format(instruction=instruction)


def format_alpaca_response(output: str, eos_token: str) -> str:
    """Return the response portion (answer plus EOS) for an Alpaca example."""
    return f"{output}{eos_token}"


def format_alpaca_example(
    instruction: str,
    input_text: str | None,
    output: str,
    eos_token: str,
) -> tuple[str, str]:
    """Return (prompt_text, response_text) for an Alpaca example."""
    prompt_text = format_alpaca_prompt(instruction=instruction, input_text=input_text)
    response_text = format_alpaca_response(output=output, eos_token=eos_token)
    return prompt_text, response_text
