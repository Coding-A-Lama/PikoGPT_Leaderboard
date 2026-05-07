"""Two complementary MC evaluation modes:
- generation_parse: matches the public leaderboard (greedy generate, parse first letter).
- loglikelihood: pure scoring of P(letter | prompt) — no generation, no invalid outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import GPT2TokenizerFast

from domain.evaluation.benchmark_types import (
    MultipleChoiceExample,
    index_for_letter,
    letter_for_index,
)
from domain.evaluation.mc_prompt import build_mc_prompt
from domain.inference.generation import (
    GenerationConfig,
    allowed_letter_token_ids,
    conditional_loglikelihood,
    detect_allowed_mc_letters,
    generate,
)


@dataclass(slots=True)
class MCExampleResult:
    example_id: str
    benchmark: str
    mode: str
    prompt: str
    gold: str
    pred: str | None
    correct: bool
    invalid: bool
    raw_generation: str | None = None
    candidate_scores: dict[str, float] | None = None


@dataclass(slots=True)
class MCBenchmarkSummary:
    benchmark: str
    mode: str
    total: int
    correct: int
    invalid: int
    accuracy_pct: float
    invalid_pct: float
    wrong_examples: list[MCExampleResult]


def _parse_first_letter(text: str, valid_letters: set[str]) -> str | None:
    for char in text:
        if char.isspace():
            continue
        upper = char.upper()
        if upper in valid_letters:
            return upper
        return None
    return None


def evaluate_mc_generation_parse(
    *,
    model: nn.Module,
    tokenizer: GPT2TokenizerFast,
    examples: list[MultipleChoiceExample],
    device: torch.device,
    max_position_embeddings: int,
    eos_token_id: int,
    max_wrong_examples: int = 25,
    auto_mc_mask: bool = False,
) -> MCBenchmarkSummary:
    """Generate up to 3 tokens with greedy decoding, parse the first letter."""
    benchmark = examples[0].source if examples else "unknown"
    config = GenerationConfig(
        max_new_tokens=3,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        eos_token_id=eos_token_id,
        stop_on_eos=True,
    )

    correct = 0
    invalid = 0
    wrong_results: list[MCExampleResult] = []
    for example in examples:
        valid_letters = {letter_for_index(i) for i in range(len(example.choices))}
        prompt = build_mc_prompt(example.question, example.choices)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) >= max_position_embeddings:
            prompt_ids = prompt_ids[-(max_position_embeddings - 1):]
        allowed_token_ids = None
        if auto_mc_mask:
            allowed_letters = detect_allowed_mc_letters(prompt)
            if allowed_letters is not None:
                allowed_token_ids = allowed_letter_token_ids(tokenizer, allowed_letters)
                if not allowed_token_ids:
                    allowed_token_ids = None

        result = generate(
            model=model,
            prompt_token_ids=prompt_ids,
            config=GenerationConfig(
                max_new_tokens=1 if allowed_token_ids else config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                eos_token_id=config.eos_token_id,
                stop_on_eos=config.stop_on_eos,
                allowed_first_token_ids=allowed_token_ids,
            ),
            device=device,
            max_position_embeddings=max_position_embeddings,
        )
        generated_text = tokenizer.decode(result.new_token_ids)
        pred_letter = _parse_first_letter(generated_text, valid_letters)
        is_invalid = pred_letter is None
        is_correct = (not is_invalid) and pred_letter == example.answer_letter

        if is_correct:
            correct += 1
        if is_invalid:
            invalid += 1

        if (not is_correct) and len(wrong_results) < max_wrong_examples:
            wrong_results.append(
                MCExampleResult(
                    example_id=example.example_id,
                    benchmark=benchmark,
                    mode="generation_parse",
                    prompt=prompt,
                    gold=example.answer_letter,
                    pred=pred_letter,
                    correct=False,
                    invalid=is_invalid,
                    raw_generation=generated_text,
                )
            )

    total = len(examples)
    return MCBenchmarkSummary(
        benchmark=benchmark,
        mode="generation_parse",
        total=total,
        correct=correct,
        invalid=invalid,
        accuracy_pct=100.0 * correct / max(total, 1),
        invalid_pct=100.0 * invalid / max(total, 1),
        wrong_examples=wrong_results,
    )


def evaluate_mc_loglikelihood(
    *,
    model: nn.Module,
    tokenizer: GPT2TokenizerFast,
    examples: list[MultipleChoiceExample],
    device: torch.device,
    max_position_embeddings: int,
    max_wrong_examples: int = 25,
) -> MCBenchmarkSummary:
    """Score each candidate letter token (e.g. ' A', ' B', ...) by conditional logprob."""
    benchmark = examples[0].source if examples else "unknown"
    correct = 0
    wrong_results: list[MCExampleResult] = []

    for example in examples:
        prompt = build_mc_prompt(example.question, example.choices)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) >= max_position_embeddings:
            prompt_ids = prompt_ids[-(max_position_embeddings - 1):]

        scores: dict[str, float] = {}
        for index in range(len(example.choices)):
            letter = letter_for_index(index)
            # Score the first BPE token of the letter as a fresh response token.
            # GPT-2 tokenizes "A" and " A" differently — the SFT response begins
            # right after "### Response:\n" without a space, so we score "A".
            cont_ids = tokenizer.encode(letter, add_special_tokens=False)
            if not cont_ids:
                continue
            score = conditional_loglikelihood(
                model=model,
                prompt_token_ids=prompt_ids,
                continuation_token_ids=cont_ids,
                device=device,
                max_position_embeddings=max_position_embeddings,
            )
            scores[letter] = score

        if not scores:
            wrong_results.append(
                MCExampleResult(
                    example_id=example.example_id,
                    benchmark=benchmark,
                    mode="loglikelihood",
                    prompt=prompt,
                    gold=example.answer_letter,
                    pred=None,
                    correct=False,
                    invalid=True,
                    candidate_scores={},
                )
            )
            continue

        pred_letter = max(scores.items(), key=lambda kv: kv[1])[0]
        is_correct = pred_letter == example.answer_letter
        if is_correct:
            correct += 1
        elif len(wrong_results) < max_wrong_examples:
            wrong_results.append(
                MCExampleResult(
                    example_id=example.example_id,
                    benchmark=benchmark,
                    mode="loglikelihood",
                    prompt=prompt,
                    gold=example.answer_letter,
                    pred=pred_letter,
                    correct=False,
                    invalid=False,
                    candidate_scores={k: float(v) for k, v in scores.items()},
                )
            )

    total = len(examples)
    return MCBenchmarkSummary(
        benchmark=benchmark,
        mode="loglikelihood",
        total=total,
        correct=correct,
        invalid=0,  # loglikelihood mode never produces invalid output
        accuracy_pct=100.0 * correct / max(total, 1),
        invalid_pct=0.0,
        wrong_examples=wrong_results,
    )
