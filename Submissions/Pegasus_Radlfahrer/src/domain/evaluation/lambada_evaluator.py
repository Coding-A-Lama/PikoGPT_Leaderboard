"""LAMBADA: predict the final word of a passage.

Two modes:
- generation: greedy generate up to 5 tokens, take the first whitespace-separated
  word, normalize, compare to gold (exact match accuracy).
- loglikelihood: report mean negative log-likelihood of the gold final word.
"""
from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass

import torch
from torch import nn
from transformers import GPT2TokenizerFast

from domain.evaluation.benchmark_types import LambadaExample
from domain.inference.generation import (
    GenerationConfig,
    conditional_loglikelihood,
    generate,
)


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_word(text: str) -> str:
    """Lowercase, strip whitespace, remove punctuation, collapse spaces."""
    text = text.strip().lower()
    text = re.split(r"\s+", text, maxsplit=1)[0] if text else text
    text = text.translate(_PUNCT_TABLE)
    return text.strip()


@dataclass(slots=True)
class LambadaExampleResult:
    example_id: str
    prompt: str
    gold: str
    pred: str
    correct: bool
    raw_generation: str
    gold_logprob: float | None = None


@dataclass(slots=True)
class LambadaSummary:
    mode: str
    total: int
    correct: int
    invalid: int
    accuracy_pct: float
    average_negative_logprob: float | None
    perplexity_like: float | None
    wrong_examples: list[LambadaExampleResult]


def evaluate_lambada_generation(
    *,
    model: nn.Module,
    tokenizer: GPT2TokenizerFast,
    examples: list[LambadaExample],
    device: torch.device,
    max_position_embeddings: int,
    eos_token_id: int,
    max_wrong_examples: int = 25,
) -> LambadaSummary:
    config = GenerationConfig(
        max_new_tokens=5,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        eos_token_id=eos_token_id,
        stop_on_eos=True,
    )

    correct = 0
    wrong_results: list[LambadaExampleResult] = []
    for example in examples:
        prompt = example.prefix
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) == 0:
            continue
        if len(prompt_ids) >= max_position_embeddings:
            prompt_ids = prompt_ids[-(max_position_embeddings - 1):]

        result = generate(
            model=model,
            prompt_token_ids=prompt_ids,
            config=config,
            device=device,
            max_position_embeddings=max_position_embeddings,
        )
        generated_text = tokenizer.decode(result.new_token_ids)
        predicted_word = normalize_word(generated_text)
        gold_word = normalize_word(example.target_word)
        is_correct = predicted_word == gold_word and gold_word != ""
        if is_correct:
            correct += 1
        elif len(wrong_results) < max_wrong_examples:
            wrong_results.append(
                LambadaExampleResult(
                    example_id=example.example_id,
                    prompt=prompt[-200:],
                    gold=gold_word,
                    pred=predicted_word,
                    correct=False,
                    raw_generation=generated_text,
                )
            )

    total = len(examples)
    return LambadaSummary(
        mode="generation",
        total=total,
        correct=correct,
        invalid=0,
        accuracy_pct=100.0 * correct / max(total, 1),
        average_negative_logprob=None,
        perplexity_like=None,
        wrong_examples=wrong_results,
    )


def evaluate_lambada_loglikelihood(
    *,
    model: nn.Module,
    tokenizer: GPT2TokenizerFast,
    examples: list[LambadaExample],
    device: torch.device,
    max_position_embeddings: int,
    max_wrong_examples: int = 25,
) -> LambadaSummary:
    """Score the gold final word continuation given the prefix.

    'Correct' here means the gold word is the most likely token sequence vs.
    a fixed greedy generation — but the simplest signal is just the average
    negative log-likelihood of the gold word, which we always report.
    """
    total_neg_logprob = 0.0
    total_examples = 0
    correct = 0
    wrong_results: list[LambadaExampleResult] = []

    for example in examples:
        prompt = example.prefix
        prompt_ids = tokenizer.encode(prompt + " ", add_special_tokens=False)  # gold word follows a space
        gold_ids = tokenizer.encode(example.target_word, add_special_tokens=False)
        if not gold_ids or not prompt_ids:
            continue
        # Build a clean (prefix-with-trailing-space, gold-word) pair.
        prompt_only = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_only) + len(gold_ids) + 1 >= max_position_embeddings:
            prompt_only = prompt_only[-(max_position_embeddings - len(gold_ids) - 1):]
        # Tokenize the leading-space gold variant if BPE produces that.
        space_gold_ids = tokenizer.encode(" " + example.target_word, add_special_tokens=False)
        cont_ids = space_gold_ids if space_gold_ids else gold_ids

        gold_logprob = conditional_loglikelihood(
            model=model,
            prompt_token_ids=prompt_only,
            continuation_token_ids=cont_ids,
            device=device,
            max_position_embeddings=max_position_embeddings,
        )

        # Crude correctness proxy: greedy-decode the same number of tokens and compare.
        config = GenerationConfig(
            max_new_tokens=len(cont_ids),
            temperature=0.0,
            top_k=None,
            top_p=None,
            repetition_penalty=1.0,
            eos_token_id=50256,
            stop_on_eos=False,
        )
        gen_result = generate(
            model=model,
            prompt_token_ids=prompt_only,
            config=config,
            device=device,
            max_position_embeddings=max_position_embeddings,
        )
        predicted_text = tokenizer.decode(gen_result.new_token_ids)
        predicted_word = normalize_word(predicted_text)
        gold_word = normalize_word(example.target_word)
        is_correct = predicted_word == gold_word and gold_word != ""

        total_neg_logprob += -gold_logprob
        total_examples += 1
        if is_correct:
            correct += 1
        elif len(wrong_results) < max_wrong_examples:
            wrong_results.append(
                LambadaExampleResult(
                    example_id=example.example_id,
                    prompt=prompt[-200:],
                    gold=gold_word,
                    pred=predicted_word,
                    correct=False,
                    raw_generation=predicted_text,
                    gold_logprob=float(gold_logprob),
                )
            )

    total = len(examples)
    avg_neg = total_neg_logprob / max(total_examples, 1) if total_examples else None
    perplexity_like = math.exp(avg_neg) if avg_neg is not None and avg_neg < 50 else None
    return LambadaSummary(
        mode="loglikelihood",
        total=total,
        correct=correct,
        invalid=0,
        accuracy_pct=100.0 * correct / max(total, 1),
        average_negative_logprob=avg_neg,
        perplexity_like=perplexity_like,
        wrong_examples=wrong_results,
    )
