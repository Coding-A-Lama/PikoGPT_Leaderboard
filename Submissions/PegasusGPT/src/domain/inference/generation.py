"""Centralized text generation with shared decoding logic.

Used by both the public-leaderboard adapter and the internal benchmark
evaluator so that decoding behavior cannot drift between code paths.

Decoding pipeline (per step):
    1. Forward the current context.
    2. Apply repetition penalty over already-generated tokens.
    3. Greedy if temperature == 0.0; otherwise temperature-scale logits,
       then optionally apply top-k and top-p filters and sample.
    4. Stop when `eos_token_id` is produced (when stop_on_eos is True) or
       when `max_new_tokens` are produced.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    eos_token_id: int = 50256
    stop_on_eos: bool = True
    allowed_first_token_ids: tuple[int, ...] | None = None


@dataclass(slots=True)
class GenerationResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    new_token_ids: list[int]
    stopped_on_eos: bool


def apply_repetition_penalty(logits: Tensor, generated_ids: list[int], penalty: float) -> Tensor:
    """Divide (positive) or multiply (negative) logits of already-generated tokens."""
    if penalty == 1.0 or not generated_ids:
        return logits
    unique_ids = torch.tensor(sorted(set(generated_ids)), dtype=torch.long, device=logits.device)
    selected = logits.index_select(0, unique_ids)
    adjusted = torch.where(selected > 0, selected / penalty, selected * penalty)
    out = logits.clone()
    out.index_copy_(0, unique_ids, adjusted)
    return out


def top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    """Mask all but the top_k highest logits with -inf (in-place safe)."""
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, top_k).values[-1]
    mask = logits < threshold
    return logits.masked_fill(mask, float("-inf"))


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    """Mask the smallest-probability tail whose cumulative probability exceeds (1 - top_p)."""
    if top_p >= 1.0 or top_p <= 0.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cumulative > top_p
    # Always keep the most likely token.
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove_indices = sorted_indices[sorted_remove]
    return logits.index_fill(-1, remove_indices, float("-inf"))


def sample_next_token(
    next_logits: Tensor,
    *,
    generated_token_ids: list[int],
    config: GenerationConfig,
    allowed_token_ids: tuple[int, ...] | None = None,
) -> int:
    """Apply the configured filters/penalties and pick the next token id."""
    if config.temperature < 0.0:
        raise ValueError("temperature must be >= 0")
    next_logits = apply_repetition_penalty(next_logits, generated_token_ids, config.repetition_penalty)
    if allowed_token_ids is not None:
        if not allowed_token_ids:
            raise ValueError("allowed_token_ids must not be empty")
        allowed = torch.tensor(allowed_token_ids, dtype=torch.long, device=next_logits.device)
        masked = torch.full_like(next_logits, float("-inf"))
        masked.index_copy_(0, allowed, next_logits.index_select(0, allowed))
        next_logits = masked

    if config.temperature == 0.0:
        return int(torch.argmax(next_logits).item())

    next_logits = next_logits / config.temperature
    if config.top_k is not None:
        next_logits = top_k_filter(next_logits, config.top_k)
    if config.top_p is not None:
        next_logits = top_p_filter(next_logits, config.top_p)
    probs = torch.softmax(next_logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
        raise ValueError("Invalid sampling probabilities after logit filtering")
    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


@torch.no_grad()
def generate(
    *,
    model: nn.Module,
    prompt_token_ids: list[int],
    config: GenerationConfig,
    device: torch.device,
    max_position_embeddings: int,
) -> GenerationResult:
    """Greedy / sampled generation with EOS stopping.

    The model is expected to expose a `forward(input_ids) -> logits` interface
    where `logits` has shape `(batch, seq_len, vocab_size)`. We always feed the
    last `max_position_embeddings` tokens because no KV-cache is implemented.
    """
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must not be empty")

    was_training = model.training
    model.eval()
    generated = list(prompt_token_ids)
    new_tokens: list[int] = []
    stopped_on_eos = False
    try:
        for _ in range(config.max_new_tokens):
            context = generated[-max_position_embeddings:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_logits = logits[0, -1, :]
            allowed_token_ids = config.allowed_first_token_ids if not new_tokens else None
            next_id = sample_next_token(
                next_logits=next_logits,
                generated_token_ids=generated,
                config=config,
                allowed_token_ids=allowed_token_ids,
            )
            if config.stop_on_eos and next_id == config.eos_token_id:
                stopped_on_eos = True
                break
            generated.append(next_id)
            new_tokens.append(next_id)
    finally:
        if was_training:
            model.train()

    return GenerationResult(
        prompt_token_ids=list(prompt_token_ids),
        generated_token_ids=generated,
        new_token_ids=new_tokens,
        stopped_on_eos=stopped_on_eos,
    )


def detect_allowed_mc_letters(prompt: str) -> list[str] | None:
    """Detect visible MC answer letters from prompt text.

    This only reads the prompt format, never labels. It supports obvious
    two-choice and four-choice prompts such as "A. ...\nB. ..." and returns
    None for non-MC prompts.
    """
    found: list[str] = []
    for letter in ("A", "B", "C", "D"):
        if re.search(rf"(?m)(?:^|\s){letter}\.\s+\S", prompt):
            found.append(letter)
    if found[:2] != ["A", "B"]:
        return None
    if len(found) >= 4:
        return found[:4]
    if len(found) == 2:
        return found
    return None


def allowed_letter_token_ids(tokenizer, allowed_letters: list[str]) -> tuple[int, ...]:
    """Collect GPT-style token IDs for first-token MC letter decoding."""
    token_ids: set[int] = set()
    for letter in allowed_letters:
        for form in (letter, f" {letter}", f"\n{letter}"):
            ids = tokenizer.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(int(ids[0]))
    return tuple(sorted(token_ids))


@torch.no_grad()
def conditional_loglikelihood(
    *,
    model: nn.Module,
    prompt_token_ids: list[int],
    continuation_token_ids: list[int],
    device: torch.device,
    max_position_embeddings: int,
) -> float:
    """Returns sum log P(continuation | prompt) under the model.

    Used for benchmark loglikelihood scoring without generating tokens.
    """
    if not continuation_token_ids:
        raise ValueError("continuation_token_ids must not be empty")

    full_ids = list(prompt_token_ids) + list(continuation_token_ids)
    if len(full_ids) > max_position_embeddings:
        # Drop tokens from the start of the prompt so that the continuation is preserved.
        overflow = len(full_ids) - max_position_embeddings
        if overflow >= len(prompt_token_ids):
            raise ValueError(
                "Continuation alone exceeds max_position_embeddings; cannot score."
            )
        prompt_token_ids = list(prompt_token_ids)[overflow:]
        full_ids = prompt_token_ids + list(continuation_token_ids)

    was_training = model.training
    model.eval()
    try:
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        logits = model(input_tensor)[0]  # (seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

        # log P(token_t) is taken from logits at position t-1.
        prompt_len = len(prompt_token_ids)
        cont_len = len(continuation_token_ids)
        start = prompt_len - 1
        if start < 0:
            raise ValueError("Prompt must have at least one token for loglikelihood scoring.")
        token_positions = torch.arange(start, start + cont_len, device=device)
        target_ids = torch.tensor(continuation_token_ids, dtype=torch.long, device=device)
        selected_log_probs = log_probs[token_positions, target_ids]
        total = float(selected_log_probs.sum().item())
    finally:
        if was_training:
            model.train()
    return total
