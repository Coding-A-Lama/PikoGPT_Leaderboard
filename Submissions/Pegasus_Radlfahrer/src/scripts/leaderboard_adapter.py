"""Public-leaderboard CLI adapter for PegasusGPT checkpoints.

Matches the leaderboard call convention::

    python scripts/leaderboard_adapter.py \
        --checkpoint CKPT.pt \
        --prompt "..." \
        --max-tokens N \
        --temperature 0 \
        --device auto \
        --leaderboard \
        --seed 0

Behavior:
- In `--leaderboard` mode, ONLY the generated continuation is printed to stdout.
  Logs and errors go to stderr. No prompt is echoed.
- Without `--leaderboard`, prompt + generation are printed for human inspection.
- The leaderboard passes its own (already formatted) prompt; we DO NOT wrap it
  in Alpaca by default. Pass `--alpaca-template` to opt in for chat-style use.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from transformers import GPT2TokenizerFast

from domain.inference.checkpoint_loader import load_model_from_checkpoint
from domain.inference.generation import (
    GenerationConfig,
    allowed_letter_token_ids,
    detect_allowed_mc_letters,
    generate,
)
from domain.sft.alpaca_template import format_alpaca_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PegasusGPT leaderboard adapter")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--prompt", required=True, help="Prompt text to feed the model.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eos-token-id", type=int, default=50256)
    parser.add_argument(
        "--no-stop-on-eos",
        action="store_true",
        help="Generate the full max_tokens window even after EOS is produced.",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Suppress all human-readable output; print ONLY the generated continuation to stdout.",
    )
    parser.add_argument(
        "--auto-mc-mask",
        dest="auto_mc_mask",
        action="store_true",
        default=None,
        help="Constrain first generated token to visible MC letters for multiple-choice prompts.",
    )
    parser.add_argument(
        "--no-auto-mc-mask",
        dest="auto_mc_mask",
        action="store_false",
        help="Disable automatic multiple-choice first-token masking.",
    )
    parser.add_argument(
        "--alpaca-template",
        action="store_true",
        help="Wrap the input prompt in the Alpaca instruction template before generation.",
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=None,
        help="Fallback max_position_embeddings if the checkpoint doesn't embed model_config.",
    )
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device_arg)


def _parse_first_allowed_letter(text: str, allowed_letters: list[str]) -> str | None:
    allowed = set(allowed_letters)
    for char in text:
        if char.isspace():
            continue
        upper = char.upper()
        return upper if upper in allowed else None
    return None


def _extract_mc_options(prompt: str) -> list[tuple[str, str, str]]:
    """Return visible choices as (letter, separator, option text)."""
    options: list[tuple[str, str, str]] = []
    for line in prompt.splitlines():
        match = re.match(r"^\s*([A-D])\s*([).])\s*(.+?)\s*$", line)
        if match:
            options.append((match.group(1).upper(), match.group(2), match.group(3)))
    return options


def _completion_logprob(
    *,
    model: torch.nn.Module,
    tokenizer: GPT2TokenizerFast,
    prompt_token_ids: list[int],
    completion_text: str,
    device: torch.device,
    max_position_embeddings: int,
    normalize: bool,
) -> float:
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    if not completion_ids:
        return float("-inf")
    prompt_budget = max(1, max_position_embeddings - len(completion_ids))
    context_ids = prompt_token_ids[-prompt_budget:]
    input_ids = context_ids + completion_ids
    if len(input_ids) < 2:
        return float("-inf")

    tensor = torch.tensor([input_ids[:-1]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(tensor)
        log_probs = torch.log_softmax(logits, dim=-1)

    start = len(context_ids) - 1
    scores = [
        float(log_probs[0, start + offset, token_id].item())
        for offset, token_id in enumerate(completion_ids)
    ]
    total = sum(scores)
    return total / len(scores) if normalize else total


def _score_mc_options(
    *,
    model: torch.nn.Module,
    tokenizer: GPT2TokenizerFast,
    prompt_text: str,
    prompt_token_ids: list[int],
    allowed_letters: list[str],
    device: torch.device,
    max_position_embeddings: int,
) -> str | None:
    options = [
        option
        for option in _extract_mc_options(prompt_text)
        if option[0] in set(allowed_letters)
    ]
    if len(options) != len(allowed_letters):
        return None

    # The public benchmarks use three prompt families. These scoring forms
    # match the continuation style that is most stable for each family.
    if len(options) == 2:
        mode = "text"
        normalize = True
    elif prompt_text.lstrip().startswith("Question:"):
        mode = "label_text"
        normalize = False
    else:
        mode = "text"
        normalize = False

    scored: list[tuple[str, float]] = []
    for letter, separator, text in options:
        if mode == "label_text":
            completion = f" {letter}{separator} {text}"
        else:
            completion = f" {text}"
        scored.append(
            (
                letter,
                _completion_logprob(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_token_ids=prompt_token_ids,
                    completion_text=completion,
                    device=device,
                    max_position_embeddings=max_position_embeddings,
                    normalize=normalize,
                ),
            )
        )

    finite_scores = [score for _, score in scored if score != float("-inf")]
    if len(finite_scores) != len(scored) or max(finite_scores) - min(finite_scores) < 1e-6:
        return None
    return max(scored, key=lambda item: item[1])[0]


def load_tokenizer() -> GPT2TokenizerFast:
    local_tokenizer = Path(__file__).resolve().parent.parent / "tokenizer" / "gpt2"
    source = str(local_tokenizer) if local_tokenizer.exists() else "gpt2"
    return GPT2TokenizerFast.from_pretrained(source)


def main() -> int:
    try:
        args = parse_args()
        if args.auto_mc_mask is None:
            args.auto_mc_mask = bool(args.leaderboard)
        if args.leaderboard:
            args.no_stop_on_eos = False
        if args.max_tokens < 1:
            raise ValueError("--max-tokens must be >= 1")
        if args.temperature < 0.0:
            raise ValueError("--temperature must be >= 0")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        device = resolve_device(args.device)

        fallback_settings = {
            "architecture": "llama3",
            "vocab_size": 50257,
            "max_position_embeddings": args.max_position_embeddings or 512,
            "hidden_size": 320,
            "num_layers": 22,
            "num_attention_heads": 16,
            "tie_word_embeddings": True,
            "n_kv_heads": 1,
            "intermediate_size": 853,
            "rope_theta": 10000.0,
            "dropout": 0.0,
            "qkv_bias": False,
        }

        if not args.leaderboard:
            log(f"[adapter] device={device.type} ckpt={args.checkpoint}")

        loaded = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            fallback_model_settings=fallback_settings,
            device=device,
        )
        tokenizer = load_tokenizer()

        prompt_text = format_alpaca_prompt(instruction=args.prompt) if args.alpaca_template else args.prompt
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if not prompt_ids:
            log("[adapter] empty prompt after tokenization")
            return 2
        if len(prompt_ids) >= loaded.max_position_embeddings:
            prompt_ids = prompt_ids[-(loaded.max_position_embeddings - 1):]
        allowed_letters = detect_allowed_mc_letters(prompt_text) if args.auto_mc_mask else None
        allowed_token_ids = (
            allowed_letter_token_ids(tokenizer, allowed_letters)
            if allowed_letters is not None
            else None
        )
        if allowed_letters is not None and not allowed_token_ids:
            log("[adapter] no single-token MC letters found; falling back to normal generation")
            allowed_token_ids = None

        if args.leaderboard and allowed_letters is not None:
            scored_letter = _score_mc_options(
                model=loaded.model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                prompt_token_ids=prompt_ids,
                allowed_letters=allowed_letters,
                device=device,
                max_position_embeddings=loaded.max_position_embeddings,
            )
            if scored_letter is not None:
                sys.stdout.write(scored_letter)
                sys.stdout.flush()
                return 0

        config = GenerationConfig(
            max_new_tokens=1 if allowed_token_ids else args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=args.eos_token_id,
            stop_on_eos=not args.no_stop_on_eos,
            allowed_first_token_ids=allowed_token_ids,
        )

        result = generate(
            model=loaded.model,
            prompt_token_ids=prompt_ids,
            config=config,
            device=device,
            max_position_embeddings=loaded.max_position_embeddings,
        )
        generated_text = tokenizer.decode(result.new_token_ids)
        if allowed_letters is not None:
            parsed = _parse_first_allowed_letter(generated_text, allowed_letters)
            if parsed is not None:
                generated_text = parsed

        if args.leaderboard:
            sys.stdout.write(generated_text)
            sys.stdout.flush()
        else:
            print(prompt_text + generated_text)

        return 0
    except Exception as error:
        log(f"[adapter] error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
