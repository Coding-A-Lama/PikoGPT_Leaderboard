from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import GPT2TokenizerFast

from domain.model.model_factory import build_model_from_config


@dataclass
class InferenceResult:
    input_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str


class GPTInferenceService:
    """Core inference routine for decoder-only language models."""

    def run(
        self,
        checkpoint_path: str,
        model_config,
        input_text: str,
        max_new_tokens: int,
        device_name: str,
        vocab_size: int | None = None,
        temperature: float = 0.8,
        top_k: int = 50,
        allowed_first_token_ids: list[int] | None = None,
    ) -> InferenceResult:
        # Resolve vocab_size: explicit param > model_config.vocab_size > default
        resolved_vocab_size = vocab_size or getattr(model_config, "vocab_size", None) or 50257

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Support both new-style (dict with architecture) and legacy (bare state_dict) checkpoints
        saved_model_config = None
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            architecture = checkpoint.get("architecture", getattr(model_config, "architecture", "gpt2"))
            state_dict = checkpoint["model_state_dict"]
            saved_model_config = checkpoint.get("model_config")
        elif isinstance(checkpoint, dict):
            architecture = getattr(model_config, "architecture", "gpt2")
            state_dict = checkpoint
        else:
            raise TypeError(f"Checkpoint at {checkpoint_path} does not contain a state dict")

        resolved_model_settings = self._resolve_model_settings(
            runtime_model_config=model_config,
            saved_model_config=saved_model_config,
            fallback_architecture=architecture,
            fallback_vocab_size=resolved_vocab_size,
        )

        model, built_config = build_model_from_config(
            architecture=resolved_model_settings["architecture"],
            vocab_size=resolved_model_settings["vocab_size"],
            max_position_embeddings=resolved_model_settings["max_position_embeddings"],
            hidden_size=resolved_model_settings["hidden_size"],
            num_layers=resolved_model_settings["num_layers"],
            num_attention_heads=resolved_model_settings["num_attention_heads"],
            tie_word_embeddings=resolved_model_settings["tie_word_embeddings"],
            mlp_hidden_size=resolved_model_settings["mlp_hidden_size"],
            qkv_bias=resolved_model_settings["qkv_bias"],
            dropout=resolved_model_settings["dropout"],
            n_kv_heads=resolved_model_settings["n_kv_heads"],
            intermediate_size=resolved_model_settings["intermediate_size"],
            rope_theta=resolved_model_settings["rope_theta"],
        )

        model.load_state_dict(state_dict)
        result = self.run_with_model(
            model=model,
            input_text=input_text,
            max_new_tokens=max_new_tokens,
            device_name=device_name,
            max_position_embeddings=built_config.max_position_embeddings,
            vocab_size=built_config.vocab_size,
            temperature=temperature,
            top_k=top_k,
            allowed_first_token_ids=allowed_first_token_ids,
        )

        return result

    def run_with_model(
        self,
        model: nn.Module,
        input_text: str,
        max_new_tokens: int,
        device_name: str,
        max_position_embeddings: int,
        vocab_size: int,
        temperature: float = 0.8,
        top_k: int = 50,
        allowed_first_token_ids: list[int] | None = None,
    ) -> InferenceResult:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        input_token_ids = tokenizer.encode(input_text, add_special_tokens=False)

        if len(input_token_ids) == 0:
            raise ValueError("input_text does not contain any tokenizable content")

        if len(input_token_ids) > max_position_embeddings:
            raise ValueError(
                "Tokenized input length exceeds max_position_embeddings configured for inference"
            )

        device = torch.device(device_name)
        model.to(device)
        was_training = model.training
        model.eval()

        generated_token_ids = list(input_token_ids)
        effective_top_k = min(top_k, vocab_size)

        try:
            with torch.no_grad():
                for step in range(max_new_tokens):
                    context_token_ids = generated_token_ids[-max_position_embeddings:]
                    input_ids = torch.tensor([context_token_ids], dtype=torch.long, device=device)
                    logits = model(input_ids)
                    next_logits = logits[0, -1, :]

                    # Restrict step-0 logits to an allowed token set when given.
                    # Used by the leaderboard adapter for MC: forces the first
                    # generated token to be one of the present-letter ids
                    # (e.g. " A"/" B"/" C"/" D"), guaranteeing the runner
                    # parses a valid letter.
                    if step == 0 and allowed_first_token_ids:
                        mask = torch.full_like(next_logits, float("-inf"))
                        mask[allowed_first_token_ids] = next_logits[allowed_first_token_ids]
                        next_logits = mask

                    if temperature == 0.0:
                        next_token_id = int(torch.argmax(next_logits).item())
                    else:
                        scaled_logits = next_logits / temperature
                        top_k_logits, top_k_indices = torch.topk(scaled_logits, effective_top_k)
                        probs = torch.softmax(top_k_logits, dim=-1)
                        sampled_index = torch.multinomial(probs, num_samples=1)
                        next_token_id = int(top_k_indices[sampled_index].item())
                    generated_token_ids.append(next_token_id)
        finally:
            if was_training:
                model.train()

        generated_text = tokenizer.decode(generated_token_ids)

        return InferenceResult(
            input_token_ids=input_token_ids,
            generated_token_ids=generated_token_ids,
            generated_text=generated_text,
        )

    def score_options_full(
        self,
        checkpoint_path: str,
        model_config,
        prompt: str,
        option_texts: list[str],
        device_name: str,
        vocab_size: int | None = None,
    ) -> int:
        """Lowest-PPL multiple-choice scoring (VL10 slide 21 canonical method).

        For each option text, compute the conditional log-probability of
        `" " + option_text` given the prompt and return the argmax index. The
        leading space is the GPT-2 BPE convention so each option's first token
        consistently encodes as a leading-space-prefixed BPE piece. This
        replicates lm-evaluation-harness's MC scoring path for non-instruction-
        tuned models — empirically +11pp on OpenBookQA at the SFT v18 scale.
        """
        # Load model with the same logic as run().
        resolved_vocab_size = vocab_size or getattr(model_config, "vocab_size", None) or 50257
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_model_config = None
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            architecture = checkpoint.get("architecture", getattr(model_config, "architecture", "gpt2"))
            state_dict = checkpoint["model_state_dict"]
            saved_model_config = checkpoint.get("model_config")
        elif isinstance(checkpoint, dict):
            architecture = getattr(model_config, "architecture", "gpt2")
            state_dict = checkpoint
        else:
            raise TypeError(f"Checkpoint at {checkpoint_path} does not contain a state dict")
        resolved_model_settings = self._resolve_model_settings(
            runtime_model_config=model_config,
            saved_model_config=saved_model_config,
            fallback_architecture=architecture,
            fallback_vocab_size=resolved_vocab_size,
        )
        model, built_config = build_model_from_config(
            architecture=resolved_model_settings["architecture"],
            vocab_size=resolved_model_settings["vocab_size"],
            max_position_embeddings=resolved_model_settings["max_position_embeddings"],
            hidden_size=resolved_model_settings["hidden_size"],
            num_layers=resolved_model_settings["num_layers"],
            num_attention_heads=resolved_model_settings["num_attention_heads"],
            tie_word_embeddings=resolved_model_settings["tie_word_embeddings"],
            mlp_hidden_size=resolved_model_settings["mlp_hidden_size"],
            qkv_bias=resolved_model_settings["qkv_bias"],
            dropout=resolved_model_settings["dropout"],
            n_kv_heads=resolved_model_settings["n_kv_heads"],
            intermediate_size=resolved_model_settings["intermediate_size"],
            rope_theta=resolved_model_settings["rope_theta"],
        )
        model.load_state_dict(state_dict)
        device = torch.device(device_name)
        model.to(device)
        model.eval()

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        max_pos = built_config.max_position_embeddings
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) >= max_pos:
            prompt_ids = prompt_ids[-(max_pos - 1):]

        # Build padded batch for all options in a single forward pass.
        # Each row: [prompt_ids ... | continuation_ids ... | pad].
        # Track per-row continuation start, length, and ids for scoring.
        rows: list[list[int]] = []
        cont_starts: list[int] = []
        cont_lens: list[int] = []
        cont_ids_per_row: list[list[int]] = []
        for option_text in option_texts:
            cont_text = " " + option_text.strip()
            cont_ids = tokenizer.encode(cont_text, add_special_tokens=False)
            if not cont_ids:
                rows.append([])
                cont_starts.append(0)
                cont_lens.append(0)
                cont_ids_per_row.append([])
                continue
            full_ids = prompt_ids + cont_ids
            if len(full_ids) > max_pos:
                keep = max_pos - len(cont_ids)
                if keep < 1:
                    rows.append([])
                    cont_starts.append(0)
                    cont_lens.append(0)
                    cont_ids_per_row.append([])
                    continue
                full_ids = prompt_ids[-keep:] + cont_ids
                cont_start = keep
            else:
                cont_start = len(prompt_ids)
            rows.append(full_ids)
            cont_starts.append(cont_start)
            cont_lens.append(len(cont_ids))
            cont_ids_per_row.append(cont_ids)

        max_len = max((len(r) for r in rows), default=0)
        if max_len == 0:
            return 0

        # Pad with 0 on the right; padding positions are masked out in scoring.
        pad_id = 0
        batch = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            if r:
                batch[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(batch)
            log_probs = torch.log_softmax(logits.float(), dim=-1)

        # Length-normalize per-token log-prob. Empirically (limit=100 OBQA
        # on dpo_v20): length-norm ON gave ~34% (close to lm-eval-harness
        # default), length-norm OFF dropped to 28%. The "no length-norm
        # for short options" trick in christof's deck assumed PMI α=4.0 to
        # debias the raw sum; without PMI, raw sum favors longer options
        # and underperforms. Always normalize is safer for our setup.
        length_normalize = True

        best_idx = 0
        best_score = float("-inf")
        for i, r in enumerate(rows):
            n = cont_lens[i]
            if n == 0 or not r:
                continue
            total = 0.0
            counted = 0
            for j in range(n):
                pos = cont_starts[i] + j
                if pos == 0:
                    continue
                total += float(log_probs[i, pos - 1, r[pos]].item())
                counted += 1
            if counted == 0:
                continue
            score = total / counted if length_normalize else total
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _resolve_model_settings(
        self,
        *,
        runtime_model_config,
        saved_model_config: dict[str, Any] | None,
        fallback_architecture: str,
        fallback_vocab_size: int,
    ) -> dict[str, Any]:
        runtime_settings = runtime_model_config.model_dump() if hasattr(runtime_model_config, "model_dump") else vars(runtime_model_config)
        saved_settings = saved_model_config if isinstance(saved_model_config, dict) else {}

        return {
            "architecture": saved_settings.get("architecture", fallback_architecture),
            "vocab_size": saved_settings.get("vocab_size", runtime_settings.get("vocab_size", fallback_vocab_size)),
            "max_position_embeddings": saved_settings.get(
                "max_position_embeddings",
                runtime_settings["max_position_embeddings"],
            ),
            "hidden_size": saved_settings.get("hidden_size", runtime_settings["hidden_size"]),
            "num_layers": saved_settings.get("num_layers", runtime_settings["num_layers"]),
            "num_attention_heads": saved_settings.get(
                "num_attention_heads",
                runtime_settings["num_attention_heads"],
            ),
            "tie_word_embeddings": saved_settings.get(
                "tie_word_embeddings",
                runtime_settings.get("tie_word_embeddings", False),
            ),
            "mlp_hidden_size": saved_settings.get("mlp_hidden_size", runtime_settings.get("mlp_hidden_size")),
            "qkv_bias": saved_settings.get("qkv_bias", runtime_settings.get("qkv_bias", False)),
            "dropout": saved_settings.get("dropout", runtime_settings.get("dropout", 0.0)),
            "n_kv_heads": saved_settings.get("n_kv_heads", runtime_settings.get("n_kv_heads")),
            "intermediate_size": saved_settings.get(
                "intermediate_size",
                runtime_settings.get("intermediate_size"),
            ),
            "rope_theta": saved_settings.get("rope_theta", runtime_settings.get("rope_theta", 10000.0)),
        }
