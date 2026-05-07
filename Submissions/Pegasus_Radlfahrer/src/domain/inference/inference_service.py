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
                for _ in range(max_new_tokens):
                    context_token_ids = generated_token_ids[-max_position_embeddings:]
                    input_ids = torch.tensor([context_token_ids], dtype=torch.long, device=device)
                    logits = model(input_ids)
                    next_logits = logits[0, -1, :]

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
