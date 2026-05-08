"""Shared checkpoint loading for inference, evaluation, and DPO.

Resolves both new-style checkpoints (dict with 'architecture' / 'model_config' /
'model_state_dict') and legacy bare state dicts. Strips the '_orig_mod.' prefix
from torch.compile state dicts so the same checkpoint loads everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from domain.model.model_factory import build_model_from_config

ORIG_MOD_PREFIX = "_orig_mod."


@dataclass(slots=True)
class LoadedModel:
    model: nn.Module
    architecture: str
    vocab_size: int
    max_position_embeddings: int


def _strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if state_dict and all(key.startswith(ORIG_MOD_PREFIX) for key in state_dict.keys()):
        return {key[len(ORIG_MOD_PREFIX):]: value for key, value in state_dict.items()}
    return state_dict


def _resolve_state_dict(checkpoint: Any) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return (
            checkpoint["model_state_dict"],
            checkpoint.get("model_config"),
            checkpoint.get("architecture"),
        )
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"], None, None
    if isinstance(checkpoint, dict):
        return checkpoint, None, None
    raise TypeError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")


def load_model_from_checkpoint(
    *,
    checkpoint_path: str,
    fallback_model_settings: dict[str, Any],
    device: torch.device,
) -> LoadedModel:
    """Load a checkpoint, preferring its embedded model_config when available.

    `fallback_model_settings` is used if the checkpoint does not embed a config
    (legacy bare state dicts). Keys: architecture, vocab_size,
    max_position_embeddings, hidden_size, num_layers, num_attention_heads,
    tie_word_embeddings, mlp_hidden_size, qkv_bias, dropout, n_kv_heads,
    intermediate_size, rope_theta.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")

    raw = torch.load(path, map_location="cpu", weights_only=False)
    state_dict, saved_config, saved_arch = _resolve_state_dict(raw)
    state_dict = _strip_compile_prefix(state_dict)

    if saved_config is not None:
        settings = dict(saved_config)
        architecture = saved_arch or settings.get("architecture") or fallback_model_settings["architecture"]
    else:
        settings = dict(fallback_model_settings)
        architecture = saved_arch or settings.get("architecture", fallback_model_settings["architecture"])

    model, built_config = build_model_from_config(
        architecture=architecture,
        vocab_size=int(settings.get("vocab_size", fallback_model_settings.get("vocab_size", 50257))),
        max_position_embeddings=int(settings["max_position_embeddings"]),
        hidden_size=int(settings["hidden_size"]),
        num_layers=int(settings["num_layers"]),
        num_attention_heads=int(settings["num_attention_heads"]),
        tie_word_embeddings=bool(settings.get("tie_word_embeddings", fallback_model_settings.get("tie_word_embeddings", False))),
        mlp_hidden_size=settings.get("mlp_hidden_size", fallback_model_settings.get("mlp_hidden_size")),
        qkv_bias=bool(settings.get("qkv_bias", fallback_model_settings.get("qkv_bias", False))),
        dropout=float(settings.get("dropout", fallback_model_settings.get("dropout", 0.0))),
        n_kv_heads=settings.get("n_kv_heads", fallback_model_settings.get("n_kv_heads")),
        intermediate_size=settings.get("intermediate_size", fallback_model_settings.get("intermediate_size")),
        rope_theta=float(settings.get("rope_theta", fallback_model_settings.get("rope_theta", 10000.0))),
        norm_type=str(settings.get("norm_type", fallback_model_settings.get("norm_type", "layernorm"))),
        norm_placement=str(settings.get("norm_placement", fallback_model_settings.get("norm_placement", "pre_norm"))),
        positional_encoding_type=str(
            settings.get(
                "positional_encoding_type",
                fallback_model_settings.get("positional_encoding_type", "learned_absolute"),
            )
        ),
        attention_type=str(settings.get("attention_type", fallback_model_settings.get("attention_type", "mha"))),
        ffn_type=str(settings.get("ffn_type", fallback_model_settings.get("ffn_type", "gelu"))),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        architecture=architecture,
        vocab_size=int(built_config.vocab_size),
        max_position_embeddings=int(built_config.max_position_embeddings),
    )
