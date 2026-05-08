from __future__ import annotations

from typing import Literal

ArchitectureName = Literal["gpt2", "llama3", "hybrid"]

SUPPORTED_ARCHITECTURES: tuple[ArchitectureName, ...] = ("gpt2", "llama3", "hybrid")
MAX_CONTEXT_WINDOW = 1024
MAX_DECODER_LAYERS = 24
MAX_PROJECT_PARAMETERS = 40_000_000


def validate_model_shape(
    *,
    architecture: ArchitectureName,
    max_position_embeddings: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    mlp_hidden_size: int | None,
    qkv_bias: bool,
    n_kv_heads: int | None,
    intermediate_size: int | None,
    norm_type: str | None = None,
    norm_placement: str | None = None,
    positional_encoding_type: str | None = None,
    attention_type: str | None = None,
    ffn_type: str | None = None,
    tie_word_embeddings: bool = False,
) -> None:
    if max_position_embeddings > MAX_CONTEXT_WINDOW:
        raise ValueError(
            f"max_position_embeddings must be <= {MAX_CONTEXT_WINDOW} for this course project"
        )
    if num_layers > MAX_DECODER_LAYERS:
        raise ValueError(
            f"num_layers must be <= {MAX_DECODER_LAYERS} for this course project"
        )
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")

    if architecture == "gpt2":
        if n_kv_heads is not None:
            raise ValueError("n_kv_heads is only valid for architecture='llama3' or 'hybrid'")
        if intermediate_size is not None:
            raise ValueError("intermediate_size is only valid for architecture='llama3' or 'hybrid'")
        return

    if architecture == "llama3":
        effective_n_kv_heads = n_kv_heads or num_attention_heads
        if num_attention_heads % effective_n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by n_kv_heads for llama3")
        if mlp_hidden_size is not None:
            raise ValueError("mlp_hidden_size is only valid for architecture='gpt2' or 'hybrid'")
        if qkv_bias:
            raise ValueError("qkv_bias must be false for architecture='llama3'")
        return

    if architecture == "hybrid":
        if norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError("norm_type must be 'layernorm' or 'rmsnorm' for architecture='hybrid'")
        if norm_placement not in ("pre_norm", "post_norm"):
            raise ValueError("norm_placement must be 'pre_norm' or 'post_norm' for architecture='hybrid'")
        if positional_encoding_type not in ("learned_absolute", "rope"):
            raise ValueError("positional_encoding_type must be 'learned_absolute' or 'rope' for architecture='hybrid'")
        if attention_type not in ("mha", "gqa"):
            raise ValueError("attention_type must be 'mha' or 'gqa' for architecture='hybrid'")
        if ffn_type not in ("gelu", "swiglu"):
            raise ValueError("ffn_type must be 'gelu' or 'swiglu' for architecture='hybrid'")

        if attention_type == "mha":
            if n_kv_heads is not None and n_kv_heads != num_attention_heads:
                raise ValueError("n_kv_heads must equal num_attention_heads when attention_type='mha'")
        else:
            effective_n_kv_heads = n_kv_heads or num_attention_heads
            if num_attention_heads % effective_n_kv_heads != 0:
                raise ValueError("num_attention_heads must be divisible by n_kv_heads for gqa")

        if ffn_type == "gelu" and intermediate_size is not None:
            raise ValueError("intermediate_size is only valid when ffn_type='swiglu'")
        if ffn_type == "swiglu" and mlp_hidden_size is not None:
            raise ValueError("mlp_hidden_size is only valid when ffn_type='gelu'")
        return

    raise ValueError(f"Unknown architecture '{architecture}'")


def estimate_decoder_parameter_count(
    *,
    architecture: ArchitectureName,
    vocab_size: int,
    max_position_embeddings: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    mlp_hidden_size: int | None,
    qkv_bias: bool,
    n_kv_heads: int | None,
    intermediate_size: int | None,
    norm_type: str | None = None,
    norm_placement: str | None = None,
    positional_encoding_type: str | None = None,
    attention_type: str | None = None,
    ffn_type: str | None = None,
    tie_word_embeddings: bool = False,
) -> int:
    if architecture == "gpt2":
        effective_mlp_hidden_size = mlp_hidden_size or (4 * hidden_size)
        embedding_params = (vocab_size * hidden_size) + (max_position_embeddings * hidden_size)
        attention_params = (4 * hidden_size * hidden_size) + hidden_size
        if qkv_bias:
            attention_params += 3 * hidden_size
        mlp_params = (2 * hidden_size * effective_mlp_hidden_size) + effective_mlp_hidden_size + hidden_size
        layer_norm_params = 4 * hidden_size
        final_layer_norm_params = 2 * hidden_size
        output_head_params = 0 if tie_word_embeddings else hidden_size * vocab_size
        return (
            embedding_params
            + (num_layers * (attention_params + mlp_params + layer_norm_params))
            + final_layer_norm_params
            + output_head_params
        )

    if architecture == "llama3":
        effective_n_kv_heads = n_kv_heads or num_attention_heads
        effective_intermediate_size = intermediate_size or int(8 * hidden_size / 3)
        head_dim = hidden_size // num_attention_heads
        kv_hidden_size = effective_n_kv_heads * head_dim
        embedding_params = vocab_size * hidden_size
        attention_params = (2 * hidden_size * hidden_size) + (2 * hidden_size * kv_hidden_size)
        feedforward_params = 3 * hidden_size * effective_intermediate_size
        layer_norm_params = 2 * hidden_size
        final_norm_params = hidden_size
        output_head_params = 0 if tie_word_embeddings else hidden_size * vocab_size
        return (
            embedding_params
            + (num_layers * (attention_params + feedforward_params + layer_norm_params))
            + final_norm_params
            + output_head_params
        )

    head_dim = hidden_size // num_attention_heads
    effective_n_kv_heads = num_attention_heads if attention_type == "mha" else (n_kv_heads or num_attention_heads)
    kv_hidden_size = effective_n_kv_heads * head_dim

    token_embedding_params = vocab_size * hidden_size
    position_embedding_params = (
        max_position_embeddings * hidden_size
        if positional_encoding_type == "learned_absolute"
        else 0
    )

    attention_weights = (2 * hidden_size * hidden_size) + (2 * hidden_size * kv_hidden_size)
    attention_bias = hidden_size
    if qkv_bias:
        attention_bias += hidden_size + (2 * kv_hidden_size)
    attention_params = attention_weights + attention_bias

    if ffn_type == "gelu":
        effective_mlp_hidden_size = mlp_hidden_size or (4 * hidden_size)
        ffn_params = (2 * hidden_size * effective_mlp_hidden_size) + effective_mlp_hidden_size + hidden_size
    else:
        effective_intermediate_size = intermediate_size or int(8 / 3 * hidden_size)
        ffn_params = 3 * hidden_size * effective_intermediate_size

    per_norm_params = 2 * hidden_size if norm_type == "layernorm" else hidden_size
    block_norm_params = 2 * per_norm_params
    final_norm_params = per_norm_params
    output_head_params = 0 if tie_word_embeddings else hidden_size * vocab_size

    return (
        token_embedding_params
        + position_embedding_params
        + (num_layers * (attention_params + ffn_params + block_norm_params))
        + final_norm_params
        + output_head_params
    )
