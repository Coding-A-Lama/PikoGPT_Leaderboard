from __future__ import annotations

from torch import nn

from domain.model.config_gpt_decoder import GPTDecoderConfig
from domain.model.config_hybrid_decoder import HybridDecoderConfig
from domain.model.config_llama_decoder import LlamaDecoderConfig
from domain.model.gpt_decoder_model import GPT2StyleDecoder
from domain.model.hybrid_decoder_model import HybridDecoder
from domain.model.llama_decoder_model import LlamaDecoder

SUPPORTED_ARCHITECTURES = ("gpt2", "llama3", "hybrid")


def build_model_from_config(
    architecture: str,
    vocab_size: int,
    max_position_embeddings: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    *,
    tie_word_embeddings: bool = False,
    mlp_hidden_size: int | None = None,
    qkv_bias: bool = False,
    dropout: float = 0.1,
    n_kv_heads: int | None = None,
    intermediate_size: int | None = None,
    rope_theta: float = 10000.0,
    norm_type: str = "layernorm",
    norm_placement: str = "pre_norm",
    positional_encoding_type: str = "learned_absolute",
    attention_type: str = "mha",
    ffn_type: str = "gelu",
) -> tuple[nn.Module, GPTDecoderConfig | LlamaDecoderConfig | HybridDecoderConfig]:
    """Build a model and its config from architecture name and hyperparameters.

    Returns (model, model_config) tuple.
    """
    if architecture == "gpt2":
        if mlp_hidden_size is None:
            mlp_hidden_size = 4 * hidden_size
        model_config = GPTDecoderConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            tie_word_embeddings=tie_word_embeddings,
            mlp_hidden_size=mlp_hidden_size,
            qkv_bias=qkv_bias,
            dropout=dropout,
        )
        return GPT2StyleDecoder(model_config), model_config

    if architecture == "llama3":
        if n_kv_heads is None:
            n_kv_heads = num_attention_heads
        if intermediate_size is None:
            intermediate_size = int(8 / 3 * hidden_size)
        model_config = LlamaDecoderConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            tie_word_embeddings=tie_word_embeddings,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            rope_theta=rope_theta,
        )
        return LlamaDecoder(model_config), model_config

    if architecture == "hybrid":
        effective_n_kv_heads = num_attention_heads if attention_type == "mha" else (n_kv_heads or num_attention_heads)
        hybrid_mlp_hidden_size = mlp_hidden_size if ffn_type == "gelu" else None
        hybrid_intermediate_size = intermediate_size if ffn_type == "swiglu" else None
        model_config = HybridDecoderConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            norm_type=norm_type,
            norm_placement=norm_placement,
            positional_encoding_type=positional_encoding_type,
            attention_type=attention_type,
            ffn_type=ffn_type,
            n_kv_heads=effective_n_kv_heads,
            mlp_hidden_size=hybrid_mlp_hidden_size,
            intermediate_size=hybrid_intermediate_size,
            qkv_bias=qkv_bias,
            dropout=dropout,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
        )
        return HybridDecoder(model_config), model_config

    raise ValueError(
        f"Unknown architecture '{architecture}'. Supported: {', '.join(SUPPORTED_ARCHITECTURES)}"
    )
