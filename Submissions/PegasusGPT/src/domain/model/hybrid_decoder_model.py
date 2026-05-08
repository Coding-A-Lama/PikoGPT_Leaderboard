from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from domain.model.config_hybrid_decoder import HybridDecoderConfig
from domain.model.gpt_decoder_model import DecoderMLP, PositionalEmbedding, TokenEmbedding
from domain.model.llama_decoder_model import (
    RMSNorm,
    SwiGLUFFN,
    _apply_rotary_embedding,
    _precompute_rope_frequencies,
)


def _make_norm(norm_type: str, hidden_size: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_size)
    if norm_type == "rmsnorm":
        return RMSNorm(hidden_size)
    raise ValueError(f"Unknown norm_type '{norm_type}'")


def _resolve_effective_ffn_size(config: HybridDecoderConfig) -> int:
    if config.ffn_type == "gelu":
        return config.mlp_hidden_size or (4 * config.hidden_size)
    return config.intermediate_size or int(8 / 3 * config.hidden_size)


def _resolve_effective_n_kv_heads(config: HybridDecoderConfig) -> int:
    if config.attention_type == "mha":
        return config.num_attention_heads
    return config.n_kv_heads or config.num_attention_heads


class HybridSelfAttention(nn.Module):
    """Causal self-attention supporting MHA/GQA and learned/RoPE positions."""

    def __init__(self, config: HybridDecoderConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_attention_heads = config.num_attention_heads
        self.n_kv_heads = _resolve_effective_n_kv_heads(config)
        if config.num_attention_heads % self.n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by n_kv_heads")

        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_group_size = config.num_attention_heads // self.n_kv_heads
        self.use_rope = config.positional_encoding_type == "rope"

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.qkv_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.qkv_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=config.qkv_bias,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.attention_dropout_rate = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)

        if self.use_rope:
            cos_table, sin_table = _precompute_rope_frequencies(
                head_dim=self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                theta=config.rope_theta,
            )
            self.register_buffer("cos_table", cos_table, persistent=False)
            self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states)
        key = key.view(batch_size, sequence_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states)
        value = value.view(batch_size, sequence_length, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos = self.cos_table[:sequence_length]
            sin = self.sin_table[:sequence_length]
            query = _apply_rotary_embedding(query, cos, sin)
            key = _apply_rotary_embedding(key, cos, sin)

        if self.kv_group_size > 1:
            key = key.repeat_interleave(self.kv_group_size, dim=1)
            value = value.repeat_interleave(self.kv_group_size, dim=1)

        context = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            dropout_p=self.attention_dropout_rate if self.training else 0.0,
            is_causal=True,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.hidden_size)
        return self.residual_dropout(self.o_proj(context))


def _make_ffn(config: HybridDecoderConfig) -> nn.Module:
    effective_size = _resolve_effective_ffn_size(config)
    if config.ffn_type == "gelu":
        return DecoderMLP(
            hidden_size=config.hidden_size,
            mlp_hidden_size=effective_size,
            dropout=config.dropout,
        )
    if config.ffn_type == "swiglu":
        return SwiGLUFFN(
            hidden_size=config.hidden_size,
            intermediate_size=effective_size,
        )
    raise ValueError(f"Unknown ffn_type '{config.ffn_type}'")


class HybridDecoderBlock(nn.Module):
    """Decoder block with configurable norm placement, attention, and FFN."""

    def __init__(self, config: HybridDecoderConfig) -> None:
        super().__init__()
        self.norm_placement = config.norm_placement
        self.attention_norm = _make_norm(config.norm_type, config.hidden_size)
        self.self_attention = HybridSelfAttention(config)
        self.mlp_norm = _make_norm(config.norm_type, config.hidden_size)
        self.mlp = _make_ffn(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.norm_placement == "pre_norm":
            hidden_states = hidden_states + self.self_attention(self.attention_norm(hidden_states))
            hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
            return hidden_states

        hidden_states = self.attention_norm(hidden_states + self.self_attention(hidden_states))
        hidden_states = self.mlp_norm(hidden_states + self.mlp(hidden_states))
        return hidden_states


class HybridDecoder(nn.Module):
    """Decoder-only transformer with independently selectable core components."""

    def __init__(self, config: HybridDecoderConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        if config.positional_encoding_type == "learned_absolute":
            self.position_embedding = PositionalEmbedding(
                hidden_size=config.hidden_size,
                max_position_embeddings=config.max_position_embeddings,
            )
        else:
            self.position_embedding = None

        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(HybridDecoderBlock(config=config) for _ in range(config.num_layers))
        self.final_norm = _make_norm(config.norm_type, config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.output_head.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError("Input sequence length exceeds max_position_embeddings configured for model")

        hidden_states = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            position_ids = torch.arange(sequence_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, sequence_length)
            hidden_states = hidden_states + self.position_embedding(position_ids)

        hidden_states = self.input_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        return self.output_head(hidden_states)
