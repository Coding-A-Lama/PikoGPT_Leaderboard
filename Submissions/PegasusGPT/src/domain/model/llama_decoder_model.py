from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from domain.model.config_llama_decoder import LlamaDecoderConfig

TIED_EMBEDDING_INIT_STD = 0.02


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama style)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(hidden_states.dtype)


def _precompute_rope_frequencies(
    head_dim: int,
    max_position_embeddings: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin tables for Rotary Position Embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_position_embeddings, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    cos_table = freqs.cos()
    sin_table = freqs.sin()
    return cos_table, sin_table


def _apply_rotary_embedding(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE rotation to a (batch, heads, seq_len, head_dim) tensor."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with Rotary Position Embeddings (no bias)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        n_kv_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by n_kv_heads")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // num_attention_heads
        self.kv_group_size = num_attention_heads // n_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        cos_table, sin_table = _precompute_rope_frequencies(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            theta=rope_theta,
        )
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, sequence_length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, sequence_length, self.n_kv_heads, self.head_dim).transpose(1, 2)

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
            is_causal=True,
        )

        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.hidden_size)
        return self.o_proj(context)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (three projections, SiLU gating, no bias)."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LlamaDecoderBlock(nn.Module):
    """Pre-norm decoder block with GQA + RoPE and SwiGLU FFN."""

    def __init__(self, config: LlamaDecoderConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.self_attention = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            n_kv_heads=config.n_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
        )
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLUFFN(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.self_attention(self.attention_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class LlamaDecoder(nn.Module):
    """Llama 3 style decoder-only transformer with RoPE, GQA, RMSNorm, and SwiGLU."""

    def __init__(self, config: LlamaDecoderConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            LlamaDecoderBlock(config=config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=TIED_EMBEDDING_INIT_STD)
            self.output_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                "Input sequence length exceeds max_position_embeddings configured for model"
            )

        hidden_states = self.token_embedding(input_ids)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        return self.output_head(hidden_states)
