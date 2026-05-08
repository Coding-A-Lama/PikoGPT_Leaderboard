from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from domain.model.config_gpt_decoder import GPTDecoderConfig

TIED_EMBEDDING_INIT_STD = 0.02


class TokenEmbedding(nn.Module):
    """Maps token ids to dense trainable vectors."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    """Learned absolute positional embedding."""

    def __init__(self, hidden_size: int, max_position_embeddings: int = 512) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, position_ids: Tensor) -> Tensor:
        return self.embedding(position_ids)


class MaskedMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention that only attends to valid history."""

    def __init__(self, hidden_size: int, num_attention_heads: int, qkv_bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout = dropout

        self.query = nn.Linear(hidden_size, hidden_size, bias = qkv_bias)
        self.key = nn.Linear(hidden_size, hidden_size, bias = qkv_bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias = qkv_bias)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

        self.attention_dropout_rate = dropout
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape

        query = self._split_heads(self.query(hidden_states), batch_size, sequence_length)
        key = self._split_heads(self.key(hidden_states), batch_size, sequence_length)
        value = self._split_heads(self.value(hidden_states), batch_size, sequence_length)

        context = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            dropout_p=self.attention_dropout_rate if self.training else 0.0,
            is_causal=True, # causal masking!
        )
        context = self._merge_heads(context, batch_size, sequence_length, hidden_size)

        projected_context = self.output_projection(context)
        return self.residual_dropout(projected_context)

    def _split_heads(self, tensor: Tensor, batch_size: int, sequence_length: int) -> Tensor:
        tensor = tensor.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(
        self,
        tensor: Tensor,
        batch_size: int,
        sequence_length: int,
        hidden_size: int,
    ) -> Tensor:
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, sequence_length, hidden_size)


class DecoderMLP(nn.Module):
    """GPT-2 style feed-forward sublayer."""

    def __init__(self, hidden_size: int, mlp_hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return self.dropout(hidden_states)


class GPTDecoderBlock(nn.Module):
    """Pre-LN decoder block with attention and MLP residual branches."""

    def __init__(self, config: GPTDecoderConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.self_attention = MaskedMultiHeadSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
            qkv_bias=config.qkv_bias,
        )
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
        self.mlp = DecoderMLP(
            hidden_size=config.hidden_size,
            mlp_hidden_size=config.mlp_hidden_size,
            dropout=config.dropout,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        attention_output = self.self_attention(
            hidden_states=self.attention_norm(hidden_states)
        )
        hidden_states = hidden_states + attention_output

        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_output
        return hidden_states


class GPT2StyleDecoder(nn.Module):
    """Decoder-only GPT pipeline with learned token/position embeddings and LM head."""

    def __init__(self, config: GPTDecoderConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.config = config
        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.position_embedding = PositionalEmbedding(
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.input_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            GPTDecoderBlock(config=config) for _ in range(config.num_layers)
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            nn.init.normal_(self.token_embedding.embedding.weight, mean=0.0, std=TIED_EMBEDDING_INIT_STD)
            self.output_head.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                "Input sequence length exceeds max_position_embeddings configured for model"
            )

        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, sequence_length)

        hidden_states = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        hidden_states = self.input_dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)
        return self.output_head(hidden_states)
