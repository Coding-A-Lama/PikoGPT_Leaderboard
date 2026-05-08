from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


NormType = Literal["layernorm", "rmsnorm"]
NormPlacementType = Literal["pre_norm", "post_norm"]
PositionalEncodingType = Literal["learned_absolute", "rope"]
AttentionType = Literal["mha", "gqa"]
FFNType = Literal["gelu", "swiglu"]


class HybridDecoderConfig(BaseModel):
    """Validated configuration for the component-search hybrid decoder."""

    model_config = ConfigDict(extra="forbid")

    vocab_size: PositiveInt = Field(..., description="Size of tokenizer vocabulary.")
    max_position_embeddings: PositiveInt = Field(..., description="Maximum supported sequence length.")
    hidden_size: PositiveInt = Field(default=256)
    num_layers: PositiveInt = Field(default=12)
    num_attention_heads: PositiveInt = Field(default=8)

    norm_type: NormType = Field(default="layernorm")
    norm_placement: NormPlacementType = Field(default="pre_norm")
    positional_encoding_type: PositionalEncodingType = Field(default="learned_absolute")
    attention_type: AttentionType = Field(default="mha")
    ffn_type: FFNType = Field(default="gelu")

    n_kv_heads: PositiveInt | None = Field(default=None)
    mlp_hidden_size: PositiveInt | None = Field(default=None)
    intermediate_size: PositiveInt | None = Field(default=None)

    qkv_bias: bool = Field(default=False)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    rope_theta: float = Field(default=10000.0)
    tie_word_embeddings: bool = Field(default=False)
