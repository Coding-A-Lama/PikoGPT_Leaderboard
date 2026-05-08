from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class GPTDecoderConfig(BaseModel):
    """Validated configuration for a GPT-2 style decoder-only transformer."""

    model_config = ConfigDict(extra="forbid")

    vocab_size: PositiveInt = Field(..., description="Size of tokenizer vocabulary.")
    max_position_embeddings: PositiveInt = Field(
        ..., description="Maximum supported sequence length."
    )
    hidden_size: PositiveInt = Field(
        default=768, description="Transformer hidden representation size."
    )
    num_layers: PositiveInt = Field(default=12, description="Number of decoder blocks.")
    num_attention_heads: PositiveInt = Field(
        default=12, description="Number of attention heads per block."
    )
    tie_word_embeddings: bool = Field(
        default=False,
        description="Share token embedding weights with the output projection head.",
    )
    mlp_hidden_size: PositiveInt = Field(
        default=3072,
        description="Intermediate MLP size for each decoder block.",
    )
    qkv_bias: bool = Field(
        default=False,
        description="Whether to include bias terms in attention query/key/value projections.",
    )
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
