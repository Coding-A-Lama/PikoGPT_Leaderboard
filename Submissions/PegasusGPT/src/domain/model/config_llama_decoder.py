from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class LlamaDecoderConfig(BaseModel):
    """Validated configuration for a Llama 3 style decoder-only transformer."""

    model_config = ConfigDict(extra="forbid")

    vocab_size: PositiveInt = Field(..., description="Size of tokenizer vocabulary.")
    max_position_embeddings: PositiveInt = Field(
        ..., description="Maximum supported sequence length."
    )
    hidden_size: PositiveInt = Field(
        default=256, description="Transformer hidden representation size."
    )
    num_layers: PositiveInt = Field(default=18, description="Number of decoder blocks.")
    num_attention_heads: PositiveInt = Field(
        default=4, description="Number of query attention heads per block."
    )
    tie_word_embeddings: bool = Field(
        default=False,
        description="Share token embedding weights with the output projection head.",
    )
    n_kv_heads: PositiveInt = Field(
        default=4,
        description="Number of key/value heads for grouped-query attention.",
    )
    intermediate_size: PositiveInt = Field(
        default=688,
        description="SwiGLU intermediate projection size (~8/3 * hidden_size).",
    )
    rope_theta: float = Field(
        default=10000.0,
        description="Base frequency for Rotary Position Embeddings.",
    )
