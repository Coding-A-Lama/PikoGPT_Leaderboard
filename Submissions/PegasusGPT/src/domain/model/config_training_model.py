from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from domain.model.model_config_validation import ArchitectureName, validate_model_shape


class TrainingModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: ArchitectureName = Field(
        default="gpt2",
        description="Model architecture: 'gpt2', 'llama3', or 'hybrid'.",
    )
    max_position_embeddings: PositiveInt = Field(
        ..., description="Maximum supported sequence length."
    )
    hidden_size: PositiveInt = Field(default=768)
    num_layers: PositiveInt = Field(default=12)
    num_attention_heads: PositiveInt = Field(default=12)
    tie_word_embeddings: bool = Field(
        default=False,
        description="Share token embedding weights with the output projection head.",
    )

    # GPT-2 / hybrid GELU FFN
    mlp_hidden_size: PositiveInt | None = Field(
        default=None,
        description="Intermediate MLP size for GELU FFN. Defaults to 4 * hidden_size.",
    )
    qkv_bias: bool = Field(default=False)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)

    # Llama 3 / hybrid GQA / SwiGLU
    n_kv_heads: PositiveInt | None = Field(
        default=None,
        description="Number of KV heads for grouped-query attention. Defaults to num_attention_heads.",
    )
    intermediate_size: PositiveInt | None = Field(
        default=None,
        description="SwiGLU intermediate size. Defaults to ~8/3 * hidden_size.",
    )
    rope_theta: float = Field(default=10000.0)

    # Hybrid component selection
    norm_type: Literal["layernorm", "rmsnorm"] = Field(
        default="layernorm",
        description="Block normalization used in the hybrid architecture.",
    )
    norm_placement: Literal["pre_norm", "post_norm"] = Field(
        default="pre_norm",
        description="Block normalization placement used in the hybrid architecture.",
    )
    positional_encoding_type: Literal["learned_absolute", "rope"] = Field(
        default="learned_absolute",
        description="Positional encoding used in the hybrid architecture.",
    )
    attention_type: Literal["mha", "gqa"] = Field(
        default="mha",
        description="Attention variant used in the hybrid architecture.",
    )
    ffn_type: Literal["gelu", "swiglu"] = Field(
        default="gelu",
        description="Feed-forward variant used in the hybrid architecture.",
    )

    @model_validator(mode="after")
    def validate_architecture_fields(self) -> "TrainingModelConfig":
        validate_model_shape(
            architecture=self.architecture,
            max_position_embeddings=self.max_position_embeddings,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            tie_word_embeddings=self.tie_word_embeddings,
            mlp_hidden_size=self.mlp_hidden_size,
            qkv_bias=self.qkv_bias,
            n_kv_heads=self.n_kv_heads,
            intermediate_size=self.intermediate_size,
            norm_type=self.norm_type if self.architecture == "hybrid" else None,
            norm_placement=self.norm_placement if self.architecture == "hybrid" else None,
            positional_encoding_type=self.positional_encoding_type if self.architecture == "hybrid" else None,
            attention_type=self.attention_type if self.architecture == "hybrid" else None,
            ffn_type=self.ffn_type if self.architecture == "hybrid" else None,
        )
        return self
