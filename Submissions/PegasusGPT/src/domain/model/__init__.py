"""Lightweight package exports for domain.model.

Keep this module import-light so config-only imports do not pull in torch-backed model
implementations unless they are needed explicitly.
"""

from domain.model.model_config_validation import (
    ArchitectureName,
    MAX_CONTEXT_WINDOW,
    MAX_DECODER_LAYERS,
    MAX_PROJECT_PARAMETERS,
    SUPPORTED_ARCHITECTURES,
)

__all__ = [
    "ArchitectureName",
    "SUPPORTED_ARCHITECTURES",
    "MAX_CONTEXT_WINDOW",
    "MAX_DECODER_LAYERS",
    "MAX_PROJECT_PARAMETERS",
]
