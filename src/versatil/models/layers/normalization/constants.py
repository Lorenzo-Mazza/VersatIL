import enum


class NormalizationType(enum.StrEnum):
    """Types of normalization layers."""

    LAYER_NORM = "layernorm"
    RMS_NORM = "rmsnorm"
