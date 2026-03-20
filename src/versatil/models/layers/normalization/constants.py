import enum


class NormalizationType(enum.StrEnum):
    """Types of normalization layers."""

    LAYER_NORM = "layernorm"
    RMS_NORM = "rmsnorm"
    ADALN = "adaptive_layernorm"
    ADARMS = "adaptive_rms"
    FROZEN_BATCHNORM2D = "frozen_batchnorm_2D"
