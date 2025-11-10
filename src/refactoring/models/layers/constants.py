import enum


class AttentionDecompositionMode(str, enum.Enum):
    """Attention computation strategies for DFormer."""
    FULL = "full"
    SEPARABLE = "separable"


class Axis(str, enum.Enum):
    HEIGHT = "height"
    WIDTH = "width"


class PositionalEncodingType(str, enum.Enum):
    """Types of positional encodings."""
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROPE = "rope"


class NormalizationType(str, enum.Enum):
    """Types of normalization layers for transformers."""
    LAYER_NORM = "layernorm"
    RMS_NORM = "rmsnorm"


class AttentionType(str, enum.Enum):
    """Types of attention mechanisms."""
    MULTI_HEAD = "mha"
    GROUPED_QUERY = "gqa"
