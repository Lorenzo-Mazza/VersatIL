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


class AttentionType(str, enum.Enum):
    """Types of attention mechanisms."""

    MULTI_HEAD = "mha"
    GROUPED_QUERY = "gqa"


class ConditioningType(str, enum.Enum):
    """Types of conditional modulation for transformers."""

    ADALN = "adaln"  # Adaptive Layer Normalization (modulate after norm)
    FILM = "film"  # Feature-wise Linear Modulation (modulate features directly)
