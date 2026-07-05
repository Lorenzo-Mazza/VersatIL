import enum


class AttentionDecompositionMode(enum.StrEnum):
    """Attention computation strategies for DFormer."""

    FULL = "full"
    SEPARABLE = "separable"


class Axis(enum.StrEnum):
    """Spatial image axes."""

    HEIGHT = "height"
    WIDTH = "width"


class PositionalEncodingType(enum.StrEnum):
    """Types of positional encodings."""

    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    ROPE = "rope"


class AttentionType(enum.StrEnum):
    """Types of attention mechanisms."""

    MULTI_HEAD = "mha"
    GROUPED_QUERY = "gqa"


class ConditioningType(enum.StrEnum):
    """Types of conditional modulation for transformers."""

    ADALN = "adaln"  # Adaptive Layer Normalization (modulate after norm)
    FILM = "film"  # Feature-wise Linear Modulation (modulate features directly)
