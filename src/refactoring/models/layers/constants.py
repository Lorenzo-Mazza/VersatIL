import enum


class AttentionDecompositionMode(str, enum.Enum):
    """Attention computation strategies."""
    FULL = "full"
    SEPARABLE = "separable"


class Axis(str, enum.Enum):
    HEIGHT = "height"
    WIDTH = "width"


class PositionalEncodingType(str, enum.Enum):
    """Types of positional encodings."""
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
