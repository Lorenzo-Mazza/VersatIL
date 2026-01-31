import enum


class ConcatDimension(str, enum.Enum):
    """Dimensions to concatenate along."""

    CHANNEL = "channel"
    HEIGHT = "height"
    WIDTH = "width"
