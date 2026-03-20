import enum


class ConcatDimension(enum.StrEnum):
    """Dimensions to concatenate along."""

    CHANNEL = "channel"
    HEIGHT = "height"
    WIDTH = "width"
