import enum


class FeatureType(str, enum.Enum):
    """Feature types for validation."""
    SPATIAL = "spatial"  # Features with (C, H, W) dimensions
    SEQUENTIAL = "sequential"  # Features with flat or (T, D) dimensions
    ANY = "any"  # Any feature type allowed


class SequentialFusionType(str, enum.Enum):
    """Types of sequential fusion methods."""
    CONCAT = 'concat'
    CROSS_ATTENTION = 'cross_attention'
    MLP = 'mlp'


class ConcatDimension(str, enum.Enum):
    """Dimensions to concatenate along."""
    CHANNEL = 'channel'
    HEIGHT = 'height'
    WIDTH = 'width'
