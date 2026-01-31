"""Constants for policy components."""

import enum



class FeatureType(str, enum.Enum):
    """Feature types for decoder validation.

    - SPATIAL: (C, H, W) - image features from CNN/ViT
    - SEQUENTIAL: (T, D) - sequence features from transformers
    - FLAT: int or (D,) - pooled/embedded features
    """
    
    SPATIAL = "spatial"
    SEQUENTIAL = "sequential"
    FLAT = "flat"
