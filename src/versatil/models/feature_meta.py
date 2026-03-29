"""Feature metadata for typed feature validation across the encoding-decoding pipeline."""

import enum
from dataclasses import dataclass


class FeatureType(enum.StrEnum):
    """Feature shape types for pipeline validation.

    Determines how downstream modules (fusion, decoder) handle a feature tensor.
    The type is set explicitly by each encoder and fusion module, not inferred
    from dimension tuples.
    """

    SPATIAL = "spatial"  # (C, H, W) — image feature maps
    SEQUENTIAL = "sequential"  # (S, D) — token/time sequences
    FLAT = "flat"  # (D,) — pooled or embedded features


@dataclass(frozen=True)
class FeatureMetadata:
    """Typed descriptor for a named feature produced by an encoder or fusion module.

    Travels through the pipeline: encoder/fusion → pipeline registry → decoder validation.

    Args:
        key: Feature name (e.g. ``"rgb"``, ``"language"``, ``"fused_rgb_language"``).
        feature_type: One of ``FeatureType`` values.
        dimension: Shape excluding batch and time dimensions. Always a tuple:
            FLAT ``(D,)``, SEQUENTIAL ``(S, D)``, SPATIAL ``(C, H, W)``.
    """

    key: str
    feature_type: str
    dimension: tuple[int, ...]


def infer_feature_type(dimension: tuple[int, ...]) -> str:
    """Infer feature type from dimension tuple length.

    Args:
        dimension: Feature dimension tuple excluding batch and time.

    Returns:
        Feature type string value.

    Raises:
        ValueError: If the dimension tuple length does not match any known type.
    """
    match len(dimension):
        case 3:
            return FeatureType.SPATIAL.value
        case 2:
            return FeatureType.SEQUENTIAL.value
        case 1:
            return FeatureType.FLAT.value
        case _:
            raise ValueError(f"Cannot infer feature type from dimension: {dimension}")
