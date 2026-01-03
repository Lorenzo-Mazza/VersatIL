"""Configurations for the raw dataset metadata used to construct a zarr store for fast parallel access.
 `dtype` across all configs refers to zarr v3 storage data type.
 zarr v3 allowed dtypes are defined here https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class DatasetMetadataConfig:
    """Configuration for the raw dataset metadata used to create the dataset zarr store.

    Attributes:
        observations: Dict of observations (ObservationMetadataConfig subclasses or CameraMetadataConfig).
            Keys are zarr store keys to assign to each observation.
        precomputed_actions: Optional dict of precomputed action configurations, indexed by the zarr store key to use.
    """

    _target_: str = "refactoring.data.raw.zarr_meta.DatasetMetadata"
    observations: dict[str, Any] = MISSING
    precomputed_actions: dict[str, Any] = field(default_factory=dict)
