"""Raw dataset metadata for creating the dataset zarr store for fast parallel access at training time.

`dtype` across all classes uses the zarr v3 type convention.
zarr v3 allowed dtypes are defined here https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass, field
from typing import Optional

from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.data.metadata import (
    ObservationMetadata,
    PositionObservationMetadata,
    OrientationObservationMetadata,
    GripperObservationMetadata,
    CameraMetadata,
    PrecomputedActionMetadata,
    PositionActionMetadata,
    OrientationActionMetadata,
    GripperActionMetadata,
)


@dataclass
class DatasetMetadata:
    """Raw dataset metadata needed for creating the dataset zarr store.

    This class aggregates all observation and action metadata from the raw dataset.
    Generic validation happens at instantiation via __post_init__.
    Dataset-specific validation should be performed by the DatasetSchema.

    Attributes:
        observations: Dict of all observation metadata, indexed by zarr store key.
            Values are CameraMetadata, or ObservationMetadata.
        precomputed_actions: Dict of precomputed action metadata, indexed by zarr store key.
            Values are PrecomputedActionMetadata.
    """

    observations: dict[str, ObservationMetadata | CameraMetadata] = field(
        default_factory=dict
    )
    precomputed_actions: dict[str, PrecomputedActionMetadata] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """Validate metadata consistency and resolve OmegaConf interpolation keys."""
        self.observations = resolve_dict_keys(self.observations)
        self.precomputed_actions = resolve_dict_keys(self.precomputed_actions)
        obs_keys = set(self.observations.keys())
        action_keys = set(self.precomputed_actions.keys())
        overlap = obs_keys & action_keys
        if overlap:
            raise ValueError(f"Keys cannot be both observations and actions: {overlap}")

        camera_keys = [
            k for k, v in self.observations.items() if isinstance(v, CameraMetadata)
        ]
        if len(camera_keys) != len(set(camera_keys)):
            raise ValueError(f"Duplicate camera keys found: {camera_keys}")

    @property
    def cameras(self) -> dict[str, CameraMetadata]:
        """Get all camera observations."""
        return {
            k: v for k, v in self.observations.items() if isinstance(v, CameraMetadata)
        }

    @property
    def position_observations(self) -> dict[str, PositionObservationMetadata]:
        """Get all position observations."""
        return {
            k: v
            for k, v in self.observations.items()
            if isinstance(v, PositionObservationMetadata)
        }

    @property
    def orientation_observations(self) -> dict[str, OrientationObservationMetadata]:
        """Get all orientation observations."""
        return {
            k: v
            for k, v in self.observations.items()
            if isinstance(v, OrientationObservationMetadata)
        }

    @property
    def gripper_observations(self) -> dict[str, GripperObservationMetadata]:
        """Get all gripper state observations."""
        return {
            k: v
            for k, v in self.observations.items()
            if isinstance(v, GripperObservationMetadata)
        }

    @property
    def proprioceptive_observations(
        self,
    ) -> dict[
        str,
        PositionObservationMetadata
        | OrientationObservationMetadata
        | GripperObservationMetadata,
    ]:
        """Get all proprioceptive observations (position, orientation, gripper)."""
        return {
            k: v
            for k, v in self.observations.items()
            if isinstance(
                v,
                (
                    PositionObservationMetadata,
                    OrientationObservationMetadata,
                    GripperObservationMetadata,
                ),
            )
        }

    @property
    def custom_observations(self) -> dict[str, ObservationMetadata]:
        """Get custom observations (not position, orientation, gripper, or camera)."""
        return {
            k: v
            for k, v in self.observations.items()
            if isinstance(v, ObservationMetadata)
            and not isinstance(
                v,
                (
                    PositionObservationMetadata,
                    OrientationObservationMetadata,
                    GripperObservationMetadata,
                ),
            )
        }

    @property
    def position_actions(self) -> dict[str, PositionActionMetadata]:
        """Get all precomputed position actions."""
        return {
            k: v
            for k, v in self.precomputed_actions.items()
            if isinstance(v, PositionActionMetadata)
        }

    @property
    def orientation_actions(self) -> dict[str, OrientationActionMetadata]:
        """Get all precomputed orientation actions."""
        return {
            k: v
            for k, v in self.precomputed_actions.items()
            if isinstance(v, OrientationActionMetadata)
        }

    @property
    def gripper_actions(self) -> dict[str, GripperActionMetadata]:
        """Get all precomputed gripper actions."""
        return {
            k: v
            for k, v in self.precomputed_actions.items()
            if isinstance(v, GripperActionMetadata)
        }

    @property
    def custom_actions(self) -> dict[str, PrecomputedActionMetadata]:
        """Get custom precomputed actions (not position, orientation, or gripper)."""
        return {
            k: v
            for k, v in self.precomputed_actions.items()
            if not isinstance(
                v,
                (
                    PositionActionMetadata,
                    OrientationActionMetadata,
                    GripperActionMetadata,
                ),
            )
        }

    def get_all_keys(self) -> list[str]:
        """Get all zarr keys (observations + actions)."""
        return list(self.observations.keys()) + list(self.precomputed_actions.keys())

    def get_camera_keys(self) -> list[str]:
        """Get list of all camera keys."""
        return list(self.cameras.keys())

    def get_proprio_dimension(self) -> int:
        """Get total proprioceptive observation dimension."""
        return sum(obs.dimension for obs in self.proprioceptive_observations.values())

    def get_gripper_dimension(self) -> int:
        """Get gripper state dimension."""
        return sum(obs.dimension for obs in self.gripper_observations.values())

    def has_precomputed_actions(self) -> bool:
        """Check if dataset has any precomputed actions."""
        return len(self.precomputed_actions) > 0

    def get_precomputed_action(self, key: str) -> Optional[PrecomputedActionMetadata]:
        """Get a specific precomputed action by key."""
        return self.precomputed_actions.get(key)

    def get_observation(
        self, key: str
    ) -> Optional[ObservationMetadata | CameraMetadata]:
        """Get a specific observation by key."""
        return self.observations.get(key)

    def has_observation(self, key: str) -> bool:
        """Check if an observation exists."""
        return key in self.observations
