"""Task space definitions for runtime data requirements.

This module defines what data an experiment uses at runtime:
"""

from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.data.constants import (
    VALID_CAMERAS,
    ActionComputationMethod,
    ObsKey,
)
from versatil.data.metadata import (
    ActionMetadata,
    CameraMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    ObservationMetadata,
    OnTheFlyActionMetadata,
    OrientationActionMetadata,
    OrientationObservationMetadata,
    PositionActionMetadata,
    PositionObservationMetadata,
    PrecomputedActionMetadata,
)
from versatil.data.raw.schemas.base import DatasetSchema


class ActionSpace:
    """Defines what actions the task will predict at runtime.


    Attributes:
        actions_metadata: Dict of all action metadata, indexed by zarr store key.
            Values are OnTheFlyActionMetadata or PrecomputedActionMetadata subclasses.
        use_gripper_class_weights: Whether to use class weights for binary gripper.
        denoise_actions: Whether to apply denoising to actions.
        denoising_percentile: Percentile for denoising threshold.
    """

    def __init__(
        self,
        actions_metadata: dict[str, ActionMetadata],
        use_gripper_class_weights: bool = False,
        denoise_actions: bool = True,
        denoising_percentile: float = 15.0,
    ):
        self.actions_metadata = resolve_dict_keys(actions_metadata)
        self.use_gripper_class_weights = use_gripper_class_weights
        self.denoise_actions = denoise_actions
        self.denoising_percentile = denoising_percentile

    @property
    def on_the_fly_actions(self) -> dict[str, OnTheFlyActionMetadata]:
        """Get actions computed on-the-fly from observations."""
        return {
            k: v
            for k, v in self.actions_metadata.items()
            if isinstance(v, OnTheFlyActionMetadata)
        }

    @property
    def precomputed_actions(self) -> dict[str, PrecomputedActionMetadata]:
        """Get precomputed actions loaded directly from zarr."""
        return {
            k: v
            for k, v in self.actions_metadata.items()
            if isinstance(v, PrecomputedActionMetadata)
        }

    @property
    def position_actions(
        self,
    ) -> dict[str, PositionActionMetadata | OnTheFlyActionMetadata]:
        """Get all position actions (precomputed or on-the-fly)."""
        result = {}
        for k, v in self.actions_metadata.items():
            if (
                isinstance(v, PositionActionMetadata)
                or isinstance(v, OnTheFlyActionMetadata)
                and isinstance(v.source_metadata, PositionObservationMetadata)
            ):
                result[k] = v
        return result

    @property
    def orientation_actions(
        self,
    ) -> dict[str, OrientationActionMetadata | OnTheFlyActionMetadata]:
        """Get all orientation actions (precomputed or on-the-fly)."""
        result = {}
        for k, v in self.actions_metadata.items():
            if (
                isinstance(v, OrientationActionMetadata)
                or isinstance(v, OnTheFlyActionMetadata)
                and isinstance(v.source_metadata, OrientationObservationMetadata)
            ):
                result[k] = v
        return result

    @property
    def gripper_actions(
        self,
    ) -> dict[str, GripperActionMetadata | OnTheFlyActionMetadata]:
        """Get all gripper actions (precomputed or on-the-fly)."""
        result = {}
        for k, v in self.actions_metadata.items():
            if (
                isinstance(v, GripperActionMetadata)
                or isinstance(v, OnTheFlyActionMetadata)
                and isinstance(v.source_metadata, GripperObservationMetadata)
            ):
                result[k] = v
        return result

    def get_total_action_dim(self) -> int:
        """Calculate total action dimension for predicted actions."""
        return sum(
            meta.prediction_dimension
            for meta in self.actions_metadata.values()
            if meta.requires_prediction_head
        )

    @property
    def position_dim(self) -> int:
        """Get total position action dimension."""
        dim = 0
        for meta in self.actions_metadata.values():
            if (
                isinstance(meta, PositionActionMetadata)
                or isinstance(meta, OnTheFlyActionMetadata)
                and isinstance(meta.source_metadata, PositionObservationMetadata)
            ):
                dim += meta.prediction_dimension
        return dim

    @property
    def orientation_dim(self) -> int:
        """Get total orientation action dimension."""
        dim = 0
        for meta in self.actions_metadata.values():
            if (
                isinstance(meta, OrientationActionMetadata)
                or isinstance(meta, OnTheFlyActionMetadata)
                and isinstance(meta.source_metadata, OrientationObservationMetadata)
            ):
                dim += meta.prediction_dimension
        return dim

    @property
    def gripper_dim(self) -> int:
        """Get total gripper action dimension."""
        dim = 0
        for meta in self.actions_metadata.values():
            if (
                isinstance(meta, GripperActionMetadata)
                or isinstance(meta, OnTheFlyActionMetadata)
                and isinstance(meta.source_metadata, GripperObservationMetadata)
            ):
                dim += meta.prediction_dimension
        return dim

    @property
    def has_on_the_fly_actions(self) -> bool:
        """Check if there are any actions to compute on-the-fly."""
        return len(self.on_the_fly_actions) > 0

    @property
    def has_precomputed_actions(self) -> bool:
        """Check if there are any precomputed actions."""
        return len(self.precomputed_actions) > 0

    @property
    def has_delta_actions(self) -> bool:
        """Check if any actions are computed as deltas."""
        return any(
            meta.computation_method == ActionComputationMethod.DELTA.value
            for meta in self.on_the_fly_actions.values()
        )

    @property
    def has_position_actions(self) -> bool:
        """Check if there are any position actions."""
        return len(self.position_actions) > 0

    @property
    def has_orientation_actions(self) -> bool:
        """Check if there are any orientation actions."""
        return len(self.orientation_actions) > 0

    @property
    def has_gripper_actions(self) -> bool:
        """Check if there are any gripper actions."""
        return len(self.gripper_actions) > 0

    @property
    def task_has_phases(self) -> bool:
        """Check if the task has phase labels."""
        return ObsKey.PHASE_LABEL.value in self.actions_metadata

    def get_required_zarr_keys(self) -> list[str]:
        """Get zarr keys needed for this action space.

        Returns:
            List of keys to load from replay buffer
        """
        return list(self.actions_metadata.keys())


class ObservationSpace:
    """Defines what observations the task will load at runtime.


    Attributes:
        observations_metadata: Dict of all observation metadata, indexed by zarr store key.
            Values are ObservationMetadata subclasses or CameraMetadata.
    """

    def __init__(
        self,
        observations_metadata: dict[str, ObservationMetadata | CameraMetadata],
    ):
        self.observations_metadata = resolve_dict_keys(observations_metadata)

    @property
    def cameras(self) -> dict[str, CameraMetadata]:
        """Get all camera observations."""
        return {
            k: v
            for k, v in self.observations_metadata.items()
            if isinstance(v, CameraMetadata)
        }

    @property
    def position_observations(self) -> dict[str, PositionObservationMetadata]:
        """Get all position observations."""
        return {
            k: v
            for k, v in self.observations_metadata.items()
            if isinstance(v, PositionObservationMetadata)
        }

    @property
    def orientation_observations(self) -> dict[str, OrientationObservationMetadata]:
        """Get all orientation observations."""
        return {
            k: v
            for k, v in self.observations_metadata.items()
            if isinstance(v, OrientationObservationMetadata)
        }

    @property
    def gripper_observations(self) -> dict[str, GripperObservationMetadata]:
        """Get all gripper state observations."""
        return {
            k: v
            for k, v in self.observations_metadata.items()
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
            for k, v in self.observations_metadata.items()
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
            for k, v in self.observations_metadata.items()
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
    def has_cameras(self) -> bool:
        """Check if any camera observations are included."""
        return len(self.cameras) > 0

    @property
    def has_gripper_state(self) -> bool:
        """Check if gripper state observation is included."""
        return len(self.gripper_observations) > 0

    @property
    def has_proprioceptive_state(self) -> bool:
        """Check if any proprioceptive observations are included."""
        return len(self.proprioceptive_observations) > 0

    @property
    def has_proprioceptive_position(self) -> bool:
        """Check if any position observations are included."""
        return len(self.position_observations) > 0

    @property
    def has_proprioceptive_orientation(self) -> bool:
        """Check if any orientation observations are included."""
        return len(self.orientation_observations) > 0

    def get_required_zarr_keys(self) -> list[str]:
        """Get all zarr keys needed for this observation space.

        Returns:
            List of keys to load from replay buffer
        """
        return list(self.observations_metadata.keys())


class TaskSpace:
    """Combines action/observation space with dataset schema for runtime.

    The task space validates that requested keys exist in the dataset schema
    and that metadata is consistent between the schema and task requirements.
    """

    def __init__(
        self,
        dataset_schema: DatasetSchema,
        dataloader: DataLoaderConfig,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        prediction_horizon: int = 16,
        observation_horizon: int = 1,
    ):
        """Initialize task space.

        Args:
            dataset_schema: Schema defining what's in the zarr store.
            dataloader: Data loading configuration.
            action_space: Actions to predict.
            observation_space: Observations to load.
            prediction_horizon: Number of timesteps to predict.
            observation_horizon: Number of history timesteps to load.
        """
        self.dataset_schema = dataset_schema
        self.dataloader = dataloader
        self.action_space = action_space
        self.observation_space = observation_space
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon
        self._validate()

    def _validate(self) -> None:
        """Validate task configuration against dataset schema."""
        zarr_keys = set(self.dataset_schema.get_required_zarr_keys())
        for key in self.action_space.get_required_zarr_keys():
            action_meta = self.action_space.actions_metadata[key]
            if isinstance(action_meta, OnTheFlyActionMetadata):
                if key not in zarr_keys:
                    raise ValueError(
                        f"On-the-fly action '{key}' references observation that doesn't exist in dataset schema."
                    )
                schema_obs = self.dataset_schema.metadata.get_observation(key)
                if schema_obs is None:
                    raise ValueError(
                        f"On-the-fly action '{key}' references observation not found in schema."
                    )
                if action_meta.source_metadata != schema_obs:
                    raise ValueError(
                        f"On-the-fly action '{key}' metadata mismatch with schema"
                    )
            else:
                if key not in zarr_keys:
                    raise ValueError(
                        f"Precomputed action '{key}' not found in dataset schema."
                    )
        for key, task_obs in self.observation_space.observations_metadata.items():
            if key not in zarr_keys:
                raise ValueError(f"Observation '{key}' not found in dataset schema.")
            schema_obs = self.dataset_schema.metadata.observations.get(key)

            if schema_obs is not None and task_obs != schema_obs:
                raise ValueError(f"Observation '{key}' metadata mismatch with schema")
        for cam_key in self.observation_space.cameras:
            if cam_key not in VALID_CAMERAS:
                raise ValueError(
                    f"Invalid camera key '{cam_key}', must be one of {VALID_CAMERAS}. "
                    f"To add custom camera keys, add them to constants.data.Cameras enum."
                )
        if self.observation_horizon < 1:
            raise ValueError(
                f"observation_horizon must be >= 1, got {self.observation_horizon}"
            )
        if self.prediction_horizon < 1:
            raise ValueError(
                f"prediction_horizon must be >= 1, got {self.prediction_horizon}"
            )
