"""Dataset schema for synthetic multimodal benchmark tasks."""

from typing import Any

import albumentations as A
import numpy as np
from pytorch_lightning import Callback

from versatil.configs.experiment import ExperimentConfig
from versatil.data.constants import (
    Cameras,
    DatasetType,
    ProprioKey,
    SyntheticObsKey,
)
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.zarr_meta import DatasetMetadata
from versatil.data.synthetic.constants import SyntheticTaskName
from versatil.training.synthetic_rollout_callback import SyntheticRolloutCallback

ALLOWED_CAMERAS = {Cameras.AGENTVIEW.value}
ALLOWED_POSITION_KEYS = {ProprioKey.SYNTHETIC_POSITION.value}
ALLOWED_ACTION_KEYS = {ProprioKey.SYNTHETIC_POSITION_ACTION.value}
ALLOWED_OBSERVATION_KEYS = {
    SyntheticObsKey.CONTEXT.value,
    SyntheticObsKey.MODE_ID.value,
}


class SyntheticSchema(DatasetSchema):
    """Schema for procedurally generated synthetic 2D navigation datasets.

    Unlike CSV or HDF5 schemas, this schema has no raw data files to read.
    Episodes are generated on-the-fly via the synthetic generators module
    and written directly to Zarr.
    """

    def __init__(
        self,
        zarr_path: str,
        metadata: DatasetMetadata,
        dataset_type: str = DatasetType.SYNTHETIC.value,
        task_name: str = SyntheticTaskName.CIRCLE.value,
        num_episodes: int = 1000,
        seed: int = 42,
        image_size: int = 64,
        num_modes: int = 2,
        trajectory_length: int = 60,
        noise_std: float = 0.01,
        num_styles: int = 1,
        mode_weights: list[float] | None = None,
    ):
        """Initialize and validate the synthetic benchmark schema.

        Args:
            zarr_path: Path to save/load the zarr store.
            metadata: Metadata for zarr array creation.
            dataset_type: Type of dataset. Must be 'synthetic'.
            task_name: SyntheticTaskName.value string identifying the task.
            num_episodes: Total episodes to generate, balanced across modes.
            seed: Random seed for reproducible generation.
            image_size: Side length in pixels of rendered images (square).
            num_modes: Number of behavioral modes.
            trajectory_length: Number of timesteps per episode.
            noise_std: Standard deviation of Gaussian trajectory noise.
            num_styles: Number of sinusoidal style variations per corridor.
            mode_weights: Per-mode sampling weights. None for uniform.
        """
        if dataset_type != DatasetType.SYNTHETIC.value:
            raise ValueError(
                f"SyntheticSchema only supports dataset_type='{DatasetType.SYNTHETIC.value}', "
                f"got '{dataset_type}'"
            )
        self.task_name = task_name
        self.num_episodes = num_episodes
        self.seed = seed
        self.image_size = image_size
        self.num_modes = num_modes
        self.trajectory_length = trajectory_length
        self.noise_std = noise_std
        self.num_styles = num_styles
        self.mode_weights = mode_weights
        self._validate_metadata(metadata)
        super().__init__(
            zarr_path=zarr_path,
            metadata=metadata,
            dataset_type=dataset_type,
        )

    @staticmethod
    def _validate_metadata(metadata: DatasetMetadata) -> None:
        """Validate synthetic-specific metadata constraints.

        Args:
            metadata: The metadata to validate against.

        Raises:
            ValueError: If validation fails.
        """
        errors = []
        camera_keys = metadata.get_camera_keys()
        invalid_cameras = set(camera_keys) - ALLOWED_CAMERAS
        if invalid_cameras:
            errors.append(
                f"Invalid cameras for SyntheticSchema: {invalid_cameras}. "
                f"Allowed cameras: {ALLOWED_CAMERAS}"
            )
        position_keys = set(metadata.position_observations.keys())
        invalid_position_keys = position_keys - ALLOWED_POSITION_KEYS
        if invalid_position_keys:
            errors.append(
                f"Invalid position observation keys: {invalid_position_keys}. "
                f"Allowed: {ALLOWED_POSITION_KEYS}"
            )
        action_keys = set(metadata.precomputed_actions.keys())
        invalid_action_keys = action_keys - ALLOWED_ACTION_KEYS
        if invalid_action_keys:
            errors.append(
                f"Invalid precomputed action keys: {invalid_action_keys}. "
                f"Allowed: {ALLOWED_ACTION_KEYS}"
            )
        custom_keys = set(metadata.custom_observations.keys())
        invalid_custom_keys = custom_keys - ALLOWED_OBSERVATION_KEYS
        if invalid_custom_keys:
            errors.append(
                f"Invalid custom observation keys: {invalid_custom_keys}. "
                f"Allowed: {ALLOWED_OBSERVATION_KEYS}"
            )
        if errors:
            raise ValueError(
                "SyntheticSchema metadata validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def get_callbacks(self, experiment_config: ExperimentConfig) -> list[Callback]:
        """Provide a rollout evaluation callback for synthetic training."""
        return [
            SyntheticRolloutCallback(
                task_name=self.task_name,
                num_modes=self.num_modes,
                num_styles=self.num_styles,
                trajectory_length=self.trajectory_length,
                noise_std=self.noise_std,
                num_rollouts=200,
                image_size=self.image_size,
                log_every_n_epochs=experiment_config.val_every,
            )
        ]

    def extract_episode(
        self,
        episode_source: Any,
        resizer: A.Resize | A.NoOp,
        depth_resizer: A.Resize | A.NoOp,
    ) -> dict[str, np.ndarray]:
        """Not applicable for synthetic data.

        Synthetic episodes are generated procedurally, not extracted from
        raw files. Use create_zarr_from_synthetic instead.

        Raises:
            NotImplementedError: Always, since synthetic data is generated not extracted.
        """
        raise NotImplementedError(
            "SyntheticSchema does not support extract_episode(). "
            "Use create_zarr_from_synthetic.create_replay_buffer_from_synthetic() instead."
        )
