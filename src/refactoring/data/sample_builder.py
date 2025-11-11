"""Sample construction for episodic dataset.

Builds the final training/validation samples by:
- Processing images
- Adding proprioceptive data
- Adding actions
- Computing padding masks
"""
from typing import Any

import numpy as np
import torch

from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.data.action_processor import ActionProcessor
from refactoring.data.augmentation_pipeline import AugmentationPipeline
from refactoring.data.constants import (
    ACTION_KEY,
    GRIPPER_ACTION_KEY,
    IS_PAD_KEY,
    LANGUAGE_KEY,
    OBSERVATION_KEY,
    PHASE_LABEL_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    PROPRIO_STATE,
    Cameras,
    GripperType,
)


class SampleBuilder:
    """Builds training samples from raw episode data."""

    def __init__(
        self,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        obs_horizon: int,
        pred_horizon: int,
        action_backward_shift: int,
        augmentation_pipeline: AugmentationPipeline,
        action_processor: ActionProcessor,
    ):
        """
        Args:
            action_space: Configuration for action space
            observation_space: Configuration for observation space
            obs_horizon: Observation history length
            pred_horizon: Action prediction horizon
            action_backward_shift: Backward shift for action timing
            augmentation_pipeline: Handles augmentations
            action_processor: Processes actions
        """

        self.action_space = action_space
        self.observation_space = observation_space
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_backward_shift = action_backward_shift
        self.augmentation_pipeline = augmentation_pipeline
        self.action_processor = action_processor


    def build_sample(
        self,
        padded_data: dict[str, np.ndarray],
        action_dict: dict[str, np.ndarray],
        start_idx: int,
        sampler_indices: np.ndarray,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Build a complete training sample.

        Args:
            padded_data: Dictionary of padded episode data
            action_dict: Dictionary of computed actions
            start_idx: Starting index in sampler
            sampler_indices: Array of sampler indices

        Note:
            Padded data layout: [historical buffer | observation window | future]
                                    [0:k]             [k:k+H]            [k+H:end]
            where k=action_backward_shift, H=obs_horizon, end= prediction_horizon+k+H

        Returns:
            Dictionary containing observation and action dictionaries. Each sub-dictionary maps keys to tensors.
        """
        sample: dict[str, dict[str, torch.Tensor]] = {OBSERVATION_KEY: {}, ACTION_KEY: {}}
        angle, rotation_matrix = self.augmentation_pipeline.setup_rotation()
        self._add_images(sample=sample, padded_data=padded_data, angle=angle)
        if self.observation_space.use_proprioceptive_data:
            self._add_proprioceptive(sample=sample, padded_data=padded_data, angle=angle, rotation_matrix=rotation_matrix)
            self._add_additional_observation_keys(sample=sample, padded_data=padded_data)

        if angle != 0 and rotation_matrix is not None and self.action_space.predict_in_camera_frame:
            action_dict = self.action_processor.rotate_actions(action_dict=action_dict, R=rotation_matrix)
        for action_key, action_array in action_dict.items():
            if action_key == GRIPPER_ACTION_KEY and self.action_space.gripper_type == GripperType.BINARY.value:
                sample[ACTION_KEY][action_key] = torch.from_numpy(action_array).long()
            else:
                sample[ACTION_KEY][action_key] = torch.from_numpy(action_array).float()
        if self.action_space.task_has_phases:
            self._add_phase_labels(sample=sample, padded_data=padded_data)
        self._add_padding_mask(sample=sample, start_idx=start_idx, sampler_indices=sampler_indices)

        return sample


    def _add_images(
        self,
        sample: dict[str, Any],
        padded_data: dict[str, np.ndarray],
        angle: float,
    ) -> None:
        """Add processed images to sample."""
        for cam in self.observation_space.camera_keys:
            # the sampler now fetches action_backward_shift extra prior observations at the start of padded_data, effectively offsetting the entire sequence
            img = padded_data[cam][self.action_backward_shift: self.action_backward_shift + self.obs_horizon]
            if cam != Cameras.DEPTH.value:
                img = img.astype(np.float32) / 255.0
                img = self.augmentation_pipeline.apply_rgb_augmentations(img, angle)
                # Convert to (T, C, H, W)
                img = np.moveaxis(img, -1, 1)
            else:
                # Depth image processing
                img = self.augmentation_pipeline.apply_depth_augmentations(img, angle)
                if len(img.shape) == 3:
                    img = img.astype(np.float32)[:, None]
            sample[OBSERVATION_KEY][cam] = torch.from_numpy(img)


    def _add_proprioceptive(
        self,
        sample: dict[str, Any],
        padded_data: dict[str, np.ndarray],
        angle: float,
        rotation_matrix: np.ndarray | None = None,
    ) -> None:
        """Add robot proprioceptive observations to sample."""
        proprio_dict = {}
        if self.observation_space.use_proprio_base_frame:
            proprio_dict[PROPRIO_OBS_ROBOT_FRAME_KEY] = torch.from_numpy(
                padded_data[PROPRIO_OBS_ROBOT_FRAME_KEY][self.action_backward_shift : self.action_backward_shift + self.obs_horizon]
            ).float()
        if self.observation_space.use_proprio_camera_frame:
            camera_frame = padded_data[PROPRIO_OBS_CAMERA_FRAME_KEY][self.action_backward_shift : self.action_backward_shift + self.obs_horizon]
            if angle > 0 and rotation_matrix is not None:
                camera_frame = self.augmentation_pipeline.rotate_proprio_data(camera_frame, rotation_matrix)
            proprio_dict[PROPRIO_OBS_CAMERA_FRAME_KEY] = torch.from_numpy(camera_frame).float()
        sample[OBSERVATION_KEY][PROPRIO_STATE] = proprio_dict


    def _add_additional_observation_keys(
        self,
        sample: dict[str, Any],
        padded_data: dict[str, np.ndarray],
    ) -> None:
        """Add additional observations to sample, such as language."""
        if self.observation_space.use_language:
            lang_data = padded_data[LANGUAGE_KEY][self.action_backward_shift : self.action_backward_shift + self.obs_horizon]
            sample[OBSERVATION_KEY][LANGUAGE_KEY] = lang_data.tolist()
        for key in self.observation_space.custom_obs_keys:
            custom_data = padded_data[key][self.action_backward_shift : self.action_backward_shift + self.obs_horizon]
            sample[OBSERVATION_KEY][key] = torch.from_numpy(custom_data).float() # Assuming float type for custom obs


    def _add_phase_labels(
        self, sample: dict[str, Any], padded_data: dict[str, np.ndarray]
    ) -> None:
        """Add phase labels to sample."""
        action_slice_start = self._get_action_slice_start()
        padded_phases = padded_data[PHASE_LABEL_KEY]
        next_phase = padded_phases[action_slice_start + 1: action_slice_start + self.pred_horizon + 1]
        sample[ACTION_KEY][PHASE_LABEL_KEY] = torch.from_numpy(next_phase).long()


    def _add_padding_mask(
        self,
        sample: dict[str, Any],
        start_idx: int,
        sampler_indices: np.ndarray,
    ) -> None:
        """Add padding mask indicating which timesteps are padded."""
        action_slice_start = self._get_action_slice_start()
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = sampler_indices[start_idx]

        action_positions = np.arange(self.pred_horizon) + action_slice_start

        if self.action_space.deltas_as_actions:
            # For deltas, both current and next positions must be valid
            is_pad = np.logical_or(
                np.logical_or(
                    action_positions < sample_start_idx,
                    action_positions >= sample_end_idx,
                ),
                np.logical_or(
                    action_positions + 1 < sample_start_idx,
                    action_positions + 1 >= sample_end_idx,
                ),
            )
        else:
            # For absolute positions, only next position must be valid
            is_pad = np.logical_or(
                action_positions + 1 < sample_start_idx,
                action_positions + 1 >= sample_end_idx,
            )

        sample[ACTION_KEY][IS_PAD_KEY] = torch.from_numpy(is_pad).bool()

    def _get_action_slice_start(self) -> int:
        """Get the starting index for action slice."""
        return self.obs_horizon - 1

