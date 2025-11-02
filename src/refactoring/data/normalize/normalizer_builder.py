import logging

import numpy as np
import torch

from refactoring.configs.task.task import ObservationSpace
from refactoring.data.action_processor import ActionProcessor
from refactoring.data.constants import (
    GRIPPER_ACTION_KEY,
    GRIPPER_STATE_OBS_KEY,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    Cameras,
    GripperType,
)
from refactoring.data.normalize.image_normalizer import (
    get_depth_image_normalizer,
    get_rgb_image_normalizer,
)
from refactoring.data.normalize.normalizer import LinearNormalizer
from refactoring.data.preprocessing.replay_buffer import ReplayBuffer


class NormalizerBuilder:
    """Builder for creating and configuring normalizers."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        action_processor: ActionProcessor,
        observation_space: ObservationSpace,
        episode_ends: np.ndarray,
        kinematics_norm_type: str,
        image_norm_type: str,
        depth_norm_type: str,
    ):
        """Initialize normalizer builder.

        Args:
            replay_buffer: Data source
            action_processor: For computing actions and applying denoising
            observation_space: The observation space configuration
            episode_ends: Episode boundaries
            kinematics_norm_type: Normalization type for kinematics
            image_norm_type: Normalization type for RGB images
            depth_norm_type: Normalization type for depth images
        """
        self.replay_buffer = replay_buffer
        self.action_processor = action_processor
        self.observation_space = observation_space
        self.episode_ends = episode_ends
        self.kinematics_norm_type = kinematics_norm_type
        self.image_norm_type = image_norm_type
        self.depth_norm_type = depth_norm_type


    def create_normalizer(
        self,
        device: torch.device | None = None,
        winsorize_depth: bool = True,
        **kwargs
    ) -> LinearNormalizer:
        """Create and fit normalizer for this dataset.

        Args:
            device: Target device for tensors
            winsorize_depth: Apply winsorization to depth values
            **kwargs: Additional arguments for normalizer fitting

        Returns:
            Fitted LinearNormalizer instance
        """
        normalizer = LinearNormalizer()
        proprio_data = self._read_proprio_data_from_buffer()
        normalizer.fit(
            data=proprio_data,
            last_n_dims=1,
            mode=self.kinematics_norm_type,
            device=device,
            range_eps=1e-10,
            **kwargs
        )
        self._setup_image_normalizers(normalizer, device, winsorize_depth)
        self._log_normalizer_stats(normalizer)

        return normalizer

    def _read_proprio_data_from_buffer(self) -> dict[str, np.ndarray]:
        """Read proprioceptive data from the replay buffer.

        Returns:
            Dictionary of proprioceptive data
        """
        action_key = PROPRIO_OBS_CAMERA_FRAME_KEY if self.action_processor.predict_in_camera_frame else PROPRIO_OBS_ROBOT_FRAME_KEY
        obs_for_actions = self.replay_buffer[action_key][:]
        if len(obs_for_actions) == 0:
            raise ValueError("Replay buffer is empty. Cannot compute normalization statistics.")

        cross_indices = self.episode_ends[:-1] - 1
        valid_mask = np.ones(len(obs_for_actions) - 1, dtype=bool)
        valid_mask[cross_indices] = False
        next_obs = obs_for_actions[1:][valid_mask]
        curr_obs = obs_for_actions[:-1][valid_mask]
        action_dict = self.action_processor.compute_actions_from_observations(curr_obs, next_obs)

        proprio_data = {}
        if self.action_processor.has_position and POSITION_ACTION_KEY in action_dict:
            proprio_data[POSITION_ACTION_KEY] = action_dict[POSITION_ACTION_KEY]
        if self.action_processor.has_orientation and ORIENTATION_ACTION_KEY in action_dict:
            proprio_data[ORIENTATION_ACTION_KEY] = action_dict[ORIENTATION_ACTION_KEY]

        if self.action_processor.has_gripper and self.action_processor.action_space.gripper_type == GripperType.CONTINUOUS:
            gripper_states = self.replay_buffer[GRIPPER_STATE_OBS_KEY][:]
            if len(gripper_states) > 1:
                gripper_curr = gripper_states[:-1][valid_mask]
                gripper_next = gripper_states[1:][valid_mask]
                gripper_actions = self.action_processor.compute_gripper_actions(
                    gripper_curr, gripper_next
                )
                proprio_data[GRIPPER_ACTION_KEY] = gripper_actions

        if self.observation_space.use_gripper_state and self.observation_space.gripper_type == GripperType.CONTINUOUS.value:
            gripper_obs = self.replay_buffer[GRIPPER_STATE_OBS_KEY][:]
            proprio_data[GRIPPER_STATE_OBS_KEY] = gripper_obs

        if self.observation_space.use_proprio_base_frame or self.observation_space.use_proprio_camera_frame:
            if self.observation_space.use_proprio_base_frame:
                proprio_data[PROPRIO_OBS_ROBOT_FRAME_KEY] = self.replay_buffer[PROPRIO_OBS_ROBOT_FRAME_KEY][:]
            if self.observation_space.use_proprio_camera_frame:
                proprio_data[PROPRIO_OBS_CAMERA_FRAME_KEY] = self.replay_buffer[PROPRIO_OBS_CAMERA_FRAME_KEY][:]

        for key in self.observation_space.custom_obs_keys:
            proprio_data[key] = self.replay_buffer[key][:]
        return proprio_data


    def _setup_image_normalizers(
        self,
        normalizer: LinearNormalizer,
        device: torch.device | None,
        winsorize_depth: bool,
    ) -> None:
        """Setup normalizers for all cameras.

        Args:
            normalizer: Normalizer to configure
            device: Target device
            winsorize_depth: Apply winsorization to depth
        """
        for cam in self.observation_space.camera_keys:
            cam_array = self.replay_buffer[cam][:]
            self._log_camera_stats(cam, cam_array)
            if cam == Cameras.DEPTH.value:
                self._setup_depth_normalizer(
                    normalizer, cam, cam_array, device, winsorize_depth
                )
            else:
                self._setup_rgb_normalizer(normalizer, cam, device)


    def _setup_depth_normalizer(
        self,
        normalizer: LinearNormalizer,
        cam: str,
        depth_arr: np.ndarray,
        device: torch.device | None,
        winsorize: bool,
    ) -> None:
        """Setup depth image normalizer with optional winsorization.

        Args:
            normalizer: Normalizer to configure
            cam: Camera name
            depth_arr: Depth array from dataset
            device: Target device
            winsorize: Apply winsorization
        """
        depth_min = depth_arr.min()
        depth_max = depth_arr.max()
        depth_mean = depth_arr.mean()
        depth_std = depth_arr.std()

        if winsorize:
            p1 = np.quantile(depth_arr, 0.01)
            p99 = np.quantile(depth_arr, 0.99)
            depth_arr_clipped = np.clip(depth_arr, p1, p99)

            depth_min = depth_arr_clipped.min()
            depth_max = depth_arr_clipped.max()
            depth_mean = depth_arr_clipped.mean()
            depth_std = depth_arr_clipped.std()

            logging.info(
                f"Depth after winsorization - "
                f"min: {depth_min}, max: {depth_max}, "
                f"mean: {depth_mean}, std: {depth_std}"
            )

        normalizer[cam] = get_depth_image_normalizer(
            input_min=depth_min,
            input_max=depth_max,
            input_mean=depth_mean,
            input_std=depth_std,
            norm_type=self.depth_norm_type,
            device=device,
        )

    def _setup_rgb_normalizer(
        self,
        normalizer: LinearNormalizer,
        cam: str,
        device: torch.device | None
    ) -> None:
        """Setup RGB image normalizer.

        Args:
            normalizer: Normalizer to configure
            cam: Camera name
            device: Target device
        """
        normalizer[cam] = get_rgb_image_normalizer(
            norm_type=self.image_norm_type,
            device=device
        )


    def _log_camera_stats(self, cam: str, cam_array: np.ndarray) -> None:
        """Log camera array statistics.

        Args:
            cam: Camera name
            cam_array: Camera data array
        """
        logging.info(
            f"Raw {cam} camera stats - "
            f"min: {cam_array.min()}, max: {cam_array.max()}, "
            f"mean: {cam_array.mean()}, std: {cam_array.std()}"
        )

    def _log_normalizer_stats(self, normalizer: LinearNormalizer) -> None:
        """Log normalizer statistics.

        Args:
            normalizer: Configured normalizer
        """
        if POSITION_ACTION_KEY in normalizer.params_dict:
            stats = normalizer[POSITION_ACTION_KEY].get_input_stats()
            logging.info(
                f"Position kinematics stats - "
                f"min: {stats['min']}, max: {stats['max']}, "
                f"mean: {stats['mean']}, std: {stats['std']}"
            )
        for cam in self.observation_space.camera_keys:
            output_stats = normalizer[cam].get_output_stats()
            logging.info(f"Normalized {cam} image stats: {output_stats}")
