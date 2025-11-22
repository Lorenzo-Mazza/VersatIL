import logging

import numpy as np
import torch

from refactoring.configs.data.tokenizer import TokenizationConfig
from refactoring.data.task import ObservationSpace
from refactoring.data.action_processor import ActionProcessor
from refactoring.data.constants import (
    GRIPPER_ACTION_KEY,
    GRIPPER_STATE_OBS_KEY,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    Cameras,
    GripperType, LANGUAGE_KEY, ACTION_KEY,
)
from refactoring.data.normalization.image_normalizer import (
    get_depth_image_normalizer,
    get_rgb_image_normalizer,
)
from refactoring.data.normalization.normalizer import LinearNormalizer
from refactoring.data.preprocessing.replay_buffer import ReplayBuffer
from refactoring.data.tokenization.tokenizer import Tokenizer
from refactoring.data.tokenization.observation_tokenizer import ObservationTokenizer
from refactoring.data.tokenization.action_tokenizer import ActionTokenizer


class NormalizerBuilder:
    """Builder for creating and configuring normalizers."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        action_processor: ActionProcessor,
        prediction_horizon: int,
        observation_space: ObservationSpace,
        episode_ends: np.ndarray,
        kinematics_norm_type: str,
        image_norm_type: str,
        depth_norm_type: str,
        depth_winsorize_quantiles: tuple[float, float] | None = (0.01, 0.99),
        kinematics_winsorize_quantiles: tuple[float, float] | None = (0.01, 0.99),
        tokenization_config: TokenizationConfig | None = None,
    ):
        """Initialize normalizer builder.

        Args:
            replay_buffer: Data source
            action_processor: For computing actions and applying denoising
            prediction_horizon: Prediction horizon for action chunking.
            observation_space: The observation space configuration
            episode_ends: Episode boundaries
            kinematics_norm_type: Normalization type for kinematics
            image_norm_type: Normalization type for RGB images
            depth_norm_type: Normalization type for depth images
            depth_winsorize_quantiles: Quantiles for depth winsorization (lower, upper).
            kinematics_winsorize_quantiles: Quantiles for kinematics winsorization.
            tokenization_config: Tokenization configuration. If None, no tokenizer created.
        """
        self.replay_buffer = replay_buffer
        self.action_processor = action_processor
        self.observation_space = observation_space
        self.episode_ends = episode_ends
        self.kinematics_norm_type = kinematics_norm_type
        self.image_norm_type = image_norm_type
        self.depth_norm_type = depth_norm_type
        self.depth_winsorize_quantiles = depth_winsorize_quantiles
        self.kinematics_winsorize_quantiles = kinematics_winsorize_quantiles
        self.tokenization_config = tokenization_config
        self.prediction_horizon = prediction_horizon


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
        proprio_data = self._read_proprio_data_from_buffer(winsorize=True)
        if self.action_processor.action_space.gripper_type == GripperType.BINARY.value:
            if GRIPPER_ACTION_KEY in proprio_data:
                proprio_data.pop(GRIPPER_ACTION_KEY)
            if GRIPPER_STATE_OBS_KEY in proprio_data:
                proprio_data.pop(GRIPPER_STATE_OBS_KEY)
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

    def _read_proprio_data_from_buffer(self, winsorize: bool = False) -> dict[str, np.ndarray]:
        """Read proprioceptive data from the replay buffer and optionally winsorize.

        Args:
            winsorize: If True, apply winsorization to clip outliers

        Returns:
            Dictionary of (winsorized) proprioceptive data
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

        if self.action_processor.has_gripper:
            gripper_states = self.replay_buffer[GRIPPER_STATE_OBS_KEY][:]
            if len(gripper_states) > 1:
                gripper_curr = gripper_states[:-1][valid_mask]
                gripper_next = gripper_states[1:][valid_mask]
                gripper_actions = self.action_processor.compute_gripper_actions(
                    gripper_curr, gripper_next
                )
                proprio_data[GRIPPER_ACTION_KEY] = gripper_actions

        if self.observation_space.use_gripper_state:
            gripper_obs = self.replay_buffer[GRIPPER_STATE_OBS_KEY][:]
            proprio_data[GRIPPER_STATE_OBS_KEY] = gripper_obs

        if self.observation_space.use_proprio_base_frame or self.observation_space.use_proprio_camera_frame:
            if self.observation_space.use_proprio_base_frame:
                proprio_data[PROPRIO_OBS_ROBOT_FRAME_KEY] = self.replay_buffer[PROPRIO_OBS_ROBOT_FRAME_KEY][:]
            if self.observation_space.use_proprio_camera_frame:
                proprio_data[PROPRIO_OBS_CAMERA_FRAME_KEY] = self.replay_buffer[PROPRIO_OBS_CAMERA_FRAME_KEY][:]

        for key in self.observation_space.custom_obs_keys:
            proprio_data[key] = self.replay_buffer[key][:]

        if winsorize and self.kinematics_winsorize_quantiles:
            proprio_data = self._apply_winsorization(
                proprio_data,
                self.kinematics_winsorize_quantiles
            )

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

        if winsorize and self.depth_winsorize_quantiles:
            lower_q, upper_q = self.depth_winsorize_quantiles
            p_lower = np.quantile(depth_arr, lower_q)
            p_upper = np.quantile(depth_arr, upper_q)
            depth_arr_clipped = np.clip(depth_arr, p_lower, p_upper)

            depth_min = depth_arr_clipped.min()
            depth_max = depth_arr_clipped.max()
            depth_mean = depth_arr_clipped.mean()
            depth_std = depth_arr_clipped.std()

            logging.info(
                f"Depth after winsorization [{lower_q}, {upper_q}] - "
                f"min: {depth_min:.4f}, max: {depth_max:.4f}, "
                f"mean: {depth_mean:.4f}, std: {depth_std:.4f}"
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

    def create_normalizer_and_tokenizer(
        self,
        device: torch.device | None = None,
        **kwargs
    ) -> tuple[LinearNormalizer, Tokenizer | None]:
        """Create and fit normalizer and optionally tokenizer to data.
        Pipeline: Raw data → Winsorize → Normalize → Tokenize

        Args:
            device: Target device for tensors
            **kwargs: Additional arguments for normalizer fitting

        Returns:
            Tuple of (normalizer, tokenizer) where tokenizer is None if not configured
        """
        normalizer = self.create_normalizer(device=device, **kwargs)

        tokenizer = None
        if self.tokenization_config and (
            self.tokenization_config.tokenize_observations or self.tokenization_config.tokenize_actions
        ):
            tokenizer = self._create_tokenizer(normalizer, device)

        return normalizer, tokenizer

    def _create_tokenizer(
        self,
        normalizer: LinearNormalizer,
        device: torch.device | None
    ) -> Tokenizer:
        """Create tokenizer fitted on normalized (and winsorized) data.

        Args:
            normalizer: Already-fitted normalizer to use for normalizing data
            device: Target device

        Returns:
            Fitted tokenizer with observation and/or action tokenizers
        """
        observation_tokenizer = None
        action_tokenizer = None

        if self.tokenization_config.tokenize_observations:
            obs_config = self.tokenization_config.observation_tokenizer
            if obs_config is None:
                raise ValueError("observation_tokenizer config must be provided when tokenize_observations=True")

            observation_tokenizer = ObservationTokenizer(
                tokenizer_model=obs_config.tokenizer_model,
                observation_keys=obs_config.observation_keys,
                bin_continuous_data=obs_config.bin_continuous_data,
                num_bins=obs_config.num_bins,
                max_token_len=obs_config.max_token_len,
                device=device,
            )
            if obs_config.bin_continuous_data:
                raw_proprio_data = self._read_proprio_data_from_buffer(winsorize=True)
                # Filter to only floating values
                proprio_keys_to_bin = [k for k in obs_config.observation_keys if k != LANGUAGE_KEY]
                raw_proprio = {k: v for k, v in raw_proprio_data.items() if k in proprio_keys_to_bin}
                if raw_proprio:
                    normalized_proprio = normalizer.normalize(raw_proprio)
                    normalized_proprio_np = {
                        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in normalized_proprio.items()
                    }
                    observation_tokenizer.fit(normalized_proprio_np)
                else:
                    observation_tokenizer.fit({}) # Pass-through
            else:
                observation_tokenizer.fit({}) # Pass-through

        if self.tokenization_config.tokenize_actions:
            action_config = self.tokenization_config.action_tokenizer
            if action_config is None:
                raise ValueError("action_tokenizer config must be provided when tokenize_actions=True")

            action_tokenizer = ActionTokenizer(
                tokenizer_chain=action_config.tokenizer_chain,
                use_pretrained_fast=action_config.use_pretrained_fast,
                language_tokenizer_model=action_config.language_tokenizer_model,
                max_token_len=action_config.max_token_len,
                device=device,
            )

            if not action_config.use_pretrained_fast:
                raw_action_data = self._read_proprio_data_from_buffer(winsorize=True)
                action_keys = [POSITION_ACTION_KEY, ORIENTATION_ACTION_KEY]
                if self.action_processor.has_gripper and GRIPPER_ACTION_KEY in raw_action_data:
                    if self.action_processor.action_space.gripper_type == GripperType.BINARY.value:
                        # Remap binary {0,1} to continuous {-1,1} for tokenization
                        raw_action_data[GRIPPER_ACTION_KEY] = 2.0 * raw_action_data[GRIPPER_ACTION_KEY] - 1.0
                    else:
                        action_keys.append(GRIPPER_ACTION_KEY)

                raw_actions = {k: v for k, v in raw_action_data.items() if k in action_keys}
                normalized_actions = normalizer.normalize(raw_actions)
                normalized_actions_np = {
                    k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in normalized_actions.items()
                }
                if self.action_processor.has_gripper and self.action_processor.action_space.gripper_type == GripperType.BINARY.value:
                    normalized_actions_np[GRIPPER_ACTION_KEY] = raw_action_data[GRIPPER_ACTION_KEY]

                action_chunks = self._create_action_chunks_for_tokenizer(
                    normalized_actions_np,
                    self.prediction_horizon
                )
                action_tokenizer.fit(action_chunks)

        return Tokenizer(observation_tokenizer=observation_tokenizer, action_tokenizer=action_tokenizer)

    def _create_action_chunks_for_tokenizer(
        self,
        action_dict: dict[str, np.ndarray],
        prediction_horizon: int
    ) -> np.ndarray:
        """Create action chunks respecting episode boundaries.

        Note: We need to create chunks here for fitting the tokenizer, since the tokenizer needs to see the full action
          dataset. However, in the dataset `_get_item()`, we will compute action chunks on-the-fly per-batch, so we don't
            store pre-chunked actions in memory.

        Args:
            action_dict: Dictionary of action arrays, each (N, D_i)
            prediction_horizon: Length of action chunks

        Returns:
            Action chunks of shape (N_chunks, prediction_horizon, total_D)
        """
        action_components = []
        for key in sorted(action_dict.keys()):
            action_components.append(action_dict[key])
        all_actions = np.concatenate(action_components, axis=-1)

        # Compute adjusted episode ends for the masked action array
        # Actions are computed from consecutive obs pairs, excluding cross-episode transitions
        # So each episode loses 1 action (the transition between episodes)
        adjusted_episode_ends = []
        cumulative = 0
        for i in range(len(self.episode_ends)):
            if i == 0:
                episode_length = self.episode_ends[i] - 1  # First episode: N obs -> N-1 actions
            else:
                episode_length = (self.episode_ends[i] - self.episode_ends[i-1]) - 1
            cumulative += episode_length
            adjusted_episode_ends.append(cumulative)

        chunks = []
        episode_start = 0
        for episode_end in adjusted_episode_ends:
            episode_actions = all_actions[episode_start:episode_end]
            episode_length = episode_end - episode_start
            if episode_length >= prediction_horizon:
                for i in range(episode_length - prediction_horizon + 1):
                    chunk = episode_actions[i:i+prediction_horizon]
                    chunks.append(chunk)
            episode_start = episode_end

        if len(chunks) == 0:
            raise ValueError(
                f"No episodes long enough for prediction_horizon={prediction_horizon}. "
                f"Longest episode has {max([adjusted_episode_ends[i] - (adjusted_episode_ends[i-1] if i > 0 else 0) for i in range(len(adjusted_episode_ends))])} steps."
            )
        return np.stack(chunks, axis=0)


    def _apply_winsorization(
        self,
        data_dict: dict[str, np.ndarray],
        quantiles: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        """Apply winsorization to clip outliers to specified quantiles.

        Args:
            data_dict: Dictionary of data arrays to winsorize
            quantiles: (lower, upper) quantiles, e.g., (0.01, 0.99)

        Returns:
            Dictionary with winsorized arrays
        """
        winsorized = {}
        lower_q, upper_q = quantiles

        for key, data in data_dict.items():
            lower_bound = np.quantile(data, lower_q, axis=0)
            upper_bound = np.quantile(data, upper_q, axis=0)

            winsorized_data = np.clip(data, lower_bound, upper_bound)
            winsorized[key] = winsorized_data

            n_clipped = np.sum(data != winsorized_data)
            if n_clipped > 0:
                logging.info(
                    f"Winsorized {key} to [{lower_q:.3f}, {upper_q:.3f}] quantiles - "
                    f"clipped {n_clipped}/{data.size} values "
                    f"({100*n_clipped/data.size:.2f}%)"
                )

        return winsorized
