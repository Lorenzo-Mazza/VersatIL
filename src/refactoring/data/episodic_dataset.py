import logging
import random

import numpy as np
import torch
import torch.utils.data as data
from threadpoolctl import threadpool_limits

from refactoring.configs.data.dataloader import DataLoaderConfig
from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.data.action_processor import ActionProcessor
from refactoring.data.augmentation.augmentation_pipeline import AugmentationPipeline
from refactoring.data.constants import (
    GRIPPER_ACTION_KEY,
    GRIPPER_STATE_OBS_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    SamplingMode,
)
from refactoring.configs.data.tokenizer import TokenizationConfig
from refactoring.data.normalization.normalizer import LinearNormalizer
from refactoring.data.normalization.normalizer_builder import NormalizerBuilder
from refactoring.data.preprocessing.replay_buffer import ReplayBuffer
from refactoring.data.preprocessing.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from refactoring.data.sample_builder import SampleBuilder
from refactoring.data.tokenization import Tokenizer

logging.basicConfig(level=logging.INFO)


class EpisodicDataset(data.Dataset):
    """PyTorch Dataset for episodic robot demonstration data.

    This class orchestrates modular components for:
    - Action processing
    - Data augmentation
    - Sample building
    - Episode splitting and management
    """


    def __init__(
            self,
            zarr_path: str,
            action_space: ActionSpace,
            observation_space: ObservationSpace,
            dataloader_config: DataLoaderConfig,
            pred_horizon: int,
            obs_horizon: int,
            train: bool = True,
            seed: int = 42,
    ):
        """Initialize episodic dataset.

        Args:
            zarr_path: Path to zarr replay buffer
            action_space: TaskSpace action space config (what to predict and how)
            observation_space: TaskSpace observation space config (what to use as observation data)
            pred_horizon: Prediction horizon, i.e. chunk size.
            obs_horizon: Observation horizon, i.e. history size.
            train: Whether to use training mode.
            seed: Random seed of the experiment.
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.sampling_mode = dataloader_config.sampling_mode
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_backward_shift = dataloader_config.action_backward_shift
        self.kinematics_norm_type = dataloader_config.kinematics_norm_type
        self.image_norm_type = dataloader_config.image_norm_type
        self.depth_norm_type = dataloader_config.depth_norm_type

        self.train = train
        self.seed = seed
        self.action_processor = ActionProcessor(action_space=action_space)
        self.augmentation_pipeline = AugmentationPipeline(
            color_augmentation=dataloader_config.color_augmentation,  # type: ignore[arg-type]
            spatial_augmentation=dataloader_config.spatial_augmentation,  # type: ignore[arg-type]
            rotation_augmentation=dataloader_config.rotation_augmentation,  # type: ignore[arg-type]
            target_height=dataloader_config.image_height,
            target_width=dataloader_config.image_width,
            train=train,
        )
        self.train = train
        self.seed = seed
        all_keys = list(set(observation_space.get_required_zarr_keys() + action_space.get_required_zarr_keys()))  # Remove duplicates
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path=zarr_path, keys=all_keys)
        logging.info(f"Total episodes in buffer: {self.replay_buffer.n_episodes}")

        if dataloader_config.center_episode_start:
            self._center_episodes_at_origin()

        # Create episode mask (train/val split)
        episode_mask = self._create_episode_mask(
            val_ratio=dataloader_config.val_ratio,
            total_ratio=dataloader_config.total_ratio,
            train=train,
            seed=seed,
        )

        # Apply downsampling if needed
        if dataloader_config.downsample_factor > 1:
            self._apply_downsampling(episode_mask, dataloader_config.downsample_factor)
            episode_mask = np.ones(self.replay_buffer.n_episodes, dtype=bool)

        self.episode_ends = self.replay_buffer.episode_ends[:]

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.obs_horizon + self.pred_horizon + self.action_backward_shift,
            pad_before=0,
            pad_after=self.pred_horizon - 1,
            episode_mask=episode_mask,
            key_first_k=dict.fromkeys(observation_space.camera_keys, self.obs_horizon + self.action_backward_shift),
            skip_initial=dataloader_config.skip_initial_episode_steps,
            pad_with_zeros=False,
        )
        self._setup_episode_indices()
        self.sample_builder = SampleBuilder(
            action_space=action_space,
            observation_space=observation_space,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_backward_shift=dataloader_config.action_backward_shift,
            augmentation_pipeline=self.augmentation_pipeline,
            action_processor=self.action_processor,
        )
        self.normalizer: LinearNormalizer | None = None


    def _create_episode_mask(
            self,
            val_ratio: float,
            total_ratio: float,
            train: bool,
            seed: int,
            max_train_episodes: int | None = None,
    ) -> np.ndarray:
        """Create boolean mask for episode selection (train/val split)."""
        n_episodes = self.replay_buffer.n_episodes

        # Apply total ratio constraint
        total_mask = np.ones(n_episodes, dtype=bool)
        if total_ratio < 1.0:
            max_total = max(1, int(n_episodes * total_ratio))
            total_mask = downsample_mask(total_mask, max_n=max_total, seed=seed)

        # Create validation mask from selected episodes
        selected_indices: np.ndarray = np.nonzero(total_mask)[0]
        n_selected = len(selected_indices)
        val_submask = get_val_mask(n_selected, val_ratio=val_ratio, seed=seed)
        val_selected_idx = selected_indices[val_submask]
        val_mask = np.zeros(n_episodes, dtype=bool)
        val_mask[val_selected_idx] = True

        # Select train or validation episodes
        if train:
            episode_mask = np.logical_and(np.logical_not(val_mask), total_mask)
        else:
            episode_mask = val_mask

        logging.info(
            f"{'Training' if train else 'Validation'} episodes: {np.sum(episode_mask)}"
        )

        # Apply max training episodes constraint
        if train and max_train_episodes is not None:
            episode_mask = downsample_mask(
                episode_mask, max_n=max_train_episodes, seed=seed
            )

        return episode_mask


    def _apply_downsampling(self, episode_mask: np.ndarray, downsample_step: int) -> None:
        """Downsample episodes by taking every n-th step."""
        subsampled_buffer = ReplayBuffer.create_empty_numpy()
        selected_episodes = np.nonzero(episode_mask)[0]

        for ep_idx in selected_episodes:
            episode = self.replay_buffer.get_episode(ep_idx)
            # Determine episode length
            if self.action_space.predict_in_camera_frame:
                ep_len = episode[PROPRIO_OBS_CAMERA_FRAME_KEY].shape[0]
            else:
                ep_len = episode[PROPRIO_OBS_ROBOT_FRAME_KEY].shape[0]
            # Create downsampling indices
            indices = np.arange(0, ep_len, downsample_step)
            # Ensure last frame is included
            if ep_len > 0 and (ep_len - 1) not in indices:
                indices = np.append(indices, ep_len - 1)
            # Downsample all arrays in episode
            downsampled_episode = {k: v[indices] for k, v in episode.items()}
            subsampled_buffer.add_episode(downsampled_episode)

        self.replay_buffer = subsampled_buffer
        self.episode_ends = self.replay_buffer.episode_ends[:]

        logging.info(
            f"After downsampling (step={downsample_step}), "
            f"episodes: {self.replay_buffer.n_episodes}, "
            f"steps: {self.replay_buffer.n_steps}"
        )


    def _center_episodes_at_origin(self) -> None:
        """Center each episode so the first observation position is at (0,0,0).

        This modifies both robot frame and camera frame proprioceptive data
        by subtracting the first observation's position from all observations
        in each episode.
        """
        current_start = 0
        for _, end in enumerate(self.replay_buffer.episode_ends):
            episode_length = end - current_start
            if self.observation_space.use_proprio_base_frame or not self.action_space.predict_in_camera_frame:
                first_obs = self.replay_buffer[PROPRIO_OBS_ROBOT_FRAME_KEY][current_start]
                first_pos = first_obs[:self.action_space.position_dim]
                episode_obs = self.replay_buffer[PROPRIO_OBS_ROBOT_FRAME_KEY][current_start:end]
                episode_obs[:, :self.action_space.position_dim] -= first_pos
                self.replay_buffer[PROPRIO_OBS_ROBOT_FRAME_KEY][current_start:end] = episode_obs
                if self.action_space.has_orientation:
                    pos_end = self.action_space.position_dim
                    ori_end = pos_end + self.action_space.orientation_dim
                    first_ori = first_obs[pos_end:ori_end]
                    episode_ori = episode_obs[:, pos_end:ori_end]
                    # Repeat first_ori for each timestep to compute relative orientations
                    first_ori_repeated = np.tile(first_ori, (episode_length, 1))
                    # This computes: relative_ori = episode_ori * first_ori^(-1)
                    centered_ori = self.action_processor._compute_orientation_deltas(
                        first_ori_repeated, episode_ori
                    )
                    episode_obs[:, pos_end:ori_end] = centered_ori

            if self.action_space.predict_in_camera_frame or self.observation_space.use_proprio_camera_frame:
                first_obs = self.replay_buffer[PROPRIO_OBS_CAMERA_FRAME_KEY][current_start]
                first_pos = first_obs[:self.action_space.position_dim]
                episode_obs = self.replay_buffer[PROPRIO_OBS_CAMERA_FRAME_KEY][current_start:end]
                episode_obs[:, :self.action_space.position_dim] -= first_pos
                self.replay_buffer[PROPRIO_OBS_CAMERA_FRAME_KEY][current_start:end] = episode_obs
                if self.action_space.has_orientation:
                    pos_end = self.action_space.position_dim
                    ori_end = pos_end + self.action_space.orientation_dim
                    first_orientation = first_obs[pos_end:ori_end]
                    episode_orientations = episode_obs[:, pos_end:ori_end]
                    first_orientation_repeated = np.tile(first_orientation, (episode_length, 1)) # Repeat for each timestep
                    centered_ori = self.action_processor._compute_orientation_deltas(first_orientation_repeated, episode_orientations)
                    episode_obs[:, pos_end:ori_end] = centered_ori

            current_start = end
        logging.info(f"Centered {len(self.replay_buffer.episode_ends)} episodes at origin")


    def _setup_episode_indices(self) -> None:
        """Setup episode-to-sample index mapping."""
        self.episode_indices = []
        current_start = 0
        for end in self.episode_ends:
            # Find sampler indices that belong to this episode
            ep_indices = [i for i, row in enumerate(self.sampler.indices) if current_start <= row[0] < end]
            self.episode_indices.append(ep_indices)
            current_start = end
        # Track which episodes have valid samples
        self.selected_episode_indices = [
            i for i, indices in enumerate(self.episode_indices) if indices
        ]


    def __len__(self) -> int:
        """Dataset length depends on sampling mode."""
        if self.sampling_mode == SamplingMode.OVERLAPPING.value:
            return len(self.sampler)
        elif self.sampling_mode == SamplingMode.RANDOM_CHUNK.value:
            return len(self.selected_episode_indices)
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")


    def __getitem__(self, idx: int) -> dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]:
        """Get a training sample."""
        threadpool_limits(1)
        start_idx = self._get_start_idx(idx)
        padded_data = self.sampler.sample_sequence(start_idx)
        action_dict = self._compute_sample_actions(padded_data)
        sample = self.sample_builder.build_sample(
            padded_data=padded_data,
            action_dict=action_dict,
            start_idx=start_idx,
            sampler_indices=self.sampler.indices,
        )
        return sample


    def _get_start_idx(self, idx: int) -> int:
        """Get replay buffer start index based on sampling mode."""
        if self.sampling_mode == SamplingMode.OVERLAPPING.value:
            return idx
        elif self.sampling_mode == SamplingMode.RANDOM_CHUNK.value:
            ep_idx = self.selected_episode_indices[idx]
            ep_indices = self.episode_indices[ep_idx]
            if not ep_indices:
                raise ValueError(f"Episode {idx} has no valid indices")
            return random.choice(ep_indices)
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")


    def _compute_sample_actions(
            self, padded_data: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Compute actions for a single sample."""
        action_key = PROPRIO_OBS_CAMERA_FRAME_KEY if self.action_processor.predict_in_camera_frame else PROPRIO_OBS_ROBOT_FRAME_KEY
        obs_for_action = padded_data[action_key]
        action_slice_start = self.obs_horizon - 1
        action_slice_end = action_slice_start + self.pred_horizon
        next_obs = obs_for_action[action_slice_start + 1:action_slice_end + 1]
        curr_obs = obs_for_action[action_slice_start:action_slice_end]
        action_dict = self.action_processor.compute_actions_from_observations(curr_obs, next_obs)

        if self.action_processor.has_gripper:
            padded_gripper = padded_data[GRIPPER_STATE_OBS_KEY]
            curr_gripper = padded_gripper[action_slice_start: action_slice_end]
            next_gripper = padded_gripper[action_slice_start + 1: action_slice_end + 1]
            gripper_actions = self.action_processor.compute_gripper_actions(curr_gripper, next_gripper)
            action_dict[GRIPPER_ACTION_KEY] = gripper_actions

        return action_dict


    def get_normalizer(
            self, device: torch.device | None = None, winsorize_depth: bool = True, **kwargs
    ) -> LinearNormalizer:
        """Get normalizer for this dataset."""
        normalizer_builder = NormalizerBuilder(
            replay_buffer=self.replay_buffer,
            action_processor=self.action_processor,
            prediction_horizon=self.pred_horizon,
            observation_space=self.observation_space,
            episode_ends=self.episode_ends,
            kinematics_norm_type=self.kinematics_norm_type,
            image_norm_type=self.image_norm_type,
            depth_norm_type=self.depth_norm_type,
        )

        return normalizer_builder.create_normalizer(
            device=device, winsorize_depth=winsorize_depth, **kwargs
        )

    def get_normalizer_and_tokenizer(
        self,
        device: torch.device | None = None,
        winsorize_depth: bool = True,
        depth_winsorize_quantiles: tuple[float, float] | None = (0.01, 0.99),
        winsorize_kinematics: bool = False,
        kinematics_winsorize_quantiles: tuple[float, float] | None = (0.01, 0.99),
        tokenization_config: TokenizationConfig | None = None,
        **kwargs
    ) -> tuple[LinearNormalizer, Tokenizer | None]:
        """Get normalizer and optionally tokenizer for this dataset.

        Args:
            device: Target device for tensors
            winsorize_depth: Apply winsorization to depth values
            depth_winsorize_quantiles: Quantiles for depth winsorization
            winsorize_kinematics: Apply winsorization to kinematics
            kinematics_winsorize_quantiles: Quantiles for kinematics winsorization
            tokenization_config: Tokenization configuration. If None, no tokenizer created.
            **kwargs: Additional arguments for normalizer fitting

        Returns:
            Tuple of (normalizer, tokenizer) where tokenizer is None if not configured
        """
        normalizer_builder = NormalizerBuilder(
            replay_buffer=self.replay_buffer,
            action_processor=self.action_processor,
            observation_space=self.observation_space,
            episode_ends=self.episode_ends,
            kinematics_norm_type=self.kinematics_norm_type,
            image_norm_type=self.image_norm_type,
            depth_norm_type=self.depth_norm_type,
            depth_winsorize_quantiles=depth_winsorize_quantiles if winsorize_depth else None,
            kinematics_winsorize_quantiles=kinematics_winsorize_quantiles if winsorize_kinematics else None,
            tokenization_config=tokenization_config,
            prediction_horizon=self.pred_horizon,
        )

        return normalizer_builder.create_normalizer_and_tokenizer(
            device=device, **kwargs
        )


    def set_tokenizer(self, tokenizer: Tokenizer | None) -> None:
        """Set tokenizer for the sample builder.

        Args:
            tokenizer: Unified tokenizer containing observation and action tokenizers
        """
        self.sample_builder.tokenizer = tokenizer

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Set normalizer for the dataset.

        Args:
            normalizer: Normalizer for observations and actions
        """
        self.sample_builder.normalizer = normalizer

    def get_gripper_positive_class_imbalance_weight(self) -> float:
        """Get class imbalance weight for gripper actions."""
        if not self.action_space.has_gripper:
            raise ValueError("Gripper actions are not being predicted")

        gripper_actions = self.replay_buffer[GRIPPER_STATE_OBS_KEY][:].squeeze(-1)
        number_of_positive_actions = gripper_actions.sum()
        number_of_negative_actions = len(gripper_actions) - number_of_positive_actions
        result: float = number_of_negative_actions / number_of_positive_actions
        return result
