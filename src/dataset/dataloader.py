import random
import shutil
from pathlib import Path

import cv2
import torch
import torch.utils.data as data
import numpy as np
from typing import List, Tuple, Optional

import albumentations as A

from legacy_constants import (Cameras, ImageNormalizationType, SamplingMode, OBSERVATION_KEY, POSITION_ACTION_KEY, GRIPPER_ACTION_KEY, IMAGENET_RGB_MEAN, IMAGENET_RGB_STD,
                              IS_PAD_KEY, ROBOT_STATE_KEY, CAMERA_FRAME_OBS_KEY, ROBOT_FRAME_OBS_KEY, KinematicsNormalizationType, DepthNormalizationType, GRIPPER_STATE_OBS_KEY, PHASE_LABEL_KEY)
from legacy_config import PolicyConfig
from model.common.normalize_util import get_image_range_normalizer, get_depth_image_normalizer, get_zero_to_one_normalizer
from model.common.normalizer import LinearNormalizer
from dataset.replay_buffer import ReplayBuffer
from dataset.sampler import SequenceSampler, get_val_mask, downsample_mask
from dataset.preprocess import create_replay_buffer
from threadpoolctl import threadpool_limits
import logging
logging.basicConfig(level=logging.INFO)

EPISODE_FILENAME = "episode.csv"

class EpisodicDataset(data.Dataset):
    def __init__(self,
                 zarr_path: str,
                 sampling_mode: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 camera_names: List[str],
                 image_height: int,
                 image_width: int,
                 image_norm_type: str = ImageNormalizationType.ZERO_TO_ONE.value,
                 depth_norm_type: str = DepthNormalizationType.ZERO_TO_ONE.value,
                 kinematics_norm_type: str = KinematicsNormalizationType.MIN_MAX.value,
                 use_color_augmentations: bool = False,
                 use_rotation_augmentations: bool = False,
                 train: bool = True,
                 action_backward_shift: int = 1,
                 predict_in_camera_frame: bool = False,  # If True, actions are in camera frame
                 obs_robot_frame: bool = False,  # If True, dataset observations contain robot state data in robot frame
                 obs_camera_frame: bool = False,  # If True, dataset observations contain robot state data in camera frame
                 deltas_as_actions: bool = False,
                 val_ratio: float = 0.0,
                 total_ratio_of_episodes: float = 1.0,  # Total ratio of episodes to use for training/validation
                 max_train_episodes: Optional[int] = None,
                 seed: int = 42,
                 skip_initial_steps: int = 0,
                 downsample_step: int = 1,
                 promote_sparsity: bool = False,  # If True, promote sparsity in actions by thresholding small movements to zero.
                 predict_gripper_action: bool = False,
                 task_has_phases: bool = False,
                 ):
        self.sampling_mode = sampling_mode
        self.image_norm_type = image_norm_type
        self.depth_norm_type = depth_norm_type
        self.kinematics_norm_type =kinematics_norm_type
        self.use_color_augmentations = use_color_augmentations
        self.use_rotation_augmentations = use_rotation_augmentations
        self.train = train
        self.action_backward_shift = action_backward_shift
        self.camera_names = camera_names
        self.use_kinematics = obs_robot_frame or obs_camera_frame
        self.deltas_as_actions = deltas_as_actions
        self.predict_in_camera_frame = predict_in_camera_frame
        self.obs_robot_frame = obs_robot_frame
        self.obs_camera_frame = obs_camera_frame
        self.image_height = image_height
        self.image_width = image_width
        self.downsample_step = downsample_step
        self.promote_sparsity = promote_sparsity

        self.predict_gripper_action = predict_gripper_action
        self.task_has_phases = task_has_phases
        # Load replay buffer into memory
        keys = camera_names + [ROBOT_FRAME_OBS_KEY, CAMERA_FRAME_OBS_KEY]
        if predict_gripper_action:
            keys += [GRIPPER_STATE_OBS_KEY]
        if task_has_phases:
            keys += [PHASE_LABEL_KEY]

        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=keys)
        logging.log(level=logging.INFO, msg=f"Total episodes in buffer: {self.replay_buffer.n_episodes}")

        self.episode_ends = self.replay_buffer.episode_ends[:]
        # Apply total_ratio_of_episodes to select a subset of episodes
        total_mask = np.ones(self.replay_buffer.n_episodes, dtype=bool)
        if total_ratio_of_episodes < 1.0:
            max_total = max(1, int(self.replay_buffer.n_episodes * total_ratio_of_episodes))
            total_mask = downsample_mask(total_mask, max_n=max_total, seed=seed)
        # Create val_mask on the selected episodes
        selected_indices = np.nonzero(total_mask)[0]
        n_selected = len(selected_indices)
        val_submask = get_val_mask(n_selected, val_ratio=val_ratio, seed=seed)
        val_selected_idx = selected_indices[val_submask]
        val_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
        val_mask[val_selected_idx] = True
        # Episode mask based on train/val
        episode_mask = val_mask if not train else np.logical_and(np.logical_not(val_mask), total_mask)
        logging.log(level=logging.INFO, msg=f"{'Training' if self.train else 'Validation'} episodes: {np.sum(episode_mask)}")

        if train:
            episode_mask = downsample_mask(episode_mask, max_n=max_train_episodes, seed=seed)
        
        if self.downsample_step > 1:
            subsampled_buffer = ReplayBuffer.create_empty_numpy()
            selected_episodes = np.nonzero(episode_mask)[0]
            for ep_idx in selected_episodes:
                episode = self.replay_buffer.get_episode(ep_idx)
                ep_len = episode[self.camera_names[0]].shape[0] if self.camera_names else episode[ROBOT_FRAME_OBS_KEY].shape[0]
                indices = np.arange(0, ep_len, self.downsample_step)
                if ep_len > 0 and (ep_len - 1) not in indices:
                    indices = np.append(indices, ep_len - 1)
                downsampled_episode = {k: v[indices] for k, v in episode.items()}
                subsampled_buffer.add_episode(downsampled_episode)
            self.replay_buffer = subsampled_buffer
            self.episode_ends = self.replay_buffer.episode_ends[:]
            episode_mask = np.ones(self.replay_buffer.n_episodes, dtype=bool)
            logging.log(level=logging.INFO, msg=f"After downsampling (step={self.downsample_step}), episodes: {self.replay_buffer.n_episodes}, steps: {self.replay_buffer.n_steps}")

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=obs_horizon + pred_horizon -1 + action_backward_shift,  # Adjust for shift
            pad_before=obs_horizon - 1,
            pad_after=pred_horizon - 1,
            episode_mask=episode_mask,
            key_first_k={cam: obs_horizon for cam in camera_names}, # Optimize image loading
            skip_initial=skip_initial_steps,
            pad_with_zeros=False,
            )

        # Augmentations (only for train)
        if train and use_color_augmentations:
            self.photometric_transform = A.Compose([
                A.ColorJitter(0.3, 0.4, 0.5, 0.1, p=0.5),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_color=(255, 255, 255), p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.CLAHE(clip_limit=4.0, p=0.3),
                A.RandomShadow(p=0.4),
                A.GaussianBlur(p=0.5),
                A.CoarseDropout(p=0.3),
                A.ShiftScaleRotate(rotate_limit=(0,0),scale_limit=(-0.5, 0.6), shift_limit=(-0.0625, 0.0625), p=0.5),
                A.ImageCompression(quality_range=(50, 100), compression_type='jpeg', p=0.2),
            ])
        else:
            self.photometric_transform = None

        # For random_chunk mode
        self.episode_indices = []
        current_start = 0
        for end in self.episode_ends:
            ep_indices = [
                i
                for i, row in enumerate(self.sampler.indices)
                if current_start <= row[0] < end
            ]
            self.episode_indices.append(ep_indices)
            current_start = end
        self.selected_episode_indices = [
            i for i, indices in enumerate(self.episode_indices) if indices
        ]
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.seed = seed

    def __len__(self):
        if self.sampling_mode == SamplingMode.OVERLAPPING.value:
            return len(self.sampler)
        elif self.sampling_mode == SamplingMode.RANDOM_CHUNK.value:
            return len(self.selected_episode_indices)
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

    def get_gripper_positive_class_imbalance_weight(self):
        if self.predict_gripper_action:
            gripper_actions = self.replay_buffer[GRIPPER_STATE_OBS_KEY][:]
            gripper_actions = gripper_actions.squeeze(-1)
            number_of_positive_actions = gripper_actions.sum()
            number_of_negative_actions = (
                len(gripper_actions) - number_of_positive_actions
            )
            return number_of_negative_actions / number_of_positive_actions
        else:
            raise ValueError(
                f"Gripper actions are not being predicted, so class weights cannot be computed"
            )

    def get_normalizer(
        self, mode="limits", device=None, recompute_depth_stats: bool = False, **kwargs
    ):
        normalizer = LinearNormalizer()
        if self.kinematics_norm_type == KinematicsNormalizationType.MIN_MAX.value:
            mode = "limits"
        elif self.kinematics_norm_type == KinematicsNormalizationType.GAUSSIAN.value:
            mode = "gaussian"
        else:
            raise ValueError(
                f"Unknown kinematics_norm_type: {self.kinematics_norm_type}"
            )
        # Select position key for actions
        action_pos_key = (
            CAMERA_FRAME_OBS_KEY
            if self.predict_in_camera_frame
            else ROBOT_FRAME_OBS_KEY
        )

        all_pos = self.replay_buffer[action_pos_key][:]
        next_pos, curr_pos = None, None
        # Compute action data based on config using vectorized operations
        if len(all_pos) == 0:
            action_data = np.zeros(
                (0, all_pos.shape[1] if len(all_pos.shape) > 1 else 3), dtype=np.float32
            )
        else:
            cross_indices = self.episode_ends[:-1] - 1
            valid_mask = np.ones(len(all_pos) - 1, dtype=bool)
            valid_mask[cross_indices] = False
            next_pos = all_pos[1:][valid_mask]
            curr_pos = all_pos[:-1][valid_mask]
            if self.deltas_as_actions:
                action_data = next_pos - curr_pos
            else:
                action_data = next_pos

        if self.promote_sparsity and len(all_pos) > 0:
            diffs = next_pos - curr_pos
            norms = np.linalg.norm(diffs, axis=1)
            non_zero_norms = norms[norms > 0]
            if len(non_zero_norms) > 0:
                self.sparsity_threshold = np.percentile(non_zero_norms, 5)
                logging.log(
                    level=logging.INFO,
                    msg=f"Computed sparsity threshold (5th percentile): {self.sparsity_threshold}",
                )

                mask = norms < self.sparsity_threshold
                next_pos[mask] = curr_pos[mask]
                if self.deltas_as_actions:
                    action_data = next_pos - curr_pos
                else:
                    action_data = next_pos
            else:
                self.sparsity_threshold = 0.0

        kinematics_data = {POSITION_ACTION_KEY: action_data}
        if self.use_kinematics:
            obs_parts = []
            if self.obs_robot_frame:
                obs_parts.append(self.replay_buffer[ROBOT_FRAME_OBS_KEY][:])
            if self.obs_camera_frame:
                obs_parts.append(self.replay_buffer[CAMERA_FRAME_OBS_KEY][:])
            robot_state_data = np.concatenate(obs_parts, axis=1)
            kinematics_data[ROBOT_STATE_KEY] = robot_state_data

        normalizer.fit(
            data=kinematics_data,
            last_n_dims=1,
            mode=mode,
            device=device,
            range_eps=1e-10,
            **kwargs,
        )
        for cam in self.camera_names:
            if cam == Cameras.DEPTH.value:
                depth_arr = self.replay_buffer[cam][:]
                depth_min, depth_max = depth_arr.min(), depth_arr.max()
                depth_mean, depth_std = depth_arr.mean(), depth_arr.std()
                logging.log(
                    level=logging.INFO,
                    msg=f"Computed depth stats from data - min: {depth_min}, max: {depth_max}, mean: {depth_mean}, std: {depth_std}",
                )
                p1 = np.quantile(depth_arr, 0.01)
                p99 = np.quantile(depth_arr, 0.99)
                depth_arr_clipped = np.clip(depth_arr, p1, p99)
                depth_min = depth_arr_clipped.min()
                depth_max = depth_arr_clipped.max()
                depth_mean = depth_arr_clipped.mean()
                depth_std = depth_arr_clipped.std()
                logging.log(
                    level=logging.INFO,
                    msg=f"After winsorization - min: {depth_min}, max: {depth_max}, mean: {depth_mean}, std: {depth_std}",
                )
                normalizer[cam] = get_depth_image_normalizer(
                    input_min=depth_min,
                    input_max=depth_max,
                    input_mean=depth_mean,
                    input_std=depth_std,
                    device=device,
                    norm_type=self.depth_norm_type,
                )
                logging.info(
                    f"Normalized depth stats (min/max/mean/std): {normalizer[Cameras.DEPTH.value].get_output_stats()['min']},"
                    f" {normalizer[Cameras.DEPTH.value].get_output_stats()['max']},"
                    f" {normalizer[Cameras.DEPTH.value].get_output_stats()['mean']}, "
                    f"{normalizer[Cameras.DEPTH.value].get_output_stats()['std']}"
                )
            else:
                if (
                    self.image_norm_type
                    == ImageNormalizationType.MINUS_ONE_TO_ONE.value
                ):
                    normalizer[cam] = get_image_range_normalizer(device=device)
                else:
                    normalizer[cam] = get_zero_to_one_normalizer(device=device)
                logging.info(
                    f"Normalized image stats (min/max/mean/std): {normalizer[cam].get_output_stats()['min']},"
                    f" {normalizer[cam].get_output_stats()['max']}, {normalizer[cam].get_output_stats()['mean']},"
                    f" {normalizer[cam].get_output_stats()['std']}"
                )

        logging.info(
            f"Action stats (min/max/mean/std): {normalizer[POSITION_ACTION_KEY].get_input_stats()['min']}, {normalizer[POSITION_ACTION_KEY].get_input_stats()['max']}, {normalizer[POSITION_ACTION_KEY].get_input_stats()['mean']}, {normalizer[POSITION_ACTION_KEY].get_input_stats()['std']}"
        )
        logging.info(
            f"Normalized action stats (min/max/mean/std): {normalizer[POSITION_ACTION_KEY].get_output_stats()['min']}, {normalizer[POSITION_ACTION_KEY].get_output_stats()['max']}, {normalizer[POSITION_ACTION_KEY].get_output_stats()['mean']}, {normalizer[POSITION_ACTION_KEY].get_output_stats()['std']}"
        )
        return normalizer

    def __getitem__(self, idx):
        threadpool_limits(1)
        if self.sampling_mode == SamplingMode.OVERLAPPING.value:
            start_idx = idx
        elif self.sampling_mode == SamplingMode.RANDOM_CHUNK.value:
            ep_idx = self.selected_episode_indices[idx]
            ep_indices = self.episode_indices[ep_idx]
            if not ep_indices:
                raise ValueError(f"Episode {idx} has no valid indices")
            # random.seed(self.seed)  # commented out for now, to increase variablity in data
            start_idx = random.choice(
                ep_indices
            )  # Randomly select an index from episode's valid starts
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.sampler.indices[
            start_idx
        ]  # Retrieve precomputed indices for this sample
        padded_data = self.sampler.sample_sequence(
            start_idx
        )  # Sample the full sequence including pads
        # Compute actions on-the-fly
        pos_key = (
            CAMERA_FRAME_OBS_KEY
            if self.predict_in_camera_frame
            else ROBOT_FRAME_OBS_KEY
        )
        padded_pos = padded_data[pos_key]

        action_slice_start = self.obs_horizon - 1 - self.action_backward_shift
        action_slice_end = action_slice_start + self.pred_horizon
        pos_len = len(padded_pos)
        pos_shape = padded_pos.shape[1:]
        pos_dtype = padded_pos.dtype

        # Extract next_pos
        next_slice_start = action_slice_start + 1
        next_slice_end = action_slice_end + 1
        next_pre_pad = max(0, -next_slice_start)
        next_post_pad = max(0, next_slice_end - pos_len)
        next_source_start = max(next_slice_start, 0)
        next_source_end = min(next_slice_end, pos_len)
        next_pos = np.empty((self.pred_horizon,) + pos_shape, dtype=pos_dtype)
        if next_source_end > next_source_start:
            next_pos[
                next_pre_pad : next_pre_pad + (next_source_end - next_source_start)
            ] = padded_pos[next_source_start:next_source_end]
        if next_pre_pad > 0:
            next_pos[:next_pre_pad] = padded_pos[0]
        if next_post_pad > 0:
            next_pos[-next_post_pad:] = padded_pos[-1]

        # Extract curr_pos
        curr_slice_start = action_slice_start
        curr_slice_end = action_slice_end
        curr_pre_pad = max(0, -curr_slice_start)
        curr_post_pad = max(0, curr_slice_end - pos_len)
        curr_source_start = max(curr_slice_start, 0)
        curr_source_end = min(curr_slice_end, pos_len)
        curr_pos = np.empty((self.pred_horizon,) + pos_shape, dtype=pos_dtype)
        if curr_source_end > curr_source_start:
            curr_pos[
                curr_pre_pad : curr_pre_pad + (curr_source_end - curr_source_start)
            ] = padded_pos[curr_source_start:curr_source_end]
        if curr_pre_pad > 0:
            curr_pos[:curr_pre_pad] = padded_pos[0]
        if curr_post_pad > 0:
            curr_pos[-curr_post_pad:] = padded_pos[-1]

        if self.promote_sparsity and self.sparsity_threshold > 0:
            diffs = next_pos - curr_pos
            mask = np.linalg.norm(diffs, axis=1) < self.sparsity_threshold
            next_pos[mask] = curr_pos[mask]

        if self.deltas_as_actions:
            action_data = next_pos - curr_pos
        else:
            action_data = next_pos

        sample = {OBSERVATION_KEY: {}}
        if self.use_rotation_augmentations:
            angle = random.uniform(-5.0, 5.0) if random.random() < 0.5 else 0.0
            if angle != 0:
                theta_rad = np.deg2rad(angle)
                cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
                R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
        else:
            angle = 0
        # Add this:
        for cam in self.camera_names:
            img = padded_data[cam][: self.obs_horizon]
            if self.train:
                if self.use_rotation_augmentations:
                    rotate_transform = A.Rotate(
                        limit=(angle, angle), p=1.0, interpolation=cv2.INTER_LINEAR
                    )
                    img = np.stack(
                        [rotate_transform(image=frame)["image"] for frame in img]
                    )
            if cam != Cameras.DEPTH.value:
                img = img.astype(np.float32) / 255.0
                if self.photometric_transform and self.train:
                    img = np.stack(
                        [
                            self.photometric_transform(image=frame)["image"]
                            for frame in img
                        ]
                    )
                if self.image_norm_type == ImageNormalizationType.IMAGENET.value:
                    mean = np.array(IMAGENET_RGB_MEAN)
                    std = np.array(IMAGENET_RGB_STD)
                    img = (img - mean) / std
                img = np.moveaxis(img, -1, 1)  # To CHW shape
            else:
                if len(img.shape) == 3:
                    img = img.astype(np.float32)[:, None]  # Add channel for depth
            sample[OBSERVATION_KEY][cam] = torch.from_numpy(img)

        if self.use_kinematics:
            obs_pos_parts = []
            if self.obs_robot_frame:
                obs_pos_parts.append(
                    padded_data[ROBOT_FRAME_OBS_KEY][: self.obs_horizon]
                )
            if self.obs_camera_frame:
                obs_pos_parts.append(
                    padded_data[CAMERA_FRAME_OBS_KEY][: self.obs_horizon]
                )
            robot_state = (
                np.concatenate(obs_pos_parts, axis=-1)
                if obs_pos_parts
                else np.empty((self.obs_horizon, 0), dtype=np.float32)
            )

            if angle != 0:  # Rotate kinematics data accordingly, if images were rotated
                if self.obs_camera_frame:
                    if self.obs_robot_frame:
                        robot_state[..., 3:6] = (
                            R @ robot_state[..., 3:6].T
                        ).T  # Rotate the camera frame position only
                    else:
                        robot_state[..., :3] = (
                            R @ robot_state[..., :3].T
                        ).T  # Rotate the camera frame position
            sample[OBSERVATION_KEY][ROBOT_STATE_KEY] = torch.from_numpy(
                robot_state
            ).float()

        if (
            angle != 0 and self.predict_in_camera_frame
        ):  # Rotate actions if in camera frame
            action_data = (R @ action_data.T).T

        if self.predict_gripper_action:
            padded_gripper = padded_data[GRIPPER_STATE_OBS_KEY]
            gripper_shape = padded_gripper.shape[1:]
            gripper_dtype = padded_gripper.dtype
            next_gripper = np.empty(
                (self.pred_horizon,) + gripper_shape, dtype=gripper_dtype
            )
            if next_source_end > next_source_start:
                next_gripper[
                    next_pre_pad : next_pre_pad + (next_source_end - next_source_start)
                ] = padded_gripper[next_source_start:next_source_end]
            if next_pre_pad > 0:
                next_gripper[:next_pre_pad] = padded_gripper[0]
            if next_post_pad > 0:
                next_gripper[-next_post_pad:] = padded_gripper[-1]
            gripper_action_data = next_gripper
            sample[GRIPPER_ACTION_KEY] = torch.from_numpy(gripper_action_data).float()

        if self.task_has_phases:
            padded_phases = padded_data[PHASE_LABEL_KEY]
            phase_shape = padded_phases.shape[1:]
            phase_dtype = padded_phases.dtype
            next_phase = np.empty((self.pred_horizon,) + phase_shape, dtype=phase_dtype)
            if next_source_end > next_source_start:
                next_phase[
                    next_pre_pad : next_pre_pad + (next_source_end - next_source_start)
                ] = padded_phases[next_source_start:next_source_end]
            if next_pre_pad > 0:
                next_phase[:next_pre_pad] = padded_phases[0]
            if next_post_pad > 0:
                next_phase[-next_post_pad:] = padded_phases[-1]
            phase_action_data = next_phase
            sample[PHASE_LABEL_KEY] = torch.from_numpy(phase_action_data).long()

        sample[POSITION_ACTION_KEY] = torch.from_numpy(action_data).float()

        # Pad mask (computed based on sampling)
        action_positions = np.arange(self.pred_horizon) + action_slice_start
        if self.deltas_as_actions:
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
            is_pad = np.logical_or(
                action_positions + 1 < sample_start_idx,
                action_positions + 1 >= sample_end_idx,
            )
        sample[IS_PAD_KEY] = torch.from_numpy(is_pad).bool()

        return sample


def get_dataloaders(config: PolicyConfig):
    datasets_paths = []
    for folder in config.dataset_folders:
        root_path = Path(folder)
        episode_dirs = [
            d
            for d in root_path.iterdir()
            if d.is_dir() and (d / EPISODE_FILENAME).exists()
        ]
        datasets_paths.extend([str(d / EPISODE_FILENAME) for d in episode_dirs])
    print(
        f"Found {len(datasets_paths)} episodes across {len(config.dataset_folders)} folders"
    )

    zarr_path = config.zarr_path
    # Preprocess if Zarr not exists or corrupted
    need_create = True
    required_keys = config.camera_names + [ROBOT_FRAME_OBS_KEY, CAMERA_FRAME_OBS_KEY]
    if config.predict_gripper_action:
        required_keys += [GRIPPER_STATE_OBS_KEY]

    if config.task_has_phases:
        required_keys += [PHASE_LABEL_KEY]

    if Path(zarr_path).exists():
        try:
            logging.log(
                level=logging.INFO,
                msg=f"Loading existing replay buffer from {zarr_path}",
            )
            ReplayBuffer.copy_from_path(zarr_path, keys=required_keys)
            need_create = False
        except Exception as e:  # Catch any exception during loading
            logging.log(
                level=logging.INFO, msg=f"Error loading {zarr_path}: {e}. Recreating..."
            )
            shutil.rmtree(zarr_path, ignore_errors=True)

    if need_create:
        logging.log(
            level=logging.INFO,
            msg=f"Creating zarr replay buffer inside path: {zarr_path}",
        )
        create_replay_buffer(
            dataset_paths=datasets_paths,
            image_height=config.image_height,
            image_width=config.image_width,
            center_crop=config.center_crop,
            center_crop_size=config.center_crop_size,
            camera_names=config.camera_names,
            center_initial_position=config.center_initial_position,
            use_rectified_images=config.use_rectified,
            downsample_factor=config.downsample_factor,
            zarr_path=zarr_path,
            predict_gripper_action=config.predict_gripper_action,
            task_has_phases=config.task_has_phases,
        )

    train_dataset = EpisodicDataset(
        zarr_path=zarr_path,
        sampling_mode=config.sampling_mode,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        image_height=config.image_height,
        image_width=config.image_width,
        image_norm_type=config.image_norm_type,
        depth_norm_type=config.depth_norm_type,
        kinematics_norm_type=config.kinematics_norm_type,
        use_color_augmentations=config.use_color_augmentations,
        use_rotation_augmentations=config.use_rotation_augmentations,
        train=True,
        action_backward_shift=config.action_backward_shift,
        camera_names=config.camera_names,
        predict_in_camera_frame=config.predict_in_camera_frame,
        obs_robot_frame=config.obs_robot_frame,
        obs_camera_frame=config.obs_camera_frame,
        deltas_as_actions=config.deltas_as_actions,
        val_ratio=config.ratio_validation_episodes,
        total_ratio_of_episodes=config.total_ratio_of_episodes,
        max_train_episodes=None,
        seed=config.seed,
        skip_initial_steps=config.skip_initial_steps,
        downsample_step=config.downsample_step,
        promote_sparsity=config.promote_sparsity,
        predict_gripper_action=config.predict_gripper_action,
        task_has_phases=config.task_has_phases,
    )
    val_dataset = EpisodicDataset(
        zarr_path=zarr_path,
        sampling_mode=config.sampling_mode,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        image_height=config.image_height,
        image_width=config.image_width,
        image_norm_type=config.image_norm_type,
        depth_norm_type=config.depth_norm_type,
        kinematics_norm_type=config.kinematics_norm_type,
        use_color_augmentations=False,
        use_rotation_augmentations=False,
        train=False,
        action_backward_shift=config.action_backward_shift,
        camera_names=config.camera_names,
        predict_in_camera_frame=config.predict_in_camera_frame,
        obs_robot_frame=config.obs_robot_frame,
        obs_camera_frame=config.obs_camera_frame,
        deltas_as_actions=config.deltas_as_actions,
        val_ratio=config.ratio_validation_episodes,
        total_ratio_of_episodes=config.total_ratio_of_episodes,
        seed=config.seed,
        skip_initial_steps=config.skip_initial_steps,
        downsample_step=config.downsample_step,
        promote_sparsity=config.promote_sparsity,
        predict_gripper_action=config.predict_gripper_action,
        task_has_phases=config.task_has_phases,
    )
    normalizer = train_dataset.get_normalizer(
        recompute_depth_stats=config.recompute_depth_stats,
        device=torch.device(config.device),
    )
    if config.promote_sparsity:
        val_dataset.sparsity_threshold = train_dataset.sparsity_threshold

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    gripper_positive_class_weights = None
    if (
        config.predict_gripper_action
        and config.calculate_gripper_positive_class_weights
    ):
        # if the positive class weights are not calculated, set to 1 because that is the "balance" case
        gripper_positive_class_weights = (
            train_dataset.get_gripper_positive_class_imbalance_weight()
        )

    if config.task_has_phases:
        selected_eps = np.where(train_dataset.sampler.episode_mask)[0]
        phase_labels = np.concatenate(
            [
                train_dataset.replay_buffer.get_episode(i)[PHASE_LABEL_KEY].flatten()
                for i in selected_eps
            ]
            if len(selected_eps) > 0
            else [np.array([])]
        )
        phase_counts = np.bincount(phase_labels, minlength=5)
        logging.info(
            f"Train phase distribution: {dict(enumerate(phase_counts.tolist()))}"
        )

        selected_eps_val = np.where(val_dataset.sampler.episode_mask)[0]
        phase_labels_val = np.concatenate(
            [
                val_dataset.replay_buffer.get_episode(i)[PHASE_LABEL_KEY].flatten()
                for i in selected_eps_val
            ]
            if len(selected_eps_val) > 0
            else [np.array([])]
        )
        phase_counts_val = np.bincount(phase_labels_val, minlength=5)
        logging.info(
            f"Validation phase distribution: {dict(enumerate(phase_counts_val.tolist()))}"
        )

    return train_loader, val_loader, normalizer, gripper_positive_class_weights


def split_episodes(
    datasets_paths: List[str], num_val_episodes: int, seed: int
) -> Tuple[List[str], List[str]]:
    random.seed(seed)

    episode_paths = [(int(Path(p).parent.name), p) for p in datasets_paths]
    episode_paths.sort(key=lambda x: x[0])

    all_indices = list(range(len(episode_paths)))
    val_indices = random.sample(all_indices, num_val_episodes)
    train_indices = [i for i in all_indices if i not in val_indices]

    train_paths = [episode_paths[i][1] for i in train_indices]
    val_paths = [episode_paths[i][1] for i in val_indices]

    return train_paths, val_paths
