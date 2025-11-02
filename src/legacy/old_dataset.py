import random
from pathlib import Path

import cv2
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

from legacy_constants import ImageNormalizationType, Cameras, SamplingMode, ACTION_KEY, OBSERVATION_KEY, IS_PAD_KEY, ROBOT_STATE_KEY
from legacy_config import ACTConfig
from model.common.normalize_util import get_image_range_normalizer, get_depth_image_normalizer, get_zero_to_one_normalizer
from model.common.normalizer import LinearNormalizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

EPISODE_FILENAME = "episode.csv"
ROBOT_FRAME_KINEMATICS_COLS = ["relative_tip_position_x", "relative_tip_position_y", "relative_tip_position_z"]
CAMERA_FRAME_KINEMATICS_COLS = ["camera_frame_tip_position_x", "camera_frame_tip_position_y", "camera_frame_tip_position_z"]
ACTION_COL_KEYS: list[str] = ["action_x", "action_y", "action_z"]
RECTIFIED_LEFT_IMAGE_PATH_KEY = "frameLeftRectifiedPath"
LEFT_IMAGE_PATH_KEY = "frameLeftPath"
RECTIFIED_RIGHT_IMAGE_PATH_KEY = "frameRightRectifiedPath"
RIGHT_IMAGE_PATH_KEY = "frameRightPath"


def create_sample_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        skip_initial: int = 0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = max(-pad_before, skip_initial)
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    return np.array(indices)




class EpisodicDataset(data.Dataset):

    def __init__(self,
                 root_folder: str,
                 dataset_paths: list[str],
                 sampling_mode: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 image_height: int,
                 image_width: int,
                 center_crop: bool,
                 center_crop_size: int,
                 downsample_factor: int,
                 deltas_as_actions: bool,
                 center_initial_position: bool,
                 camera_space_actions: bool,
                 camera_space_obs: bool,
                 robot_space_obs: bool,
                 use_rectified_images: bool,
                 use_depth: bool,
                 camera_names: List[str],
                 skip_initial_steps: int = 5,
                 image_norm_type: str = ImageNormalizationType.ZERO_TO_ONE.value,
                 use_augmentations: bool = False,
                 train: bool = True,
                 action_backward_shift: int = 1,
                 ):
        self.root_folder = root_folder
        self.dataset_paths = dataset_paths
        self.sampling_mode = sampling_mode
        self.downsample_factor = downsample_factor
        self.deltas_as_actions = deltas_as_actions
        self.camera_space_actions = camera_space_actions
        self.camera_space_obs = camera_space_obs
        self.robot_space_obs = robot_space_obs
        self.use_rectified_images = use_rectified_images
        self.center_initial_position = center_initial_position
        self.use_depth = use_depth
        self.camera_names = camera_names
        self.skip_initial_steps = skip_initial_steps
        self.image_norm_type = image_norm_type
        self.use_augmentations = use_augmentations
        self.train = train
        self.action_backward_shift = action_backward_shift

        self.observation_cols = []
        if self.robot_space_obs:
            self.observation_cols += ROBOT_FRAME_KINEMATICS_COLS
        if self.camera_space_obs:
            self.observation_cols += CAMERA_FRAME_KINEMATICS_COLS


        dataset, self.episode_ends = self.load_dataset()
        self.dataset = self.filter_columns(dataset)
        self.use_kinematics = robot_space_obs or camera_space_obs

        if center_crop:
            crop_transform = [A.CenterCrop(height=center_crop_size, width=center_crop_size)]
        else:
            crop_transform = []

        self.rgb_transform = A.Compose(
            crop_transform + [
                A.Resize(height=image_height, width=image_width),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0) if image_norm_type != ImageNormalizationType.IMAGENET.value else
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

        self.depth_transform = A.Compose(
            crop_transform + [
                A.Resize(height=image_height, width=image_width),
                ToTensorV2(),
            ]
        )

        if train and use_augmentations:
            self.augmentation_transform = A.Compose(
                [
                    A.ColorJitter(0.3, 0.4, 0.5, 0.1, p=0.3),
                    A.GaussianBlur(p=0.2),
                    A.CoarseDropout(p=0.2),
                ],
            )
        else:
            self.augmentation_transform = None

        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=pred_horizon - 1,
            skip_initial=self.skip_initial_steps
        )
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.episode_indices = []
        current_start = 0
        for end in self.episode_ends:
            ep_indices = [i for i, row in enumerate(self.indices) if current_start <= row[0] < end]
            self.episode_indices.append(ep_indices)
            current_start = end


    @property
    def camera_name_to_col(self) -> Dict[str, str]:
        """Mapping from camera names to their respective column keys."""
        return {
            Cameras.LEFT.value: LEFT_IMAGE_PATH_KEY if not self.use_rectified_images else RECTIFIED_LEFT_IMAGE_PATH_KEY,
            Cameras.RIGHT.value: RIGHT_IMAGE_PATH_KEY if not self.use_rectified_images else RECTIFIED_RIGHT_IMAGE_PATH_KEY,
            Cameras.DEPTH.value: Cameras.DEPTH.value
        }


    def __len__(self):
        if self.sampling_mode == SamplingMode.OVERLAPPING.value:
            return len(self.indices)
        elif self.sampling_mode == SamplingMode.RANDOM_CHUNK.value:
            return len(self.episode_ends)
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")


    def load_dataset(self) -> Tuple[pd.DataFrame, np.ndarray]:
        episodes_data = []
        episode_ends = []
        cumulative_length = 0
        for path in sorted(self.dataset_paths, key=lambda x: int(Path(x).parent.name)):
            episode_df = pd.read_csv(path)
            indices = list(range(0, len(episode_df), self.downsample_factor))
            if len(episode_df) - 1 not in indices:
                indices.append(len(episode_df) - 1)
            episode_df = episode_df.iloc[indices]
            if self.use_depth:
                episode_df[Cameras.DEPTH.value] = episode_df[LEFT_IMAGE_PATH_KEY].apply(lambda x: x.replace("framesLeft", "stereoDepth").replace(".png", ".tiff"))
            if self.center_initial_position:
                col_list = CAMERA_FRAME_KINEMATICS_COLS + ROBOT_FRAME_KINEMATICS_COLS
                initial_pos = episode_df.loc[0, col_list].astype(float)
                episode_df[col_list] = episode_df[col_list].astype(float).sub(initial_pos, axis=1)
            target_columns = CAMERA_FRAME_KINEMATICS_COLS if self.camera_space_actions else ROBOT_FRAME_KINEMATICS_COLS
            if self.deltas_as_actions:
                episode_df[ACTION_COL_KEYS] = np.nan_to_num(np.array(episode_df[target_columns].shift(-1)) -
                                                            np.array(episode_df[target_columns]))
            else:
                episode_df[ACTION_COL_KEYS] = episode_df[target_columns].shift(-1).ffill()

            episodes_data.append(episode_df)
            cumulative_length += len(episode_df)
            episode_ends.append(cumulative_length)
        return pd.concat(episodes_data).reset_index(drop=True), np.array(episode_ends)


    def filter_columns(self, dataset: pd.DataFrame):
        """Filter out columns that are not needed for the dataset."""
        keys = []
        for cam_name in self.camera_names:
            keys.append(self.camera_name_to_col[cam_name])
        return dataset[keys + self.observation_cols + ACTION_COL_KEYS]


    def compute_depth_stats(self):
        """Compute statistics for depth images in the dataset."""
        depth_mean = 0
        depth_var = 0
        depth_min = float('inf')
        depth_max = float('-inf')
        num_depth_images = 0
        for image_path in self.dataset[Cameras.DEPTH]:
            depth_image = self._read_image(image_path=image_path, is_disparity=True)
            depth_tensor = self.depth_transform(image=depth_image)['image'].unsqueeze(0)  # Apply image transform (resize, to tensor)
            batch_size = depth_tensor.numel()  # Total number of elements in the tensor
            new_depth_mean = depth_tensor.mean().item()
            new_depth_var = depth_tensor.var().item()
            depth_min = min(depth_min, depth_tensor.min().item())
            depth_max = max(depth_max, depth_tensor.max().item())
            # Update the running mean and variance
            depth_mean = (depth_mean * num_depth_images + new_depth_mean * batch_size) / (num_depth_images + batch_size)
            depth_var = (depth_var * num_depth_images + new_depth_var * batch_size) / (num_depth_images + batch_size)
            num_depth_images += batch_size
        return depth_mean, depth_var, depth_min, depth_max, num_depth_images

    def get_normalizer(self, mode = 'limits', device = None, recompute_depth_stats: bool = False, **kwargs):
        # Create normalizer for actions and states
        kinematics_data = {
            ACTION_KEY: self.dataset[ACTION_COL_KEYS].values,
        }
        if self.use_kinematics:
            kinematics_data[ROBOT_STATE_KEY] = self.dataset[self.observation_cols].values

        normalizer = LinearNormalizer()
        normalizer.fit(data=kinematics_data, last_n_dims=1, mode=mode, device=device, **kwargs)

        # Create normalizer for images
        for camera in self.camera_names:
            if Cameras.DEPTH.value == camera:
                if recompute_depth_stats:
                    depth_mean, depth_var, depth_min, depth_max, num_depth_images = self.compute_depth_stats()
                    normalizer[camera] = get_depth_image_normalizer(
                        min=depth_min,
                        max=depth_max,
                        mean=depth_mean,
                        std=np.sqrt(depth_var),
                        device=device,
                        use_mean_std=True if self.image_norm_type == ImageNormalizationType.ZERO_TO_ONE.value else False
                    )
                else:
                    '''Hardcoded values coming from the latest depth stats
                    # import pickle
                    # dataset_stats_path = '/mnt/cluster/workspaces/mazzalore/iros/act_checkpoints/act_training_20250710/dataset_stats.pkl'
                    # with open(dataset_stats_path, 'rb') as f:
                    #    dataset_stats = pickle.load(f)'''
                    normalizer[camera] = get_depth_image_normalizer(min=0.007743687900641262,
                                                                    max=0.1426304120135479,
                                                                    mean=0.025363063708876345,
                                                                    std=0.007381618954241276,
                                                                    device=device,
                                                                    use_mean_std=True if self.image_norm_type == ImageNormalizationType.ZERO_TO_ONE.value else False
                                                                    )
            else:
                if self.image_norm_type == ImageNormalizationType.MINUS_ONE_TO_ONE.value:
                    normalizer[camera] = get_image_range_normalizer(device=device)
                else:
                    normalizer[camera] = get_zero_to_one_normalizer(device=device)
        return normalizer


    @staticmethod
    def _read_image(image_path: str, is_disparity: bool = False):
        # BGR → RGB for colour; keep depth as is
        if is_disparity:
            disparity_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            depth = 1. / disparity_image
            return depth
        else:
            return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


    def sample_sequence(self, data: dict[str, pd.Series | np.ndarray], sequence_length: int, sample_start_idx: int, sample_end_idx: int
                        ) -> dict[str, pd.Series | np.ndarray]:
        result = {}
        for key, input_arr in data.items():
            if isinstance(input_arr, pd.Series):
                input_arr = input_arr.values
            data_arr = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype
            )

            if sample_start_idx > 0:
                data_arr[:sample_start_idx] = input_arr[0] if not self.deltas_as_actions else 0.
            if sample_end_idx < sequence_length:
                data_arr[sample_end_idx:] = input_arr[-1] if not self.deltas_as_actions else 0.
            data_arr[sample_start_idx:sample_end_idx] = input_arr

            result[key] = data_arr

        is_pad = np.zeros(sequence_length, dtype=bool)
        if sample_start_idx > 0:
            is_pad[:sample_start_idx] = True
        if sample_end_idx < sequence_length:
            is_pad[sample_end_idx:] = True
        result[IS_PAD_KEY] = is_pad
        return result


    def __getitem__(self, episode_idx):
        if self.sampling_mode == SamplingMode.OVERLAPPING.value:
            buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[episode_idx]
            buffer_start_idx_action= buffer_start_idx
            buffer_end_idx_action = buffer_end_idx
        elif self.sampling_mode == SamplingMode.RANDOM_CHUNK.value:
            episode_indices = self.episode_indices[episode_idx]
            if not episode_indices:
                raise ValueError(f"Episode {episode_idx} has no valid indices")
            rand_ind = random.choice(episode_indices)
            buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[rand_ind]
            episode_start = 0 if episode_idx == 0 else self.episode_ends[episode_idx - 1]
            buffer_start_idx_action = max(buffer_start_idx - self.action_backward_shift, episode_start)
            buffer_end_idx_action = buffer_start_idx_action + (buffer_end_idx - buffer_start_idx)
        else:
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

        sequence = self.dataset.iloc[buffer_start_idx:buffer_end_idx]

        data_dict = {}
        if self.use_kinematics:
            data_dict[OBSERVATION_KEY] = sequence[self.observation_cols].values
        for camera in self.camera_names:
            data_dict[camera] = sequence[self.camera_name_to_col[camera]]
        act_sequence = self.dataset.iloc[buffer_start_idx_action:buffer_end_idx_action]
        data_dict[ACTION_KEY] = act_sequence[ACTION_COL_KEYS].values

        padded_sample = self.sample_sequence(
            data=data_dict,
            sequence_length=self.pred_horizon,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        camera_to_image_tensors = {cam: [] for cam in self.camera_names}
        for i in range(self.obs_horizon):
            for camera in self.camera_names:
                image_path = padded_sample[camera][i]
                if camera == Cameras.DEPTH.value:
                    depth_np = self._read_image(image_path=image_path, is_disparity=True)
                    image_tensor = self.depth_transform(image=depth_np)['image'] # Apply image transform (resize, to tensor)
                    image_tensor = image_tensor.unsqueeze(0) if len(image_tensor.shape) == 2 else image_tensor
                else:
                    image_np = self._read_image(image_path=image_path, is_disparity=False)
                    if self.augmentation_transform and self.train:
                        image_np = self.augmentation_transform(image=image_np)['image']  # Apply augmentation if needed
                    image_tensor = self.rgb_transform(image=image_np)['image']  # Apply image transform (resize, to tensor)
                camera_to_image_tensors[camera].append(image_tensor)

        sample = {OBSERVATION_KEY: {}}
        for camera in  self.camera_names:
            sample[OBSERVATION_KEY][camera] = torch.stack(camera_to_image_tensors[camera])
        if self.use_kinematics:
            sample[OBSERVATION_KEY][ROBOT_STATE_KEY] = torch.from_numpy(padded_sample[OBSERVATION_KEY][:self.obs_horizon, :]).float()
        sample[ACTION_KEY] = torch.from_numpy(padded_sample[ACTION_KEY][:self.pred_horizon, :]).float()
        sample[IS_PAD_KEY] = torch.from_numpy(padded_sample[IS_PAD_KEY]).bool()
        return sample


def get_dataloaders(config: ACTConfig):
    root_path = Path(config.dataset_folder)
    episode_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    datasets_paths = [str(d / EPISODE_FILENAME) for d in episode_dirs]
    episode_nums = [int(Path(p).parent.name) for p in datasets_paths]
    sorted_paths = [p for _, p in sorted(zip(episode_nums, datasets_paths))]
    num_val_episodes = min(int(len(sorted_paths) * config.ratio_validation_episodes), len(sorted_paths) - 1)
    train_paths, val_paths = split_episodes(
        sorted_paths,
        num_val_episodes=num_val_episodes,
        seed=config.seed
    )

    train_dataset = EpisodicDataset(
        root_folder=str(root_path),
        dataset_paths=train_paths,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        image_height=config.image_height,
        image_width=config.image_width,
        center_crop=config.center_crop,
        center_crop_size=config.center_crop_size,
        downsample_factor=config.downsample_factor,
        skip_initial_steps=config.skip_initial_steps,
        image_norm_type= config.image_norm_type,
        use_augmentations=config.use_augmentations,
        deltas_as_actions=config.deltas_as_actions,
        camera_space_actions=config.predict_in_camera_frame,
        robot_space_obs=config.obs_robot_frame,
        camera_space_obs=config.obs_camera_frame,
        use_depth=config.use_depth,
        use_rectified_images=config.use_rectified,
        center_initial_position=config.center_initial_position,
        camera_names=config.camera_names,
        sampling_mode=config.sampling_mode,
        action_backward_shift= config.action_backward_shift,
    )

    val_dataset = EpisodicDataset(
        root_folder=str(root_path),
        dataset_paths=val_paths,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        image_height= config.image_height,
        image_width=config.image_width,
        center_crop=config.center_crop,
        center_crop_size=config.center_crop_size,
        downsample_factor=config.downsample_factor,
        skip_initial_steps=config.skip_initial_steps,
        image_norm_type= config.image_norm_type,
        deltas_as_actions=config.deltas_as_actions,
        camera_space_actions=config.predict_in_camera_frame,
        robot_space_obs=config.obs_robot_frame,
        camera_space_obs=config.obs_camera_frame,
        use_depth=config.use_depth,
        use_rectified_images=config.use_rectified,
        center_initial_position=config.center_initial_position,
        camera_names=config.camera_names,
        sampling_mode=config.sampling_mode,
        action_backward_shift=config.action_backward_shift,
        use_augmentations=False,
        train=False
    )

    normalizer = train_dataset.get_normalizer(recompute_depth_stats=config.recompute_depth_stats,device=torch.device(config.device))

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

    return train_loader, val_loader, normalizer


def split_episodes(
        datasets_paths: List[str],
        num_val_episodes: int,
        seed: int
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