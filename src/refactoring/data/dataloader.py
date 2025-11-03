import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from hydra.utils import instantiate

from refactoring.configs import MainConfig
from refactoring.data.constants import (
    EPISODE_FILENAME,
    PHASE_LABEL_KEY,
)
from refactoring.data.episodic_dataset import EpisodicDataset
from refactoring.data.normalize.normalizer import LinearNormalizer
from refactoring.data.preprocessing.create_zarr import create_replay_buffer
from refactoring.data.preprocessing.replay_buffer import ReplayBuffer
from refactoring.data.schemas.base import DatasetSchema


def get_dataloaders(
    config: MainConfig,
) -> tuple[data.DataLoader, data.DataLoader, LinearNormalizer, float | None]:
    """Create train and validation dataloaders with normalizer.

    Args:
        config: Main configuration object from Hydra

    Returns:
        Tuple of (train_loader, val_loader, normalizer, gripper_class_weights)
    """
    schema: DatasetSchema = instantiate(config.task.dataset_schema)
    logging.info(f"Using dataset schema: {schema.__class__.__name__}")
    datasets_paths = _collect_dataset_paths(schema.dataset_folders)
    logging.info(f"Found {len(datasets_paths)} episodes across {len(schema.dataset_folders)} folders")

    _ensure_zarr_exists(schema=schema, datasets_paths=datasets_paths)

    action_space = instantiate(config.task.action_space)
    observation_space = instantiate(config.task.observation_space)

    train_dataset = EpisodicDataset(
        zarr_path=schema.zarr_path,
        pred_horizon=config.task.prediction_horizon,
        obs_horizon=config.task.observation_horizon,
        dataloader_config=config.task.dataloader,
        train=True,
        seed=config.experiment.seed,
        action_space=action_space,
        observation_space=observation_space,
    )

    val_dataset = EpisodicDataset(
        zarr_path=schema.zarr_path,
        pred_horizon=config.task.prediction_horizon,
        obs_horizon=config.task.observation_horizon,
        dataloader_config=config.task.dataloader,
        train=False,
        seed=config.experiment.seed,
        action_space=action_space,
        observation_space=observation_space,
    )

    # Get normalizer
    device = torch.device(config.experiment.device)
    normalizer = train_dataset.get_normalizer(
        winsorize_depth=config.task.dataloader.winsorize_depth,
        device=device,
    )

    # Share denoising thresholds with validation dataset
    if config.task.action_space.denoise_actions:
        val_dataset.action_processor.action_denoising_threshold = (
            train_dataset.action_processor.action_denoising_threshold
        )
        val_dataset.action_processor.orientation_denoising_threshold = (
            train_dataset.action_processor.orientation_denoising_threshold
        )

    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.task.dataloader.batch_size,
        shuffle=config.task.dataloader.shuffle,
        num_workers=config.task.dataloader.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config.task.dataloader.batch_size,
        shuffle=False,
        num_workers=min(4, config.task.dataloader.num_workers),
        pin_memory=True,
        persistent_workers=True,
    )

    gripper_positive_class_weights = None
    if config.task.action_space.has_gripper and config.task.action_space.use_gripper_class_weights:
        gripper_positive_class_weights = train_dataset.get_gripper_positive_class_imbalance_weight()

    if config.task.action_space.task_has_phases:
        _log_phase_distributions(train_dataset, val_dataset)

    return train_loader, val_loader, normalizer, gripper_positive_class_weights


def _collect_dataset_paths(dataset_folders: list[str]) -> list[str]:
    """Collect all episode CSV paths from dataset folders."""
    datasets_paths = []
    for folder in dataset_folders:
        root_path = Path(folder)
        episode_dirs = [
            d for d in root_path.iterdir() if d.is_dir() and (d / EPISODE_FILENAME).exists()
        ]
        datasets_paths.extend([str(d / EPISODE_FILENAME) for d in episode_dirs])
    return datasets_paths


def _ensure_zarr_exists(
        schema: DatasetSchema,
        datasets_paths: list[str],
) -> None:
    """Create zarr if it doesn't exist or is invalid."""
    zarr_path = schema.zarr_path
    need_create = True
    required_keys = schema.get_required_zarr_keys()
    if Path(zarr_path).exists():
        try:
            logging.info(f"Loading existing replay buffer from {zarr_path}")
            ReplayBuffer.copy_from_path(zarr_path, keys=required_keys)
            need_create = False
        except Exception as e:
            logging.info(f"Error loading {zarr_path}: {e}. Recreating...")
            shutil.rmtree(zarr_path, ignore_errors=True)

    if need_create:
        logging.info(f"Creating zarr replay buffer at: {zarr_path}")
        create_replay_buffer(schema=schema, datasets_paths=datasets_paths)


def _log_phase_distributions(
    train_dataset: EpisodicDataset, val_dataset: EpisodicDataset
) -> None:
    """Log phase label distributions for train and val."""
    selected_eps = np.where(train_dataset.sampler.episode_mask)[0]
    if len(selected_eps) > 0:
        phase_labels = np.concatenate(
            [
                train_dataset.replay_buffer.get_episode(i)[PHASE_LABEL_KEY].flatten()
                for i in selected_eps
            ]
        )
        phase_counts = np.bincount(phase_labels, minlength=5)
        logging.info(f"Train phase distribution: {dict(enumerate(phase_counts.tolist()))}")  # type: ignore[arg-type]

    selected_eps_val = np.where(val_dataset.sampler.episode_mask)[0]
    if len(selected_eps_val) > 0:
        phase_labels_val = np.concatenate(
            [
                val_dataset.replay_buffer.get_episode(i)[PHASE_LABEL_KEY].flatten()
                for i in selected_eps_val
            ]
        )
        phase_counts_val = np.bincount(phase_labels_val, minlength=5)
        logging.info(f"Val phase distribution: {dict(enumerate(phase_counts_val.tolist()))}")  # type: ignore[arg-type]
