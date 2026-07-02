import logging
import shutil
from pathlib import Path

import torch
import torch.utils.data as data
from omegaconf import DictConfig

from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.configs.data.tokenizer import TokenizationConfig
from versatil.data.constants import (
    ActionDiscretizerType,
    BinningStrategy,
    KinematicsNormalizationType,
)
from versatil.data.episodic_dataset import EpisodicDataset
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.preprocessing.create_zarr_from_csv import create_replay_buffer
from versatil.data.preprocessing.create_zarr_from_hdf5 import (
    create_replay_buffer_from_hdf5,
)
from versatil.data.preprocessing.create_zarr_from_lerobot import (
    create_replay_buffer_from_lerobot,
)
from versatil.data.preprocessing.create_zarr_from_synthetic import (
    create_replay_buffer_from_synthetic,
)
from versatil.data.preprocessing.replay_buffer import ReplayBuffer
from versatil.data.raw.schemas import CsvDatasetSchema
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.raw.schemas.custom.synthetic import SyntheticSchema
from versatil.data.raw.schemas.hdf5 import Hdf5DatasetSchema
from versatil.data.raw.schemas.lerobot import LeRobotDatasetSchemaV30
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization.tokenizer import Tokenizer, validate_tokenizer_config


def get_dataloaders(
    config: DictConfig,
) -> tuple[
    data.DataLoader,
    data.DataLoader | None,
    LinearNormalizer,
    Tokenizer | None,
    float | None,
]:
    """Create train and validation dataloaders with normalizer and optional tokenizer.

    Args:
        config: Main configuration object instantiated by Hydra

    Returns:
        Tuple of (train_loader, val_loader, normalizer, tokenizer, gripper_class_weights).
        val_loader is None when val_ratio is 0.

    Note: The type hint for `config` indicates `DictConfig`, but at runtime hydra instantiates a `MainConfig` with
      all target fields resolved into python objects.
    """
    schema: DatasetSchema = config.task.dataset_schema
    action_space: ActionSpace = config.task.action_space
    observation_space: ObservationSpace = config.task.observation_space
    dataloader_config: DataLoaderConfig = config.task.dataloader
    tokenization_config: TokenizationConfig = dataloader_config.tokenization

    validate_dataloader_config(dataloader_config)
    validate_tokenizer_config(tokenization_config)

    logging.info(f"Using dataset schema: {schema.__class__.__name__}")
    _ensure_zarr_exists(
        schema=schema, preload_in_memory=dataloader_config.preload_data_in_memory
    )
    skip_validation = dataloader_config.val_ratio == 0

    train_dataset = EpisodicDataset(
        zarr_path=schema.zarr_path,
        pred_horizon=config.task.prediction_horizon,
        obs_horizon=config.task.observation_horizon,
        dataloader_config=dataloader_config,
        train=True,
        seed=config.experiment.seed,
        action_space=action_space,
        observation_space=observation_space,
    )

    normalizer, tokenizer = train_dataset.get_normalizer_and_tokenizer(
        winsorize_depth=dataloader_config.winsorize_depth,
        depth_winsorize_quantiles=dataloader_config.depth_winsorize_quantiles,
        winsorize_kinematics=dataloader_config.winsorize_kinematics,
        kinematics_winsorize_quantiles=dataloader_config.kinematics_winsorize_quantiles,
        tokenization_config=tokenization_config,
        clamp_kinematics_range=dataloader_config.clamp_kinematics_range,
        min_kinematics_std=dataloader_config.min_kinematics_std,
        min_kinematics_range=dataloader_config.min_kinematics_range,
        action_sample_size=dataloader_config.action_sample_size,
        device=torch.device("cpu"),  # Keep on CPU for DataLoader workers
    )
    train_dataset.set_normalizer(normalizer)
    train_dataset.set_tokenizer(tokenizer)

    num_workers = config.task.dataloader.num_workers
    use_multiprocessing = num_workers > 0
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.task.dataloader.batch_size,
        shuffle=config.task.dataloader.shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_multiprocessing,
        prefetch_factor=2 if use_multiprocessing else None,
    )

    val_loader: data.DataLoader | None = None
    if not skip_validation:
        val_dataset = EpisodicDataset(
            zarr_path=schema.zarr_path,
            pred_horizon=config.task.prediction_horizon,
            obs_horizon=config.task.observation_horizon,
            dataloader_config=dataloader_config,
            train=False,
            seed=config.experiment.seed,
            action_space=action_space,
            observation_space=observation_space,
        )
        val_dataset.set_normalizer(normalizer)
        val_dataset.set_tokenizer(tokenizer)

        if action_space.denoise_actions:
            val_dataset.action_processor.denoising_thresholds = (
                train_dataset.action_processor.denoising_thresholds.copy()
            )
            val_dataset.action_processor._denoising_thresholds_computed = True

        val_num_workers = min(4, config.task.dataloader.num_workers)
        val_use_multiprocessing = val_num_workers > 0
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=config.task.dataloader.batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=True,
            persistent_workers=val_use_multiprocessing,
            prefetch_factor=2 if val_use_multiprocessing else None,
        )
    else:
        logging.info("Validation disabled (val_ratio=0). Training without validation.")

    gripper_positive_class_weights = None
    if (
        config.task.action_space.has_gripper_actions
        and config.task.action_space.use_gripper_class_weights
    ):
        gripper_positive_class_weights = (
            train_dataset.get_gripper_positive_class_imbalance_weight()
        )

    return (
        train_loader,
        val_loader,
        normalizer,
        tokenizer,
        gripper_positive_class_weights,
    )


def validate_dataloader_config(config: DataLoaderConfig) -> None:
    """Validate Dataloader configuration."""
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    if config.num_workers < 0:
        raise ValueError(f"num_workers cannot be negative, got {config.num_workers}")
    if not 0 <= config.val_ratio < 1:
        raise ValueError(f"val_ratio must be in range [0, 1), got {config.val_ratio}")
    if not 0 < config.total_ratio <= 1:
        raise ValueError(
            f"total_ratio must be in range (0, 1], got {config.total_ratio}"
        )
    if config.skip_initial_episode_steps < 0:
        raise ValueError(
            f"skip_initial_episode_steps cannot be negative, "
            f"got {config.skip_initial_episode_steps}"
        )
    if config.downsample_factor < 1:
        raise ValueError(
            f"downsample_factor must be >= 1, got {config.downsample_factor}"
        )
    if config.action_backward_shift < 0:
        raise ValueError(
            f"action_backward_shift cannot be negative, "
            f"got {config.action_backward_shift}"
        )
    _validate_uniform_binning_normalization(config=config)


def _validate_uniform_binning_normalization(config: DataLoaderConfig) -> None:
    """Reject uniform action binning without min-max kinematics normalization.

    Uniform bins span a fixed [-1, 1] range, so action values outside it
    (gaussian or demeaned normalization) would silently clip into the edge
    bins and corrupt every tokenized action.
    """
    tokenization = config.tokenization
    if not tokenization.tokenize_actions or tokenization.action_tokenizer is None:
        return
    action_discretizer = tokenization.action_tokenizer.action_discretizer
    if action_discretizer.type != ActionDiscretizerType.BINNED.value:
        return
    if action_discretizer.binning_strategy != BinningStrategy.UNIFORM.value:
        return
    if config.kinematics_norm_type != KinematicsNormalizationType.MIN_MAX.value:
        raise ValueError(
            "Uniform action binning requires min-max kinematics normalization: "
            "uniform bins cover the fixed [-1, 1] range and would clip "
            f"'{config.kinematics_norm_type}'-normalized actions at the edge "
            "bins. Set kinematics_norm_type to "
            f"'{KinematicsNormalizationType.MIN_MAX.value}' or switch the "
            f"action discretizer to binning_strategy="
            f"'{BinningStrategy.QUANTILE.value}'."
        )


def _collect_dataset_paths(
    dataset_folders: list[str], episode_filename: str
) -> list[str]:
    """Collect all episode CSV paths from dataset folders."""
    datasets_paths = []
    for folder in dataset_folders:
        root_path = Path(folder)
        episode_dirs = [
            d
            for d in root_path.iterdir()
            if d.is_dir() and (d / episode_filename).exists()
        ]
        datasets_paths.extend([str(d / episode_filename) for d in episode_dirs])
    return datasets_paths


def _ensure_zarr_exists(schema: DatasetSchema, preload_in_memory: bool = False) -> None:
    """Create zarr if it doesn't exist or is invalid. Optionally, preload in memory.

    Dispatches to the appropriate creation function based on schema type:
    - Hdf5DatasetSchema: Uses hdf5_paths from schema directly
    - CsvDatasetSchema: Collects episode CSV paths from dataset_folders
    """
    zarr_path = schema.zarr_path
    need_create = True
    required_keys = schema.get_required_zarr_keys()

    if Path(zarr_path).exists():
        try:
            if preload_in_memory:
                logging.info(f"Preloading replay buffer into memory from {zarr_path}")
                ReplayBuffer.copy_from_path(zarr_path, keys=required_keys)
            else:
                logging.info(f"Loading existing replay buffer from {zarr_path}")
                buffer = ReplayBuffer.create_from_path(zarr_path)
                missing_keys = set(required_keys) - set(buffer.keys())
                if missing_keys:
                    raise KeyError(f"Missing required keys: {missing_keys}")
            need_create = False
        except Exception as e:
            logging.info(f"Error loading {zarr_path}: {e}. Recreating...")
            shutil.rmtree(zarr_path, ignore_errors=True)

    if need_create:
        logging.info(f"Creating zarr replay buffer at: {zarr_path}")
        if isinstance(schema, Hdf5DatasetSchema):
            logging.info(f"Processing {len(schema.hdf5_paths)} HDF5 files")
            create_replay_buffer_from_hdf5(schema=schema)
        elif isinstance(schema, CsvDatasetSchema):
            datasets_paths = _collect_dataset_paths(
                dataset_folders=schema.dataset_folders,
                episode_filename=schema.dataset_filename,
            )
            logging.info(
                f"Found {len(datasets_paths)} episodes across {len(schema.dataset_folders)} folders"
            )
            create_replay_buffer(schema=schema, datasets_paths=datasets_paths)
        elif isinstance(schema, LeRobotDatasetSchemaV30):
            create_replay_buffer_from_lerobot(schema=schema)
        elif isinstance(schema, SyntheticSchema):
            create_replay_buffer_from_synthetic(schema=schema)
        else:
            raise NotImplementedError(
                f"Zarr creation not implemented for schema type: {type(schema)}"
            )
