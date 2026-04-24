from dataclasses import dataclass, field

from versatil.configs.data.augmentations import AugmentationPipelineConfig
from versatil.configs.data.tokenizer import TokenizationConfig
from versatil.data.constants import (
    ImageNormalizationType,
    KinematicsNormalizationType,
)


@dataclass
class DataLoaderConfig:
    """Hydra config for dataset loading, normalization, augmentation, and sampling behavior."""

    preload_data_in_memory: bool = False  # Whether to preload the entire zarr into RAM, speeds up training considerably but works only for small datasets.
    # Batching
    batch_size: int = 64
    num_workers: int = 16
    shuffle: bool = True
    # Data processing
    image_norm_type: str = ImageNormalizationType.MINUS_ONE_TO_ONE.value
    depth_norm_type: str = ImageNormalizationType.MINUS_ONE_TO_ONE.value
    kinematics_norm_type: str = KinematicsNormalizationType.MIN_MAX.value
    winsorize_depth: bool = True
    depth_winsorize_quantiles: tuple[float, float] = (0.01, 0.99)
    winsorize_kinematics: bool = True
    kinematics_winsorize_quantiles: tuple[float, float] = (0.01, 0.99)
    # Kinematics normalization clamping - useful for datasets with very small action deltas
    clamp_kinematics_range: bool = True
    min_kinematics_std: float = 1e-2
    min_kinematics_range: float = 1e-2
    # Tokenization
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    color_augmentation: AugmentationPipelineConfig = field(
        default_factory=AugmentationPipelineConfig
    )
    spatial_augmentation: AugmentationPipelineConfig | None = field(
        default_factory=AugmentationPipelineConfig
    )
    rotation_augmentation: AugmentationPipelineConfig | None = None
    #: Whether to skip the initial n steps of each episode due to recording artifacts.
    skip_initial_episode_steps: int = 0
    #: Whether to downsample each dataset episode by taking every n-th step.
    downsample_factor: int = 1
    #: Number of steps to shift actions backward in the sequence, to compensate for hardware latency.
    action_backward_shift: int = 0
    #: Max trailing padding allowed per sampled window. Valid starts per episode:
    #: ``episode_length - sequence_length + trailing_padded_actions + 1``. If 0, only windows that fit
    #: entirely within the episode are sampled (no right-side padding). When None (default), resolves
    #: to ``pred_horizon - 1``.
    trailing_padded_actions: int | None = None
    #: Ratio of dataset episodes to use for validation.
    val_ratio: float = 0.1
    #: Ratio of total dataset episodes to use (for ablation studies on varying dataset sizes).
    total_ratio: float = 1.0
    #: Number of action rows to stash on the normalizer per action key for downstream
    #: data-aware initialization (e.g. mixture-density head k-means++). Set to 0 to
    #: disable. Memory cost per action key is ``action_sample_size * action_dim *
    #: bytes_per_element``.
    action_sample_size: int = 2048
