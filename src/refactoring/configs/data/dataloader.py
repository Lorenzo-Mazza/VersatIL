from dataclasses import dataclass, field

from refactoring.configs.data.augmentations import AugmentationPipelineConfig
from refactoring.configs.data.tokenizer import TokenizationConfig
from refactoring.data.constants import (
    ImageNormalizationType,
    KinematicsNormalizationType,
)


@dataclass
class DataLoaderConfig:
    # Batching
    batch_size: int = 64
    num_workers: int = 16
    shuffle: bool = True
    # Data processing
    # If the image dimensions of the zarr differ from these, a resize needs to be added in the augmentation module.
    image_height: int = 270
    image_width: int = 480
    image_norm_type: str = ImageNormalizationType.MINUS_ONE_TO_ONE.value
    depth_norm_type: str = ImageNormalizationType.MINUS_ONE_TO_ONE.value
    kinematics_norm_type: str = KinematicsNormalizationType.MIN_MAX
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
    action_backward_shift: int = 1
    #: Ratio of dataset episodes to use for validation.
    val_ratio: float = 0.1
    #: Ratio of total dataset episodes to use (for ablation studies on varying dataset sizes).
    total_ratio: float = 1.0
