from dataclasses import dataclass, field

from refactoring.configs.task.augmentations import (
    AugmentationPipelineConfig,
    ColorAugmentationPipeline,
    RotateConfig,
    SpatialAugmentationPipeline,
)
from refactoring.data.constants import (
    ImageNormalizationType,
    KinematicsNormalizationType,
    SamplingMode,
)


@dataclass
class TokenizationConfig:
    """Tokenization configuration for actions and proprioceptive observations.
    Tokenization converts continuous normalized values into discrete tokens,
    enabling vocabulary-based action prediction with transformers.
    """
    enabled: bool = False
    tokenize_actions: bool = False
    use_pretrained_action_tokenizer: bool = True
    tokenize_proprio_obs: bool = False
    proprio_num_bins: int = 256


@dataclass
class DataloaderConfig:
    """Data loading and preprocessing configuration."""
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

    # Tokenization
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)

    # Augmentations
    color_augmentation: AugmentationPipelineConfig | None = field(
        default_factory=lambda: ColorAugmentationPipeline()
    )
    spatial_augmentation: AugmentationPipelineConfig | None = field(
        default_factory=lambda: SpatialAugmentationPipeline()
    )
    rotation_augmentation: RotateConfig | None = field(
        default_factory=lambda: RotateConfig()
    )

    #: Sampling mode
    sampling_mode: str = SamplingMode.OVERLAPPING.value
    #: Whether to skip the initial n steps of each episode due to recording artifacts.
    skip_initial_episode_steps: int = 0
    #: Whether to downsample each dataset episode by taking every n-th step.
    downsample_factor: int = 1
    #: Whether to center kinematics data around zero for each episode start.
    center_episode_start: bool = False
    #: Number of steps to shift actions backward in the sequence, to compensate for hardware latency.
    action_backward_shift: int = 1
    #: Ratio of dataset episodes to use for validation.
    val_ratio: float = 0.1
    #: Ratio of total dataset episodes to use (for ablation studies on varying dataset sizes).
    total_ratio: float = 1.0


    def __post_init__(self):
        """Validate configuration after initialization.

        This catches simple self-contained errors immediately,
        reducing the need for external validation.
        """
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers cannot be negative, got {self.num_workers}")
        if self.image_height <= 0:
            raise ValueError(f"image_height must be positive, got {self.image_height}")
        if self.image_width <= 0:
            raise ValueError(f"image_width must be positive, got {self.image_width}")
        if not 0 < self.val_ratio < 1:
            raise ValueError(f"val_ratio must be in range (0, 1), got {self.val_ratio}")
        if not 0 < self.total_ratio <= 1:
            raise ValueError(f"total_ratio must be in range (0, 1], got {self.total_ratio}")
        if self.skip_initial_episode_steps < 0:
            raise ValueError(f"skip_initial_episode_steps cannot be negative, "f"got {self.skip_initial_episode_steps}")
        if self.downsample_factor < 1:
            raise ValueError(f"downsample_factor must be >= 1, got {self.downsample_factor}")
        if self.action_backward_shift < 0:
            raise ValueError(f"action_backward_shift cannot be negative, "f"got {self.action_backward_shift}")

        # Validate enum values
        valid_image_norms = [e.value for e in ImageNormalizationType]
        if self.image_norm_type not in valid_image_norms:
            raise ValueError(f"image_norm_type must be one of {valid_image_norms}, "f"got {self.image_norm_type}")
        if self.depth_norm_type not in valid_image_norms:
            raise ValueError(f"depth_norm_type must be one of {valid_image_norms}, "f"got {self.depth_norm_type}")
        valid_sampling_modes = [e.value for e in SamplingMode]
        if self.sampling_mode not in valid_sampling_modes:
            raise ValueError(f"sampling_mode must be one of {valid_sampling_modes}, " f"got {self.sampling_mode}")
