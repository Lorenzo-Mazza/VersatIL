import abc
import json
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from legacy_constants import (
    ACTION_KEY,
    OBSERVATION_KEY,
    ROBOT_STATE_KEY,
    SHAPE_KEY,
    Cameras,
    DepthFusionStrategy,
    DepthNormalizationType,
    DiffusionArchitecture,
    DiffusionScheduler,
    ImageNormalizationType,
    KinematicsNormalizationType,
    PolicyType,
    SamplingMode,
)


@dataclass
class TaskBaseConfig(abc.ABC):
    # Experiment parameters
    exp_name: str = 'needle_driving_v5_100_fixed_angle'  # Policy name and timestamp of the experiment will be prepended to this name.
    checkpoint_folder: str = '/mnt/cluster/workspaces/mazzalore/iros'  # Policy name and time of the experiment will be appended to this path.
    resume_from_checkpoint: str | None = None  # Path of the checkpoint to resume from, if any
    use_wandb: bool = True
    total_ratio_of_episodes: float = 1.0  # Ratio of episodes to use for training, validation and testing
    val_every: int = 1
    checkpoint_every: int = 100
    plot_every: int = 200
    device: str = 'cuda'
    distributed_training: bool = False

    # Data parameters
    dataset_folders: list[str] = field(default_factory=lambda: ["/mnt/cluster/datasets/threading_il/v7",
                                  "/mnt/cluster/datasets/threading_il/v8",
                                  "/mnt/cluster/datasets/threading_il/v8_recovery"])
    zarr_path: str = '/mnt/cluster/datasets/threading_il/v9/dataset.zarr'  # Path to the Zarr archive to create or load
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 16
    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    ratio_validation_episodes: float = 0.1
    seed: int = 42
    image_height: int = 270
    image_width: int = 480
    center_crop: bool = False
    center_crop_size: int | None = None
    downsample_factor: int = 1
    skip_initial_steps: int = 5
    image_norm_type: str = ImageNormalizationType.MINUS_ONE_TO_ONE.value
    depth_norm_type: str = DepthNormalizationType.MINUS_ONE_TO_ONE.value
    kinematics_norm_type: str | None = KinematicsNormalizationType.MIN_MAX  # 'min_max' or '0_1'
    use_color_augmentations: bool = True
    use_rotation_augmentations: bool = False
    sampling_mode: str = SamplingMode.OVERLAPPING.value
    action_backward_shift: int = 1 # Number of steps to shift actions backward in the sequence ( it's a "hack" of the original ACT code to compensate for latency)
    recompute_depth_stats: bool = True
    downsample_step: int = 1  # Number of steps to downsample the dataset to
    promote_sparsity: bool = True  # Whether to promote sparsity in the dataset by thresholding small movements to zero

    predict_gripper_action: bool = False
    calculate_gripper_positive_class_weights: bool = True # Used only when predict_gripper_action is set to True
    #: Whether to use depth images
    use_depth: bool = False
    #: Whether to use rectified images
    use_rectified: bool = True
    #: Whether to use deltas as actions or next positions
    deltas_as_actions: bool = False
    #: Whether to center the initial position of each episode to be in (0, 0, 0)
    center_initial_position: bool = False
    #: Whether to request the observation in the robot frame
    obs_robot_frame: bool = True
    #: Whether to request the observation in the camera frame
    obs_camera_frame: bool = False
    #: Whether to predict actions in the camera frame
    predict_in_camera_frame: bool = False
    #: Dimension of the action
    action_dim: int = 3

    # Training parameters
    num_epochs: int = 600
    gradient_accumulate_every: int = 1
    use_ema: bool = True  # Whether to use Exponential Moving Average for model weights
    ema_power: float = 0.75
    backbone: str = "resnet18"
    depth_fusion: str = DepthFusionStrategy.LEFT_CHANNEL_WISE.value  # Strategy for depth fusion, if applicable
    # Model weights parameter for the DFormer model, used only if Depth Fusion Strategy is Geometric Attention.
    dformer_checkpoint_path: str = "/mnt/cluster/workspaces/mazzalore/iros/pretrained_dformer/DFormerv2_Small_NYU.pth"
    freeze_dformer: bool = True # Whether to freeze the whole DFormer model or fine-tune the last layer.
    # Optimizer parameters
    learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-6
    clip_gradient_norm: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    backbone_has_separate_lr: bool = False  # Whether to use a separate learning rate for the backbone
    # Learning rate scheduler parameters
    lr_schedule: str | None = None
    lr_warmup_steps: int = 5000

    # Inference parameters
    temporal_agg: bool = True
    update_rate_hz: float = 3.0  # Frequency at which to update the policy during inference

    task_has_phases: bool = False

    _overrides: dict[str, Any] = field(default_factory=dict)  # Not used; overrides are class vars in subclasses


    @property
    def state_dim(self) -> int:
        """Dimension of the observation state."""
        return 3 * (int(self.obs_robot_frame) + int(self.obs_camera_frame))

    @property
    def camera_names(self) -> list[str]:
        """List of camera names used in the dataset."""
        return [Cameras.LEFT.value, Cameras.RIGHT.value] + ([Cameras.DEPTH.value] if self.use_depth else [])

    @property
    def checkpoint_dir(self) -> str:
        """Directory where checkpoints are saved.

        The directory tree is constructed using the checkpoint_folder, policy_name, and current experiment name.
        """
        return f"{self.checkpoint_folder}/{self.policy_name}_checkpoints"



    @property
    def shape_meta(self):
        shapes = {OBSERVATION_KEY: {}, ACTION_KEY: {SHAPE_KEY: (self.action_dim,)}}
        for cam in self.camera_names:
            channels = 1 if cam == Cameras.DEPTH.value else 3
            shapes[OBSERVATION_KEY][cam] = {
                SHAPE_KEY: (channels, self.image_height, self.image_width),
            }
        if self.state_dim > 0:
            shapes[OBSERVATION_KEY][ROBOT_STATE_KEY] = {SHAPE_KEY: (self.state_dim,)}
        return shapes


    @property
    @abstractmethod
    def policy_name(self) -> str:
        """Name of the policy."""
        raise NotImplementedError("Subclasses must implement the policy_name property.")


    def __post_init__(self):
        if self.distributed_training and self.device != 'cuda':
            raise ValueError("Distributed training is only supported in GPU.")
        for key, value in type(self)._overrides.items():
            setattr(self, key, value)
        self.exp_name = f"{self.policy_name}_" + self.exp_name


@dataclass
class DiffusionConfig(TaskBaseConfig):
    # Policy-specific overrides
    _overrides = {
        "obs_horizon": 2,
        "pred_horizon": 16,
        "action_horizon": 8,
        "obs_camera_frame": True,
        "deltas_as_actions": False,
        "sampling_mode": SamplingMode.OVERLAPPING.value,
        "image_norm_type": ImageNormalizationType.MINUS_ONE_TO_ONE.value,
        "depth_fusion": DepthFusionStrategy.LEFT_CHANNEL_WISE.value,
        "use_ema": True,
        "ema_power": 0.75,
        "num_epochs": 600,
        "learning_rate": 1.0e-6,
        "weight_decay": 1.0e-6,
        "lr_schedule": "cosine",
        "betas": (0.95, 0.999),
    }
    # Diffusion parameters
    random_crop: bool = False  # We handle random cropping inside the vision encoder
    pretrained_backbone: bool = False  # End-to-end training from scratch
    action_model_architecture: str = DiffusionArchitecture.UNET.value # Like Surgical Robot Transformer paper
    diffusion_scheduler: DiffusionScheduler = DiffusionScheduler.DDIM.value
    num_train_timesteps: int = 100
    num_inference_steps: int = 10  # Fast inference
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    scheduler_variance_type: str = "fixed_small"
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    prediction_type: str = "epsilon"
    down_dims: tuple[int, int, int] = (256, 512, 1024)


    @property
    def policy_name(self) -> str:
        """Name of the policy."""
        return PolicyType.DIFFUSION_POLICY.value


@dataclass
class FlowMatchingConfig(TaskBaseConfig):
    _overrides = {
        "obs_horizon": 2,
        "pred_horizon": 16,
        "action_horizon": 8,
        "obs_camera_frame": True,
        "deltas_as_actions": False,
        "sampling_mode": SamplingMode.OVERLAPPING.value,
        "image_norm_type": ImageNormalizationType.MINUS_ONE_TO_ONE.value,
        "depth_fusion": DepthFusionStrategy.LEFT_CHANNEL_WISE.value,
        "use_ema": True,
        "ema_power": 0.75,
        "num_epochs": 600,
        "learning_rate": 1.0e-6,
        "weight_decay": 1.0e-6,
        "lr_schedule": "cosine",
        "betas": (0.95, 0.999),
    }
    # Flow matching parameters
    random_crop: bool = True  # We handle random cropping in the dataset
    pretrained_backbone: bool = False
    action_model_architecture: str = DiffusionArchitecture.UNET.value
    num_inference_steps: int = 10
    sigma: float = 0.0
    down_dims: tuple[int, int, int] = (256, 512, 1024)

    @property
    def policy_name(self) -> str:
        """Name of the policy."""
        return PolicyType.FLOW_MATCHING.value


@dataclass
class ACTConfig(TaskBaseConfig):
    """Configuration for the ACT policy."""
    _overrides = {
        "obs_horizon": 1,
        "action_horizon": 30,
        "pred_horizon": 30,
        "deltas_as_actions": True,
        "obs_camera_frame": False,
        "action_backward_shift": 1,
        "sampling_mode": SamplingMode.RANDOM_CHUNK.value,
        "image_norm_type": ImageNormalizationType.ZERO_TO_ONE.value,
        "kinematics_norm_type": KinematicsNormalizationType.GAUSSIAN.value,
        "depth_fusion": DepthFusionStrategy.GEOMETRIC_ATTENTION.value,
        "depth_norm_type": DepthNormalizationType.ZERO_TO_ONE.value,
        "use_ema": False,
        "ema_power": None,
        "num_epochs": 8000,
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
        "lr_schedule": None,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "backbone_has_separate_lr": True,

    }
    use_fake_proprio: bool = False
    # * Backbone
    dilation: bool = False
    position_embedding: str = 'sine'
    # * Transformer
    enc_layers: int = 4
    dec_layers: int = 7
    dim_feedforward: int = 3200
    hidden_dim: int = 512
    dropout: float = 0.1
    nheads: int = 8
    pre_norm: bool = False
    # * Segmentation
    masks: bool = False

    # Optimization parameters
    lr_backbone: float = 1e-5
    lr_drop: int = 200
    clip_max_norm: float = 0.1
    sinkhorn_weight: float = 0.2
    mse_weight: float = 0.8
    length_weight: float = 0.001
    bce_weight: float = 0.05
    l1_weight: float = 0.0
    kl_weight: float = 0.0 # default in ACT paper is 100

    @property
    def policy_name(self) -> str:
        """Name of the policy."""
        return PolicyType.ACT.value

@dataclass
class PhaseACTConfig(ACTConfig):
    n_phases: int = 5 # Number of phases to divide the action sequence into
    phase_ce_weight: float = 0.1 # Weight for the phase classification loss
    entropy_weight: float = 0.005 # Weight for the entropy (regularization term) of the predicted phase probabilities
    phase_learnable_temperature: float = 100.0 # Learnable parameter for the logits temperature (confidence regularizer)

    @property
    def policy_name(self) -> str:
        """Name of the policy."""
        return PolicyType.PHASE_ACT.value

@dataclass
class MoEConfig(TaskBaseConfig):
    n_experts: int = 5 # Number of experts in the Mixture of Experts model
    routing_method: str = 'soft' # Routing method: 'top_k' or 'soft'
    top_k: int = 2 # Number of top experts to use for each input
    encoder: str = 'cvae'
    decoder: str = 'transformer'
    expert_network = 'mlp'

    @property
    def policy_name(self) -> str:
        """Name of the policy."""
        return PolicyType.MOE.value

PolicyConfig = DiffusionConfig | FlowMatchingConfig | ACTConfig | PhaseACTConfig


def save_config(config: PolicyConfig, output_dir: Path | str):
    """Save ACTConfig to a JSON file in the given directory.

    Args:
        config: The configuration to save
        output_dir: Directory where to save the config.json file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = vars(config)
    config_dict['policy_name'] = config.policy_name  # Explicitly add the properties
    config_dict['state_dim'] = config.state_dim
    config_dict['camera_names'] = config.camera_names
    config_dict['shape_meta'] = config.shape_meta
    config_dict['checkpoint_dir'] = config.checkpoint_dir
    with open(output_dir / "config.json", "w") as f:
        # convert non-serializable objects to string
        json.dump(config_dict, f, indent=2, default=str)


def load_config(config_dir: Path | str) -> PolicyConfig:
    """Load policy config from a JSON file in the given directory.

    Args:
        config_dir: Directory containing the config.json file

    Returns:
        An ACTConfig instance with the loaded settings
    """
    output_dir = Path(config_dir)
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        data = json.load(f)
    mapping = {
        PolicyType.ACT.value: ACTConfig,
        PolicyType.DIFFUSION_POLICY.value: DiffusionConfig,
        PolicyType.FLOW_MATCHING.value: FlowMatchingConfig,
        PolicyType.PHASE_ACT.value : PhaseACTConfig,
    }
    policy = data.get("policy_name")
    if not policy:
        raise ValueError("Missing policy_name field in config.json")
    config_class = mapping.get(policy)
    if not config_class:
        raise ValueError(f"Unknown policy config for policy_name {policy}")
    cfg = config_class()
    for key, value in data.items():
        if key in ["policy_name","state_dim", "camera_names", "shape_meta", "checkpoint_dir"]:
            continue
        setattr(cfg, key, value)
    return cfg
