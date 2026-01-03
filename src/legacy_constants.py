import enum


class Cameras(enum.Enum):
    """Enum for camera names."""

    LEFT = "left"
    RIGHT = "right"
    DEPTH = "depth"


class ImageNormalizationType(str, enum.Enum):
    """Enum for image normalization types."""

    MINUS_ONE_TO_ONE = "minus_one_to_one"  # Normalize to [-1, 1]
    ZERO_TO_ONE = "zero_to_one"  # Normalize to [0, 1]
    IMAGENET = "imagenet"  # Normalize between 0 and 1 and center using ImageNet stats (mean and std)


class KinematicsNormalizationType(str, enum.Enum):
    """Enum for kinematics normalization types."""

    MIN_MAX = "min_max"  # Normalize to [0, 1]
    GAUSSIAN = "gaussian"  # Standardize to mean 0 and std 1


class DepthNormalizationType(str, enum.Enum):
    """Enum for depth maps normalization."""

    MINUS_ONE_TO_ONE = "minus_one_to_one"  # Normalize to [-1, 1]
    ZERO_TO_ONE = "zero_to_one"  # Normalize to [0, 1]
    IMAGENET = "imagenet"  # Normalize between 0 and 1 and center using ImageNet stats (mean and std)


class SamplingMode(str, enum.Enum):
    """Enum for sampling mode of the dataset.

    -Overlapping means that the samples can overlap in time.
    -Random_chunk means that the samples are randomly selected chunks of the episode.
    """

    OVERLAPPING = "overlapping"
    RANDOM_CHUNK = "random_chunk"


class DepthFusionStrategy(str, enum.Enum):
    """Enum for depth fusion strategies in multi-modal image processing (e.g., RGB + depth).

    - LEFT_CHANNEL_WISE: Fuse the depth map channel-wise (along the channel dimension) with the left RGB camera's features.
      This assumes the depth map aligns spatially with the left view (common in stereo setups where left is the reference),
      allowing mid-level fusion after backbones for better cross-modal integration without spatial mismatch.

    - WIDTH: Fuse by concatenating depth features spatially along the width dimension (dim=3), treating depth as a separate
      'view' alongside left and right RGB. This is a simple multi-view concatenation but may not exploit pixel-wise
      correspondences between depth and RGB, potentially leading to less optimal learning of 3D-aware features.
    """

    LEFT_CHANNEL_WISE = "left_channel_wise"
    WIDTH = "width"
    SEPARATE = "separate"
    ATTENTION = "attention"  # Use attention to fuse depth and RGB features
    GEOMETRIC_ATTENTION = "geometric_attention"  # Use dformerv2's geometric attention for depth and left RGB fusion


class DiffusionScheduler(enum.Enum):
    DDIM = "DDIM"  # Denoising diffusion implicit models
    DDPM = "DDPM"  # Denoising diffusion probabilistic models


class DiffusionArchitecture(enum.Enum):
    """Enum for different diffusion architectures."""

    UNET = "UNET"  # Standard UNet architecture
    TRANSFORMER = "Transformer"  # Transformer-based architecture


class PolicyType(enum.Enum):
    """Enum for different policy types."""

    ACT = "ACT"
    DIFFUSION_POLICY = "DIFFUSION_POLICY"
    FLOW_MATCHING = "FLOW_MATCHING"
    PHASE_ACT = "PHASE_ACT"
    MOE = "MOE"


class MoERoutingType(enum.Enum):
    """Enum for different Mixture of Experts (MoE) routing strategies."""

    TOP_K = "top_k"  # Select the top-k experts based on gating scores
    SOFT = "soft"  # Weighted combination of all experts based on gating scores


# Buffer key for robot and camera frame observations
ROBOT_FRAME_OBS_KEY = "robot_frame_obs"
CAMERA_FRAME_OBS_KEY = "camera_frame_obs"
GRIPPER_STATE_OBS_KEY = "gripper_state_obs"

ACTION_KEY = "action"  # Used only for defining the shape of the action. It is not used in the policy.

POSITION_ACTION_KEY = "position_action"
GRIPPER_ACTION_KEY = "gripper_action"

ROBOT_STATE_KEY = "robot_state"
OBSERVATION_KEY = "observation"
IS_PAD_KEY = "is_pad"
SHAPE_KEY = "shape"
PHASE_LABEL_KEY = "phase_label"


class ExplanationType(enum.Enum):
    GRADCAM = "gradcam"
    GRADCAM_PLUS_PLUS = "gradcam++"
    ABLATION_CAM = "ablation_cam"
    SALIENCY_MAP = "saliency_map"
    INTEGRATED_GRADIENT = "integrated_gradient"


IMAGENET_RGB_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_RGB_STD: list[float] = [0.229, 0.224, 0.225]
IMAGENET_DEPTH_MEAN: float = 0.48  # Cf. https://github.com/VCIP-RGBD/RGBD-Pretrain/blob/main/data/constants.py#L3
IMAGENET_DEPTH_STD: float = 0.28
