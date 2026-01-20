"""Centralized test fixtures.
"""
import os
from pathlib import Path
CACHE_DIR = Path("/mnt/cluster/workspaces/mazzalore/pretrained_models")

def setup_cache_directories():
    """Configure cache directories for model downloads before any test run."""
    os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
    for cache_path in [
        CACHE_DIR / "huggingface" / "transformers",
        CACHE_DIR / "huggingface" / "hub",
        CACHE_DIR / "torch" / "hub",
    ]:
        cache_path.mkdir(parents=True, exist_ok=True)
setup_cache_directories()

import pytest
import torch
import numpy as np
import zarr
from pathlib import Path
import tempfile
from omegaconf import OmegaConf


from versatil.data.constants import (
    Cameras,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    ProprioKey,
)
from versatil.configs.main import MainConfig
from versatil.configs.experiment import ExperimentConfig
from versatil.configs.data.task import TaskSpaceConfig
from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.configs.training import TrainingConfig, AdamWConfig
from versatil.configs.policy import PolicyConfig
from versatil.configs.inference import InferenceConfig
from versatil.data.constants import (
    OBSERVATION_KEY,
    ACTION_KEY,
    POSITION_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
)
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import OrientationRepresentation, GripperType
from versatil.models.encoding.encoders.base import EncoderOutput, EncoderInput
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.encoding.fusion.base import FusionModule, FusionInput, FusionOutput
from versatil.models.decoding.algorithm.behavior_cloning import BehavioralCloning
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.models.policy import Policy
from versatil.metrics.composite import ActionReconstructionLoss



class DummyNormalizer(torch.nn.Module):
    """Dummy normalizer that acts as pass-through (no normalization)."""

    def __init__(self):
        super().__init__()
        self.params_dict = torch.nn.ParameterDict()

    def __getitem__(self, key):
        """Support subscripting to match LinearNormalizer API."""
        return self

    def normalize(self, x):
        """Pass-through normalization (identity function)."""
        return x

    def unnormalize(self, x):
        """Pass-through unnormalization (identity function)."""
        return x

    def unnormalize_actions(self, x):
        """Pass-through action unnormalization (identity function)."""
        return x


class DummyRGBEncoder(torch.nn.Module):
    """Minimal RGB encoder for testing."""

    def __init__(self):
        super().__init__()
        self.name = "rgb"
        self.conv = torch.nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.input_specification = EncoderInput(keys=["rgb"])

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=["features"],
            dimensions={"features": (256, 7, 7)}
        )

    def forward(self, inputs: dict) -> dict:
        input_key = self.input_specification.keys[0]
        rgb = inputs[input_key]

        # Handle temporal dimension: (B, T, C, H, W) -> (B*T, C, H, W)
        if rgb.ndim == 5:
            B, T, C, H, W = rgb.shape
            rgb = rgb.reshape(B * T, C, H, W)
            x = self.conv(rgb)
            x = self.pool(x)
            # Reshape back: (B*T, C', H', W') -> (B, T, C', H', W')
            _, C_out, H_out, W_out = x.shape
            x = x.reshape(B, T, C_out, H_out, W_out)
        else:
            x = self.conv(rgb)
            x = self.pool(x)

        return {"features": x}


class DummyDepthEncoder(torch.nn.Module):
    """Minimal depth encoder for testing."""

    def __init__(self):
        super().__init__()
        self.name = "depth"
        self.conv = torch.nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.input_specification = EncoderInput(keys=["depth"])

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=["features"],
            dimensions={"features": (128, 7, 7)}
        )

    def forward(self, inputs: dict) -> dict:
        input_key = self.input_specification.keys[0]
        depth = inputs[input_key]
        x = self.conv(depth)
        x = self.pool(x)
        return {"features": x}


class DummyProprioEncoder(torch.nn.Module):
    """Minimal proprioceptive encoder for testing."""

    def __init__(self, input_dim=7, output_dim=128):
        super().__init__()
        self.name = "proprio"
        self.output_dim = output_dim
        self.mlp = torch.nn.Linear(input_dim, output_dim)
        self.input_specification = EncoderInput(keys=["proprio"])

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=["features"],
            dimensions={"features": self.output_dim}
        )

    def forward(self, inputs: dict) -> dict:
        proprio = inputs["proprio"]
        if proprio.dim() == 3:
            B, T, D = proprio.shape
            x = self.mlp(proprio.reshape(B*T, D))
            x = x.reshape(B, T, -1)
            x = x[:, -1, :]
        else:
            x = self.mlp(proprio)
        return {"features": x}


class DummyFusion(torch.nn.Module):
    """Minimal fusion for testing - simple pooling and concat."""

    def __init__(self, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.output_name = "fused"
        self.input_features = ["rgb_features", "proprio_features"]

    def get_output_specification(self) -> FusionOutput:
        return FusionOutput(
            output_name="fused",
            output_dim=self.output_dim
        )

    def forward(self, input_features: list) -> torch.Tensor:
        """Forward pass that accepts a list of feature tensors.

        Args:
            input_features: List of tensors [rgb_features, proprio_features]

        Returns:
            Fused tensor
        """
        feats_to_concat = []

        for feat in input_features:
            if feat is None:
                continue

            # Handle spatial+temporal features (B, T, C, H, W)
            if feat.dim() == 5:
                B, T, C, H, W = feat.shape
                feat = feat.reshape(B*T, C, H, W)
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat = feat.reshape(B, T, -1)
                feat = feat[:, -1, :]
            # Handle spatial features (B, C, H, W)
            elif feat.dim() == 4:
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
            # Handle flat features (B, D) - already in correct format
            # No additional processing needed

            feats_to_concat.append(feat)

        if len(feats_to_concat) > 0:
            fused = torch.cat(feats_to_concat, dim=-1)
            if fused.shape[-1] != self.output_dim:
                proj = torch.nn.Linear(fused.shape[-1], self.output_dim).to(fused.device)
                fused = proj(fused)
        else:
            batch_size = 1
            fused = torch.zeros(batch_size, self.output_dim)

        return fused


class MockActionDecoder(ActionDecoder):
    """Mock action decoder for testing."""

    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        feature_dim: int = 128,
        prediction_horizon: int = 10,
        device: str = "cpu",
    ):
        decoder_input = DecoderInput(keys=["fused"])

        action_heads = {}
        if action_space.has_position:
            action_heads[POSITION_ACTION_KEY] = ActionHead(
                input_dim=feature_dim,
                output_dim=action_space.position_dim,
                blocks=[],
            )
        if action_space.has_orientation:
            action_heads[ORIENTATION_ACTION_KEY] = ActionHead(
                input_dim=feature_dim,
                output_dim=action_space.orientation_dim,
                blocks=[],
            )
        if action_space.has_gripper:
            action_heads[GRIPPER_ACTION_KEY] = ActionHead(
                input_dim=feature_dim,
                output_dim=action_space.gripper_dim,
                blocks=[],
            )

        super().__init__(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=1,
            prediction_horizon=prediction_horizon,
        )

        self.feature_dim = feature_dim
        self.feature_proj = torch.nn.Linear(feature_dim, feature_dim).to(self.device)

    def forward(self, features: dict, actions=None) -> dict:
        feature_tensor = features.get("fused", next(iter(features.values())))
        batch_size = feature_tensor.shape[0]

        projected = self.feature_proj(feature_tensor)

        predictions = {}
        if self.use_position_actions:
            predictions[POSITION_ACTION_KEY] = self.action_heads[POSITION_ACTION_KEY](projected).unsqueeze(1).repeat(1, self.prediction_horizon, 1)
        if self.use_orientation_actions:
            predictions[ORIENTATION_ACTION_KEY] = self.action_heads[ORIENTATION_ACTION_KEY](projected).unsqueeze(1).repeat(1, self.prediction_horizon, 1)
        if self.use_gripper_actions:
            predictions[GRIPPER_ACTION_KEY] = self.action_heads[GRIPPER_ACTION_KEY](projected).unsqueeze(1).repeat(1, self.prediction_horizon, 1)

        return predictions



def generate_synthetic_positions(
        num_samples: int = 10,
        num_dims: int = 3,
        position_range: tuple = (-1.0, 1.0),
        seed: int = 42
) -> np.ndarray:
    """Generate synthetic position data.

    Args:
        num_samples: Number of timesteps
        num_dims: Number of position dimensions (usually 3)
        position_range: (min, max) range for positions
        seed: Random seed

    Returns:
        Array of shape (num_samples, num_dims)
    """
    np.random.seed(seed)
    return np.random.uniform(
        position_range[0], position_range[1], (num_samples, num_dims)
    ).astype(np.float32)


def generate_synthetic_quaternions(
        num_samples: int = 10,
        seed: int = 42
) -> np.ndarray:
    """Generate synthetic unit quaternions (w, action_embedding, y, z).

    Args:
        num_samples: Number of timesteps
        seed: Random seed

    Returns:
        Array of shape (num_samples, 4) with normalized quaternions
    """
    np.random.seed(seed)
    quats = np.random.randn(num_samples, 4).astype(np.float32)
    # Normalize to unit quaternions
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    return quats


def generate_synthetic_euler_angles(
        num_samples: int = 10,
        angle_range: tuple = (-np.pi, np.pi),
        seed: int = 42
) -> np.ndarray:
    """Generate synthetic Euler angles (xyz convention).

    Args:
        num_samples: Number of timesteps
        angle_range: (min, max) range for angles in radians
        seed: Random seed

    Returns:
        Array of shape (num_samples, 3)
    """
    np.random.seed(seed)
    return np.random.uniform(
        angle_range[0], angle_range[1], (num_samples, 3)
    ).astype(np.float32)


def generate_synthetic_gripper_states(
        num_samples: int = 10,
        gripper_type: str = "binary",
        seed: int = 42
) -> np.ndarray:
    """Generate synthetic gripper states.

    Args:
        num_samples: Number of timesteps
        gripper_type: "binary" (0/1) or "continuous" (0.0-1.0)
        seed: Random seed

    Returns:
        Array of shape (num_samples, 1)
    """
    np.random.seed(seed)
    if gripper_type == "binary":
        return np.random.randint(0, 2, (num_samples, 1)).astype(np.float32)
    elif gripper_type == "continuous":
        return np.random.uniform(0.0, 1.0, (num_samples, 1)).astype(np.float32)
    else:
        raise ValueError(f"Unknown gripper_type: {gripper_type}")


def generate_synthetic_rgb_images(
        num_timesteps: int = 5,
        height: int = 64,
        width: int = 64,
        seed: int = 42
) -> np.ndarray:
    """Generate synthetic RGB images.

    Args:
        num_timesteps: Number of timesteps (T)
        height: Image height (H)
        width: Image width (W)
        seed: Random seed

    Returns:
        Array of shape (T, H, W, 3) with values in [0, 1]
    """
    np.random.seed(seed)
    return np.random.rand(num_timesteps, height, width, 3).astype(np.float32)


def generate_synthetic_depth_images(
        num_timesteps: int = 5,
        height: int = 64,
        width: int = 64,
        depth_range: tuple = (0.5, 5.0),
        seed: int = 42
) -> np.ndarray:
    """Generate synthetic depth images.

    Args:
        num_timesteps: Number of timesteps (T)
        height: Image height (H)
        width: Image width (W)
        depth_range: (min, max) depth values in meters
        seed: Random seed

    Returns:
        Array of shape (T, H, W) with depth values
    """
    np.random.seed(seed)
    return np.random.uniform(
        depth_range[0], depth_range[1], (num_timesteps, height, width)
    ).astype(np.float32)


def generate_synthetic_episode(
        num_timesteps: int = 10,
        position_dim: int = 3,
        orientation_dim: int = 4,
        has_gripper: bool = True,
        cameras: list = None,
        image_height: int = 64,
        image_width: int = 64,
        seed: int = 42
) -> dict:
    """Generate a complete synthetic episode.

    Args:
        num_timesteps: Number of timesteps in episode
        position_dim: Position dimensionality (usually 3)
        orientation_dim: Orientation dimensionality (4 for quat, 3 for euler, 1 for roll)
        has_gripper: Whether to include gripper states
        cameras: List of camera names (e.g., [Cameras.LEFT.value, Cameras.RIGHT.value])
        image_height: Image height
        image_width: Image width
        seed: Random seed

    Returns:
        Dictionary with episode data matching replay buffer structure
    """
    if cameras is None:
        cameras = [Cameras.LEFT.value, Cameras.RIGHT.value]

    episode = {}

    # Generate proprioceptive observations
    positions = generate_synthetic_positions(num_timesteps, position_dim, seed=seed)

    if orientation_dim == 4:
        orientations = generate_synthetic_quaternions(num_timesteps, seed=seed)
    elif orientation_dim == 3:
        orientations = generate_synthetic_euler_angles(num_timesteps, seed=seed)
    elif orientation_dim == 1:
        orientations = generate_synthetic_euler_angles(num_timesteps, seed=seed)[:, :1]
    else:
        orientations = np.zeros((num_timesteps, 0), dtype=np.float32)

    # Combine position and orientation
    proprio = np.concatenate([positions, orientations], axis=1)
    episode[PROPRIO_OBS_ROBOT_FRAME_KEY] = proprio
    episode[PROPRIO_OBS_CAMERA_FRAME_KEY] = proprio.copy()  # Same for simplicity

    # Generate gripper states
    if has_gripper:
        episode[GRIPPER_STATE_OBS_KEY] = generate_synthetic_gripper_states(
            num_timesteps, gripper_type="binary", seed=seed
        )

    # Generate images
    for cam in cameras:
        if cam == Cameras.DEPTH.value:
            episode[cam] = generate_synthetic_depth_images(
                num_timesteps, image_height, image_width, seed=seed
            )
        else:
            # RGB images (need to be uint8 [0, 255])
            rgb = generate_synthetic_rgb_images(
                num_timesteps, image_height, image_width, seed=seed
            )
            episode[cam] = (rgb * 255).astype(np.uint8)

    return episode


def create_synthetic_replay_buffer(
        num_episodes: int = 5,
        num_timesteps_per_episode: int = 10,
        position_dim: int = 3,
        orientation_dim: int = 4,
        has_gripper: bool = True,
        cameras: list = None,
        image_height: int = 64,
        image_width: int = 64,
        seed: int = 42
) -> tuple:
    """Create a synthetic replay buffer in Zarr format.

    Args:
        num_episodes: Number of episodes
        num_timesteps_per_episode: Timesteps per episode
        position_dim: Position dimensionality
        orientation_dim: Orientation dimensionality
        has_gripper: Whether to include gripper
        cameras: List of camera names
        image_height: Image height
        image_width: Image width
        seed: Random seed

    Returns:
        Tuple of (zarr_path, episode_ends)
    """
    if cameras is None:
        cameras = [Cameras.LEFT.value, Cameras.RIGHT.value]

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    zarr_path = Path(temp_dir) / "test_replay_buffer.zarr"

    # Create zarr groups
    store = zarr.storage.LocalStore(str(zarr_path))
    root = zarr.open_group(store=store, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    # Calculate total timesteps
    total_timesteps = num_episodes * num_timesteps_per_episode
    proprio_dim = position_dim + orientation_dim

    # Create arrays
    data_group.create_array(
        PROPRIO_OBS_ROBOT_FRAME_KEY,
        shape=(total_timesteps, proprio_dim),
        chunks=(100, proprio_dim),
        dtype=np.float32,
    )

    data_group.create_array(
        PROPRIO_OBS_CAMERA_FRAME_KEY,
        shape=(total_timesteps, proprio_dim),
        chunks=(100, proprio_dim),
        dtype=np.float32,
    )

    if has_gripper:
        data_group.create_array(
            GRIPPER_STATE_OBS_KEY,
            shape=(total_timesteps, 1),
            chunks=(100, 1),
            dtype=np.float32,
        )

    for cam in cameras:
        if cam == Cameras.DEPTH.value:
            data_group.create_array(
                cam,
                shape=(total_timesteps, image_height, image_width),
                chunks=(1, image_height, image_width),
                dtype=np.float32,
            )
        else:
            data_group.create_array(
                cam,
                shape=(total_timesteps, image_height, image_width, 3),
                chunks=(1, image_height, image_width, 3),
                dtype=np.uint8,
            )

    # Generate and append episodes
    episode_ends = []
    current_idx = 0

    for ep_idx in range(num_episodes):
        episode = generate_synthetic_episode(
            num_timesteps=num_timesteps_per_episode,
            position_dim=position_dim,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            cameras=cameras,
            image_height=image_height,
            image_width=image_width,
            seed=seed + ep_idx,  # Different seed per episode
        )

        # Append to arrays
        end_idx = current_idx + num_timesteps_per_episode
        data_group[PROPRIO_OBS_ROBOT_FRAME_KEY][current_idx:end_idx] = episode[PROPRIO_OBS_ROBOT_FRAME_KEY]
        data_group[PROPRIO_OBS_CAMERA_FRAME_KEY][current_idx:end_idx] = episode[PROPRIO_OBS_CAMERA_FRAME_KEY]

        if has_gripper:
            data_group[GRIPPER_STATE_OBS_KEY][current_idx:end_idx] = episode[GRIPPER_STATE_OBS_KEY]

        for cam in cameras:
            data_group[cam][current_idx:end_idx] = episode[cam]

        episode_ends.append(end_idx)
        current_idx = end_idx

    # Save metadata
    meta_group.create_array(
        'episode_ends',
        data=np.array(episode_ends),
        chunks=(len(episode_ends),),
    )

    return str(zarr_path), episode_ends


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_episode():
    """Generate a single synthetic episode with default parameters."""
    return generate_synthetic_episode(
        num_timesteps=10,
        position_dim=3,
        orientation_dim=4,
        has_gripper=True,
        cameras=[Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value],
        image_height=64,
        image_width=64,
        seed=42
    )


@pytest.fixture
def synthetic_replay_buffer():
    """Create a synthetic replay buffer with default parameters.

    Returns:
        Tuple of (zarr_path, episode_ends)
    """
    return create_synthetic_replay_buffer(
        num_episodes=5,
        num_timesteps_per_episode=10,
        position_dim=3,
        orientation_dim=4,
        has_gripper=True,
        cameras=[Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value],
        image_height=64,
        image_width=64,
        seed=42
    )


@pytest.fixture
def synthetic_replay_buffer_small():
    """Create a small synthetic replay buffer (2 episodes, 5 timesteps each)."""
    return create_synthetic_replay_buffer(
        num_episodes=2,
        num_timesteps_per_episode=5,
        position_dim=3,
        orientation_dim=4,
        has_gripper=True,
        cameras=[Cameras.LEFT.value],
        image_height=32,
        image_width=32,
        seed=42
    )


@pytest.fixture
def synthetic_positions():
    """Generate synthetic position data (10 samples, 3D)."""
    return generate_synthetic_positions(num_samples=10, num_dims=3, seed=42)


@pytest.fixture
def synthetic_quaternions():
    """Generate synthetic quaternions (10 samples)."""
    return generate_synthetic_quaternions(num_samples=10, seed=42)


@pytest.fixture
def synthetic_euler_angles():
    """Generate synthetic Euler angles (10 samples)."""
    return generate_synthetic_euler_angles(num_samples=10, seed=42)


@pytest.fixture
def synthetic_rgb_images():
    """Generate synthetic RGB images (5 timesteps, 64x64)."""
    return generate_synthetic_rgb_images(
        num_timesteps=5, height=64, width=64, seed=42
    )


@pytest.fixture
def synthetic_depth_images():
    """Generate synthetic depth images (5 timesteps, 64x64)."""
    return generate_synthetic_depth_images(
        num_timesteps=5, height=64, width=64, seed=42
    )








@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def temporal_length():
    """Default temporal sequence length."""
    return 2


@pytest.fixture
def image_size():
    """Default image size (H, W)."""
    return (224, 224)


@pytest.fixture
def rgb_image_4d(batch_size, image_size):
    """4D RGB image batch (B, C, H, W)."""
    H, W = image_size
    return torch.randn(batch_size, 3, H, W)


@pytest.fixture
def rgb_image_5d(batch_size, temporal_length, image_size):
    """5D RGB image batch with temporal dimension (B, T, C, H, W)."""
    H, W = image_size
    return torch.randn(batch_size, temporal_length, 3, H, W)


@pytest.fixture
def input_dict_4d(rgb_image_4d):
    """Input dictionary with 4D RGB images."""
    return {"rgb": rgb_image_4d}


@pytest.fixture
def input_dict_5d(rgb_image_5d):
    """Input dictionary with 5D RGB images (temporal)."""
    return {"rgb": rgb_image_5d}


@pytest.fixture
def observation_dict_multi_camera(batch_size, image_size):
    """Observation dictionary with multiple camera views."""
    H, W = image_size
    return {
        "left_rgb": torch.randn(batch_size, 3, H, W),
        "right_rgb": torch.randn(batch_size, 3, H, W),
        "wrist_rgb": torch.randn(batch_size, 3, H, W),
    }


@pytest.fixture
def observation_dict_temporal_multi_camera(batch_size, temporal_length, image_size):
    """Temporal observation dictionary with multiple camera views."""
    H, W = image_size
    return {
        "left_rgb": torch.randn(batch_size, temporal_length, 3, H, W),
        "right_rgb": torch.randn(batch_size, temporal_length, 3, H, W),
        "wrist_rgb": torch.randn(batch_size, temporal_length, 3, H, W),
    }


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(params=[
    {"num_timesteps": 5, "image_height": 32, "image_width": 32},
    {"num_timesteps": 10, "image_height": 64, "image_width": 64},
    {"num_timesteps": 20, "image_height": 128, "image_width": 128},
])
def parametrized_episode(request):
    """Parametrized episode with different sizes."""
    return generate_synthetic_episode(**request.param, seed=42)


@pytest.fixture(params=[
    [Cameras.LEFT.value],
    [Cameras.LEFT.value, Cameras.RIGHT.value],
    [Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value],
])
def parametrized_cameras(request):
    """Parametrized camera configurations."""
    return request.param


# ==============================================================================
# Training Component Fixtures
# ==============================================================================

@pytest.fixture
def simple_training_config():
    """Create a simple training configuration for testing."""

    optimizer_config = AdamWConfig(
        lr=1e-4,
        weight_decay=1e-6,
    )

    return TrainingConfig(
        num_epochs=2,
        gradient_accumulate_every=1,
        optimizer=optimizer_config,
        clip_gradient_norm=False,
        lr_schedule=None,
        use_ema=False,  # Disabled for simpler tests
        ema_power=0.75,
    )


@pytest.fixture
def simple_observation_space():
    """Create a simple observation space for testing."""
    return ObservationSpace(
        camera_keys=[Cameras.LEFT.value],
        use_proprio_base_frame=True,
        use_proprio_camera_frame=False,
        use_language=False,
        use_gripper_state=False,
        gripper_type=GripperType.BINARY.value,
    )


@pytest.fixture
def simple_action_space():
    """Create a simple action space for testing."""
    return ActionSpace(
        has_position=True,
        position_dim=3,
        has_orientation=True,
        orientation_dim=4,
        orientation_repr=OrientationRepresentation.QUATERNION.value,
        has_gripper=True,
        gripper_type=GripperType.BINARY.value,
        predict_in_camera_frame=False,
        deltas_as_actions=False,
        denoise_actions=False,
        task_has_phases=False,
    )


@pytest.fixture
def synthetic_training_batch(batch_size, device):
    """Create a synthetic training batch matching dummy encoder expectations."""
    obs_horizon = 2
    pred_horizon = 4

    batch = {
        OBSERVATION_KEY: {
            "rgb": torch.randn(batch_size, obs_horizon, 3, 64, 64, device=device),
            "proprio": torch.randn(batch_size, obs_horizon, 7, device=device),
        },
        ACTION_KEY: {
            POSITION_ACTION_KEY: torch.randn(batch_size, pred_horizon, 3, device=device),
            ORIENTATION_ACTION_KEY: torch.randn(batch_size, pred_horizon, 4, device=device),
            GRIPPER_ACTION_KEY: torch.randint(0, 2, (batch_size, pred_horizon, 1), device=device, dtype=torch.float32),
        },
    }

    return batch


@pytest.fixture
def simple_policy(simple_observation_space, simple_action_space, device):
    """Create a real minimal policy for testing (not MagicMock)."""
    feature_dim = 256
    prediction_horizon = 4

    rgb_encoder = DummyRGBEncoder()
    proprio_encoder = DummyProprioEncoder(input_dim=7, output_dim=128)

    encoders = torch.nn.ModuleDict({
        "rgb": rgb_encoder,
        "proprio": proprio_encoder,
    })

    encoder_outputs = {
        "rgb": rgb_encoder.get_output_specification(),
        "proprio": proprio_encoder.get_output_specification(),
    }

    fusion = DummyFusion(output_dim=feature_dim)
    fusion_stages = torch.nn.ModuleList([fusion])

    feature_keys_to_dims = {
        "rgb_features": (256, 7, 7),
        "proprio_features": 128,
        "fused": feature_dim,
    }

    encoding_pipeline = EncodingPipeline.__new__(EncodingPipeline)
    torch.nn.Module.__init__(encoding_pipeline)
    encoding_pipeline.encoders = encoders
    encoding_pipeline.conditional_encoders = torch.nn.ModuleDict()
    encoding_pipeline.fusion_stages = fusion_stages
    encoding_pipeline.encoders_to_outputs = encoder_outputs
    encoding_pipeline._feature_keys_to_dims = feature_keys_to_dims
    encoding_pipeline._consumed_features = {"rgb_features", "proprio_features"}  # Fusion consumes these

    def _flatten_observation_dict(self, observation):
        return observation
    encoding_pipeline._flatten_observation_dict = _flatten_observation_dict.__get__(encoding_pipeline, EncodingPipeline)

    algorithm = BehavioralCloning()
    decoder = MockActionDecoder(
        observation_space=simple_observation_space,
        action_space=simple_action_space,
        feature_dim=feature_dim,
        prediction_horizon=prediction_horizon,
        device=device,
    )

    loss = ActionReconstructionLoss(
        action_keys=[POSITION_ACTION_KEY, ORIENTATION_ACTION_KEY, GRIPPER_ACTION_KEY],
        mse_weight=1.0,
        gripper_bce_weight=1.0,
        use_vae=False,
    )

    policy = Policy(
        encoding_pipeline=encoding_pipeline,
        algorithm=algorithm,
        decoder=decoder,
        observation_space=simple_observation_space,
        action_space=simple_action_space,
        prediction_horizon=prediction_horizon,
        loss=loss,
        device=device,
        validate_loss_keys=True,
    )

    policy.normalizer = DummyNormalizer()
    policy.to(device)
    return policy


@pytest.fixture
def mock_main_config(tmp_path, simple_observation_space, simple_action_space):
    """Create a mock MainConfig for Workspace testing."""

    experiment_config = ExperimentConfig(
        name="test_experiment",
        seed=42,
        checkpoint_folder=str(tmp_path),
        use_wandb=False,
        device="cpu",
        distributed=False,
        checkpoint_every=1,
        val_every=1,
    )

    dataloader_config = DataLoaderConfig(
        batch_size=2,
        num_workers=0,  # No multiprocessing in tests
        shuffle=True,
        image_height=64,
        image_width=64,
    )

    task_config = TaskSpaceConfig(
        observation_space=simple_observation_space,
        action_space=simple_action_space,
        observation_horizon=2,
        prediction_horizon=4,
        dataloader=dataloader_config,
    )

    optimizer_config = AdamWConfig(
        lr=1e-4,
    )

    training_config = TrainingConfig(
        num_epochs=1,  # Short for testing
        optimizer=optimizer_config,
        use_ema=False,
    )

    # Mock policy config (will be replaced with actual policy in tests)
    policy_config = PolicyConfig()

    inference_config = InferenceConfig()

    config = MainConfig(
        experiment=experiment_config,
        task=task_config,
        training=training_config,
        policy=policy_config,
        inference=inference_config,
    )

    return config


@pytest.fixture
def minimal_yaml_config_factory():
    """Factory for creating minimal OmegaConf DictConfig for Workspace original_yaml_config parameter."""
    def factory(**kwargs):
        defaults = {
            "experiment": {
                "name": "test_experiment",
                "seed": 42,
                "checkpoint_folder": "/tmp/test_checkpoints",
                "device": "cpu",
                "use_wandb": False,
            },
            "task": {"_target_": "versatil.data.task.TaskSpace",
                "action_space": {"_target_": "versatil.data.task.ActionSpace",
                    "has_position": True, "position_dim": 3, "has_gripper": True, "gripper_dim": 1},
                "observation_space": {"_target_": "versatil.data.task.ObservationSpace",
                    "camera_keys": ["left"], "use_proprio_base_frame": True},
                "observation_horizon": 2,
                "prediction_horizon": 4,
                "dataloader": {"batch_size": 2, "image_height": 4, "image_width": 4},
                "dataset_schema": {
                    "_target_": "versatil.data.schemas.custom.bowel_retraction.BowelRetractionSchema",
                    "dataset_folders": [],
                    "zarr_path": ""
                }
            },
            "training": {
                "num_epochs": 10,
                "optimizer": {
                    "target_class": "torch.optim.AdamW",
                    "lr": 1e-4,
                    "weight_decay": 1e-4,
                }
            },
            "policy": {
                "_target_": "versatil.models.policy.Policy",
                "observation_space": "${task.observation_space}",
                "action_space": "${task.action_space}",
                "prediction_horizon": "${task.prediction_horizon}",
                "device": "${experiment.device}",
                "validate_loss_keys": True,
                "encoding_pipeline": {
                    "_target_": "versatil.models.encoding.pipeline.EncodingPipeline",
                    "encoders": {
                        "left_rgb": {
                            "_target_": "versatil.models.encoding.encoders.rgb.cnn.CNNEncoder",
                            "input_keys": "${cameras: LEFT}",
                            "backbone": "${rgb_backbone: RESNET18}",
                            "pretrained": "false",
                            "frozen": "false",
                            "pooling_method": "none",
                            "use_group_norm": "true",
                            "image_height": "${task.dataloader.image_height}",
                            "image_width": "${task.dataloader.image_width}",
                            },
                    },
                    "fusion_stages": [
                        {"_target_": "versatil.models.encoding.fusion.spatial.SpatialFusion",
                        "input_features": ["left_rgb", ],
                                        "output_name": "visual_features",
                                        "hidden_dim": 512,
                         }
                    ],
                },
                "algorithm": {
                    "_target_": "versatil.models.decoding.algorithm.behavior_cloning.BehavioralCloning",
                },
                "decoder": {
                    "_target_": "versatil.models.decoding.decoders.factory.act.ACT",
                    "input_keys": ["visual_features"],
                    "embedding_dimension": 512,
                    "number_of_heads": 8,
                    "feedforward_dimension": 3200,
                    "number_of_encoder_layers": 4,
                    "number_of_decoder_layers": 7,
                    "observation_horizon": "${task.observation_horizon}",
                    "prediction_horizon": "${task.prediction_horizon}",
                    "action_space": "${task.action_space}",
                    "observation_space": "${task.observation_space}",
                    "device": "${experiment.device}",
                    "action_heads": {
                        "position_action": {
                            "_target_": "versatil.models.decoding.action_heads.ActionHead",
                            "input_dim": "${policy.decoder.embedding_dimension}",
                            "output_dim": "${task.action_space.position_dim}",
                            "blocks": None,
                        },
                        "gripper_action": {
                            "_target_": "versatil.models.decoding.action_heads.ActionHead",
                            "input_dim": "${policy.decoder.embedding_dimension}",
                            "output_dim": "${task.action_space.gripper_dim}",
                            "blocks": None,
                        },
                    },
                },
                "loss": {
                    "_target_": "versatil.metrics.ActionReconstructionLoss",
                    "action_keys": None,
                    "mse_weight": 1.0,
                    "gripper_bce_weight": 1.0,
                    "use_vae": False,
                },
            },
            "inference": {
                "temporal_agg": True,
                "update_rate_hz": 3.0,
            }
        }

        def deep_merge(base, override):
            """Recursively merge override into base."""
            for key, value in override.items():
                if key in base and isinstance(value, dict) and isinstance(base[key], dict):
                    # Recursively merge nested dicts
                    base[key] = deep_merge(base[key], value)
                else:
                    # Override or add new key
                    base[key] = value
            return base

        for key, value in kwargs.items():
            if key in defaults and isinstance(value, dict) and isinstance(defaults[key], dict):
                # Recursively merge nested dicts
                defaults[key] = deep_merge(defaults[key], value)
            else:
                defaults[key] = value

        return OmegaConf.create(defaults)
    return factory