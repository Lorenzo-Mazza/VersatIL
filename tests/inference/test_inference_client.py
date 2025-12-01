"""Tests for inference client with real config and synthetic data."""
import pytest
import torch
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
from imitation_learning_toolkit.sockets.model_client import Action

from refactoring.data.constants import (
    Cameras,
    POSITION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    GripperType,
)
from refactoring.configs.experiment import ExperimentConfig
from refactoring.configs.data.task import TaskSpaceConfig
from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.configs.data.dataloader import DataLoaderConfig
from refactoring.configs.data.dataset.schema import DatasetSchemaConfig
from refactoring.configs.training import TrainingConfig, OptimizerConfig, AdamWConfig
from refactoring.models.encoding.encoders.rgb.cnn import CNNEncoder
from refactoring.models.encoding.encoders.constants import RGBBackboneType, PoolingMethod
from refactoring.models.encoding.pipeline import EncodingPipeline
from refactoring.models.decoding.algorithm.behavior_cloning import BehavioralCloning
from refactoring.models.decoding.decoders.base import DecoderInput
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.policy import Policy
from refactoring.training.lightning_policy import LightningPolicy
from refactoring.metrics.composite import ActionReconstructionLoss
from refactoring.inference.client import InferenceClient
from tests.conftest import DummyNormalizer
from refactoring.models.decoding.decoders.base import ActionDecoder

import numpy as np


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints."""
    checkpoint_path = tmp_path / "test_checkpoint"
    checkpoint_path.mkdir()
    return checkpoint_path


@pytest.fixture
def test_config():
    """Create a complete test configuration."""
    observation_space = ObservationSpace(
        camera_keys=[Cameras.LEFT.value],
        use_proprioceptive_data=True,
        use_proprio_base_frame=True,
        use_proprio_camera_frame=False,
        use_gripper_state=False,
        gripper_type=GripperType.BINARY.value,
    )

    action_space = ActionSpace(
        has_position=True,
        position_dim=3,
        has_orientation=False,
        orientation_dim=0,
        has_gripper=True,
        gripper_type=GripperType.BINARY.value,
        predict_in_camera_frame=False,
        deltas_as_actions=True,
        denoise_actions=False,
        task_has_phases=False,
    )

    dataloader_config = DataLoaderConfig(
        batch_size=2,
        num_workers=0,
        image_height=224,
        image_width=224,
    )

    dataset_schema_config = DatasetSchemaConfig(
        _target_="refactoring.data.schemas.base.DatasetSchema",
        dataset_folders=["dummy"],
        zarr_path="dummy.zarr",
    )

    task_config = TaskSpaceConfig(
        dataset_schema=dataset_schema_config,
        dataloader=dataloader_config,
        action_space=action_space,
        observation_space=observation_space,
        observation_horizon=1,
        prediction_horizon=10,
    )

    optimizer_config = AdamWConfig(
        lr=1e-4,
        weight_decay=1e-6,
    )

    training_config = TrainingConfig(
        num_epochs=1,
        gradient_accumulate_every=1,
        optimizer=optimizer_config,
    )

    experiment_config = ExperimentConfig(
        name="test_inference",
        seed=42,
        use_wandb=False,
        device="cpu",
    )

    config_dict = {
        "task": task_config,
        "training": training_config,
        "experiment": experiment_config,
        "policy": {},
    }

    config = OmegaConf.create(config_dict)

    return config


@pytest.fixture
def test_policy(test_config, device):
    """Create a minimal test policy with real components."""
    observation_space = OmegaConf.to_object(test_config.task.observation_space)
    action_space = OmegaConf.to_object(test_config.task.action_space)
    prediction_horizon = test_config.task.prediction_horizon

    rgb_encoder = CNNEncoder(
        input_keys=Cameras.LEFT.value,
        backbone=RGBBackboneType.RESNET18.value,
        pooling_method=PoolingMethod.SPATIAL_SOFTMAX.value,
        use_group_norm=True,
        pretrained=False,
        frozen=False,
        image_height=224,
        image_width=224,
    ).to(device)

    encoders = torch.nn.ModuleDict({"rgb": rgb_encoder})
    encoder_outputs = {"rgb": rgb_encoder.get_output_specification()}

    feature_keys_to_dims = {}
    for encoder_name, output_spec in encoder_outputs.items():
        for feature_name in output_spec.features:
            full_name = f"{encoder_name}_{feature_name}"
            feature_keys_to_dims[full_name] = output_spec.dimensions[feature_name]

    encoding_pipeline = EncodingPipeline.__new__(EncodingPipeline)
    torch.nn.Module.__init__(encoding_pipeline)
    encoding_pipeline.encoders = encoders
    encoding_pipeline.conditional_encoders = torch.nn.ModuleDict()
    encoding_pipeline.fusion_stages = torch.nn.ModuleList([])
    encoding_pipeline.encoders_to_outputs = encoder_outputs
    encoding_pipeline._feature_keys_to_dims = feature_keys_to_dims
    encoding_pipeline._consumed_features = set()

    def _flatten_observation_dict(self, observation):
        return observation

    encoding_pipeline._flatten_observation_dict = _flatten_observation_dict.__get__(
        encoding_pipeline, EncodingPipeline
    )

    position_head = ActionHead(
        input_dim=512,
        output_dim=action_space.position_dim,
        blocks=[],
    ).to(device)

    gripper_head = ActionHead(
        input_dim=512,
        output_dim=action_space.gripper_dim,
        blocks=[],
    ).to(device)

    action_heads = {
        POSITION_ACTION_KEY: position_head,
        GRIPPER_ACTION_KEY: gripper_head,
    }


    class SimpleDecoder(ActionDecoder):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.fc = None

        def forward(self, features, actions=None):
            feat = features["rgb_image"]
            batch_size = feat.shape[0]
            feat_flat = feat.reshape(batch_size, -1)

            if self.fc is None:
                self.fc = torch.nn.Linear(feat_flat.shape[-1], 512).to(feat.device)

            hidden = self.fc(feat_flat)

            outputs = {}
            outputs[POSITION_ACTION_KEY] = self.action_heads[POSITION_ACTION_KEY](
                hidden
            ).unsqueeze(1).expand(-1, self.prediction_horizon, -1)
            outputs[GRIPPER_ACTION_KEY] = torch.sigmoid(
                self.action_heads[GRIPPER_ACTION_KEY](hidden)
            ).unsqueeze(1).expand(-1, self.prediction_horizon, -1)

            return outputs

    decoder_input = DecoderInput(
        keys=["rgb_image"],
        required_types=[],
        requires_actions=False,
    )

    decoder = SimpleDecoder(
        decoder_input=decoder_input,
        observation_space=observation_space,
        action_space=action_space,
        action_heads=action_heads,
        device=str(device),
        observation_horizon=1,
        prediction_horizon=prediction_horizon,
    ).to(device)

    algorithm = BehavioralCloning()

    loss_module = ActionReconstructionLoss(
        action_keys=[POSITION_ACTION_KEY, GRIPPER_ACTION_KEY],
        mse_weight=1.0,
        l1_weight=0.0,
        gripper_bce_weight=1.0,
        kl_weight=0.0,
        gripper_type=GripperType.BINARY.value,
        use_vae=False,
    )

    policy = Policy(
        encoding_pipeline=encoding_pipeline,
        algorithm=algorithm,
        decoder=decoder,
        observation_space=observation_space,
        action_space=action_space,
        prediction_horizon=prediction_horizon,
        loss=loss_module,
        device=str(device),
        validate_loss_keys=True,
    )

    normalizer = DummyNormalizer()
    policy.normalizer = normalizer

    return policy


@pytest.fixture
def saved_checkpoint(checkpoint_dir, test_config, test_policy, device):
    """Save a policy checkpoint and config to disk."""
    training_config = TrainingConfig(
        num_epochs=1,
        gradient_accumulate_every=1,
        optimizer=AdamWConfig(
            lr=1e-4,
        ),
    )

    lightning_policy = LightningPolicy(
        policy=test_policy,
        training_config=training_config,
    )

    checkpoint_file = checkpoint_dir / "latest.ckpt"

    checkpoint = {
        'state_dict': lightning_policy.state_dict(),
        'hyper_parameters': {
            'training_config': training_config,
        },
        'epoch': 0,
        'global_step': 0,
    }

    torch.save(checkpoint, checkpoint_file)

    config_file = checkpoint_dir / "config.yaml"
    OmegaConf.save(test_config, config_file)

    return checkpoint_dir


@pytest.fixture
def device():
    """Get available device for tests - always use CPU to avoid OOM."""
    return torch.device("cpu")


@pytest.mark.unit
class TestInferenceClientInitialization:
    """Test InferenceClient initialization and config loading."""

    @patch('refactoring.inference.client.AbstractModelClient.__init__')
    @patch('refactoring.inference.client.LightningPolicy.load_from_checkpoint')
    def test_client_loads_config_and_model(self, mock_load_checkpoint, mock_super_init, saved_checkpoint, test_policy, device):
        """Test that InferenceClient properly loads config and model from checkpoint."""
        mock_super_init.return_value = None

        mock_lightning_policy = MagicMock()
        mock_lightning_policy.policy = test_policy
        mock_lightning_policy.eval = MagicMock()
        mock_load_checkpoint.return_value = mock_lightning_policy

        client = InferenceClient(
            device=device,
            checkpoint_path=str(saved_checkpoint),
            temporal_agg=True,
            update_rate_hz=10.0,
        )

        assert client.config is not None
        assert client.config.task.dataloader.image_height == 224
        assert client.config.task.dataloader.image_width == 224
        assert client.config.task.observation_horizon == 1
        assert client.config.task.prediction_horizon == 10

        assert client.model is not None
        assert client.policy is not None
        assert client.image_height == 224
        assert client.image_width == 224
        assert client.prediction_horizon == 10

        mock_super_init.assert_called_once()
        mock_load_checkpoint.assert_called_once()

    @patch('refactoring.inference.client.AbstractModelClient.__init__')
    def test_client_raises_error_missing_config(self, mock_super_init, tmp_path, device):
        """Test that InferenceClient raises error when config.yaml is missing."""
        mock_super_init.return_value = None

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            InferenceClient(
                device=device,
                checkpoint_path=str(empty_dir),
                temporal_agg=True,
            )

    @patch('refactoring.inference.client.AbstractModelClient.__init__')
    def test_client_raises_error_missing_checkpoint(self, mock_super_init, tmp_path, device, test_config):
        """Test that InferenceClient raises error when checkpoint is missing."""
        mock_super_init.return_value = None

        incomplete_dir = tmp_path / "incomplete"
        incomplete_dir.mkdir()

        config_file = incomplete_dir / "config.yaml"
        OmegaConf.save(test_config, config_file)

        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            InferenceClient(
                device=device,
                checkpoint_path=str(incomplete_dir),
                temporal_agg=True,
            )


@pytest.mark.unit
class TestInferenceClientPrediction:
    """Test InferenceClient prediction functionality with synthetic data."""

    @patch('refactoring.inference.client.AbstractModelClient.__init__')
    @patch('refactoring.inference.client.LightningPolicy.load_from_checkpoint')
    def test_get_actions_from_model(self, mock_load_checkpoint, mock_super_init, saved_checkpoint, test_policy, device):
        """Test that client can generate actions from synthetic observations."""
        mock_super_init.return_value = None

        mock_lightning_policy = MagicMock()
        mock_lightning_policy.policy = test_policy
        mock_lightning_policy.eval = MagicMock()
        mock_load_checkpoint.return_value = mock_lightning_policy

        client = InferenceClient(
            device=device,
            checkpoint_path=str(saved_checkpoint),
            temporal_agg=True,
            update_rate_hz=10.0,
        )

        observation_horizon = client.policy.decoder.observation_horizon

        client.observation_buffer_size = observation_horizon
        client.left_image_buffer = []
        client.right_image_buffer = []
        client.robot_state_buffer = []
        client.obs_robot_frame = True
        client.obs_camera_frame = False
        client.request_depth = False
        client.predicts_delta = True
        client.predicts_in_camera_frame = False
        client.enable_logging = False

        for _ in range(observation_horizon):
            left_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            robot_state = np.random.randn(3).astype(np.float32)

            client.left_image_buffer.append(left_img)
            client.right_image_buffer.append(right_img)
            client.robot_state_buffer.append(robot_state)

        actions = client.get_actions_from_model()

        assert isinstance(actions, list)
        assert len(actions) > 0

        assert isinstance(actions[0], Action)
        assert actions[0].robot_action is not None
        assert len(actions[0].robot_action) == 4
        assert actions[0].gripper_action is not None

    @patch('refactoring.inference.client.AbstractModelClient.__init__')
    @patch('refactoring.inference.client.LightningPolicy.load_from_checkpoint')
    def test_multiple_prediction_cycles(self, mock_load_checkpoint, mock_super_init, saved_checkpoint, test_policy, device):
        """Test that client can perform multiple prediction cycles."""
        mock_super_init.return_value = None

        mock_lightning_policy = MagicMock()
        mock_lightning_policy.policy = test_policy
        mock_lightning_policy.eval = MagicMock()
        mock_load_checkpoint.return_value = mock_lightning_policy

        client = InferenceClient(
            device=device,
            checkpoint_path=str(saved_checkpoint),
            temporal_agg=True,
            update_rate_hz=10.0,
        )

        observation_horizon = client.policy.decoder.observation_horizon

        client.observation_buffer_size = observation_horizon
        client.left_image_buffer = []
        client.right_image_buffer = []
        client.robot_state_buffer = []
        client.obs_robot_frame = True
        client.obs_camera_frame = False
        client.request_depth = False
        client.predicts_delta = True
        client.predicts_in_camera_frame = False
        client.enable_logging = False

        num_cycles = 5
        all_actions = []

        for cycle in range(num_cycles):
            left_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            robot_state = np.random.randn(3).astype(np.float32)

            client.left_image_buffer.append(left_img)
            client.right_image_buffer.append(right_img)
            client.robot_state_buffer.append(robot_state)

            actions = client.get_actions_from_model()
            all_actions.append(actions)

        assert len(all_actions) == num_cycles

        for actions in all_actions:
            assert isinstance(actions, list)
            assert len(actions) > 0

    @patch('refactoring.inference.client.AbstractModelClient.__init__')
    @patch('refactoring.inference.client.LightningPolicy.load_from_checkpoint')
    def test_temporal_aggregation_disabled(self, mock_load_checkpoint, mock_super_init, saved_checkpoint, test_policy, device):
        """Test that client works with temporal aggregation disabled."""
        mock_super_init.return_value = None

        mock_lightning_policy = MagicMock()
        mock_lightning_policy.policy = test_policy
        mock_lightning_policy.eval = MagicMock()
        mock_load_checkpoint.return_value = mock_lightning_policy

        client = InferenceClient(
            device=device,
            checkpoint_path=str(saved_checkpoint),
            temporal_agg=False,
            update_rate_hz=10.0,
        )

        observation_horizon = client.policy.decoder.observation_horizon
        prediction_horizon = client.policy.prediction_horizon

        client.observation_buffer_size = observation_horizon
        client.left_image_buffer = []
        client.right_image_buffer = []
        client.robot_state_buffer = []
        client.obs_robot_frame = True
        client.obs_camera_frame = False
        client.request_depth = False
        client.predicts_delta = True
        client.predicts_in_camera_frame = False
        client.enable_logging = False

        for _ in range(observation_horizon):
            left_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            robot_state = np.random.randn(3).astype(np.float32)

            client.left_image_buffer.append(left_img)
            client.right_image_buffer.append(right_img)
            client.robot_state_buffer.append(robot_state)

        actions = client.get_actions_from_model()

        assert isinstance(actions, list)
        assert len(actions) == prediction_horizon
