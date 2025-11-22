"""End-to-end tests for the inference client endpoint."""

import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf
from unittest.mock import Mock, patch, MagicMock

from refactoring.configs.experiment import ExperimentConfig
from refactoring.configs.main import MainConfig
from refactoring.configs.data.task import TaskSpaceConfig
from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.configs.data.dataloader import DataLoaderConfig
from refactoring.configs.training import TrainingConfig
from refactoring.configs.policy import PolicyConfig
from refactoring.configs.inference import InferenceConfig
from refactoring.workspace import Workspace
from refactoring.inference.client import InferenceClient
from refactoring.data.constants import Cameras, OrientationRepresentation, GripperType


@pytest.mark.unit
class TestInferenceClientConfigLoading:
    """Test that inference client loads config correctly."""

    def test_inference_client_requires_config_yaml(self, tmp_path):
        """Test that InferenceClient raises error if config.yaml is missing."""
        # Create checkpoint directory without config.yaml
        checkpoint_dir = tmp_path / "no_config"
        checkpoint_dir.mkdir()

        # Mock the model server to avoid network calls
        with patch("refactoring.inference.client.AbstractModelClient.__init__"):
            # Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError, match="Config file not found"):
                InferenceClient(
                    device=torch.device("cpu"),
                    checkpoint_path=str(checkpoint_dir),
                )

    def test_inference_client_loads_config_from_checkpoint_dir(self, tmp_path, simple_policy, minimal_yaml_config_factory):
        """Test that InferenceClient successfully loads config.yaml."""
        # Create workspace to save config
        config = MainConfig(
            experiment=ExperimentConfig(
                name="inference_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskSpaceConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                ),
                observation_horizon=1,
                prediction_horizon=4,
                dataloader=DataLoaderConfig(
                    batch_size=2,
                    image_height=270,
                    image_width=480,
                ),
            ),
            training=TrainingConfig(num_epochs=1),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        original_yaml = minimal_yaml_config_factory(
            task={
                "action_space": {"has_position": True},
                "observation_space": {"camera_keys": ["left"]},
                "observation_horizon": 1,
                "prediction_horizon": 4,
                "dataloader": {"batch_size": 2, "image_height": 270, "image_width": 480}
            }
        )
        workspace = Workspace(config, original_yaml_config=original_yaml)

        # Create a fake checkpoint file
        checkpoint_path = workspace.output_dir / "latest.ckpt"
        checkpoint_path.touch()

        # Mock AbstractModelClient and LightningPolicy
        with patch("refactoring.inference.client.AbstractModelClient.__init__") as mock_abstract_init:
            mock_abstract_init.return_value = None

            with patch("refactoring.inference.client.LightningPolicy.load_from_checkpoint") as mock_load:
                mock_model = Mock()
                mock_model.policy = simple_policy
                mock_model.eval = Mock()
                mock_load.return_value = mock_model

                # Create inference client - should not raise
                client = InferenceClient(
                    device=torch.device("cpu"),
                    checkpoint_path=str(workspace.output_dir),
                )

                # Verify config was loaded
                assert hasattr(client, "config")
                assert client.config is not None

    def test_inference_client_uses_config_for_parameters(self, tmp_path, minimal_yaml_config_factory):
        """Test that InferenceClient uses config parameters correctly."""
        # Create config with specific parameters
        obs_space = ObservationSpace(
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value],
            use_proprio_base_frame=True,
            use_proprio_camera_frame=True,
        )
        action_space = ActionSpace(
            has_position=True,
            position_dim=3,
            has_orientation=True,
            orientation_dim=4,
            orientation_repr=OrientationRepresentation.QUATERNION.value,
            has_gripper=True,
            gripper_type=GripperType.BINARY.value,
            deltas_as_actions=True,
            predict_in_camera_frame=True,
        )

        config = MainConfig(
            experiment=ExperimentConfig(
                name="params_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskSpaceConfig(
                observation_space=obs_space,
                action_space=action_space,
                observation_horizon=2,
                prediction_horizon=16,
                dataloader=DataLoaderConfig(
                    batch_size=32,
                    image_height=480,
                    image_width=640,
                ),
            ),
            training=TrainingConfig(num_epochs=100),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        original_yaml = minimal_yaml_config_factory(
            task={
                "action_space": {"has_position": True, "has_orientation": False, "has_gripper": True},
                "observation_space": {"camera_keys": ["left", "right", "depth"]},
                "observation_horizon": 2,
                "prediction_horizon": 16,
                "dataloader": {"batch_size": 32, "image_height": 480, "image_width": 640}
            },
            inference={"temperature": 1.0}
        )
        workspace = Workspace(config, original_yaml_config=original_yaml)
        checkpoint_path = workspace.output_dir / "latest.ckpt"
        checkpoint_path.touch()

        with patch("refactoring.inference.client.AbstractModelClient.__init__") as mock_abstract_init:
            mock_abstract_init.return_value = None

            with patch("refactoring.inference.client.LightningPolicy.load_from_checkpoint") as mock_load:
                # Create mock policy with correct observation/action spaces
                mock_policy = Mock()
                mock_policy.observation_space = obs_space
                mock_policy.action_space = action_space
                mock_policy.prediction_horizon = 16

                # Mock normalizer for depth statistics
                mock_depth_normalizer = Mock()
                mock_depth_normalizer.params_dict = {
                    'input_stats': {
                        'min': torch.tensor(0.0),
                        'max': torch.tensor(10.0)
                    }
                }
                mock_normalizer = Mock()
                mock_normalizer.__getitem__ = Mock(return_value=mock_depth_normalizer)
                mock_policy.normalizer = mock_normalizer
                mock_policy.set_tokenizer = Mock()

                mock_model = Mock()
                mock_model.policy = mock_policy
                mock_model.eval = Mock()
                mock_load.return_value = mock_model

                client = InferenceClient(
                    device=torch.device("cpu"),
                    checkpoint_path=str(workspace.output_dir),
                )

                # Verify AbstractModelClient was initialized with correct params
                assert mock_abstract_init.called
                call_kwargs = mock_abstract_init.call_args.kwargs

                # Check parameters derived from config
                assert call_kwargs["request_depth"] is True  # DEPTH in camera_keys
                assert call_kwargs["request_gripper_state"] is True  # has_gripper=True
                assert call_kwargs["predicts_in_camera_frame"] is True
                assert call_kwargs["predicts_delta"] is True
                assert call_kwargs["obs_robot_frame"] is True
                assert call_kwargs["obs_camera_frame"] is True

    def test_inference_client_image_dimensions_from_config(self, tmp_path, simple_policy, minimal_yaml_config_factory):
        """Test that InferenceClient reads image dimensions from config."""
        image_h, image_w = 360, 540
        config = MainConfig(
            experiment=ExperimentConfig(
                name="img_dims_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskSpaceConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                ),
                observation_horizon=1,
                prediction_horizon=4,
                dataloader=DataLoaderConfig(
                    batch_size=2,
                    image_height=image_h,
                    image_width=image_w,
                ),
            ),
            training=TrainingConfig(num_epochs=1),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        original_yaml = minimal_yaml_config_factory(
            task={
                "action_space": {"has_position": True},
                "observation_space": {"camera_keys": ["left"]},
                "dataloader": {"image_height": image_h, "image_width": image_w}
            }
        )
        workspace = Workspace(config, original_yaml_config=original_yaml)
        checkpoint_path = workspace.output_dir / "latest.ckpt"
        checkpoint_path.touch()

        with patch("refactoring.inference.client.AbstractModelClient.__init__") as mock_abstract_init:
            mock_abstract_init.return_value = None

            with patch("refactoring.inference.client.LightningPolicy.load_from_checkpoint") as mock_load:
                mock_model = Mock()
                mock_model.policy = simple_policy
                mock_model.eval = Mock()
                mock_load.return_value = mock_model

                client = InferenceClient(
                    device=torch.device("cpu"),
                    checkpoint_path=str(workspace.output_dir),
                )

                # Verify image dimensions match config
                assert client.image_height == image_h
                assert client.image_width == image_w


@pytest.mark.unit
class TestInferenceClientIntegrationWithWorkspace:
    """Test that inference client works with workspace-saved config."""

    def test_end_to_end_config_flow(self, tmp_path, simple_policy, minimal_yaml_config_factory):
        """Test complete flow: workspace saves config -> inference client loads it."""
        # Step 1: Create and save config via workspace
        config = MainConfig(
            experiment=ExperimentConfig(
                name="e2e_inference_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                seed=777,
                use_wandb=False,
            ),
            task=TaskSpaceConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
                    use_proprio_base_frame=True,
                    use_proprio_camera_frame=False,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                    has_gripper=True,
                    gripper_type=GripperType.BINARY.value,
                    deltas_as_actions=False,
                    predict_in_camera_frame=False,
                ),
                observation_horizon=1,
                prediction_horizon=8,
                dataloader=DataLoaderConfig(
                    batch_size=16,
                    image_height=224,
                    image_width=224,
                ),
            ),
            training=TrainingConfig(num_epochs=50),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        original_yaml = minimal_yaml_config_factory(
            experiment={"seed": 777},
            task={
                "action_space": {"has_position": True, "has_gripper": True},
                "observation_space": {"camera_keys": ["left", "right"]},
                "observation_horizon": 1,
                "prediction_horizon": 8,
                "dataloader": {"batch_size": 16, "image_height": 224, "image_width": 224}
            }
        )
        workspace = Workspace(config, original_yaml_config=original_yaml)

        # Step 2: Verify config was saved
        config_path = workspace.output_dir / "config.yaml"
        assert config_path.exists()

        # Step 3: Create fake checkpoint
        checkpoint_path = workspace.output_dir / "latest.ckpt"
        checkpoint_path.touch()

        # Step 4: Load config via inference client
        with patch("refactoring.inference.client.AbstractModelClient.__init__") as mock_abstract_init:
            mock_abstract_init.return_value = None

            with patch("refactoring.inference.client.LightningPolicy.load_from_checkpoint") as mock_load:
                mock_model = Mock()
                mock_model.policy = simple_policy
                mock_model.eval = Mock()
                mock_load.return_value = mock_model

                client = InferenceClient(
                    device=torch.device("cpu"),
                    checkpoint_path=str(workspace.output_dir),
                )

                # Step 5: Verify client loaded correct config values
                assert client.config.experiment.seed == 777
                assert client.config.task.observation_horizon == 1
                assert client.config.task.prediction_horizon == 8
                assert client.image_height == 224
                assert client.image_width == 224

                # Verify AbstractModelClient init params match config
                call_kwargs = mock_abstract_init.call_args.kwargs
                assert call_kwargs["request_depth"] is False  # No DEPTH in camera_keys
                assert call_kwargs["request_gripper_state"] is True
                assert call_kwargs["predicts_in_camera_frame"] is False
                assert call_kwargs["predicts_delta"] is False
                assert call_kwargs["obs_robot_frame"] is True
                assert call_kwargs["obs_camera_frame"] is False

    def test_checkpoint_not_found_error(self, tmp_path, minimal_yaml_config_factory):
        """Test that InferenceClient raises error if no checkpoint file exists."""
        # Create workspace to save config
        config = MainConfig(
            experiment=ExperimentConfig(
                name="no_ckpt_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskSpaceConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                ),
                observation_horizon=1,
                prediction_horizon=4,
                dataloader=DataLoaderConfig(batch_size=2),
            ),
            training=TrainingConfig(num_epochs=1),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        original_yaml = minimal_yaml_config_factory(
            task={
                "action_space": {"has_position": True},
                "observation_space": {"camera_keys": ["left"]},
                "observation_horizon": 1,
                "prediction_horizon": 4
            }
        )
        workspace = Workspace(config, original_yaml_config=original_yaml)

        # Don't create checkpoint file

        with patch("refactoring.inference.client.AbstractModelClient.__init__") as mock_abstract_init:
            mock_abstract_init.return_value = None

            # Should raise FileNotFoundError for missing checkpoint
            with pytest.raises(FileNotFoundError, match="No checkpoint found"):
                InferenceClient(
                    device=torch.device("cpu"),
                    checkpoint_path=str(workspace.output_dir),
                )
