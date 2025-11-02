"""End-to-end tests for the explainer endpoint."""

import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf
from unittest.mock import Mock, patch

from refactoring.configs.experiment import ExperimentConfig
from refactoring.configs.main import MainConfig
from refactoring.configs.task.task import TaskConfig, ActionSpace, ObservationSpace
from refactoring.configs.task.dataloader import DataloaderConfig
from refactoring.configs.task.dataset.schema import DatasetSchemaConfig
from refactoring.configs.training import TrainingConfig
from refactoring.configs.policy import PolicyConfig
from refactoring.configs.inference import InferenceConfig
from refactoring.workspace import Workspace
from refactoring.endpoints.explain import ModelExplainer
from refactoring.data.constants import Cameras, OrientationRepresentation


@pytest.mark.unit
class TestExplainerConfigLoading:
    """Test that explainer endpoint loads config correctly."""

    def test_explainer_requires_config_yaml(self, tmp_path):
        """Test that ModelExplainer raises error if config.yaml is missing."""
        # Create checkpoint directory without config.yaml
        checkpoint_dir = tmp_path / "no_config"
        checkpoint_dir.mkdir()

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            ModelExplainer(
                device=torch.device("cpu"),
                checkpoint_path=str(checkpoint_dir),
            )

    def test_explainer_loads_config_from_checkpoint_dir(self, tmp_path, simple_policy):
        """Test that ModelExplainer successfully loads config.yaml."""
        # Create workspace to save config
        config = MainConfig(
            experiment=ExperimentConfig(
                name="explainer_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                ),
                observation_horizon=2,
                prediction_horizon=4,
                dataloader=DataloaderConfig(
                    batch_size=2,
                    image_height=270,
                    image_width=480,
                ),
                dataset_schema=DatasetSchemaConfig(
                    _target_="refactoring.data.schemas.bowel_retraction.BowelRetractionSchema",
                    dataset_folders=[],
                    zarr_path="",
                ),
            ),
            training=TrainingConfig(num_epochs=1),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)

        # Create a fake checkpoint file
        checkpoint_path = workspace.output_dir / "latest.ckpt"
        checkpoint_path.touch()

        # Mock LightningPolicy.load_from_checkpoint to avoid loading real model
        with patch("refactoring.endpoints.explain.LightningPolicy.load_from_checkpoint") as mock_load:
            # Setup mock
            mock_model = Mock()
            mock_model.policy = simple_policy
            mock_load.return_value = mock_model

            # Create explainer - should not raise
            explainer = ModelExplainer(
                device=torch.device("cpu"),
                checkpoint_path=str(workspace.output_dir),
            )

            # Verify config was loaded
            assert hasattr(explainer, "config")
            assert explainer.config is not None

    def test_explainer_uses_config_for_image_dimensions(self, tmp_path, simple_policy):
        """Test that ModelExplainer uses image dimensions from config."""
        # Create config with specific image dimensions
        image_h, image_w = 224, 384
        config = MainConfig(
            experiment=ExperimentConfig(
                name="img_dims_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskConfig(
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
                dataloader=DataloaderConfig(
                    batch_size=2,
                    image_height=image_h,
                    image_width=image_w,
                ),
                dataset_schema=DatasetSchemaConfig(
                    _target_="refactoring.data.schemas.bowel_retraction.BowelRetractionSchema",
                    dataset_folders=[],
                    zarr_path="",
                ),
            ),
            training=TrainingConfig(num_epochs=1),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)
        checkpoint_path = workspace.output_dir / "latest.ckpt"
        checkpoint_path.touch()

        with patch("refactoring.endpoints.explain.LightningPolicy.load_from_checkpoint") as mock_load:
            mock_model = Mock()
            mock_model.policy = simple_policy
            mock_load.return_value = mock_model

            explainer = ModelExplainer(
                device=torch.device("cpu"),
                checkpoint_path=str(workspace.output_dir),
            )

            # Check that transform uses config dimensions
            # The transform should have a Resize operation with these dimensions
            assert explainer.transform is not None
            resize_transform = explainer.transform.transforms[0]
            assert resize_transform.height == image_h
            assert resize_transform.width == image_w

    def test_explainer_uses_config_for_observation_horizon(self, tmp_path, simple_policy):
        """Test that ModelExplainer uses observation_horizon from config."""
        obs_horizon = 3
        config = MainConfig(
            experiment=ExperimentConfig(
                name="obs_horizon_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                ),
                observation_horizon=obs_horizon,
                prediction_horizon=4,
                dataloader=DataloaderConfig(
                    batch_size=2,
                    image_height=270,
                    image_width=480,
                ),
                dataset_schema=DatasetSchemaConfig(
                    _target_="refactoring.data.schemas.bowel_retraction.BowelRetractionSchema",
                    dataset_folders=[],
                    zarr_path="",
                ),
            ),
            training=TrainingConfig(num_epochs=1),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)
        checkpoint_path = workspace.output_dir / "latest.ckpt"
        checkpoint_path.touch()

        with patch("refactoring.endpoints.explain.LightningPolicy.load_from_checkpoint") as mock_load:
            mock_model = Mock()
            mock_model.policy = simple_policy
            mock_load.return_value = mock_model

            explainer = ModelExplainer(
                device=torch.device("cpu"),
                checkpoint_path=str(workspace.output_dir),
            )

            # Verify observation_horizon matches config
            assert explainer.observation_horizon == obs_horizon

    def test_explainer_instantiates_dataset_schema(self, tmp_path, simple_policy):
        """Test that ModelExplainer instantiates the dataset schema from config."""
        config = MainConfig(
            experiment=ExperimentConfig(
                name="schema_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                ),
                observation_horizon=1,
                prediction_horizon=4,
                dataloader=DataloaderConfig(
                    batch_size=2,
                    image_height=270,
                    image_width=480,
                ),
                dataset_schema=DatasetSchemaConfig(
                    _target_="refactoring.data.schemas.bowel_retraction.BowelRetractionSchema",
                    dataset_folders=["/fake/path"],
                    zarr_path="/fake/zarr.zarr",
                ),
            ),
            training=TrainingConfig(num_epochs=1),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)
        checkpoint_path = workspace.output_dir / "latest.ckpt"
        checkpoint_path.touch()

        with patch("refactoring.endpoints.explain.LightningPolicy.load_from_checkpoint") as mock_load:
            mock_model = Mock()
            mock_model.policy = simple_policy
            mock_load.return_value = mock_model

            explainer = ModelExplainer(
                device=torch.device("cpu"),
                checkpoint_path=str(workspace.output_dir),
            )

            # Verify dataset_schema was instantiated
            assert hasattr(explainer, "dataset_schema")
            assert explainer.dataset_schema is not None

            # Verify it has expected methods
            assert hasattr(explainer.dataset_schema, "get_image_path_column")
            assert hasattr(explainer.dataset_schema, "compute_depth_path")


@pytest.mark.unit
class TestExplainerIntegrationWithWorkspace:
    """Test that explainer works with workspace-saved config."""

    def test_end_to_end_config_flow(self, tmp_path, simple_policy):
        """Test complete flow: workspace saves config -> explainer loads it."""
        # Step 1: Create and save config via workspace
        config = MainConfig(
            experiment=ExperimentConfig(
                name="e2e_test",
                checkpoint_folder=str(tmp_path),
                device="cpu",
                seed=999,
                use_wandb=False,
            ),
            task=TaskConfig(
                observation_space=ObservationSpace(
                    camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
                    use_proprio_base_frame=True,
                ),
                action_space=ActionSpace(
                    has_position=True,
                    position_dim=3,
                    has_orientation=True,
                    orientation_dim=4,
                    orientation_repr=OrientationRepresentation.QUATERNION.value,
                ),
                observation_horizon=2,
                prediction_horizon=16,
                dataloader=DataloaderConfig(
                    batch_size=32,
                    image_height=480,
                    image_width=640,
                ),
                dataset_schema=DatasetSchemaConfig(
                    _target_="refactoring.data.schemas.bowel_retraction.BowelRetractionSchema",
                    dataset_folders=[],
                    zarr_path="",
                ),
            ),
            training=TrainingConfig(num_epochs=100),
            policy=PolicyConfig(),
            inference=InferenceConfig(),
        )

        workspace = Workspace(config)

        # Step 2: Verify config was saved
        config_path = workspace.output_dir / "config.yaml"
        assert config_path.exists()

        # Step 3: Create fake checkpoint
        checkpoint_path = workspace.output_dir / "latest.ckpt"
        checkpoint_path.touch()

        # Step 4: Load config via explainer
        with patch("refactoring.endpoints.explain.LightningPolicy.load_from_checkpoint") as mock_load:
            mock_model = Mock()
            mock_model.policy = simple_policy
            mock_load.return_value = mock_model

            explainer = ModelExplainer(
                device=torch.device("cpu"),
                checkpoint_path=str(workspace.output_dir),
            )

            # Step 5: Verify explainer loaded correct config values
            assert explainer.config.experiment.seed == 999
            assert explainer.config.task.observation_horizon == 2
            assert explainer.config.task.prediction_horizon == 16
            assert explainer.config.task.dataloader.image_height == 480
            assert explainer.config.task.dataloader.image_width == 640
            assert explainer.observation_horizon == 2

            # Verify transform dimensions
            resize_transform = explainer.transform.transforms[0]
            assert resize_transform.height == 480
            assert resize_transform.width == 640
