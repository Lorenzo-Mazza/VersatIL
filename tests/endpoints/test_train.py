import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from refactoring.endpoints import train


@pytest.fixture
def mock_hydra_config_factory(minimal_yaml_config_factory):
    """Factory for creating mock Hydra configs for testing train.py."""
    def factory(**overrides):
        return minimal_yaml_config_factory(**overrides)
    return factory


@pytest.fixture
def mock_workspace_factory():
    """Factory for creating mock Workspace instances."""
    def factory():
        workspace = MagicMock()
        workspace.output_dir = Path("/tmp/test_output")
        return workspace
    return factory


@pytest.fixture
def mock_instantiated_config_factory(tmp_path):
    """Factory for creating mock instantiated configs."""
    def factory(resume_from=None, distributed=False):
        config = MagicMock()
        config.experiment.resume_from = resume_from
        config.experiment.distributed = distributed
        config.experiment.checkpoint_folder = str(tmp_path)
        return config
    return factory


@pytest.fixture(autouse=True)
def mock_omega_conf_to_yaml():
    """Auto-mock OmegaConf.to_yaml to avoid DictConfig validation issues."""
    with patch("refactoring.endpoints.train.OmegaConf") as mock:
        mock.to_yaml.return_value = "mocked_yaml_output"
        yield mock


@pytest.mark.unit
class TestTrainEndpointConfigInstantiation:
    """Test that train.py correctly instantiates configs from YAML."""

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.logger")
    def test_empty_config_raises_error(
        self, mock_logger, mock_instantiate, mock_validate, mock_workspace_class
    ):
        with pytest.raises(ValueError, match="No configuration specified"):
            train.main.__wrapped__(None)

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.OmegaConf")
    def test_hydra_instantiate_is_called(
        self,
        mock_omega_conf,
        mock_instantiate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        instantiated = mock_instantiated_config_factory(resume_from=None)
        workspace = mock_workspace_factory()

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        mock_instantiate.assert_called_once_with(config)

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.logger")
    def test_validate_config_is_called_with_instantiated_config(
        self,
        mock_logger,
        mock_instantiate,
        mock_validate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        instantiated = mock_instantiated_config_factory(resume_from=None)
        workspace = mock_workspace_factory()

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        mock_validate.assert_called_once_with(instantiated)

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    def test_workspace_created_with_both_configs(
        self,
        mock_instantiate,
        mock_validate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        instantiated = mock_instantiated_config_factory(resume_from=None)
        workspace = mock_workspace_factory()

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        mock_workspace_class.assert_called_once_with(
            instantiated, original_yaml_config=config
        )

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    def test_workspace_run_is_invoked(
        self,
        mock_instantiate,
        mock_validate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        instantiated = mock_instantiated_config_factory(resume_from=None)
        workspace = mock_workspace_factory()

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        workspace.run.assert_called_once()


@pytest.mark.unit
class TestTrainEndpointResumeCheckpoint:
    """Test checkpoint resume functionality."""

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.Path")
    def test_load_checkpoint_called_when_resume_from_exists(
        self,
        mock_path_class,
        mock_instantiate,
        mock_validate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        checkpoint_path = "/path/to/checkpoint.pt"
        instantiated = mock_instantiated_config_factory(resume_from=checkpoint_path)
        workspace = mock_workspace_factory()

        mock_checkpoint_path = MagicMock()
        mock_checkpoint_path.exists.return_value = True
        mock_checkpoint_path.__str__.return_value = checkpoint_path
        mock_path_class.return_value = mock_checkpoint_path

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        workspace.load_checkpoint.assert_called_once_with(checkpoint_path)
        workspace.run.assert_called_once()

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.Path")
    @patch("refactoring.endpoints.train.logger")
    def test_warning_logged_when_checkpoint_missing(
        self,
        mock_logger,
        mock_path_class,
        mock_instantiate,
        mock_validate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        checkpoint_path = "/path/to/missing.pt"
        instantiated = mock_instantiated_config_factory(resume_from=checkpoint_path)
        workspace = mock_workspace_factory()

        mock_checkpoint_path = MagicMock()
        mock_checkpoint_path.exists.return_value = False
        mock_checkpoint_path.__str__.return_value = checkpoint_path
        mock_path_class.return_value = mock_checkpoint_path

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "Checkpoint not found" in warning_message

        workspace.load_checkpoint.assert_not_called()
        workspace.run.assert_called_once()

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    def test_no_load_checkpoint_when_resume_from_is_none(
        self,
        mock_instantiate,
        mock_validate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        instantiated = mock_instantiated_config_factory(resume_from=None)
        workspace = mock_workspace_factory()

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        workspace.load_checkpoint.assert_not_called()
        workspace.run.assert_called_once()


@pytest.mark.unit
class TestTrainEndpointDistributedDetection:
    """Test distributed training detection from environment variables."""

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.logger")
    @patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=False)
    def test_world_size_env_var_sets_distributed_true(
        self,
        mock_logger,
        mock_instantiate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        config.experiment.distributed = False

        instantiated = mock_instantiated_config_factory(distributed=False)
        workspace = mock_workspace_factory()

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        assert config.experiment.distributed is True

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.logger")
    @patch.dict(os.environ, {}, clear=True)
    def test_no_world_size_keeps_distributed_false(
        self,
        mock_logger,
        mock_instantiate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        assert "WORLD_SIZE" not in os.environ

        config = mock_hydra_config_factory()
        config.experiment.distributed = False

        instantiated = mock_instantiated_config_factory(distributed=False)
        workspace = mock_workspace_factory()

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        assert config.experiment.distributed is False

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.logger")
    @patch.dict(
        os.environ,
        {
            "WORLD_SIZE": "8",
            "SLURM_PROCID": "0",
            "SLURM_GPUS_ON_NODE": "2",
        },
        clear=False,
    )
    def test_distributed_detection_logs_world_size(
        self,
        mock_logger,
        mock_instantiate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        config.experiment.distributed = False

        instantiated = mock_instantiated_config_factory(distributed=False)
        workspace = mock_workspace_factory()

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        assert config.experiment.distributed is True

        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("WORLD_SIZE=8" in call for call in info_calls)


@pytest.mark.unit
class TestTrainEndpointExecutionFlow:
    """Integration tests verifying complete execution flow."""

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.logger")
    def test_full_flow_without_resume(
        self,
        mock_logger,
        mock_instantiate,
        mock_validate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        instantiated = mock_instantiated_config_factory(resume_from=None)
        workspace = mock_workspace_factory()

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        assert mock_instantiate.called
        assert mock_validate.called
        assert mock_workspace_class.called
        assert workspace.run.called

        workspace.load_checkpoint.assert_not_called()

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.Path")
    @patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=False)
    def test_full_flow_with_resume_and_distributed(
        self,
        mock_path_class,
        mock_instantiate,
        mock_validate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        config.experiment.distributed = False
        checkpoint_path = "/path/to/checkpoint.pt"

        instantiated = mock_instantiated_config_factory(resume_from=checkpoint_path)
        workspace = mock_workspace_factory()

        mock_checkpoint_path = MagicMock()
        mock_checkpoint_path.exists.return_value = True
        mock_checkpoint_path.__str__.return_value = checkpoint_path
        mock_path_class.return_value = mock_checkpoint_path

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        train.main.__wrapped__(config)

        assert config.experiment.distributed is True

        workspace.load_checkpoint.assert_called_once_with(checkpoint_path)
        workspace.run.assert_called_once()

    @patch("refactoring.endpoints.train.Workspace")
    @patch("refactoring.endpoints.train.validate_config")
    @patch("hydra.utils.instantiate")
    @patch("refactoring.endpoints.train.logger")
    def test_execution_order_is_correct(
        self,
        mock_logger,
        mock_instantiate,
        mock_validate,
        mock_workspace_class,
        mock_hydra_config_factory,
        mock_instantiated_config_factory,
        mock_workspace_factory,
    ):
        config = mock_hydra_config_factory()
        instantiated = mock_instantiated_config_factory(resume_from=None)
        workspace = mock_workspace_factory()

        mock_instantiate.return_value = instantiated
        mock_workspace_class.return_value = workspace

        call_order = []
        mock_instantiate.side_effect = lambda x: (call_order.append("instantiate"), instantiated)[1]
        mock_validate.side_effect = lambda x: call_order.append("validate")
        mock_workspace_class.side_effect = lambda *args, **kwargs: (call_order.append("workspace"), workspace)[1]
        workspace.run.side_effect = lambda: call_order.append("run")

        train.main.__wrapped__(config)

        assert call_order == ["instantiate", "validate", "workspace", "run"]