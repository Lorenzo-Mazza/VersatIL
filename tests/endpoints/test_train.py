"""Tests for versatil.endpoints.train module."""

import os
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from versatil.endpoints.train import main


@pytest.mark.unit
@patch("versatil.endpoints.train.Workspace")
@patch("versatil.endpoints.train.validate_experiment")
@patch("versatil.endpoints.train.hydra.utils.instantiate")
def test_main_instantiates_validates_and_runs_workspace(
    mock_instantiate,
    mock_validate,
    mock_workspace_class,
):
    yaml_config = OmegaConf.create(
        {
            "policy": {"_target_": "fake"},
            "task": {},
            "training": {},
            "experiment": {"distributed": False},
        }
    )
    mock_config = MagicMock()
    mock_instantiate.return_value = mock_config
    mock_workspace = MagicMock()
    mock_workspace_class.return_value = mock_workspace

    main(yaml_config)

    mock_instantiate.assert_called_once_with(yaml_config)
    mock_validate.assert_called_once_with(mock_config)
    mock_workspace_class.assert_called_once_with(
        mock_config, original_yaml_config=yaml_config
    )
    mock_workspace.run.assert_called_once()


@pytest.mark.unit
@patch("versatil.endpoints.train.Workspace")
@patch("versatil.endpoints.train.validate_experiment")
@patch("versatil.endpoints.train.hydra.utils.instantiate")
def test_main_sets_distributed_when_world_size_in_env(
    mock_instantiate,
    mock_validate,
    mock_workspace_class,
):
    yaml_config = OmegaConf.create(
        {
            "policy": {"_target_": "fake"},
            "task": {},
            "training": {},
            "experiment": {"distributed": False},
        }
    )
    mock_instantiate.return_value = MagicMock()
    mock_workspace_class.return_value = MagicMock()

    with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
        main(yaml_config)

    assert yaml_config.experiment.distributed is True


@pytest.mark.unit
def test_main_raises_on_empty_config():
    with pytest.raises(ValueError, match="No configuration specified"):
        main(DictConfig({}))
