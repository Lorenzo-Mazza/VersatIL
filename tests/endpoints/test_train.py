"""Tests for versatil.endpoints.train module."""

import os
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, OmegaConf

from versatil.endpoints.train import _set_cuda_device_from_local_rank, main


@pytest.mark.unit
@pytest.mark.parametrize(
    ("environment", "expected_device_index"),
    [
        ({"LOCAL_RANK": "2"}, 2),
        ({"SLURM_LOCALID": "3"}, 3),
        ({"LOCAL_RANK": "1", "SLURM_LOCALID": "3"}, 1),
    ],
)
@patch("versatil.endpoints.train.torch.cuda.set_device")
@patch("versatil.endpoints.train.torch.cuda.is_available", return_value=True)
def test_set_cuda_device_from_local_rank_binds_rank_before_model_construction(
    mock_cuda_is_available,
    mock_set_device,
    environment,
    expected_device_index,
):
    with patch.dict(os.environ, environment, clear=True):
        _set_cuda_device_from_local_rank(device="cuda")

    mock_cuda_is_available.assert_called_once_with()
    mock_set_device.assert_called_once_with(expected_device_index)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("device", "environment", "cuda_available"),
    [
        ("cpu", {"LOCAL_RANK": "1"}, True),
        ("cuda", {}, True),
        ("cuda", {"LOCAL_RANK": "1"}, False),
    ],
)
@patch("versatil.endpoints.train.torch.cuda.set_device")
@patch("versatil.endpoints.train.torch.cuda.is_available")
def test_set_cuda_device_from_local_rank_skips_unavailable_bindings(
    mock_cuda_is_available,
    mock_set_device,
    device,
    environment,
    cuda_available,
):
    mock_cuda_is_available.return_value = cuda_available

    with patch.dict(os.environ, environment, clear=True):
        _set_cuda_device_from_local_rank(device=device)

    mock_set_device.assert_not_called()


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
            "experiment": {"device": "cuda", "distributed": False},
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
            "experiment": {"device": "cuda", "distributed": False},
        }
    )
    mock_instantiate.return_value = MagicMock()
    mock_workspace_class.return_value = MagicMock()

    with (
        patch.dict(os.environ, {"WORLD_SIZE": "4"}),
        patch(
            "versatil.endpoints.train._set_cuda_device_from_local_rank"
        ) as mock_set_cuda_device,
    ):
        main(yaml_config)

    assert yaml_config.experiment.distributed is True
    mock_set_cuda_device.assert_called_once_with(device="cuda")


@pytest.mark.unit
def test_main_raises_on_empty_config():
    with pytest.raises(ValueError, match="No configuration specified"):
        main(DictConfig({}))
