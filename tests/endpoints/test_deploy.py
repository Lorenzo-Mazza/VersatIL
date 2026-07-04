"""Tests for versatil.endpoints.deploy module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from versatil.endpoints.deploy import main


def _deployment_config(
    device: str | None = "cpu",
    temporal_aggregation: bool = True,
    update_rate_hz: float | None = 10.0,
    request_timeout_seconds: float | None = 2.5,
) -> DictConfig:
    return OmegaConf.create(
        {
            "checkpoint_path": "/tmp/ckpt",
            "checkpoint_name": "best.ckpt",
            "device": device,
            "max_steps": 100,
            "compile_model": False,
            "client": {
                "model_server_address": "10.0.0.1",
                "model_server_port": 5556,
                "temporal_aggregation": temporal_aggregation,
                "action_execution_horizon": None,
                "update_rate_hz": update_rate_hz,
                "temporal_max_timesteps": 800,
                "timing_log": True,
                "request_timeout_seconds": request_timeout_seconds,
            },
        }
    )


@pytest.mark.unit
@patch("versatil.endpoints.deploy.InferenceClient")
@patch("versatil.endpoints.deploy.SocketActionTransport")
@patch("versatil.endpoints.deploy.SocketObservationTransport")
@patch("versatil.endpoints.deploy.load_policy")
def test_main_wires_client_from_config(
    mock_load_policy,
    mock_obs_transport_class,
    mock_action_transport_class,
    mock_client_class,
):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    main(_deployment_config())

    mock_load_policy.assert_called_once_with(
        checkpoint_path="/tmp/ckpt",
        device=torch.device("cpu"),
        checkpoint_name="best.ckpt",
        compile_model=False,
    )
    mock_obs_transport_class.assert_called_once_with(
        server_address="10.0.0.1",
        server_port=5556,
        request_timeout_seconds=2.5,
    )
    mock_action_transport_class.assert_called_once_with(
        server_address="10.0.0.1",
        server_port=5556,
        request_timeout_seconds=2.5,
    )
    mock_client_class.assert_called_once()
    call_kwargs = mock_client_class.call_args.kwargs
    assert call_kwargs["temporal_aggregation"] is True
    assert call_kwargs["timing_log"] is True
    assert call_kwargs["update_rate_hz"] == 10.0

    mock_client.run_episode.assert_called_once_with(max_steps=100)
    mock_client.shutdown.assert_called_once()


@pytest.mark.unit
@patch("versatil.endpoints.deploy.InferenceClient")
@patch("versatil.endpoints.deploy.SocketActionTransport")
@patch("versatil.endpoints.deploy.SocketObservationTransport")
@patch("versatil.endpoints.deploy.load_policy")
def test_main_defaults_to_cpu_without_cuda(
    mock_load_policy,
    mock_obs_transport_class,
    mock_action_transport_class,
    mock_client_class,
):
    mock_client_class.return_value = MagicMock()

    with patch("versatil.endpoints.deploy.torch.cuda.is_available", return_value=False):
        main(_deployment_config(device=None))

    device_used = mock_load_policy.call_args.kwargs["device"]
    assert device_used == torch.device("cpu")


@pytest.mark.unit
@patch("versatil.endpoints.deploy.InferenceClient")
@patch("versatil.endpoints.deploy.SocketActionTransport")
@patch("versatil.endpoints.deploy.SocketObservationTransport")
@patch("versatil.endpoints.deploy.load_policy")
def test_main_calls_shutdown_even_on_keyboard_interrupt(
    mock_load_policy,
    mock_obs_transport_class,
    mock_action_transport_class,
    mock_client_class,
):
    mock_client = MagicMock()
    mock_client.run_episode.side_effect = KeyboardInterrupt
    mock_client_class.return_value = mock_client

    main(_deployment_config())

    mock_client.shutdown.assert_called_once()
