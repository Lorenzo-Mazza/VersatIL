"""Tests for versatil.endpoints.test module."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.endpoints.test import main


@pytest.mark.unit
@patch("versatil.endpoints.test.InferenceClient")
@patch("versatil.endpoints.test.SocketActionTransport")
@patch("versatil.endpoints.test.SocketObservationTransport")
@patch("versatil.endpoints.test.load_policy")
@patch("versatil.endpoints.test.parse_args")
def test_main_creates_policy_loader_with_parsed_args(
    mock_parse_args,
    mock_load_policy,
    mock_obs_transport_class,
    mock_action_transport_class,
    mock_client_class,
):
    mock_parse_args.return_value = MagicMock(
        checkpoint_path="/tmp/ckpt",
        checkpoint_name="best.ckpt",
        device="cpu",
        model_server_address="10.0.0.1",
        model_server_port=5556,
        temporal_aggregation=True,
        action_execution_horizon=None,
        timing_log=True,
        update_frequency=10.0,
        max_steps=100,
        request_timeout=2.5,
    )
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    main()

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
@patch("versatil.endpoints.test.InferenceClient")
@patch("versatil.endpoints.test.SocketActionTransport")
@patch("versatil.endpoints.test.SocketObservationTransport")
@patch("versatil.endpoints.test.load_policy")
@patch("versatil.endpoints.test.parse_args")
def test_main_defaults_to_cuda_when_available(
    mock_parse_args,
    mock_load_policy,
    mock_obs_transport_class,
    mock_action_transport_class,
    mock_client_class,
):
    mock_parse_args.return_value = MagicMock(
        checkpoint_path="/tmp/ckpt",
        checkpoint_name="last.ckpt",
        device=None,
        model_server_address="127.0.0.1",
        model_server_port=5555,
        temporal_aggregation=False,
        action_execution_horizon=None,
        timing_log=False,
        update_frequency=None,
        max_steps=1000,
    )
    mock_client_class.return_value = MagicMock()

    with patch("versatil.endpoints.test.torch.cuda.is_available", return_value=False):
        main()

    device_used = mock_load_policy.call_args.kwargs["device"]
    assert device_used == torch.device("cpu")


@pytest.mark.unit
@patch("versatil.endpoints.test.InferenceClient")
@patch("versatil.endpoints.test.SocketActionTransport")
@patch("versatil.endpoints.test.SocketObservationTransport")
@patch("versatil.endpoints.test.load_policy")
@patch("versatil.endpoints.test.parse_args")
def test_main_calls_shutdown_even_on_keyboard_interrupt(
    mock_parse_args,
    mock_load_policy,
    mock_obs_transport_class,
    mock_action_transport_class,
    mock_client_class,
):
    mock_parse_args.return_value = MagicMock(
        checkpoint_path="/tmp/ckpt",
        checkpoint_name="last.ckpt",
        device="cpu",
        model_server_address="127.0.0.1",
        model_server_port=5555,
        temporal_aggregation=False,
        action_execution_horizon=None,
        timing_log=False,
        update_frequency=None,
        max_steps=1000,
    )
    mock_client = MagicMock()
    mock_client.run_episode.side_effect = KeyboardInterrupt
    mock_client_class.return_value = mock_client

    main()

    mock_client.shutdown.assert_called_once()
