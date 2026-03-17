"""Tests for versatil.inference.socket_transport module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
from tso_robotics_sockets import InferenceRequestKey, ServerRoute
from tso_robotics_sockets.client import SocketClient

from versatil.inference.socket_transport import (
    SocketActionTransport,
    SocketObservationTransport,
)

SOCKET_CLIENT_PATH = "versatil.inference.socket_transport.SocketClient"


@pytest.fixture
def mock_socket_client() -> MagicMock:
    return MagicMock(spec=SocketClient)


@pytest.fixture
def observation_transport_factory(
    mock_socket_client: MagicMock,
) -> Callable[..., SocketObservationTransport]:
    def factory(
        server_address: str = "192.168.1.1",
        server_port: int = 6000,
    ) -> SocketObservationTransport:
        with patch(SOCKET_CLIENT_PATH) as patched_client_class:
            patched_client_class.return_value = mock_socket_client
            transport = SocketObservationTransport(
                server_address=server_address,
                server_port=server_port,
            )
        return transport

    return factory


@pytest.fixture
def action_transport_factory(
    mock_socket_client: MagicMock,
) -> Callable[..., SocketActionTransport]:
    def factory(
        server_address: str = "192.168.1.1",
        server_port: int = 6000,
    ) -> SocketActionTransport:
        with patch(SOCKET_CLIENT_PATH) as patched_client_class:
            patched_client_class.return_value = mock_socket_client
            transport = SocketActionTransport(
                server_address=server_address,
                server_port=server_port,
            )
        return transport

    return factory


@pytest.mark.unit
class TestSocketObservationTransportInitialization:
    def test_creates_socket_client_with_provided_address_and_port(
        self,
    ):
        with patch(SOCKET_CLIENT_PATH) as patched_client_class:
            SocketObservationTransport(
                server_address="10.0.0.5",
                server_port=7777,
            )
            patched_client_class.assert_called_once_with(
                server_address="10.0.0.5",
                server_port=7777,
            )

    def test_creates_socket_client_with_default_address_and_port(
        self,
    ):
        with patch(SOCKET_CLIENT_PATH) as patched_client_class:
            SocketObservationTransport()
            patched_client_class.assert_called_once_with(
                server_address="127.0.0.1",
                server_port=5555,
            )


@pytest.mark.unit
class TestSocketObservationTransportReceive:
    def test_sends_correct_route_name(
        self,
        observation_transport_factory,
        mock_socket_client,
    ):
        transport = observation_transport_factory()
        mock_socket_client.send_request.return_value = {}

        transport.receive(
            requested_keys=["left"],
            compression_type="jpeg",
        )

        call_kwargs = mock_socket_client.send_request.call_args
        assert call_kwargs.kwargs["route_name"] == ServerRoute.GET_OBSERVATION.value

    def test_sends_requested_keys(
        self,
        observation_transport_factory,
        mock_socket_client,
    ):
        transport = observation_transport_factory()
        mock_socket_client.send_request.return_value = {}
        requested_keys = ["left", "right", "proprio_robot_frame"]

        transport.receive(
            requested_keys=requested_keys,
            compression_type="jpeg",
        )

        call_kwargs = mock_socket_client.send_request.call_args
        sent_data = call_kwargs.kwargs["dict_data"]
        assert sent_data[InferenceRequestKey.REQUESTED_KEYS.value] == requested_keys

    def test_sends_compression_type(
        self,
        observation_transport_factory,
        mock_socket_client,
    ):
        transport = observation_transport_factory()
        mock_socket_client.send_request.return_value = {}

        transport.receive(
            requested_keys=["left"],
            compression_type="png",
        )

        call_kwargs = mock_socket_client.send_request.call_args
        sent_data = call_kwargs.kwargs["dict_data"]
        assert sent_data[InferenceRequestKey.COMPRESSION_TYPE.value] == "png"

    def test_returns_server_response(
        self,
        observation_transport_factory,
        mock_socket_client,
    ):
        transport = observation_transport_factory()
        expected_response = {"left": [1, 2, 3], "status": "ok"}
        mock_socket_client.send_request.return_value = expected_response

        result = transport.receive(
            requested_keys=["left"],
            compression_type="jpeg",
        )

        assert result == expected_response


@pytest.mark.unit
class TestSocketObservationTransportRegister:
    def test_sends_correct_route_name(
        self,
        observation_transport_factory,
        mock_socket_client,
    ):
        transport = observation_transport_factory()
        mock_socket_client.send_request.return_value = {}

        transport.register(client_name="test_policy")

        call_kwargs = mock_socket_client.send_request.call_args
        assert call_kwargs.kwargs["route_name"] == ServerRoute.REGISTER_CLIENT.value

    def test_sends_client_name(
        self,
        observation_transport_factory,
        mock_socket_client,
    ):
        transport = observation_transport_factory()
        mock_socket_client.send_request.return_value = {}

        transport.register(client_name="my_checkpoint_path")

        call_kwargs = mock_socket_client.send_request.call_args
        sent_data = call_kwargs.kwargs["dict_data"]
        assert sent_data[InferenceRequestKey.CLIENT_NAME.value] == "my_checkpoint_path"

    def test_returns_server_response(
        self,
        observation_transport_factory,
        mock_socket_client,
    ):
        transport = observation_transport_factory()
        expected_response = {"acknowledged": True}
        mock_socket_client.send_request.return_value = expected_response

        result = transport.register(client_name="test_policy")

        assert result == expected_response


@pytest.mark.unit
class TestSocketObservationTransportClose:
    def test_delegates_to_socket_close(
        self,
        observation_transport_factory,
        mock_socket_client,
    ):
        transport = observation_transport_factory()

        transport.close()

        mock_socket_client.close.assert_called_once()


@pytest.mark.unit
class TestSocketActionTransportInitialization:
    def test_creates_socket_client_with_provided_address_and_port(
        self,
    ):
        with patch(SOCKET_CLIENT_PATH) as patched_client_class:
            SocketActionTransport(
                server_address="10.0.0.5",
                server_port=7777,
            )
            patched_client_class.assert_called_once_with(
                server_address="10.0.0.5",
                server_port=7777,
            )

    def test_creates_socket_client_with_default_address_and_port(
        self,
    ):
        with patch(SOCKET_CLIENT_PATH) as patched_client_class:
            SocketActionTransport()
            patched_client_class.assert_called_once_with(
                server_address="127.0.0.1",
                server_port=5555,
            )


@pytest.mark.unit
class TestSocketActionTransportSend:
    def test_sends_correct_route_name(
        self,
        action_transport_factory,
        mock_socket_client,
    ):
        transport = action_transport_factory()
        mock_socket_client.send_request.return_value = {}

        transport.send(
            actions={0: [1.0, 2.0, 3.0]},
            action_metadata={"position": {"dimension": 3}},
        )

        call_kwargs = mock_socket_client.send_request.call_args
        assert call_kwargs.kwargs["route_name"] == ServerRoute.SEND_ACTION.value

    def test_sends_actions(
        self,
        action_transport_factory,
        mock_socket_client,
    ):
        transport = action_transport_factory()
        mock_socket_client.send_request.return_value = {}
        actions = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]}

        transport.send(
            actions=actions,
            action_metadata={"position": {"dimension": 3}},
        )

        call_kwargs = mock_socket_client.send_request.call_args
        sent_data = call_kwargs.kwargs["dict_data"]
        assert sent_data[InferenceRequestKey.ACTIONS.value] == actions

    def test_sends_action_metadata(
        self,
        action_transport_factory,
        mock_socket_client,
    ):
        transport = action_transport_factory()
        mock_socket_client.send_request.return_value = {}
        action_metadata = {
            "position": {"dimension": 3},
            "gripper": {"dimension": 1, "gripper_type": "binary"},
        }

        transport.send(
            actions={0: [1.0, 2.0, 3.0, 0.5]},
            action_metadata=action_metadata,
        )

        call_kwargs = mock_socket_client.send_request.call_args
        sent_data = call_kwargs.kwargs["dict_data"]
        assert sent_data[InferenceRequestKey.ACTION_METADATA.value] == action_metadata

    def test_returns_server_response(
        self,
        action_transport_factory,
        mock_socket_client,
    ):
        transport = action_transport_factory()
        expected_response = {"status": "ok"}
        mock_socket_client.send_request.return_value = expected_response

        result = transport.send(
            actions={0: [1.0, 2.0]},
            action_metadata={},
        )

        assert result == expected_response


@pytest.mark.unit
class TestSocketActionTransportClose:
    def test_delegates_to_socket_close(
        self,
        action_transport_factory,
        mock_socket_client,
    ):
        transport = action_transport_factory()

        transport.close()

        mock_socket_client.close.assert_called_once()
