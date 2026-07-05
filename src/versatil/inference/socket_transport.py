"""Concrete transport implementations for inference client communication."""

import zmq
from tso_robotics_sockets import (
    InferenceRequestKey,
    ServerRoute,
    SocketClient,
)


class SocketObservationTransport:
    """Observation transport using tso_robotics_sockets.SocketClient.

    Handles registration, observation requests, and image decompression
    over a ZMQ socket connection.
    """

    def __init__(
        self,
        server_address: str = "127.0.0.1",
        server_port: int = 5555,
        request_timeout_seconds: float | None = None,
    ):
        """Initialize socket observation transport.

        Args:
            server_address: Address of the environment server.
            server_port: Port of the environment server.
            request_timeout_seconds: Per-request timeout after which receive
                raises ``TimeoutError`` instead of blocking forever on a dead
                server. ``None`` keeps the fully blocking behavior.
        """
        self.socket = SocketClient(
            server_address=server_address,
            server_port=server_port,
            request_timeout_seconds=request_timeout_seconds,
        )

    def receive(self, requested_keys: list[str], compression_type: str) -> dict:
        """Request observations from the server.

        Args:
            requested_keys: Observation keys to request.
            compression_type: Compression format for image data.

        Returns:
            Server response dict with observation data and status.
        """
        return self.socket.send_request(
            route_name=ServerRoute.GET_OBSERVATION.value,
            dict_data={
                InferenceRequestKey.REQUESTED_KEYS.value: requested_keys,
                InferenceRequestKey.COMPRESSION_TYPE.value: compression_type,
            },
        )

    def register(self, client_name: str) -> dict:
        """Register the client with the server.

        Args:
            client_name: Identifier for this client.

        Returns:
            Server acknowledgement response.
        """
        return self.socket.send_request(
            route_name=ServerRoute.REGISTER_CLIENT.value,
            dict_data={
                InferenceRequestKey.CLIENT_NAME.value: client_name,
            },
        )

    def close(self) -> None:
        """Close the socket connection without blocking."""
        self.socket.request_socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()


class SocketActionTransport:
    """Action transport using tso_robotics_sockets.SocketClient.

    Sends raw action predictions plus metadata to the server.
    The server handles any coordinate conversion (e.g. delta computation).
    """

    def __init__(
        self,
        server_address: str = "127.0.0.1",
        server_port: int = 5555,
        request_timeout_seconds: float | None = None,
    ):
        """Initialize socket action transport.

        Args:
            server_address: Address of the environment server.
            server_port: Port of the environment server.
            request_timeout_seconds: Per-request timeout after which send
                raises ``TimeoutError`` instead of blocking forever on a dead
                server. ``None`` keeps the fully blocking behavior.
        """
        self.socket = SocketClient(
            server_address=server_address,
            server_port=server_port,
            request_timeout_seconds=request_timeout_seconds,
        )

    def send(self, actions: dict, action_metadata: dict) -> dict:
        """Send actions and metadata to the server.

        Args:
            actions: Dict mapping environment index to a structured action
                dict keyed by ``ActionComponent`` value, as produced by
                ``ActionPostprocessor.format_action``.
            action_metadata: Dict describing the action space, keyed by action
                component with ``ActionMetadataField`` entries.

        Returns:
            Server acknowledgement response.
        """
        return self.socket.send_request(
            route_name=ServerRoute.SEND_ACTION.value,
            dict_data={
                InferenceRequestKey.ACTIONS.value: actions,
                InferenceRequestKey.ACTION_METADATA.value: action_metadata,
            },
        )

    def close(self) -> None:
        """Close the socket connection without blocking."""
        self.socket.request_socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
