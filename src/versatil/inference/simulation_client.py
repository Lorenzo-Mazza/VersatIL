"""Simulation inference client using standardized server protocol."""

import enum
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from imitation_learning_toolkit.sockets.client import SocketClient
from imitation_learning_toolkit.sockets.compression import (
    CompressionType,
    decompress_array,
)

from versatil.data.constants import GripperType, ObsKey
from versatil.data.metadata import (
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
)
from versatil.inference.base_client import BaseInferenceClient
from versatil.inference.observation_buffer import ObservationBuffer
from versatil.inference.temporal_aggregation import TemporalAggregator
from versatil.training.constants import PrecisionType


class ServerRoute(str, enum.Enum):
    """Standardized routes for simulation servers."""

    GET_OBSERVATION = "get_observation"
    SEND_ACTION = "send_action"
    REGISTER_CLIENT = "register_client"


class ServerStatus(str, enum.Enum):
    """Standardized status values from simulation servers."""

    FINISHED = "FINISHED"
    ERROR = "ERROR"
    WAITING_ACTION = "WAITING_ACTION"
    PROCESSING = "PROCESSING"
    CREATING_ENV = "CREATING_ENV"


class ServerResponseKey(str, enum.Enum):
    """Standardized response keys from simulation servers."""

    STATUS = "status"
    ERROR_MSG = "error_msg"
    RESET_ENVIRONMENT_INDICES = "reset_environment_indices"
    TIMESTEP = "timestep"
    IMAGE_HEIGHT = "image_height"
    IMAGE_WIDTH = "image_width"


class ServerRequestKey(str, enum.Enum):
    """Standardized request keys for simulation servers."""

    REQUESTED_KEYS = "requested_keys"
    ACTIONS = "actions"
    CLIENT_NAME = "client_name"
    COMPRESSION_TYPE = "compression_type"


@dataclass
class EnvironmentState:
    """Per-environment observation buffer and temporal aggregator."""

    observation_buffer: ObservationBuffer
    temporal_aggregator: TemporalAggregator | None = None


class SimulationClient(BaseInferenceClient):
    """Inference client for simulation servers.

    Observation and action keys are derived from the policy metadata.
    Supports single-env and multi-env servers.
    """

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        checkpoint_name: str = "last.ckpt",
        server_address: str = "127.0.0.1",
        server_port: int = 5555,
        temporal_agg: bool = True,
        favor_more_recent: bool = True,
        exponential_decay: float = 0.01,
        precision: str = PrecisionType.BF16_MIXED.value,
        seed: int = 42,
        enable_logging: bool = False,
        compression_type: str = CompressionType.RAW.value,
    ):
        """Initialize simulation client.

        Args:
            device: Device to run inference on.
            checkpoint_path: Path to checkpoint directory.
            checkpoint_name: Name of the checkpoint file.
            server_address: Address of the simulation server.
            server_port: Port of the simulation server.
            temporal_agg: Whether to use temporal aggregation.
            favor_more_recent: Weight newer predictions more heavily.
            exponential_decay: Decay factor for temporal aggregation.
            precision: Precision type for model inference.
            seed: Random seed for reproducibility.
            enable_logging: Enable debug logging.
            compression_type: Compression type for image data.
        """
        super().__init__(
            device=device,
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            precision=precision,
            seed=seed,
        )
        self.socket = SocketClient(
            server_address=server_address, server_port=server_port
        )
        self.temporal_agg = temporal_agg
        self.favor_more_recent = favor_more_recent
        self.exponential_decay = exponential_decay
        self.enable_logging = enable_logging
        self.compression_type = compression_type
        self.camera_keys = list(self.observation_space.cameras.keys())
        self.proprioceptive_keys = list(
            self.observation_space.proprioceptive_observations.keys()
        )
        self.has_language = (
            ObsKey.LANGUAGE.value
            in self.observation_space.observations_metadata
        )
        self.all_observation_keys = list(
            self.observation_space.observations_metadata.keys()
        )
        self.action_keys_to_dimensions = {
            key: metadata.prediction_dimension
            for key, metadata in self.action_space.actions_metadata.items()
            if metadata.requires_prediction_head
        }
        self.gripper_action_keys = set(
            self.action_space.gripper_actions.keys()
        )
        self.rotate_images = self.config.inference.rotate_images
        self.image_height = self.config.task.dataloader.image_height
        self.image_width = self.config.task.dataloader.image_width
        self._setup_image_transform()
        self.environment_states: dict[int, EnvironmentState] = {}

    def _setup_image_transform(self) -> None:
        """Setup albumentations transform for all camera keys."""
        additional_targets = {}
        for camera_key in self.camera_keys[1:]:
            additional_targets[camera_key] = "image"
        self.image_transform = A.Compose(
            [
                A.Resize(
                    height=self.image_height, width=self.image_width
                ),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )

    def _create_environment_state(self) -> EnvironmentState:
        """Create a new environment state with buffer and aggregator."""
        buffer_keys = self.camera_keys + self.proprioceptive_keys
        if self.has_language:
            buffer_keys = buffer_keys + [ObsKey.LANGUAGE.value]

        observation_buffer = ObservationBuffer(
            buffer_size=self.observation_horizon,
            required_keys=buffer_keys,
        )
        temporal_aggregator = None
        if self.temporal_agg:
            temporal_aggregator = TemporalAggregator(
                device=self.device,
                action_keys_to_dimensions=self.action_keys_to_dimensions,
                prediction_horizon=self.prediction_horizon,
                exponential_decay=self.exponential_decay,
                favor_more_recent=self.favor_more_recent,
            )
        return EnvironmentState(
            observation_buffer=observation_buffer,
            temporal_aggregator=temporal_aggregator,
        )

    def update_loop(self) -> None:
        """Request observations, run inference, send actions in a loop."""
        self.socket.send_request(
            route_name=ServerRoute.REGISTER_CLIENT.value,
            dict_data={
                ServerRequestKey.CLIENT_NAME.value: str(
                    Path(self.checkpoint_path) / Path(self.checkpoint_name).stem
                ),
            },
        )
        while True:
            response = self._request_observation()
            status = response.get(ServerResponseKey.STATUS.value)
            if status == ServerStatus.FINISHED.value:
                break
            if status == ServerStatus.ERROR.value:
                raise RuntimeError(
                    f"Server error: "
                    f"{response.get(ServerResponseKey.ERROR_MSG.value)}"
                )
            if status in (
                ServerStatus.PROCESSING.value,
                ServerStatus.CREATING_ENV.value,
            ):
                continue
            self._handle_reset_signal(response)
            per_environment_observations = self._parse_observations(response)
            self._update_environment_states(per_environment_observations)
            self._remove_inactive_environments(per_environment_observations)
            actions_by_environment = (
                self._get_actions_for_ready_environments()
            )
            if actions_by_environment:
                self._send_actions(actions_by_environment)

    def _request_observation(self) -> dict:
        """Request observations from the simulation server."""
        return self.socket.send_request(
            route_name=ServerRoute.GET_OBSERVATION.value,
            dict_data={
                ServerRequestKey.REQUESTED_KEYS.value: (
                    self.all_observation_keys
                ),
                ServerRequestKey.COMPRESSION_TYPE.value: (
                    self.compression_type
                ),
            },
        )

    def _send_actions(
        self, actions_by_environment: dict[int, list[float]]
    ) -> None:
        """Send actions to the simulation server."""
        self.socket.send_request(
            route_name=ServerRoute.SEND_ACTION.value,
            dict_data={
                ServerRequestKey.ACTIONS.value: actions_by_environment,
            },
        )

    def _parse_observations(
        self, response: dict
    ) -> dict[int, dict[str, np.ndarray | str]]:
        """Parse server response into per-environment observation dicts.

        Note:
            Multi-env responses have dict-valued observation data keyed
            by environment index. Single-env has scalar data.
        """
        first_camera = self.camera_keys[0] if self.camera_keys else None
        is_multi_environment = first_camera is not None and isinstance(
            response.get(first_camera), dict
        )
        if is_multi_environment:
            return self._parse_multi_environment(response)
        return self._parse_single_environment(response)

    def _parse_single_environment(
        self, response: dict
    ) -> dict[int, dict[str, np.ndarray | str]]:
        """Parse single-environment response, wrapped as environment 0."""
        observations: dict[str, np.ndarray | str] = {}
        for camera_key in self.camera_keys:
            if camera_key in response:
                image = decompress_array(
                    response[camera_key], method=self.compression_type
                )
                if self.rotate_images:
                    image = np.ascontiguousarray(image[::-1, ::-1])
                observations[camera_key] = image

        for key in self.proprioceptive_keys:
            if key in response:
                observations[key] = np.array(
                    response[key], dtype=np.float32
                )
        if self.has_language and ObsKey.LANGUAGE.value in response:
            observations[ObsKey.LANGUAGE.value] = response[
                ObsKey.LANGUAGE.value
            ]
        return {0: observations}

    def _parse_multi_environment(
        self, response: dict
    ) -> dict[int, dict[str, np.ndarray | str]]:
        """Parse multi-environment response keyed by environment index."""
        first_camera = self.camera_keys[0]
        environment_indices = [
            int(key) for key in response[first_camera].keys()
        ]
        per_environment = {}
        for environment_index in environment_indices:
            index_string = str(environment_index)
            observations: dict[str, np.ndarray | str] = {}
            for camera_key in self.camera_keys:
                image = decompress_array(
                    response[camera_key][index_string],
                    method=self.compression_type,
                )
                if self.rotate_images:
                    image = np.ascontiguousarray(image[::-1, ::-1])
                observations[camera_key] = image
            for key in self.proprioceptive_keys:
                if key in response:
                    observations[key] = np.array(
                        response[key][index_string], dtype=np.float32
                    )
            if self.has_language and ObsKey.LANGUAGE.value in response:
                observations[ObsKey.LANGUAGE.value] = response[
                    ObsKey.LANGUAGE.value
                ][index_string]
            per_environment[environment_index] = observations

        return per_environment

    def _update_environment_states(
        self,
        per_environment_observations: dict[
            int, dict[str, np.ndarray | str]
        ],
    ) -> None:
        """Add parsed observations to per-environment buffers."""
        for environment_index, observations in (
            per_environment_observations.items()
        ):
            if environment_index not in self.environment_states:
                self.environment_states[environment_index] = (
                    self._create_environment_state()
                )
            self.environment_states[
                environment_index
            ].observation_buffer.add(observations=observations)

    def _handle_reset_signal(self, response: dict) -> None:
        """Reset environment states and buffers that the server signals to reset."""
        reset_indices = response.get(
            ServerResponseKey.RESET_ENVIRONMENT_INDICES.value, []
        )
        for environment_index in reset_indices:
            environment_index = int(environment_index)
            if environment_index in self.environment_states:
                self._reset_environment(environment_index)

    def _reset_environment(self, environment_index: int) -> None:
        """Reset observation buffer and temporal aggregator for one env."""
        state = self.environment_states[environment_index]
        state.observation_buffer.reset()
        if state.temporal_aggregator is not None:
            state.temporal_aggregator.reset()

    def _reset_all_environments(self) -> None:
        """Reset all environment states."""
        for environment_index in list(self.environment_states.keys()):
            self._reset_environment(environment_index)

    def _remove_inactive_environments(
        self,
        per_environment_observations: dict[
            int, dict[str, np.ndarray | str]
        ],
    ) -> None:
        """Remove environments no longer present in server responses."""
        active_indices = set(per_environment_observations.keys())
        inactive_indices = [
            index
            for index in self.environment_states
            if index not in active_indices
        ]
        for index in inactive_indices:
            del self.environment_states[index]

    def _get_actions_for_ready_environments(
        self,
    ) -> dict[int, list[float]]:
        """Run inference for environments with full observation buffers.

        Returns:
            Dict mapping environment index to flat action list.
        """
        ready_indices = []
        camera_batches: dict[str, list[torch.Tensor]] = {
            key: [] for key in self.camera_keys
        }
        proprioceptive_batches: dict[str, list[torch.Tensor]] = {
            key: [] for key in self.proprioceptive_keys
        }
        language_batch: list[list[str]] = []

        for environment_index, state in self.environment_states.items():
            if not state.observation_buffer.is_ready():
                continue
            ready_indices.append(environment_index)
            recent = state.observation_buffer.get_recent()

            camera_tensors = self._transform_camera_observations(
                recent_observations=recent
            )
            for camera_key in self.camera_keys:
                camera_batches[camera_key].append(
                    camera_tensors[camera_key]
                )

            for key in self.proprioceptive_keys:
                proprioceptive_tensor = torch.tensor(
                    np.array(recent[key]), dtype=torch.float32
                )
                proprioceptive_batches[key].append(proprioceptive_tensor)

            if self.has_language:
                language_batch.append(recent[ObsKey.LANGUAGE.value])

        if not ready_indices:
            return {}

        observation_dict = {}
        for camera_key in self.camera_keys:
            # (batch, observation_horizon, C, H, W)
            observation_dict[camera_key] = torch.stack(
                camera_batches[camera_key]
            )
        for key in self.proprioceptive_keys:
            # (batch, observation_horizon, dimension)
            observation_dict[key] = torch.stack(
                proprioceptive_batches[key]
            )
        if self.has_language:
            observation_dict[ObsKey.LANGUAGE.value] = language_batch

        action_dict = self._run_inference(obs_dict=observation_dict)

        return self._distribute_actions(
            action_dict=action_dict, ready_indices=ready_indices
        )

    def _distribute_actions(
        self,
        action_dict: dict[str, torch.Tensor],
        ready_indices: list[int],
    ) -> dict[int, list[float]]:
        """Split batched inference results per environment."""
        actions_by_environment: dict[int, list[float]] = {}
        for batch_index, environment_index in enumerate(ready_indices):
            environment_predictions = {
                key: action_dict[key][batch_index]
                for key in self.action_keys_to_dimensions
            }
            state = self.environment_states[environment_index]

            if self.temporal_agg and state.temporal_aggregator is not None:
                averaged = state.temporal_aggregator.store_and_average(
                    current_predictions=environment_predictions
                )
                actions_by_environment[environment_index] = (
                    self._format_action(action_dict=averaged)
                )
            else:
                single_step = {
                    key: tensor[0]
                    for key, tensor in environment_predictions.items()
                }
                actions_by_environment[environment_index] = (
                    self._format_action(action_dict=single_step)
                )

        return actions_by_environment

    def _transform_camera_observations(
        self, recent_observations: dict[str, list]
    ) -> dict[str, torch.Tensor]:
        """Transform a temporal sequence of camera images.

        Note:
            All cameras are transformed together per timestep so they
            receive the same spatial augmentation.

        Returns:
            Dict mapping camera key to tensor (observation_horizon, C, H, W).
        """
        camera_tensors: dict[str, list[torch.Tensor]] = {
            key: [] for key in self.camera_keys
        }
        first_camera_key = self.camera_keys[0]
        observation_count = len(recent_observations[first_camera_key])
        for timestep in range(observation_count):
            transform_kwargs = {
                "image": recent_observations[first_camera_key][timestep]
            }
            for other_key in self.camera_keys[1:]:
                transform_kwargs[other_key] = recent_observations[
                    other_key
                ][timestep]
            transformed = self.image_transform(**transform_kwargs)
            camera_tensors[first_camera_key].append(
                transformed["image"] / 255.0
            )
            for other_key in self.camera_keys[1:]:
                camera_tensors[other_key].append(
                    transformed[other_key] / 255.0
                )
        return {
            key: torch.stack(tensors)
            for key, tensors in camera_tensors.items()
        }


    def _format_action(
        self, action_dict: dict[str, torch.Tensor]
    ) -> list[float]:
        """Format action tensors into a flat list for the server.

        Iterates over action space metadata in order, applying
        gripper postprocessing where needed.
        """
        parts = []
        for key, metadata in self.action_space.actions_metadata.items():
            if not metadata.requires_prediction_head:
                continue
            value = (
                action_dict[key].cpu().detach().float().numpy().flatten()
            )
            if key in self.gripper_action_keys:
                value = self._postprocess_gripper_action(
                    raw_value=value, action_key=key
                )
            parts.append(value)
        return np.concatenate(parts).tolist()

    def _postprocess_gripper_action(
        self, raw_value: np.ndarray, action_key: str
    ) -> np.ndarray:
        """Apply metadata-driven gripper postprocessing.

        Note:
            Binary grippers: sigmoid maps logit to probability, then
            scales to [-1, 1] for the server.
        """
        gripper_type = self._get_gripper_type(action_key=action_key)
        if gripper_type == GripperType.BINARY.value:
            probability = 1.0 / (1.0 + np.exp(-raw_value[0]))
            return np.array([2.0 * probability - 1.0])
        return raw_value

    def _get_gripper_type(self, action_key: str) -> str | None:
        """Get gripper type from action metadata."""
        metadata = self.action_space.actions_metadata[action_key]
        if isinstance(metadata, GripperActionMetadata):
            return metadata.gripper_type
        if isinstance(metadata, OnTheFlyActionMetadata):
            if isinstance(
                metadata.source_metadata, GripperObservationMetadata
            ):
                return metadata.source_metadata.gripper_type
        return None

    def shutdown(self) -> None:
        """Close the ZMQ socket connection."""
        self.socket.close()