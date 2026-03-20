"""Inference client connecting trained policies to environments via transports."""

import enum
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tso_robotics_sockets import (
    CompressionType,
    InferenceResponseKey,
    ServerStatus,
    TransportKey,
)
from versatil_constants.shared import ObsKey

from versatil.inference.action_postprocessor import ActionPostprocessor
from versatil.inference.observation_buffer import ObservationBuffer
from versatil.inference.observation_preprocessor import ObservationPreprocessor
from versatil.inference.policy_loader import PolicyLoader
from versatil.inference.protocol import ActionTransport, ObservationTransport
from versatil.inference.temporal_aggregation import TemporalAggregator


class EpisodeStatus(enum.StrEnum):
    """Status values controlling the episode loop."""

    CONTINUE = "continue"
    FINISHED = "finished"
    SKIP = "skip"


@dataclass
class EnvironmentState:
    """Per-environment observation buffer and temporal aggregator."""

    observation_buffer: ObservationBuffer
    temporal_aggregator: TemporalAggregator | None = None


class InferenceClient:
    """Connects a trained policy to a simulation or real-world robot environment via custom transport protocols."""

    def __init__(
        self,
        policy_loader: PolicyLoader,
        observation_transport: ObservationTransport,
        action_transport: ActionTransport,
        temporal_aggregation: bool = False,
        favor_more_recent: bool = True,
        exponential_decay: float = 0.01,
        compression_type: str = CompressionType.RAW.value,
        max_timesteps: int = 800,
        timing_log: bool = False,
        update_rate_hz: float | None = None,
    ):
        """Initialize the inference client.

        Args:
            policy_loader: Loaded policy providing inference and metadata.
            observation_transport: Transport for receiving observations.
            action_transport: Transport for sending actions.
            temporal_aggregation: Whether to use temporal aggregation.
            favor_more_recent: Weight newer predictions more heavily.
            exponential_decay: Decay factor for temporal aggregation.
            compression_type: Compression type for image data transfer.
            max_timesteps: Maximum episode length for temporal aggregation.
            timing_log: Whether to log per-step timing breakdown.
            update_rate_hz: Target inference frequency in Hz.
        """
        self.policy_loader = policy_loader
        self.observation_transport = observation_transport
        self.action_transport = action_transport
        self.temporal_aggregation = temporal_aggregation
        self.compression_type = compression_type
        self.timing_log = timing_log
        self.update_rate_hz = update_rate_hz
        self.timestep = 0
        observation_space = policy_loader.observation_space
        action_space = policy_loader.action_space
        self.camera_keys = list(observation_space.cameras.keys())
        self.proprioceptive_keys = list(
            observation_space.proprioceptive_observations.keys()
        )
        self.has_language = (
            ObsKey.LANGUAGE.value in observation_space.observations_metadata
        )
        self.all_observation_keys = list(observation_space.observations_metadata.keys())
        self.action_keys_to_dimensions = {
            key: metadata.prediction_dimension
            for key, metadata in action_space.actions_metadata.items()
            if metadata.requires_prediction_head
        }
        self.observation_preprocessor = ObservationPreprocessor(
            camera_keys=self.camera_keys,
            proprioceptive_keys=self.proprioceptive_keys,
            has_language=self.has_language,
            image_height=policy_loader.config.task.dataloader.image_height,
            image_width=policy_loader.config.task.dataloader.image_width,
            compression_type=compression_type,
            rotate_images=policy_loader.config.inference.rotate_images,
            depth_clamp_range=policy_loader.depth_clamp_range,
        )
        self.action_postprocessor = ActionPostprocessor(
            action_space=action_space,
            denoising_thresholds=policy_loader.denoising_thresholds,
        )
        self._temporal_config = {
            "favor_more_recent": favor_more_recent,
            "exponential_decay": exponential_decay,
            "max_timesteps": max_timesteps,
        }
        self.environment_states: dict[int, EnvironmentState] = {}

    def run_episode(self, max_steps: int) -> None:
        """Run a full inference episode.

        Args:
            max_steps: Maximum number of steps in the episode.
        """
        self.observation_transport.register(
            client_name=self.policy_loader.checkpoint_path
        )
        for _ in range(max_steps):
            status = self.step()
            if status == EpisodeStatus.FINISHED.value:
                break

    def step(self) -> str:
        """Execute a single step: receive, predict, send.

        Returns:
            Status string indicating episode state.
        """
        step_start = time.time() if self.timing_log else None

        response = self.observation_transport.receive(
            requested_keys=self.all_observation_keys,
            compression_type=self.compression_type,
        )
        status = self._check_status(response=response)
        if status != EpisodeStatus.CONTINUE.value:
            return status

        if self.timing_log:
            preprocessing_start = time.time()

        self._handle_reset_signal(response=response)
        per_environment_observations = self.observation_preprocessor.parse_response(
            response=response
        )
        self._update_environment_states(
            per_environment_observations=per_environment_observations
        )
        self._remove_inactive_environments(
            per_environment_observations=per_environment_observations
        )

        if self.timing_log:
            preprocessing_duration = time.time() - preprocessing_start
            inference_start = time.time()

        actions_by_environment = self._get_actions_for_ready_environments()

        if self.timing_log:
            inference_duration = time.time() - inference_start
            postprocessing_start = time.time()

        if actions_by_environment:
            action_metadata = self.action_postprocessor.build_action_metadata()
            self.action_transport.send(
                actions=actions_by_environment,
                action_metadata=action_metadata,
            )

        if self.timing_log:
            postprocessing_duration = time.time() - postprocessing_start
            total_duration = time.time() - step_start
            logging.info(
                "[TIMING] Step %d: preprocess=%.4fs inference=%.4fs "
                "postprocess=%.4fs total=%.4fs fps=%.1f",
                self.timestep,
                preprocessing_duration,
                inference_duration,
                postprocessing_duration,
                total_duration,
                1.0 / total_duration,
            )

        self.timestep += 1

        if self.update_rate_hz is not None:
            time.sleep(1.0 / self.update_rate_hz)

        return EpisodeStatus.CONTINUE.value

    def reset(self) -> None:
        """Reset all environment states for a new episode."""
        for state in self.environment_states.values():
            state.observation_buffer.reset()
            if state.temporal_aggregator is not None:
                state.temporal_aggregator.reset()
        self.environment_states.clear()

    @staticmethod
    def _check_status(response: dict) -> str:
        """Check server response status and return episode status.

        Args:
            response: Server response dictionary.

        Returns:
            Episode status string.

        Raises:
            RuntimeError: If server reports an error.
        """
        status = response.get(TransportKey.STATUS.value)
        if status == ServerStatus.FINISHED.value:
            return EpisodeStatus.FINISHED.value
        if status == ServerStatus.ERROR.value:
            raise RuntimeError(
                f"Server error: {response.get(TransportKey.ERROR_MSG.value)}"
            )
        if status in (
            ServerStatus.PROCESSING.value,
            ServerStatus.CREATING_ENV.value,
        ):
            return EpisodeStatus.SKIP.value
        return EpisodeStatus.CONTINUE.value

    def _handle_reset_signal(self, response: dict) -> None:
        """Reset environment states that the server signals to reset.

        Args:
            response: Server response potentially containing reset indices.
        """
        reset_indices = response.get(
            InferenceResponseKey.RESET_ENVIRONMENT_INDICES.value, []
        )
        for environment_index in reset_indices:
            environment_index = int(environment_index)
            if environment_index in self.environment_states:
                state = self.environment_states[environment_index]
                state.observation_buffer.reset()
                if state.temporal_aggregator is not None:
                    state.temporal_aggregator.reset()

    def _update_environment_states(
        self,
        per_environment_observations: dict[int, dict[str, np.ndarray | str]],
    ) -> None:
        """Add parsed observations to per-environment buffers.

        Args:
            per_environment_observations: Dict mapping environment index
                to observation dict.
        """
        for environment_index, observations in per_environment_observations.items():
            if environment_index not in self.environment_states:
                self.environment_states[environment_index] = (
                    self._create_environment_state()
                )
            self.environment_states[environment_index].observation_buffer.add(
                observations=observations
            )

    def _remove_inactive_environments(
        self,
        per_environment_observations: dict[int, dict[str, np.ndarray | str]],
    ) -> None:
        """Remove environments no longer present in server responses.

        Args:
            per_environment_observations: Currently active environment observations.
        """
        active_indices = set(per_environment_observations.keys())
        inactive_indices = [
            index for index in self.environment_states if index not in active_indices
        ]
        for index in inactive_indices:
            del self.environment_states[index]

    def _create_environment_state(self) -> EnvironmentState:
        """Create a new environment state with buffer and optional aggregator."""
        buffer_keys = self.camera_keys + self.proprioceptive_keys
        if self.has_language:
            buffer_keys = buffer_keys + [ObsKey.LANGUAGE.value]

        observation_buffer = ObservationBuffer(
            buffer_size=self.policy_loader.observation_horizon,
            required_keys=buffer_keys,
        )
        temporal_aggregator = None
        if self.temporal_aggregation:
            temporal_aggregator = TemporalAggregator(
                device=self.policy_loader.device,
                action_keys_to_dimensions=self.action_keys_to_dimensions,
                prediction_horizon=self.policy_loader.prediction_horizon,
                exponential_decay=self._temporal_config["exponential_decay"],
                favor_more_recent=self._temporal_config["favor_more_recent"],
                max_timesteps=self._temporal_config["max_timesteps"],
            )
        return EnvironmentState(
            observation_buffer=observation_buffer,
            temporal_aggregator=temporal_aggregator,
        )

    def _get_actions_for_ready_environments(
        self,
    ) -> dict[int, dict[str, list[float]]]:
        """Run inference for environments with full observation buffers.

        Returns:
            Dict mapping environment index to structured action dict.
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

            camera_tensors = (
                self.observation_preprocessor.transform_camera_observations(
                    recent_observations=recent
                )
            )
            for camera_key in self.camera_keys:
                camera_batches[camera_key].append(camera_tensors[camera_key])

            for key in self.proprioceptive_keys:
                proprioceptive_tensor = torch.tensor(
                    np.array(recent[key]), dtype=torch.float32
                )
                proprioceptive_batches[key].append(proprioceptive_tensor)

            if self.has_language:
                language_batch.append(recent[ObsKey.LANGUAGE.value])

        if not ready_indices:
            return {}

        observation_dict: dict[str, Any] = {}
        for camera_key in self.camera_keys:
            observation_dict[camera_key] = torch.stack(camera_batches[camera_key])
        for key in self.proprioceptive_keys:
            observation_dict[key] = torch.stack(proprioceptive_batches[key])
        if self.has_language:
            observation_dict[ObsKey.LANGUAGE.value] = language_batch

        action_dict = self.policy_loader.run_inference(obs_dict=observation_dict)

        return self._distribute_actions(
            action_dict=action_dict, ready_indices=ready_indices
        )

    def _distribute_actions(
        self,
        action_dict: dict[str, torch.Tensor],
        ready_indices: list[int],
    ) -> dict[int, dict[str, list[float]]]:
        """Split batched inference results per environment.

        Args:
            action_dict: Batched action predictions from the policy.
            ready_indices: Environment indices that were included in the batch.

        Returns:
            Dict mapping environment index to structured action dict.
        """
        actions_by_environment: dict[int, dict[str, list[float]]] = {}
        for batch_index, environment_index in enumerate(ready_indices):
            environment_predictions = {
                key: action_dict[key][batch_index]
                for key in self.action_keys_to_dimensions
            }
            state = self.environment_states[environment_index]

            if self.temporal_aggregation and state.temporal_aggregator is not None:
                averaged = state.temporal_aggregator.store_and_average(
                    current_predictions=environment_predictions
                )
                actions_by_environment[environment_index] = (
                    self.action_postprocessor.format_action(action_dict=averaged)
                )
            else:
                single_step = {
                    key: tensor[0] for key, tensor in environment_predictions.items()
                }
                actions_by_environment[environment_index] = (
                    self.action_postprocessor.format_action(action_dict=single_step)
                )

        return actions_by_environment

    def shutdown(self) -> None:
        """Close transport connections."""
        self.observation_transport.close()
        if hasattr(self.action_transport, "close"):
            self.action_transport.close()
