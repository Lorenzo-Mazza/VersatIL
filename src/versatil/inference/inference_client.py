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
from versatil.inference.protocol import (
    ActionTransport,
    ObservationTransport,
    PolicyInference,
)
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
        policy_loader: PolicyInference,
        observation_transport: ObservationTransport,
        action_transport: ActionTransport,
        temporal_aggregation: bool = False,
        action_execution_horizon: int | None = None,
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
            observation_transport: Protocol for receiving observations from the environment.
            action_transport: Protocol for sending actions to the environment.
            temporal_aggregation: Whether to use temporal ensemble. When enabled,
                the policy is queried every step and the overlapping action predictions
                from consecutive chunks are averaged with exponential weighting.
                When disabled, the policy predicts a chunk of actions and all of them
                are sent to the environment before re-querying.
            action_execution_horizon: How many actions from each predicted chunk to execute
                when temporal_aggregation is False. Defaults to prediction_horizon.
            favor_more_recent: In temporal ensemble, weight predictions from later inference
                calls higher than earlier ones for the same timestep.
            exponential_decay: Exponential decay rate for the temporal ensemble weights.
                Higher values discount older predictions more aggressively.
            compression_type: Compression type for image data transfer.
            max_timesteps: Maximum episode length for temporal aggregation.
            timing_log: Whether to log per-step timing breakdown.
            update_rate_hz: Target inference frequency in Hz.
        """
        self.policy_loader = policy_loader
        self.observation_transport = observation_transport
        self.action_transport = action_transport
        self.temporal_aggregation = temporal_aggregation
        self.action_execution_horizon = (
            action_execution_horizon
            if action_execution_horizon is not None
            else policy_loader.prediction_horizon
        )
        if self.action_execution_horizon > policy_loader.prediction_horizon:
            raise ValueError(
                f"action_execution_horizon ({self.action_execution_horizon}) cannot exceed "
                f"prediction_horizon ({policy_loader.prediction_horizon})."
            )
        self.compression_type = compression_type
        self.timing_log = timing_log
        self.update_rate_hz = update_rate_hz
        self.timestep = 0
        observation_space = policy_loader.observation_space
        action_space = policy_loader.action_space
        self.camera_keys, self.state_keys, self.has_language = (
            self._bucket_observation_keys(observation_space=observation_space)
        )
        self.all_observation_keys = list(observation_space.observations_metadata.keys())
        self.action_keys_to_dimensions = {
            key: metadata.prediction_dimension
            for key, metadata in action_space.actions_metadata.items()
            if metadata.requires_prediction_head
        }
        self.observation_preprocessor = ObservationPreprocessor(
            camera_keys=self.camera_keys,
            state_keys=self.state_keys,
            has_language=self.has_language,
            camera_metadata=policy_loader.observation_space.cameras,
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

    @staticmethod
    def _bucket_observation_keys(
        observation_space: Any,
    ) -> tuple[list[str], list[str], bool]:
        """Bucket observation keys by metadata type.

        Returns:
            camera_keys: Keys whose metadata is `CameraMetadata`.
            state_keys: Keys whose metadata is a numerical `ObservationMetadata`
                (robot proprioception + any other non-image numerical state).
            has_language: Whether the language instruction key is present.
        """
        camera_keys = list(observation_space.cameras.keys())
        state_keys = list(observation_space.numerical_observations.keys())
        has_language = ObsKey.LANGUAGE.value in observation_space.observations_metadata
        covered = set(camera_keys) | set(state_keys)
        if has_language:
            covered.add(ObsKey.LANGUAGE.value)
        unsupported = set(observation_space.observations_metadata.keys()) - covered
        if unsupported:
            raise TypeError(
                f"Observations {sorted(unsupported)} have no inference dispatch; "
                "expected CameraMetadata, numerical ObservationMetadata, or the "
                f"language key '{ObsKey.LANGUAGE.value}'."
            )
        return camera_keys, state_keys, has_language

    def run_episode(self, max_steps: int) -> None:
        """Run a full inference episode.

        Args:
            max_steps: Maximum number of steps in the episode.
        """
        self.observation_transport.register(
            client_name=self.policy_loader.checkpoint_path
        )
        for _step_idx in range(max_steps):
            try:
                status = self.step()
            except Exception:
                logging.exception("Fatal error at step %d", _step_idx)
                raise
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
            # All environments have the same number of actions per chunk.
            # Send one step at a time — the server steps the environment on each send.
            num_steps = len(next(iter(actions_by_environment.values())))
            for step in range(num_steps):
                step_actions = {
                    env_idx: action_list[step]
                    for env_idx, action_list in actions_by_environment.items()
                }
                self.action_transport.send(
                    actions=step_actions,
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
        buffer_keys = self.camera_keys + self.state_keys
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
        state_batches: dict[str, list[torch.Tensor]] = {
            key: [] for key in self.state_keys
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

            for key in self.state_keys:
                state_tensor = torch.tensor(np.array(recent[key]), dtype=torch.float32)
                state_batches[key].append(state_tensor)

            if self.has_language:
                language_batch.append(recent[ObsKey.LANGUAGE.value])

        if not ready_indices:
            return {}

        observation_dict: dict[str, Any] = {}
        for camera_key in self.camera_keys:
            observation_dict[camera_key] = torch.stack(camera_batches[camera_key])
        for key in self.state_keys:
            observation_dict[key] = torch.stack(state_batches[key])
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
    ) -> dict[int, list[dict[str, list[float]]]]:
        """Split batched inference results into per-environment action sequences.

        With temporal ensemble: returns a single averaged action per environment.
        Without temporal ensemble: returns action_execution_horizon formatted
        actions per environment from the predicted chunk.

        Args:
            action_dict: Batched policy output. Each value has shape
                (batch_size, prediction_horizon, action_dim).
            ready_indices: Environment indices included in the inference batch.

        Returns:
            Dict mapping environment index to a list of formatted action dicts.
            Length is 1 for temporal ensemble, action_execution_horizon otherwise.
        """
        actions_by_environment: dict[int, list[dict[str, list[float]]]] = {}
        for batch_index, environment_index in enumerate(ready_indices):
            environment_predictions = {
                key: action_dict[key][batch_index]  # (prediction_horizon, action_dim)
                for key in self.action_keys_to_dimensions
            }
            state = self.environment_states[environment_index]

            if self.temporal_aggregation and state.temporal_aggregator is not None:
                averaged = state.temporal_aggregator.store_and_average(
                    current_predictions=environment_predictions
                )
                actions_by_environment[environment_index] = [
                    self.action_postprocessor.format_action(action_dict=averaged)
                ]
            else:
                chunk = []
                for step in range(self.action_execution_horizon):
                    single_step = {
                        key: tensor[step]  # (action_dim,)
                        for key, tensor in environment_predictions.items()
                    }
                    chunk.append(
                        self.action_postprocessor.format_action(action_dict=single_step)
                    )
                actions_by_environment[environment_index] = chunk

        return actions_by_environment

    def shutdown(self) -> None:
        """Close transport connections."""
        try:
            self.observation_transport.close()
        except Exception:
            logging.warning("Error closing observation transport", exc_info=True)
        try:
            if hasattr(self.action_transport, "close"):
                self.action_transport.close()
        except Exception:
            logging.warning("Error closing action transport", exc_info=True)
