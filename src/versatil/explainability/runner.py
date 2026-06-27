"""Hydra-instantiated runner for policy explainability workflows."""

import logging
from datetime import datetime
from pathlib import Path

import torch
from tso_robotics_sockets import CompressionType

from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
from versatil.common.tensor_ops import to_device
from versatil.explainability.constants import (
    VALID_EXPLANATION_SOURCE_TYPES,
    VALID_EXPLANATION_TYPES,
    ExplanationSourceType,
)
from versatil.explainability.explanation_heatmaps import to_explanation_heatmaps
from versatil.explainability.online_runtime import ExplainabilityPolicyRuntime
from versatil.explainability.sources.dataset import DatasetExplanationSource
from versatil.explainability.sources.online import OnlineInferenceExplanationSource
from versatil.explainability.sources.typedefs import ExplanationBatch
from versatil.explainability.typedefs import ObservationBatch
from versatil.explainability.writer import ExplanationWriter
from versatil.inference.inference_client import InferenceClient
from versatil.inference.socket_transport import (
    SocketActionTransport,
    SocketObservationTransport,
)
from versatil.models.policy import Policy
from versatil.training.constants import CheckpointFilename

DEFAULT_ONLINE_MAX_STEPS = 1_000_000


class ExplainabilityRunner:
    """Generate xAI insights for the predictions of a trained policy."""

    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value,
        output_directory: str | None = None,
        device: str = "auto",
        source: str = ExplanationSourceType.DATASET.value,
        split: str = "all",
        sample_stride: int = 50,
        max_samples: int | None = None,
        data_path_override: str | list[str] | None = None,
        batch_size: int = 1,
        model_server_address: str = "127.0.0.1",
        model_server_port: int = 5555,
        temporal_aggregation: bool = False,
        action_execution_horizon: int | None = None,
        update_rate_hz: float | None = None,
        temporal_max_timesteps: int = 800,
        timing_log: bool = False,
        compression_type: str = CompressionType.RAW.value,
        channel_batch_size: int = 32,
        explanation_types: list[str] | None = None,
        target_camera_keys: list[str] | None = None,
        target_vision_module_names: list[str] | None = None,
        save_raw_heatmaps: bool = False,
        save_overlays: bool = True,
        image_weight: float = 0.5,
        overlay_image_format: str = "png",
    ) -> None:
        """Initialize the runner.

        Args:
            checkpoint_path: Directory containing the training ``config.yaml``
                and the model checkpoint.
            checkpoint_name: Checkpoint filename to restore trained policy.
            output_directory: Directory for written explanation files. ``None``
                writes under ``checkpoint_path/explainability/<timestamp>``.
            device: Device name, or ``auto`` to prefer CUDA when available.
            source: Explanation source type. Standalone ``run()`` supports
                ``dataset`` and ``online_inference``.
            split: Dataset split for offline explanations.
            sample_stride: Explanation interval. In offline dataset mode, keep
                every Nth episodic dataset sample. In online inference mode,
                explain every Nth inference timestep.
            max_samples: Optional cap on the number of observation windows to
                explain. Offline mode applies this after ``sample_stride``. Online
                mode applies it to ready inference windows and derives the
                inference-loop step budget from ``sample_stride``.
            data_path_override: Optional offline input location to explain
                instead of the data path stored in the checkpoint task config.
                ``None`` uses the checkpoint schema as-is. A single ``.zarr``
                path is sampled directly. A non-zarr path is raw data in the
                same dataset schema as the checkpoint. A list is only for raw
                schemas that already accept multiple inputs, such as CSV
                folders or HDF5 files; multiple zarr paths are rejected. Raw
                inputs are converted to ``offline_dataset.zarr`` beside the
                first override path unless ``zarr_cache_directory`` is set by
                the dataset source.
            batch_size: Number of samples per attribution call.
            model_server_address: Environment server address used by
                ``source=online_inference``.
            model_server_port: Environment server port used by
                ``source=online_inference``.
            temporal_aggregation: Whether online inference should average
                overlapping action predictions from consecutive policy calls.
            action_execution_horizon: Number of actions sent from each predicted
                chunk when temporal aggregation is disabled. ``None`` uses the
                checkpoint prediction horizon.
            update_rate_hz: Optional action-send rate limit for online
                inference. ``None`` sends actions as soon as they are available.
            temporal_max_timesteps: Maximum episode length tracked by temporal
                aggregation state.
            timing_log: Whether to log per-step timing breakdowns in online
                mode.
            compression_type: Image compression format requested from the online
                environment server.
            channel_batch_size: Number of channels per Ablation-CAM forward.
            explanation_types: Explanation methods to run. If None, it runs all
                supported methods.
            target_camera_keys: Optional camera-key allowlist for generated
                heatmaps.
            target_vision_module_names: Optional visual module allowlist. Names
                include encoding-pipeline entries and decoder-owned VLM vision
                tower paths.
            save_raw_heatmaps: Whether to save on disk raw heatmap tensors.
            save_overlays: Whether to save on disk heatmaps with image overlays.
            image_weight: Original-image blend weight for overlays.
            overlay_image_format: Image file format for saved overlays.

        Raises:
            ValueError: If explanation types or source data are invalid.
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name
        self.output_directory = self._resolve_output_directory(
            checkpoint_path=checkpoint_path,
            output_directory=output_directory,
        )
        self.device = self._resolve_device(device=device)
        self.source = source
        self.split = split
        self.sample_stride = sample_stride
        self.max_samples = max_samples
        self.data_path_override = data_path_override
        self.batch_size = batch_size
        self.model_server_address = model_server_address
        self.model_server_port = model_server_port
        self.temporal_aggregation = temporal_aggregation
        self.action_execution_horizon = action_execution_horizon
        self.update_rate_hz = update_rate_hz
        self.temporal_max_timesteps = temporal_max_timesteps
        self.timing_log = timing_log
        self.compression_type = compression_type
        self.explanation_types = self._resolve_explanation_types(
            explanation_types=explanation_types
        )
        self.target_camera_keys = target_camera_keys
        self.target_vision_module_names = target_vision_module_names
        self.save_raw_heatmaps = save_raw_heatmaps
        self.save_overlays = save_overlays
        self.channel_batch_size = channel_batch_size
        self.explanation_heatmaps = to_explanation_heatmaps(
            channel_batch_size=channel_batch_size
        )
        self.image_weight = image_weight
        self.overlay_image_format = overlay_image_format
        self._validate_source(source=source)
        self._validate_sampling_configuration(
            sample_stride=sample_stride,
            max_samples=max_samples,
        )
        if source == ExplanationSourceType.ONLINE_INFERENCE.value:
            self._validate_online_configuration(
                model_server_port=model_server_port,
                action_execution_horizon=action_execution_horizon,
                update_rate_hz=update_rate_hz,
                temporal_max_timesteps=temporal_max_timesteps,
                compression_type=compression_type,
            )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.writer = ExplanationWriter(
            output_directory=self.output_directory,
            image_weight=self.image_weight,
            overlay_image_format=self.overlay_image_format,
        )

        checkpoint_loader = FloatCheckpointLoader(
            device=self.device,
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
        )
        self.checkpoint_loader = checkpoint_loader
        self.config = checkpoint_loader.config
        self.policy: Policy = checkpoint_loader.policy
        self.policy.eval()
        self._batch_counter = 0

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve the torch device used for attribution."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _resolve_output_directory(
        checkpoint_path: str,
        output_directory: str | None,
    ) -> Path:
        """Resolve where generated files should be written."""
        if output_directory is not None:
            return Path(output_directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(checkpoint_path) / "explainability" / timestamp

    @staticmethod
    def _resolve_explanation_types(
        explanation_types: list[str] | None,
    ) -> list[str]:
        """Validate configured explanation methods and expand the default set."""
        if explanation_types is None:
            explanation_types = list(VALID_EXPLANATION_TYPES)
        invalid_types = [
            explanation_type
            for explanation_type in explanation_types
            if explanation_type not in VALID_EXPLANATION_TYPES
        ]
        if invalid_types:
            raise ValueError(
                f"Unsupported explanation_types {invalid_types}. "
                f"Use one or more of: {list(VALID_EXPLANATION_TYPES)}"
            )
        return explanation_types

    @staticmethod
    def _validate_source(source: str) -> None:
        """Validate that source is a supported explainability mode."""
        if source not in VALID_EXPLANATION_SOURCE_TYPES:
            raise ValueError(
                f"source must be one of {list(VALID_EXPLANATION_SOURCE_TYPES)}. "
                f"Got: {source}"
            )

    @staticmethod
    def _validate_sampling_configuration(
        sample_stride: int,
        max_samples: int | None,
    ) -> None:
        """Validate sample selection settings shared by all sources."""
        if sample_stride <= 0:
            raise ValueError(f"sample_stride must be positive. Got: {sample_stride}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError(
                f"max_samples must be positive when set. Got: {max_samples}"
            )

    @staticmethod
    def _resolve_online_max_steps(
        sample_stride: int,
        max_samples: int | None,
    ) -> int:
        """Translate the explanation sample cap into an online step budget."""
        if max_samples is None:
            return DEFAULT_ONLINE_MAX_STEPS
        return (max_samples - 1) * sample_stride + 1

    @staticmethod
    def _validate_online_configuration(
        model_server_port: int,
        action_execution_horizon: int | None,
        update_rate_hz: float | None,
        temporal_max_timesteps: int,
        compression_type: str,
    ) -> None:
        """Validate online inference settings before opening transports.

        Args:
            model_server_port: Environment server port.
            action_execution_horizon: Optional number of predicted actions sent
                per policy call when temporal aggregation is disabled.
            update_rate_hz: Optional action-send rate limit.
            temporal_max_timesteps: Temporal aggregation buffer length.
            compression_type: Requested observation image compression format.

        Raises:
            ValueError: If a numeric limit is non-positive or compression is not
                supported by ``tso_robotics_sockets``.
        """
        if model_server_port <= 0:
            raise ValueError(
                f"model_server_port must be positive. Got: {model_server_port}"
            )
        if action_execution_horizon is not None and action_execution_horizon <= 0:
            raise ValueError(
                "action_execution_horizon must be positive when set. "
                f"Got: {action_execution_horizon}"
            )
        if update_rate_hz is not None and update_rate_hz <= 0:
            raise ValueError(
                f"update_rate_hz must be positive when set. Got: {update_rate_hz}"
            )
        if temporal_max_timesteps <= 0:
            raise ValueError(
                "temporal_max_timesteps must be positive. "
                f"Got: {temporal_max_timesteps}"
            )
        valid_compression_types = [member.value for member in CompressionType]
        if compression_type not in valid_compression_types:
            raise ValueError(
                f"compression_type must be one of {valid_compression_types}. "
                f"Got: {compression_type}"
            )

    def run(self) -> None:
        """Run the configured explainability workflow.

        Raises:
            ValueError: If ``source`` is not a supported explainability mode.
        """
        match self.source:
            case ExplanationSourceType.ONLINE_INFERENCE.value:
                self._run_online_inference()
            case ExplanationSourceType.DATASET.value:
                self._run_dataset_source()
            case _:
                self._validate_source(source=self.source)
                raise ValueError(f"Unhandled explainability source: {self.source}")

    def _run_dataset_source(self) -> None:
        """Explain offline episodic windows from the configured dataset source."""
        source = DatasetExplanationSource(
            config=self.config,
            policy=self.policy,
            split=self.split,
            batch_size=self.batch_size,
            sample_stride=self.sample_stride,
            max_samples=self.max_samples,
            data_path_override=self.data_path_override,
        )
        for batch in source:
            self.explain_batch(batch=batch)
        logging.info("Saved explainability files to %s", self.output_directory)

    def _run_online_inference(self) -> None:
        """Run online inference and explain ready observation windows.

        Online mode delegates observation transport, preprocessing, buffering,
        action postprocessing, and action transport to ``InferenceClient``. The
        attached online source receives the exact observation batch passed to
        policy inference, so the explainer is independent of how the robot or
        simulator stores raw recordings.
        """
        policy_runtime = ExplainabilityPolicyRuntime(
            checkpoint_loader=self.checkpoint_loader,
            checkpoint_name=self.checkpoint_name,
        )
        observation_transport = SocketObservationTransport(
            server_address=self.model_server_address,
            server_port=self.model_server_port,
        )
        action_transport = SocketActionTransport(
            server_address=self.model_server_address,
            server_port=self.model_server_port,
        )
        client = InferenceClient(
            policy_runtime=policy_runtime,
            observation_transport=observation_transport,
            action_transport=action_transport,
            temporal_aggregation=self.temporal_aggregation,
            action_execution_horizon=self.action_execution_horizon,
            compression_type=self.compression_type,
            max_timesteps=self.temporal_max_timesteps,
            timing_log=self.timing_log,
            update_rate_hz=self.update_rate_hz,
            online_explanation_source=self.build_online_source(),
        )
        try:
            client.run_episode(
                max_steps=self._resolve_online_max_steps(
                    sample_stride=self.sample_stride,
                    max_samples=self.max_samples,
                )
            )
        finally:
            client.shutdown()
        logging.info("Saved explainability files to %s", self.output_directory)

    def build_online_source(self) -> OnlineInferenceExplanationSource:
        """Build an online inference adapter for ``InferenceClient``.

        Returns:
            Online source that delegates accepted inference windows to this
            runner.
        """
        return OnlineInferenceExplanationSource(
            consumer=self,
            sample_stride=self.sample_stride,
            max_samples=self.max_samples,
        )

    def explain_batch(self, batch: ExplanationBatch) -> None:
        """Generate configured explanations for one batch.

        Args:
            batch: Observation window and metadata from a supported source.
        """
        moved_observation = to_device(data=batch.observation, device=self.device)
        if not isinstance(moved_observation, dict):
            raise RuntimeError(
                f"Expected observation dictionary, got {type(moved_observation)}."
            )
        observation = dict(moved_observation)
        actions = None
        if batch.actions is not None:
            moved_actions = to_device(data=batch.actions, device=self.device)
            if not isinstance(moved_actions, dict):
                raise RuntimeError(
                    f"Expected action dictionary, got {type(moved_actions)}."
                )
            actions = dict(moved_actions)
        for explanation_type in self.explanation_types:
            heatmaps = self._compute_heatmaps(
                observation=observation,
                actions=actions,
                explanation_type=explanation_type,
                preprocess_observation=batch.preprocess_observation,
            )
            if self.save_raw_heatmaps:
                self.writer.save_raw_heatmaps(
                    heatmaps=heatmaps,
                    explanation_type=explanation_type,
                    metadata=batch.metadata,
                    batch_counter=self._batch_counter,
                )
            if self.save_overlays:
                self.writer.save_overlays(
                    heatmaps=heatmaps,
                    explanation_type=explanation_type,
                    batch=batch,
                    batch_counter=self._batch_counter,
                )
        self._batch_counter += 1

    def _compute_heatmaps(
        self,
        observation: ObservationBatch,
        actions: dict[str, torch.Tensor] | None,
        explanation_type: str,
        preprocess_observation: bool,
    ) -> dict[str, torch.Tensor]:
        """Compute heatmaps for one method and the configured visual targets.

        Args:
            observation: Observation batch on the runner device.
            actions: Optional action batch on the runner device.
            explanation_type: Explanation method to run.
            preprocess_observation: Whether the explainer should preprocess
                observations before attribution.

        Returns:
            Explainable heatmaps keyed by camera name.

        Raises:
            ValueError: If ``explanation_type`` is not registered.
        """
        if explanation_type not in self.explanation_heatmaps:
            supported_types = list(self.explanation_heatmaps)
            raise ValueError(
                f"Unsupported explanation_type '{explanation_type}'. "
                f"Use one of: {supported_types}"
            )
        heatmap_function = self.explanation_heatmaps[explanation_type]
        heatmaps: dict[str, torch.Tensor] = {}
        target_cameras = self._get_target_cameras()
        for target_camera in target_cameras:
            current_heatmaps = heatmap_function(
                policy=self.policy,
                observation=observation,
                actions=actions,
                target_camera=target_camera,
                target_vision_module_names=self.target_vision_module_names,
                preprocess_observation=preprocess_observation,
            )
            heatmaps.update(current_heatmaps)
        return heatmaps

    def _get_target_cameras(self) -> list[str | None]:
        """Return configured camera filters for attribution calls.

        Returns:
            ``[None]`` when all cameras should be explained; otherwise the
            configured camera keys.

        Raises:
            ValueError: If the configured camera allowlist is empty.
        """
        if self.target_camera_keys is None:
            return [None]
        if not self.target_camera_keys:
            raise ValueError("target_camera_keys must not be empty when set.")
        return self.target_camera_keys
