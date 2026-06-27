"""Hydra configuration for the `explain` endpoint."""

from dataclasses import dataclass, field

from omegaconf import MISSING
from tso_robotics_sockets import CompressionType

from versatil.explainability.constants import (
    VALID_EXPLANATION_TYPES,
    ExplanationSourceType,
)
from versatil.training.constants import CheckpointFilename


@dataclass
class ExplainabilityConfig:
    """Hydra config for generating xAI insights on policy predictions.

    Attributes:
        _target_: Import path instantiated by Hydra.
        checkpoint_path: Directory containing the checkpoint, config, and
            normalizer/tokenizer files used to restore the policy.
        checkpoint_name: Checkpoint filename inside ``checkpoint_path``.
        output_directory: Optional directory for written explanation files.
            When ``None``, outputs are written under
            ``checkpoint_path/explainability`` with a timestamped subdirectory.
        device: Torch device for attribution, or ``auto`` to prefer CUDA when
            available.
        source: Explanation source. ``dataset`` samples offline episodic
            windows from the checkpoint schema or ``data_path_override``;
            ``online_inference`` starts the same inference client loop used by
            the test endpoint and explains ready observation windows before
            action prediction.
        split: Dataset split for offline explanations: ``train``, ``val``, or
            ``all``.
        sample_stride: Explanation interval. In offline dataset mode, keep every
            Nth episodic dataset sample. In online inference mode, explain every
            Nth inference timestep.
        max_samples: Optional cap on the number of offline dataset samples.
        data_path_override: Optional offline input location to explain instead
            of the data path stored in the checkpoint task config.
            ``None`` keeps the checkpoint's original ``task.dataset_schema``
            paths. A single path ending in ``.zarr`` is treated as an existing
            replay buffer and sampled directly. A non-zarr path is treated as
            raw data in the same schema format as the checkpoint, for example a
            CSV episode-folder root, an HDF5 file, or a LeRobot dataset root.
            A list is only for raw schemas that already support multiple raw
            inputs, such as CSV ``dataset_folders`` or HDF5 ``hdf5_paths``;
            multiple zarr paths are not supported. Raw overrides are converted
            to ``offline_dataset.zarr`` beside the first override path before
            episodic windows are sampled.
        batch_size: Number of sampled windows explained per attribution call.
        model_server_address: Environment server address for online inference
            mode.
        model_server_port: Environment server port for online inference mode.
        temporal_aggregation: Whether online inference should average
            overlapping action predictions from consecutive policy calls.
        action_execution_horizon: Number of actions sent from each predicted
            chunk when temporal aggregation is disabled. ``None`` uses the
            checkpoint prediction horizon.
        update_rate_hz: Optional action-send rate limit for online inference.
            ``None`` sends actions as soon as they are available.
        max_steps: Maximum number of online inference loop steps before the
            explain endpoint exits.
        temporal_max_timesteps: Maximum episode length tracked by temporal
            aggregation state.
        timing_log: Whether to log per-step preprocessing, inference, and
            postprocessing timings in online mode.
        compression_type: Image compression format requested from the online
            environment server.
        explanation_types: Visual attribution methods to run. ``gradcam``
            handles both CNN feature maps and ViT patch-token maps internally.
        target_camera_keys: Optional camera-key allowlist. ``None`` explains all
            cameras exposed by visual modules.
        target_vision_module_names: Optional visual module allowlist. Names
            include encoding-pipeline entries and decoder-owned VLM vision tower
            paths.
        save_raw_heatmaps: Whether to save raw heatmap tensors as ``.pt`` files.
        save_overlays: Whether to save image overlays for displayable camera
            observations.
        channel_batch_size: Number of feature channels ablated per forward pass
            for Ablation-CAM.
        image_weight: Blend weight for the original image when saving overlays.
        overlay_image_format: File extension for overlay images, with or
            without a leading dot.
    """

    _target_: str = "versatil.explainability.runner.ExplainabilityRunner"
    checkpoint_path: str = MISSING
    checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value
    output_directory: str | None = None
    device: str = "auto"
    source: str = ExplanationSourceType.DATASET.value
    split: str = "all"
    sample_stride: int = 50
    max_samples: int | None = None
    data_path_override: str | list[str] | None = None
    batch_size: int = 1
    model_server_address: str = "127.0.0.1"
    model_server_port: int = 5555
    temporal_aggregation: bool = False
    action_execution_horizon: int | None = None
    update_rate_hz: float | None = None
    max_steps: int = 1000000
    temporal_max_timesteps: int = 800
    timing_log: bool = False
    compression_type: str = CompressionType.RAW.value
    explanation_types: list[str] = field(
        default_factory=lambda: list(VALID_EXPLANATION_TYPES)
    )
    target_camera_keys: list[str] | None = None
    target_vision_module_names: list[str] | None = None
    save_raw_heatmaps: bool = False
    save_overlays: bool = True
    channel_batch_size: int = 32
    image_weight: float = 0.5
    overlay_image_format: str = "png"
