"""Hydra configuration for the `explain` endpoint."""

from dataclasses import dataclass, field

from omegaconf import MISSING

from versatil.configs.inference_client import InferenceClientConfig
from versatil.explainability.constants import (
    VALID_EXPLANATION_TYPES,
    ExplanationSourceType,
)
from versatil.training.constants import CheckpointFilename


@dataclass
class ExplanationWriterConfig:
    """Settings for how explanations are written to disk.

    Attributes:
        save_raw_heatmaps: Whether raw heatmap tensors are saved.
        save_overlays: Whether overlay images are rendered and saved.
        image_weight: Blend weight of the camera image in overlays.
        overlay_image_format: Overlay image file extension without a dot.
    """

    save_raw_heatmaps: bool = False
    save_overlays: bool = True
    image_weight: float = 0.5
    overlay_image_format: str = "png"


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
        max_samples: Optional cap on the number of observation windows to
            explain. Offline mode applies the cap after ``sample_stride``. Online
            mode applies it to ready inference windows and derives the
            inference-loop step budget from ``sample_stride``.
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
        explanation_types: Visual attribution methods to run. ``gradcam``
            handles both CNN feature maps and ViT patch-token maps internally.
        target_camera_keys: Optional camera-key allowlist. ``None`` explains all
            cameras exposed by visual modules.
        target_vision_module_names: Optional visual module allowlist. Names
            include encoding-pipeline entries and decoder-owned VLM vision tower
            paths.
        channel_batch_size: Number of feature channels ablated per forward pass
            for Ablation-CAM.
        online: Socket inference client settings for the online source, nested
            under the ``online`` key (e.g. ``online.model_server_address``,
            ``online.compression_type``). See ``InferenceClientConfig``.
        writer: Explanation writer settings, nested under the ``writer`` key
            (e.g. ``writer.save_raw_heatmaps``, ``writer.image_weight``). See
            ``ExplanationWriterConfig``.
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
    explanation_types: list[str] = field(
        default_factory=lambda: list(VALID_EXPLANATION_TYPES)
    )
    target_camera_keys: list[str] | None = None
    target_vision_module_names: list[str] | None = None
    channel_batch_size: int = 32
    online: InferenceClientConfig = field(default_factory=InferenceClientConfig)
    writer: ExplanationWriterConfig = field(default_factory=ExplanationWriterConfig)
