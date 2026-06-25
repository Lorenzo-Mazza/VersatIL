"""Hydra configuration dataclasses for the post-training compression endpoint."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from versatil.training.constants import CheckpointFilename


@dataclass
class PreparationConfig:
    """Pre-quantization model preparation settings."""

    replace_frozen_batchnorm: bool = True
    fuse_conv_batchnorm: bool = True


@dataclass
class BasePrunerConfig:
    """Base config for pruning strategies."""

    amount: float = MISSING


@dataclass
class UnstructuredPrunerConfig(BasePrunerConfig):
    """Global L1 unstructured weight pruning.

    Note:
        When layer_types is null, targets all modules with a weight
        parameter. Use ${prunable_layer:*} resolver in YAML to constrain which layer to prune.
    """

    _target_: str = "versatil.post_training_compression.pruning.UnstructuredPruner"
    layer_types: list[str] | None = None


@dataclass
class StructuredPrunerConfig(BasePrunerConfig):
    """Per-layer structured weight pruning using Lp-norm channel ranking.

    Note:
        When layer_types is null, defaults to Conv1d, Linear and Conv2d.
        Use ${prunable_layer:*} resolver in YAML to add types.
    """

    _target_: str = "versatil.post_training_compression.pruning.StructuredPruner"
    norm_order: int = 1
    dimension: int = 0
    layer_types: list[str] | None = None


@dataclass
class CompressionTargetConfig:
    """Per-module preparation and pruning scheme with inheritance.

    Note:
        Absent fields in YAML inherit from the global config via Hydra
        interpolation defaults. Explicit null means skip.
    """

    _target_: str = (
        "versatil.post_training_compression.compression_target.CompressionTarget"
    )
    module_path: str = MISSING
    preparation: PreparationConfig | None = "${preparation}"  #
    pruning: list[Any] | None = "${pruning}"  # list[BasePrunerConfig] | None


@dataclass
class TorchInductorBackendConfig:
    """Torch inductor deployment backend that writes .pt2 artifacts."""

    _target_: str = "versatil.post_training_compression.deployment_backends.torch_inductor.TorchInductorBackend"


@dataclass
class ExecutorchXNNPACKBackendConfig:
    """ExecuTorch backend that lowers artifacts to XNNPACK .pte files."""

    _target_: str = (
        "versatil.post_training_compression.deployment_backends.executorch_xnnpack."
        "ExecutorchXNNPACKBackend"
    )
    max_batch_size: int = 32


@dataclass
class PostTrainingCompressorConfig:
    """Top-level config for the post-training compression endpoint.

    Note:
        Global fields serve as defaults inherited by per-module configs
        via Hydra interpolation.
    """

    _target_: str = (
        "versatil.post_training_compression.compressor.PostTrainingCompressor"
    )
    checkpoint_path: str = MISSING
    checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value
    output_directory: str | None = None
    device: str = "cpu"  # Device for export and calibration ("cpu" or "cuda")
    calibration_steps: int = 32
    generate_report: bool = False
    modules: list[CompressionTargetConfig] = field(default_factory=list)
    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    pruning: list[Any] | None = None  #  list[BasePrunerConfig] | None
    quantization: Any | None = (
        None  # PT2EQuantizationWorkflowConfig | EagerQuantizationWorkflowConfig | None
    )
    deployment_backend: Any | None = None
