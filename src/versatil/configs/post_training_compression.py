"""Hydra configuration dataclasses for the post-training compression endpoint."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from versatil.training.constants import CheckpointFilename


@dataclass
class PreparationConfig:
    """Pre-quantization model preparation settings.

    Attributes:
        replace_frozen_batchnorm: Whether FrozenBatchNorm layers become plain BatchNorm
            before fusion.
        fuse_conv_batchnorm: Whether Conv and BatchNorm pairs are fused.
    """

    replace_frozen_batchnorm: bool = True
    fuse_conv_batchnorm: bool = True


@dataclass
class BasePrunerConfig:
    """Base config for pruning strategies.

    Attributes:
        amount: Fraction of weights to prune, in (0, 1).
    """

    amount: float = MISSING


@dataclass
class UnstructuredPrunerConfig(BasePrunerConfig):
    """Global L1 unstructured weight pruning.

    Note:
        When layer_types is null, convolution and linear layers are pruned.
        Use the ${prunable_layer:*} resolver in YAML to constrain the set.

    Attributes:
        _target_: Import path instantiated by Hydra.
        layer_types: PrunableLayerType values to target. Defaults to convolution and
            linear layers (normalization scales and embedding tables are usually not
            good pruning targets).
    """

    _target_: str = "versatil.post_training_compression.pruning.UnstructuredPruner"
    layer_types: list[str] | None = None


@dataclass
class StructuredPrunerConfig(BasePrunerConfig):
    """Per-layer structured weight pruning using Lp-norm channel ranking.

    Note:
        When layer_types is null, defaults to Conv1d, Linear and Conv2d.
        Use ${prunable_layer:*} resolver in YAML to add types.

    Attributes:
        _target_: Import path instantiated by Hydra.
        norm_order: The p in Lp-norm used to rank channels (e.g., 1 for L1, 2 for L2).
        dimension: Weight tensor dimension along which to prune.
        layer_types: PrunableLayerType values to target. Defaults to Conv1d, Conv2d, and
            Linear.
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

    Attributes:
        _target_: Import path instantiated by Hydra.
        module_path: Dotted path to the target submodule, or empty string for the full
            policy.
        preparation: BN replacement and fusion settings.
        pruning: Pruning strategies to apply sequentially.
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
    """ExecuTorch backend that lowers artifacts to XNNPACK .pte files.

    Attributes:
        _target_: Import path instantiated by Hydra.
        max_batch_size: Upper bound for dynamic batch execution in the serialized
            ExecuTorch program.
    """

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

    Attributes:
        _target_: Import path instantiated by Hydra.
        checkpoint_path: Path to the training checkpoint directory.
        checkpoint_name: Checkpoint filename inside the directory.
        output_directory: Where to save compressed output. Defaults to
            checkpoint_path/compressed/<timestamp>.
        calibration_steps: Number of calibration batches for static quantization.
        generate_report: Whether to generate a quantization report after saving.
            Disabled by default since it runs additional forward passes for
            benchmarking.
        modules: Per-module compression schemes (empty = global).
        preparation: Global preparation settings.
        pruning: Global pruning strategies (inherited by modules).
        quantization: Quantization workflow. ``None`` exports the float model without
            quantization.
        deployment_backend: Deployment backend that owns artifact format and lowering.
            Defaults to torch inductor.
    """

    _target_: str = (
        "versatil.post_training_compression.compressor.PostTrainingCompressor"
    )
    checkpoint_path: str = MISSING
    checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value
    output_directory: str | None = None
    calibration_steps: int = 32
    generate_report: bool = False
    modules: list[CompressionTargetConfig] = field(default_factory=list)
    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    pruning: list[Any] | None = None  #  list[BasePrunerConfig] | None
    quantization: Any | None = (
        None  # PT2EQuantizationWorkflowConfig | EagerQuantizationWorkflowConfig | None
    )
    deployment_backend: Any | None = None
