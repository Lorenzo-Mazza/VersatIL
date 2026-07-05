"""Hydra configuration dataclasses for quantization workflows and backends."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class BasePT2EBackendConfig:
    """Shared settings for PT2E quantization backends.

    Attributes:
        is_dynamic: Whether activations are quantized dynamically.
        is_qat: Whether the backend prepares quantization-aware training observers.
    """

    is_dynamic: bool = False
    is_qat: bool = False


@dataclass
class X86InductorBackendConfig(BasePT2EBackendConfig):
    """X86 Inductor backend for PT2E quantized operator lowering.

    Attributes:
        _target_: Import path instantiated by Hydra.
        reduce_range: Reduce quantization range for older CPUs without VNNI.
    """

    _target_: str = (
        "versatil.quantization.pt2e.backends.x86_inductor.X86InductorBackend"
    )
    reduce_range: bool = False


@dataclass
class XNNPACKPT2EBackendConfig(BasePT2EBackendConfig):
    """XNNPACK backend for PT2E quantization and ExecuTorch deployment.

    Attributes:
        _target_: Import path instantiated by Hydra.
        is_per_channel: Use per-channel symmetric weight quantization.
    """

    _target_: str = "versatil.quantization.pt2e.backends.xnnpack.XNNPACKPT2EBackend"
    is_per_channel: bool = True


@dataclass
class Int8DynamicQuantizeConfig:
    """Dynamic int8 activation + int8 weight quantization (`quantize_` API)."""

    _target_: str = "torchao.quantization.Int8DynamicActivationInt8WeightConfig"


@dataclass
class Int4WeightOnlyQuantizeConfig:
    """Int4 weight-only quantization with groupwise scaling (`quantize_` API).

    Attributes:
        _target_: Import path instantiated by Hydra.
        group_size: Rows sharing one quantization scale.
    """

    _target_: str = "torchao.quantization.Int4WeightOnlyConfig"
    group_size: int = 128


@dataclass
class EagerQuantizationModuleTargetConfig:
    """Module target for eager torchao quantization.

    Attributes:
        _target_: Import path instantiated by Hydra.
        module_path: Dotted path to the target module, or ``""`` for root.
        quantize_config: torchao eager quantization config applied to this target.
    """

    _target_: str = "versatil.quantization.module_target.EagerQuantizationModuleTarget"
    module_path: str = ""
    quantize_config: Any = MISSING


@dataclass
class PT2EQuantizationModuleTargetConfig:
    """Module target for PT2E quantization.

    Attributes:
        _target_: Import path instantiated by Hydra.
        module_path: Dotted path to the target module, or ``""`` for root.
        pt2e_backend: PT2E backend that creates the quantizer for this target.
    """

    _target_: str = "versatil.quantization.module_target.PT2EQuantizationModuleTarget"
    module_path: str = ""
    pt2e_backend: BasePT2EBackendConfig = field(
        default_factory=X86InductorBackendConfig
    )


@dataclass
class PT2EQuantizationWorkflowConfig:
    """Graph-level quantization with operator fusion via torch.export.

    Attributes:
        _target_: Import path instantiated by Hydra.
        targets: module-level PT2E quantization targets.
    """

    _target_: str = "versatil.quantization.workflows.pt2e.PT2EQuantizationWorkflow"
    targets: list[Any] = field(
        default_factory=lambda: [PT2EQuantizationModuleTargetConfig()]
    )


@dataclass
class EagerQuantizationWorkflowConfig:
    """Eager torchao quantization via quantize_().

    Attributes:
        _target_: Import path instantiated by Hydra.
        targets: Module-level eager quantization targets.
        is_qat: Whether this workflow is used for QAT checkpoint training and
            conversion.
        auto_filter_incompatible_linears: Whether to skip linears whose ``in_features``
            are incompatible with the config group size.
    """

    _target_: str = "versatil.quantization.workflows.eager.EagerQuantizationWorkflow"
    targets: list[Any] = MISSING
    is_qat: bool = False
    auto_filter_incompatible_linears: bool = True
