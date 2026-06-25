"""Hydra configuration dataclasses for quantization workflows and backends."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class BasePT2EBackendConfig:
    """Shared settings for PT2E quantization backends."""

    is_dynamic: bool = False
    is_qat: bool = False
    reduce_range: bool = False


@dataclass
class X86InductorBackendConfig(BasePT2EBackendConfig):
    """X86 Inductor backend for PT2E quantized operator lowering."""

    _target_: str = (
        "versatil.quantization.pt2e.backends.x86_inductor.X86InductorBackend"
    )


@dataclass
class Int8DynamicQuantizeConfig:
    """Dynamic int8 activation + int8 weight quantization (`quantize_` API)."""

    _target_: str = "torchao.quantization.Int8DynamicActivationInt8WeightConfig"


@dataclass
class Int4WeightOnlyQuantizeConfig:
    """Int4 weight-only quantization with groupwise scaling (`quantize_` API)."""

    _target_: str = "torchao.quantization.Int4WeightOnlyConfig"
    group_size: int = 128


@dataclass
class EagerQuantizationModuleTargetConfig:
    """Module target for eager torchao quantization."""

    _target_: str = "versatil.quantization.module_target.EagerQuantizationModuleTarget"
    module_path: str = ""
    quantize_config: Any = MISSING


@dataclass
class PT2EQuantizationModuleTargetConfig:
    """Module target for PT2E quantization."""

    _target_: str = "versatil.quantization.module_target.PT2EQuantizationModuleTarget"
    module_path: str = ""
    pt2e_backend: BasePT2EBackendConfig = field(
        default_factory=X86InductorBackendConfig
    )


@dataclass
class PT2EQuantizationWorkflowConfig:
    """Graph-level quantization with operator fusion via torch.export."""

    _target_: str = "versatil.quantization.workflows.pt2e.PT2EQuantizationWorkflow"
    targets: list[Any] = field(
        default_factory=lambda: [PT2EQuantizationModuleTargetConfig()]
    )


@dataclass
class EagerQuantizationWorkflowConfig:
    """Eager torchao quantization via quantize_()."""

    _target_: str = "versatil.quantization.workflows.eager.EagerQuantizationWorkflow"
    targets: list[Any] = MISSING
    is_qat: bool = False
    auto_filter_incompatible_linears: bool = True
