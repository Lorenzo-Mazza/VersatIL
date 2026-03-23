"""Hydra configuration dataclasses for quantization strategies and backends."""

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

    _target_: str = "versatil.quantization.backends.x86_inductor.X86InductorBackend"


@dataclass
class PT2EStrategyConfig:
    """Graph-level quantization with operator fusion via torch.export.

    Uses a PT2E backend (e.g. X86 Inductor) to quantize and lower
    operators after export. Static backends require calibration data.
    """

    _target_: str = "versatil.quantization.strategies.PT2EStrategy"
    pt2e_backend: BasePT2EBackendConfig = field(
        default_factory=X86InductorBackendConfig
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
class QuantizeApiStrategyConfig:
    """Eager mode quantization via torchao quantize_() API.

    Applies quantization in-place without operator fusion.
    Supports dynamic activation and weight-only configs.

    Note:
        Currently, we only support quantization of linear layers.
        For other use-cases, use the PT2E API.
    """

    _target_: str = "versatil.quantization.strategies.QuantizeApiStrategy"
    quantize_config: Any = MISSING  # AOBaseConfig subclass via _target_
