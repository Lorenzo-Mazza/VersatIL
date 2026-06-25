"""Constants and enums for quantization configuration."""

from enum import StrEnum


class PT2EBackendName(StrEnum):
    """Target backend for PT2E quantized operator lowering.

    These backends are specific to the PT2E quantization workflow.
    The eager workflow does not use PT2E backend selection.
    """

    X86_INDUCTOR = "x86_inductor"


class QuantizationMode(StrEnum):
    """Quantization workflow used to produce the deployable model."""

    NONE = "none"
    PT2E = "pt2e"
    EAGER = "eager"


class QuantizableOperatorType(StrEnum):
    """Operator types targeted for quantization coverage analysis."""

    CONV2D = "conv2d"
    LINEAR = "linear"


class FXNodePattern(StrEnum):
    """FX graph node target patterns for operator classification."""

    ADDMM = "addmm"
    DEQUANTIZE = "dequantize"
    QUANTIZE_PER_TENSOR = "quantize_per_tensor"


class ReportMetricKey(StrEnum):
    """Dict keys returned by QuantizationReport analysis methods."""

    QUANTIZED = "quantized"
    TOTAL = "total"
    MAX_DIFFERENCE = "max_difference"
    MEAN_DIFFERENCE = "mean_difference"
    FLOAT_BYTES = "float_bytes"
    QUANTIZED_BYTES = "quantized_bytes"
    COMPRESSION_RATIO = "compression_ratio"
    FLOAT_MS = "float_milliseconds"
    QUANTIZED_MS = "quantized_milliseconds"
    SPEEDUP = "speedup"
