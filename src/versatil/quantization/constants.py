"""Constants and enums for quantization configuration."""

from enum import StrEnum


class QuantizationBackend(StrEnum):
    """Target backend for PT2E quantized operator lowering.

    These backends are specific to the PT2E quantization path.
    The quantize_() API path does not use backend selection.
    """

    X86_INDUCTOR = "x86_inductor"


class QuantizableOperatorType(StrEnum):
    """Operator types targeted for quantization coverage analysis."""

    CONV2D = "conv2d"
    LINEAR = "linear"


class QuantizationMetadataKey(StrEnum):
    """Keys used in legacy quantization metadata JSON files."""

    WEIGHTS_FILE = "weights_file"
    OBSERVATION_KEYS = "observation_keys"
    ACTION_KEYS = "action_keys"
    TORCHAO_VERSION = "torchao_version"
    TORCH_VERSION = "torch_version"
    IS_DYNAMIC = "is_dynamic"
    IS_QAT = "is_qat"
    REDUCE_RANGE = "reduce_range"
    TRAINING_CHECKPOINT_PATH = "training_checkpoint_path"


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
