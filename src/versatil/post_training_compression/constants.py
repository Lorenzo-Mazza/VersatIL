"""Constants and enums for post-training compression."""

from enum import StrEnum

from torch import nn


class QuantizationStrategy(StrEnum):
    """Quantization strategy used during compression."""

    PT2E = "pt2e"
    QUANTIZE_API = "quantize_api"


class CompressionMetadataKey(StrEnum):
    """Keys used in compression metadata JSON files (.pt2 format)."""

    MODEL_FILE = "model_file"
    NORMALIZER_FILE = "normalizer_file"
    INPUT_KEYS = "input_keys"
    OUTPUT_KEYS = "output_keys"
    TORCHAO_VERSION = "torchao_version"
    TORCH_VERSION = "torch_version"
    TRAINING_CHECKPOINT_PATH = "training_checkpoint_path"
    QUANTIZATION_STRATEGY = "quantization_strategy"
    IS_DYNAMIC = "is_dynamic"
    IS_QAT = "is_qat"
    REDUCE_RANGE = "reduce_range"


class CompressionFilename(StrEnum):
    """Standard filenames in compressed checkpoint directories."""

    QUANTIZATION_CONFIG = "quantization_config.yaml"
    COMPRESSION_METADATA = "compression_metadata.json"
    COMPRESSED_MODEL = "compressed_policy.pt2"
    NORMALIZER = "normalizer.pt"
    TOKENIZER_DIR = "tokenizer"


class PrunableLayerType(StrEnum):
    """Common layer types targeted by pruning strategies."""

    CONV1D = "conv1d"
    CONV2D = "conv2d"
    LINEAR = "linear"

    def to_module_type(self) -> type[nn.Module]:
        """Resolve to the corresponding torch.nn.Module class."""
        mapping: dict[str, type[nn.Module]] = {
            PrunableLayerType.CONV1D.value: nn.Conv1d,
            PrunableLayerType.CONV2D.value: nn.Conv2d,
            PrunableLayerType.LINEAR.value: nn.Linear,
        }
        return mapping[self.value]


class PruningTargetAttribute(StrEnum):
    """PyTorch module attribute names targeted by pruning."""

    WEIGHT = "weight"
