"""Constants and enums for post-training compression."""

from enum import StrEnum

from torch import nn


class QuantizationWorkflow(StrEnum):
    """Quantization workflow used during compression."""

    PT2E = "pt2e"
    EAGER = "eager"
    NONE = "none"


class ArtifactFormat(StrEnum):
    """Deployment artifact format emitted by deployment backends."""

    TORCH_EXPORT_PT2 = "torch_export_pt2"
    EXECUTORCH_PTE = "executorch_pte"


class DeploymentBackendName(StrEnum):
    """Deployment backend identifiers stored in metadata."""

    TORCH_INDUCTOR = "torch_inductor"
    EXECUTORCH_XNNPACK = "executorch_xnnpack"


class CompressionMetadataKey(StrEnum):
    """Keys used in compression metadata JSON files."""

    MODEL_FILE = "model_file"
    NORMALIZER_FILE = "normalizer_file"
    ARTIFACT_FORMAT = "artifact_format"
    DEPLOYMENT_BACKEND = "deployment_backend"
    INPUT_KEYS = "input_keys"
    OUTPUT_KEYS = "output_keys"
    TORCHAO_VERSION = "torchao_version"
    TORCH_VERSION = "torch_version"
    TRAINING_CHECKPOINT_PATH = "training_checkpoint_path"
    QUANTIZATION_WORKFLOW = "quantization_workflow"
    DENOISING_THRESHOLDS = "denoising_thresholds"


class CompressionFilename(StrEnum):
    """Standard filenames in compressed checkpoint directories."""

    QUANTIZATION_CONFIG = "quantization_config.yaml"
    COMPRESSION_METADATA = "compression_metadata.json"
    COMPRESSED_MODEL = "compressed_policy.pt2"
    EXECUTORCH_MODEL = "compressed_policy.pte"
    NORMALIZER = "normalizer.pt"
    TOKENIZER_DIR = "tokenizer"


class PrunableLayerType(StrEnum):
    """Layer types targeted by pruning strategies."""

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
