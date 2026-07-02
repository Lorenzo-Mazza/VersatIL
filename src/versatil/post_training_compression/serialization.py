"""Save and load compressed models with metadata."""

import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    CompressionMetadataKey,
    DeploymentBackendName,
)
from versatil.post_training_compression.export import _export_with_dynamic_batch
from versatil.training.constants import CheckpointFilename


def save_compressed_model(
    converted_model: nn.Module | None,
    example_inputs: tuple[torch.Tensor, ...],
    save_directory: str,
    input_keys: list[str],
    output_keys: list[str],
    normalizer: LinearNormalizer,
    training_checkpoint_path: str,
    quantization_config: Any,
    quantization_workflow: str,
    model_filename: str = CompressionFilename.COMPRESSED_MODEL.value,
    normalizer_filename: str = CompressionFilename.NORMALIZER.value,
    artifact_format: str = ArtifactFormat.TORCH_EXPORT_PT2.value,
    backend_name: str = DeploymentBackendName.TORCH_INDUCTOR.value,
    model_bytes: bytes | None = None,
    denoising_thresholds: dict[str, float] | None = None,
) -> Path:
    """Save compressed model artifact with normalizer and metadata.

    Saves the deployment artifact, normalizer, quantization config, training
    config, optional tokenizer files, and compression metadata.

    Args:
        converted_model: The converted model, used for Torch Export artifacts.
        example_inputs: Example input tensors for torch.export.
        save_directory: Directory to save into (created if needed).
        input_keys: Sorted input (observation) key ordering.
        output_keys: Sorted output (action) key ordering.
        normalizer: The policy's normalizer module.
        training_checkpoint_path: Path to the original training checkpoint
            directory used as the source for compression.
        quantization_config: The QuantizationConfig used for quantization.
        quantization_workflow: The workflow used (QuantizationWorkflow value).
        model_filename: Filename for the saved deployment artifact.
        normalizer_filename: Filename for the saved normalizer state.
        artifact_format: Serialized artifact format identifier.
        backend_name: Serialized deployment backend identifier.
        model_bytes: Optional pre-lowered artifact bytes, used for .pte.
        denoising_thresholds: Per-action-key denoising thresholds from the
            source policy, persisted so compressed deployments zero small
            deltas exactly like the float runtime.

    Returns:
        Path to the save directory.
    """
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / model_filename
    if model_bytes is not None:
        model_path.write_bytes(model_bytes)
    else:
        if converted_model is None:
            raise ValueError(
                "converted_model is required when model_bytes is not provided."
            )
        exported_program = _export_with_dynamic_batch(
            model=converted_model,
            example_inputs=example_inputs,
        )
        torch.export.save(exported_program, str(model_path))
    torch.save(normalizer.state_dict(), save_path / normalizer_filename)
    config_omega = OmegaConf.structured(quantization_config)
    OmegaConf.save(
        config=config_omega,
        f=save_path / CompressionFilename.QUANTIZATION_CONFIG.value,
    )
    tokenizer_source = (
        Path(training_checkpoint_path) / CheckpointFilename.TOKENIZER_DIR.value
    )
    tokenizer_dest = save_path / CompressionFilename.TOKENIZER_DIR.value
    if tokenizer_source.exists():
        if tokenizer_dest.exists():
            shutil.rmtree(tokenizer_dest)
        shutil.copytree(tokenizer_source, tokenizer_dest)
    config_source = Path(training_checkpoint_path) / CheckpointFilename.CONFIG.value
    if config_source.exists():
        shutil.copy2(config_source, save_path / CheckpointFilename.CONFIG.value)
    metadata = {
        CompressionMetadataKey.MODEL_FILE.value: model_filename,
        CompressionMetadataKey.NORMALIZER_FILE.value: normalizer_filename,
        CompressionMetadataKey.ARTIFACT_FORMAT.value: artifact_format,
        CompressionMetadataKey.DEPLOYMENT_BACKEND.value: backend_name,
        CompressionMetadataKey.INPUT_KEYS.value: input_keys,
        CompressionMetadataKey.OUTPUT_KEYS.value: output_keys,
        CompressionMetadataKey.TORCHAO_VERSION.value: _get_torchao_version(),
        CompressionMetadataKey.TORCH_VERSION.value: torch.__version__,
        CompressionMetadataKey.TRAINING_CHECKPOINT_PATH.value: training_checkpoint_path,
        CompressionMetadataKey.QUANTIZATION_WORKFLOW.value: quantization_workflow,
        CompressionMetadataKey.DENOISING_THRESHOLDS.value: denoising_thresholds or {},
    }
    with open(save_path / CompressionFilename.COMPRESSION_METADATA.value, "w") as file:
        json.dump(metadata, file, indent=2)

    return save_path


def load_compression_metadata(metadata_path: str) -> dict[str, Any]:
    """Load compression metadata from a checkpoint directory.

    Loads compression_metadata.json and optionally merges
    quantization_config.yaml fields.

    Args:
        metadata_path: Path to compression_metadata.json.

    Returns:
        Dict with runtime metadata + optional config fields.
    """
    metadata_file = Path(metadata_path)
    with open(metadata_file) as file:
        metadata = json.load(file)
    config_path = metadata_file.parent / CompressionFilename.QUANTIZATION_CONFIG.value
    if config_path.exists():
        config = OmegaConf.load(config_path)
        config_dict = OmegaConf.to_container(config, resolve=True)
        quantization_fields = _extract_quantization_fields(config_dict=config_dict)
        metadata.update(quantization_fields)
    metadata.setdefault(
        CompressionMetadataKey.ARTIFACT_FORMAT.value,
        ArtifactFormat.TORCH_EXPORT_PT2.value,
    )
    metadata.setdefault(
        CompressionMetadataKey.DEPLOYMENT_BACKEND.value,
        DeploymentBackendName.TORCH_INDUCTOR.value,
    )

    return metadata


def _extract_quantization_fields(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract quantization flags from a saved config dict.

    Handles full compressor configs with a nested ``quantization`` block and
    workflow configs saved directly.

    Args:
        config_dict: Resolved quantization config as a plain dict.

    Returns:
        Dict with is_dynamic, is_qat, and reduce_range values.
    """
    backend_keys = [
        CompressionMetadataKey.IS_DYNAMIC.value,
        CompressionMetadataKey.IS_QAT.value,
        CompressionMetadataKey.REDUCE_RANGE.value,
    ]
    quantization_config = config_dict.get("quantization", config_dict)
    if not isinstance(quantization_config, dict):
        return dict.fromkeys(backend_keys, False)
    targets = quantization_config.get("targets")
    if isinstance(targets, list) and targets:
        first_target = targets[0]
        if isinstance(first_target, dict):
            backend_config = first_target.get("pt2e_backend")
            if isinstance(backend_config, dict):
                return {key: backend_config.get(key, False) for key in backend_keys}
    return {key: quantization_config.get(key, False) for key in backend_keys}


def _get_torchao_version() -> str:
    """Get installed torchao version.

    Returns:
        Version string of the installed torchao package.
    """
    return version("torchao")
