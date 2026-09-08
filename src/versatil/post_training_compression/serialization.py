"""Save and load compressed models with metadata."""

import json
import shutil
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.exportable.metadata import PolicyExportMetadata
from versatil.post_training_compression.constants import (
    ArtifactFormat,
    CompressionFilename,
    CompressionMetadataKey,
    DeploymentBackendName,
)
from versatil.post_training_compression.export import _export_with_dynamic_batch
from versatil.quantization.metadata import QuantizationTargetMetadata
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
    pt2e_backend_config: dict[str, Any] | None = None,
    quantization_targets: list[QuantizationTargetMetadata] | None = None,
    calibration_batches: int | None = None,
    export_metadata: PolicyExportMetadata | None = None,
    tokenizer: Tokenizer | None = None,
) -> Path:
    """Save compressed model artifact with normalizer and metadata.

    Args:
        converted_model: The converted model, used for Torch Export artifacts.
        example_inputs: Observation tensors followed by any additional graph
            inputs required by the export metadata, in argument order.
        save_directory: Directory to save into (created if needed).
        input_keys: Sorted input (observation) key ordering.
        output_keys: Graph output names in tensor order: action keys or the
            action-token key.
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
        pt2e_backend_config: Instantiable config node of the PT2E quantizer
            backend, persisted so inference can rebuild the backend without
            depending on the full compressor config schema.
        quantization_targets: Per-target conversion settings, selected/skipped
            layers and resulting weight types supplied by the workflow.
        calibration_batches: Consumed representative observation batches, or None
            when the workflow leaves this count unspecified.
        export_metadata: Graph output format and additional tensor inputs.
            Defaults to normalized action outputs and observation-only inputs.
        tokenizer: Loaded observation and action tokenizers to save with their
            processor implementations, replacing the destination tokenizer bundle.
            Omitted values use the original checkpoint's tokenizer directory when
            available.

    Returns:
        Path to the save directory.

    Note:
        The directory contains the deployment artifact, normalizer, quantization
        config, training config, optional tokenizer files and compression metadata.
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
    if tokenizer is not None:
        if tokenizer_dest.exists():
            shutil.rmtree(tokenizer_dest)
        tokenizer.save_pretrained(path=tokenizer_dest)
    elif tokenizer_source.exists():
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
        CompressionMetadataKey.PT2E_BACKEND.value: pt2e_backend_config,
        CompressionMetadataKey.QUANTIZATION_TARGETS.value: (
            [asdict(target) for target in quantization_targets]
            if quantization_targets is not None
            else None
        ),
        CompressionMetadataKey.CALIBRATION_BATCHES.value: calibration_batches,
        CompressionMetadataKey.POLICY_EXPORT_METADATA.value: (
            export_metadata or PolicyExportMetadata()
        ).to_dict(),
    }
    with open(save_path / CompressionFilename.COMPRESSION_METADATA.value, "w") as file:
        json.dump(metadata, file, indent=2)

    return save_path


def load_compression_metadata(metadata_path: str) -> dict[str, Any]:
    """Load compression metadata from a checkpoint directory.

    Args:
        metadata_path: Path to compression_metadata.json.

    Returns:
        Dict with runtime metadata.
    """
    with open(metadata_path) as file:
        metadata: dict[str, Any] = json.load(file)
    return metadata


def _get_torchao_version() -> str:
    """Get installed torchao version.

    Returns:
        Version string of the installed torchao package.
    """
    return version("torchao")
