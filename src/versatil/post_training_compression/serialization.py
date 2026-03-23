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
    CompressionFilename,
    CompressionMetadataKey,
)
from versatil.post_training_compression.export import _export_with_dynamic_batch
from versatil.training.constants import CheckpointFilename


def save_compressed_model(
    converted_model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    save_directory: str,
    input_keys: list[str],
    output_keys: list[str],
    normalizer: LinearNormalizer,
    training_checkpoint_path: str,
    quantization_config: object,
    quantization_strategy: str,
    model_filename: str = CompressionFilename.COMPRESSED_MODEL.value,
    normalizer_filename: str = CompressionFilename.NORMALIZER.value,
) -> Path:
    """Save compressed model as .pt2 archive with normalizer and metadata.

    Saves:
    - .pt2 archive: torch.export.export() -> torch.export.save()
    - normalizer.pt: Normalizer state_dict for standalone loading.
    - quantization_config.yaml: The QuantizationConfig via OmegaConf.
    - compression_metadata.json: Runtime artifacts (keys, versions,
      filenames, training checkpoint path, quantization strategy).

    Args:
        converted_model: The converted (pre-lowered) quantized model.
        example_inputs: Example input tensors for torch.export.
        save_directory: Directory to save into (created if needed).
        input_keys: Sorted input (observation) key ordering.
        output_keys: Sorted output (action) key ordering.
        normalizer: The policy's normalizer module.
        training_checkpoint_path: Path to the original training checkpoint
            directory used as the source for compression.
        quantization_config: The QuantizationConfig used for quantization.
        quantization_strategy: The strategy used (QuantizationStrategy value).
        model_filename: Filename for the saved .pt2 archive.
        normalizer_filename: Filename for the saved normalizer state.

    Returns:
        Path to the save directory.
    """
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    exported_program = _export_with_dynamic_batch(
        model=converted_model,
        example_inputs=example_inputs,
    )
    torch.export.save(exported_program, str(save_path / model_filename))
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
        CompressionMetadataKey.INPUT_KEYS.value: input_keys,
        CompressionMetadataKey.OUTPUT_KEYS.value: output_keys,
        CompressionMetadataKey.TORCHAO_VERSION.value: _get_torchao_version(),
        CompressionMetadataKey.TORCH_VERSION.value: torch.__version__,
        CompressionMetadataKey.TRAINING_CHECKPOINT_PATH.value: training_checkpoint_path,
        CompressionMetadataKey.QUANTIZATION_STRATEGY.value: quantization_strategy,
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
        backend_fields = _extract_backend_fields(config_dict=config_dict)
        metadata.update(backend_fields)

    return metadata


def _extract_backend_fields(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract backend quantization fields from a config dict.

    Handles PT2E configs where fields are nested under ``pt2e_backend``
    and flat configs where fields are at the top level.

    Args:
        config_dict: Resolved quantization config as a plain dict.

    Returns:
        Dict with is_dynamic, is_qat, and reduce_range fields.
    """
    backend_keys = [
        CompressionMetadataKey.IS_DYNAMIC.value,
        CompressionMetadataKey.IS_QAT.value,
        CompressionMetadataKey.REDUCE_RANGE.value,
    ]
    nested = config_dict.get("pt2e_backend")
    if isinstance(nested, dict):
        return {key: nested.get(key, False) for key in backend_keys}
    else:
        return {key: config_dict.get(key, False) for key in backend_keys}


def _get_torchao_version() -> str:
    """Get installed torchao version.

    Returns:
        Version string of the installed torchao package.
    """
    return version("torchao")
