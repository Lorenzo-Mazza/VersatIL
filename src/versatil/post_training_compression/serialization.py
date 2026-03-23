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
from versatil.quantization.constants import QuantizationMetadataKey
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
    model_filename: str = CompressionFilename.COMPRESSED_MODEL.value,
    normalizer_filename: str = CompressionFilename.NORMALIZER.value,
) -> Path:
    """Save compressed model as .pt2 archive with normalizer and metadata.

    Saves:
    - .pt2 archive: torch.export.export() -> torch.export.save()
    - normalizer.pt: Normalizer state_dict for standalone loading.
    - quantization_config.yaml: The QuantizationConfig via OmegaConf.
    - compression_metadata.json: Runtime artifacts (keys, versions,
      filenames, training checkpoint path).

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


def save_quantized_model(
    quantized_model: nn.Module,
    save_directory: str,
    observation_keys: list[str],
    action_keys: list[str],
    quantization_config: object,
    training_checkpoint_path: str,
    weights_filename: str = CompressionFilename.QUANTIZED_WEIGHTS.value,
    strip_redundant_float_weights: bool = True,
) -> Path:
    """Save quantized model state_dict and metadata (legacy format).

    Saves:
    - Weights file: State dict (optionally stripped of redundant float
      weights).
    - quantization_config.yaml: The QuantizationConfig via OmegaConf.
    - quantization_metadata.json: Runtime artifacts (keys, versions,
      weights filename, training checkpoint path).

    Args:
        quantized_model: The quantized model.
        save_directory: Directory to save into (created if needed).
        observation_keys: Sorted observation key ordering.
        action_keys: Sorted action key ordering.
        quantization_config: The QuantizationConfig used for quantization.
        training_checkpoint_path: Path to the original training checkpoint
            directory used as the source for quantization.
        weights_filename: Filename for the saved weights file.
        strip_redundant_float_weights: If True, remove float32 weight
            tensors that have int8 counterparts (same numel).

    Returns:
        Path to the save directory.
    """
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    state_dict = quantized_model.state_dict()
    if strip_redundant_float_weights:
        state_dict = _strip_redundant_weights(state_dict=state_dict)

    torch.save(state_dict, save_path / weights_filename)
    config_omega = OmegaConf.structured(quantization_config)
    OmegaConf.save(
        config=config_omega,
        f=save_path / CompressionFilename.QUANTIZATION_CONFIG.value,
    )
    metadata = {
        QuantizationMetadataKey.WEIGHTS_FILE.value: weights_filename,
        QuantizationMetadataKey.OBSERVATION_KEYS.value: observation_keys,
        QuantizationMetadataKey.ACTION_KEYS.value: action_keys,
        QuantizationMetadataKey.TORCHAO_VERSION.value: _get_torchao_version(),
        QuantizationMetadataKey.TORCH_VERSION.value: torch.__version__,
        QuantizationMetadataKey.TRAINING_CHECKPOINT_PATH.value: training_checkpoint_path,
    }
    with open(save_path / CompressionFilename.QUANTIZATION_METADATA.value, "w") as file:
        json.dump(metadata, file, indent=2)

    return save_path


def load_quantization_metadata(metadata_path: str) -> dict[str, Any]:
    """Load quantization metadata from a checkpoint directory (legacy format).

    Loads both quantization_metadata.json and quantization_config.yaml,
    merging them into a single dict.

    For PT2E configs, backend fields (is_dynamic, is_qat, reduce_range)
    are nested under ``pt2e_backend``. For other config structures, falls
    back to top-level lookup.

    Args:
        metadata_path: Path to quantization_metadata.json.

    Returns:
        Dict with runtime metadata + config fields
        (is_dynamic, is_qat, reduce_range).
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
        QuantizationMetadataKey.IS_DYNAMIC.value,
        QuantizationMetadataKey.IS_QAT.value,
        QuantizationMetadataKey.REDUCE_RANGE.value,
    ]
    # PT2E configs nest backend fields under pt2e_backend
    nested = config_dict.get("pt2e_backend")
    if isinstance(nested, dict):
        return {key: nested.get(key, False) for key in backend_keys}
    else:
        return {key: config_dict.get(key, False) for key in backend_keys}


def verify_reload_fidelity(
    original_model: nn.Module,
    reloaded_model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
) -> bool:
    """Verify exact numerical match between original and reloaded model.

    Args:
        original_model: The model before save.
        reloaded_model: The model after loading from disk.
        example_inputs: Inputs to run through both models.

    Returns:
        True if all output tensors match exactly.
    """
    with torch.no_grad():
        original_outputs = original_model(*example_inputs)
        reloaded_outputs = reloaded_model(*example_inputs)
    if len(original_outputs) != len(reloaded_outputs):
        return False

    return all(
        torch.equal(original, reloaded)
        for original, reloaded in zip(original_outputs, reloaded_outputs)
    )


def _strip_redundant_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove float32 weight tensors that have int8 counterparts.

    Args:
        state_dict: Model state dict to clean.

    Returns:
        State dict with redundant float32 weights removed.
    """
    int8_numels: set[int] = {
        value.numel() for value in state_dict.values() if value.dtype == torch.int8
    }
    clean: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if (
            value.dtype == torch.float32
            and key.endswith(".weight")
            and value.numel() in int8_numels
        ):
            continue
        clean[key] = value
    return clean


def _get_torchao_version() -> str:
    """Get installed torchao version.

    Returns:
        Version string of the installed torchao package.
    """
    return version("torchao")
