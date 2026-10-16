"""Policy context loading for post-training compression."""

import logging

import torch

from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
from versatil.checkpoint_loading.qat_policy import _QATCheckpointLoader
from versatil.post_training_compression.constants import CompressionDeviceType
from versatil.post_training_compression.policy_context import PolicyContext
from versatil.quantization.workflows.base import BaseQuantizationWorkflow


def load_float_policy_context(
    checkpoint_path: str,
    checkpoint_name: str,
    device: torch.device = torch.device(CompressionDeviceType.CPU.value),
) -> PolicyContext:
    """Restore a float policy and preprocessing state on the requested device.

    Args:
        checkpoint_path: Directory containing the training checkpoint.
        checkpoint_name: Checkpoint filename to restore.
        device: Device for the policy, normalizer and tokenizer.

    Returns:
        Loaded policy context for compression.
    """
    logging.info(f"Loading policy from {checkpoint_path}")
    checkpoint_loader = FloatCheckpointLoader(
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
    )
    return PolicyContext(
        policy=checkpoint_loader.policy,
        config=checkpoint_loader.config,
        tokenizer=checkpoint_loader.tokenizer,
        observation_space=checkpoint_loader.observation_space,
        observation_horizon=checkpoint_loader.observation_horizon,
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
    )


def load_qat_policy_context(
    checkpoint_path: str,
    checkpoint_name: str,
    quantization: BaseQuantizationWorkflow,
    device: torch.device = torch.device(CompressionDeviceType.CPU.value),
) -> PolicyContext:
    """Restore a QAT policy and preprocessing state on the requested device.

    Args:
        checkpoint_path: Directory containing the training checkpoint.
        checkpoint_name: Checkpoint filename to restore.
        quantization: Workflow that restores fake-quantized modules.
        device: Device for the policy, normalizer and tokenizer.

    Returns:
        Prepared policy context for QAT conversion.
    """
    checkpoint_loader = _QATCheckpointLoader(
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
        quantization=quantization,
    )
    return PolicyContext(
        policy=checkpoint_loader.policy,
        config=checkpoint_loader.config,
        tokenizer=checkpoint_loader.tokenizer,
        observation_space=checkpoint_loader.observation_space,
        observation_horizon=checkpoint_loader.observation_horizon,
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
    )
