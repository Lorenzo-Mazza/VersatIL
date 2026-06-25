"""Policy context loading for post-training compression."""

import logging

import torch

from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
from versatil.checkpoint_loading.qat_policy import _QATCheckpointLoader
from versatil.post_training_compression.policy_context import PolicyContext
from versatil.quantization.workflows.base import BaseQuantizationWorkflow


def load_float_policy_context(
    checkpoint_path: str,
    checkpoint_name: str,
) -> PolicyContext:
    """Load a policy context from a float checkpoint."""
    logging.info("Loading policy from %s", checkpoint_path)
    checkpoint_loader = FloatCheckpointLoader(
        device=torch.device("cpu"),
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
) -> PolicyContext:
    """Load a policy context from a QAT checkpoint."""
    checkpoint_loader = _QATCheckpointLoader(
        device=torch.device("cpu"),
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
