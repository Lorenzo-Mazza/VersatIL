"""Tests for versatil.post_training_compression.policy_loading module."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.post_training_compression import policy_loading
from versatil.post_training_compression.policy_context import PolicyContext
from versatil.training.constants import CheckpointFilename

POLICY_LOADING_MODULE = "versatil.post_training_compression.policy_loading"


@pytest.mark.unit
@pytest.mark.parametrize(
    "function_name, loader_name, uses_quantization",
    [
        ("load_float_policy_context", "FloatCheckpointLoader", False),
        ("load_qat_policy_context", "_QATCheckpointLoader", True),
    ],
    ids=["float", "qat"],
)
def test_policy_context_loaders_build_context(
    function_name,
    loader_name,
    uses_quantization,
) -> None:
    checkpoint_path = "/tmp/checkpoint"
    checkpoint_name = CheckpointFilename.DEFAULT_CHECKPOINT.value
    loader = MagicMock()
    loader.policy = MagicMock()
    loader.config = MagicMock()
    loader.tokenizer = MagicMock()
    loader.observation_space = MagicMock()
    loader.observation_horizon = 2
    quantization = MagicMock()

    with patch(
        f"{POLICY_LOADING_MODULE}.{loader_name}",
        return_value=loader,
    ) as loader_class:
        load_context = getattr(policy_loading, function_name)
        if uses_quantization:
            context = load_context(
                checkpoint_path=checkpoint_path,
                checkpoint_name=checkpoint_name,
                quantization=quantization,
            )
        else:
            context = load_context(
                checkpoint_path=checkpoint_path,
                checkpoint_name=checkpoint_name,
            )

    expected_call_kwargs = {
        "device": torch.device("cpu"),
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": checkpoint_name,
    }
    if uses_quantization:
        expected_call_kwargs["quantization"] = quantization
    loader_class.assert_called_once_with(**expected_call_kwargs)
    assert isinstance(context, PolicyContext)
    assert context.policy is loader.policy
    assert context.config is loader.config
    assert context.tokenizer is loader.tokenizer
    assert context.observation_space is loader.observation_space
    assert context.observation_horizon == loader.observation_horizon
    assert context.checkpoint_path == checkpoint_path
    assert context.checkpoint_name == checkpoint_name
