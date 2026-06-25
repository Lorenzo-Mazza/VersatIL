"""Tests for versatil.checkpoint_loading.float_policy module."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
from versatil.training.constants import CheckpointFilename, CheckpointKey

FLOAT_POLICY_MODULE = "versatil.checkpoint_loading.float_policy"


@pytest.mark.unit
@pytest.mark.parametrize(
    "has_tokenizer", [False, True], ids=["without_tokenizer", "with_tokenizer"]
)
def test_float_checkpoint_loader_restores_policy(
    checkpoint_config_factory,
    checkpoint_payload_factory,
    lightning_module_factory,
    has_tokenizer,
) -> None:
    checkpoint_path = "/tmp/checkpoint"
    checkpoint_name = CheckpointFilename.DEFAULT_CHECKPOINT.value
    checkpoint_file = os.path.join(checkpoint_path, checkpoint_name)
    config_path = os.path.join(checkpoint_path, CheckpointFilename.CONFIG.value)
    tokenizer_path = os.path.join(
        checkpoint_path,
        CheckpointFilename.TOKENIZER_DIR.value,
    )
    tokenizer = MagicMock() if has_tokenizer else None
    config = checkpoint_config_factory()
    checkpoint = checkpoint_payload_factory()
    state_dict = checkpoint[CheckpointKey.STATE_DICT.value]
    lightning_module = lightning_module_factory(state_dict=state_dict)

    with (
        patch(f"{FLOAT_POLICY_MODULE}.os.path.exists", return_value=True),
        patch.object(
            FloatCheckpointLoader,
            "_load_config",
            return_value=config,
        ) as mock_load_config,
        patch.object(
            FloatCheckpointLoader,
            "_load_tokenizer",
            return_value=tokenizer,
        ) as mock_load_tokenizer,
        patch(
            f"{FLOAT_POLICY_MODULE}.torch.load", return_value=checkpoint
        ) as mock_load,
        patch(
            f"{FLOAT_POLICY_MODULE}.LightningPolicy",
            return_value=lightning_module,
        ) as mock_lightning_policy,
        patch.object(
            FloatCheckpointLoader,
            "_validate_checkpoint_loading",
        ) as mock_validate,
    ):
        loader = FloatCheckpointLoader(
            device=torch.device("cpu"),
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
        )

    mock_load_config.assert_called_once_with(config_path=config_path)
    mock_load_tokenizer.assert_called_once_with(tokenizer_path=tokenizer_path)
    mock_load.assert_called_once_with(
        checkpoint_file,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    mock_lightning_policy.assert_called_once_with(
        policy=config.policy,
        training_config=config.training,
    )
    lightning_module.load_state_dict.assert_called_once_with(state_dict, strict=False)
    mock_validate.assert_called_once_with(
        checkpoint_state_dict=state_dict,
        model_state_dict=lightning_module.state_dict.return_value,
    )
    config.policy.to.assert_called_once_with(torch.device("cpu"))
    config.policy.eval.assert_called_once_with()
    if has_tokenizer:
        tokenizer.to.assert_called_once_with(torch.device("cpu"))
        config.policy.set_tokenizer.assert_called_once_with(tokenizer)
    else:
        config.policy.set_tokenizer.assert_not_called()
    assert loader.policy is config.policy
