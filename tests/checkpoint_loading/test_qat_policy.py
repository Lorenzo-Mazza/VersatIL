"""Tests for versatil.checkpoint_loading.qat_policy module."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.checkpoint_loading.qat_policy import _QATCheckpointLoader
from versatil.training.constants import CheckpointFilename, CheckpointKey

QAT_POLICY_MODULE = "versatil.checkpoint_loading.qat_policy"


@pytest.mark.unit
def test_qat_checkpoint_loader_prepares_model_before_weight_loading(
    checkpoint_config_factory,
    checkpoint_payload_factory,
    lightning_module_factory,
) -> None:
    checkpoint_path = "/tmp/checkpoint"
    checkpoint_name = CheckpointFilename.DEFAULT_CHECKPOINT.value
    checkpoint_file = os.path.join(checkpoint_path, checkpoint_name)
    config_path = os.path.join(checkpoint_path, CheckpointFilename.CONFIG.value)
    tokenizer_path = os.path.join(
        checkpoint_path,
        CheckpointFilename.TOKENIZER_DIR.value,
    )
    config = checkpoint_config_factory()
    quantization = MagicMock()
    checkpoint = checkpoint_payload_factory()
    state_dict = checkpoint[CheckpointKey.STATE_DICT.value]
    call_order = []
    lightning_module = lightning_module_factory(
        state_dict=state_dict,
        call_order=call_order,
    )
    materialize = MagicMock(side_effect=lambda: call_order.append("materialize"))
    quantization.prepare_model.side_effect = lambda model: call_order.append("prepare")

    with (
        patch(f"{QAT_POLICY_MODULE}.os.path.exists", return_value=True),
        patch.object(
            _QATCheckpointLoader,
            "_load_config",
            return_value=config,
        ) as mock_load_config,
        patch.object(
            _QATCheckpointLoader,
            "_load_tokenizer",
            return_value=None,
        ) as mock_load_tokenizer,
        patch.object(
            _QATCheckpointLoader,
            "_materialize_lazy_modules",
            materialize,
        ),
        patch(f"{QAT_POLICY_MODULE}.torch.load", return_value=checkpoint) as mock_load,
        patch(
            f"{QAT_POLICY_MODULE}.LightningPolicy",
            return_value=lightning_module,
        ) as mock_lightning_policy,
        patch.object(
            _QATCheckpointLoader,
            "_validate_checkpoint_loading",
        ) as mock_validate,
    ):
        loader = _QATCheckpointLoader(
            device=torch.device("cpu"),
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            quantization=quantization,
        )

    mock_load_config.assert_called_once_with(config_path=config_path)
    mock_load_tokenizer.assert_called_once_with(tokenizer_path=tokenizer_path)
    config.policy.encoding_pipeline.set_output_dtype.assert_called_once_with(
        torch.float32
    )
    quantization.prepare_model.assert_called_once_with(model=config.policy)
    mock_load.assert_called_once_with(
        checkpoint_file,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    mock_lightning_policy.assert_called_once_with(
        policy=config.policy,
        training_config=config.training,
    )
    mock_validate.assert_called_once_with(
        checkpoint_state_dict=state_dict,
        model_state_dict=lightning_module.state_dict.return_value,
        missing_keys=[],
        unexpected_keys=[],
    )
    assert call_order == ["materialize", "prepare", "load_state_dict"]
    assert loader.policy is config.policy
