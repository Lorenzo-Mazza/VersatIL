"""Tests for versatil.metrics.losses.classification module."""

import math
import re
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.losses.classification import (
    ActionTokenLoss,
    PhaseClassificationLoss,
)
from versatil.models.decoding.constants import DecoderOutputKey


@pytest.fixture
def tokenizer_with_action_ids_factory() -> Callable[[list[int]], MagicMock]:
    def factory(action_token_ids: list[int]) -> MagicMock:
        action_tokenizer = MagicMock()
        action_tokenizer.action_discretizer.token_count = len(action_token_ids)
        action_tokenizer.token_id_mapping.encode.return_value = np.asarray(
            action_token_ids
        )
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.action_tokenizer = action_tokenizer
        return tokenizer

    return factory


@pytest.mark.unit
class TestPhaseClassificationLossGetRequiredKeys:
    def test_returns_phase_key(self):
        loss = PhaseClassificationLoss(key="phase_label")
        assert loss.get_required_keys() == {"phase_label"}


@pytest.mark.unit
class TestPhaseClassificationLossForward:
    def test_perfect_predictions_produce_low_cross_entropy(self):
        batch_size, horizon, num_phases = 2, 3, 4
        labels = torch.zeros(batch_size, horizon, dtype=torch.long)
        logits = torch.zeros(batch_size, horizon, num_phases)
        logits[:, :, 0] = 100.0  # strong signal for class 0
        predictions = {"phase_label": logits}
        targets = {"phase_label": labels}
        loss = PhaseClassificationLoss(
            key="phase_label",
            cross_entropy_weight=1.0,
            entropy_weight=0.0,
            label_smoothing=0.0,
        )
        output = loss(predictions, targets)
        assert output.total_loss.item() < 0.01

    def test_random_predictions_produce_higher_loss(self, rng):
        batch_size, horizon, num_phases = 4, 5, 3
        logits_data = rng.standard_normal((batch_size, horizon, num_phases)).astype(
            np.float32
        )
        logits = torch.from_numpy(logits_data)
        labels = torch.zeros(batch_size, horizon, dtype=torch.long)
        predictions = {"phase_label": logits}
        targets = {"phase_label": labels}
        loss = PhaseClassificationLoss(
            key="phase_label",
            cross_entropy_weight=1.0,
            entropy_weight=0.0,
            label_smoothing=0.0,
        )
        output = loss(predictions, targets)
        assert output.total_loss.item() > 0.1

    def test_entropy_regularization_subtracts_from_loss(self, rng):
        batch_size, horizon, num_phases = 4, 5, 3
        logits_data = rng.standard_normal((batch_size, horizon, num_phases)).astype(
            np.float32
        )
        logits = torch.from_numpy(logits_data)
        labels = torch.zeros(batch_size, horizon, dtype=torch.long)
        predictions = {"phase_label": logits}
        targets = {"phase_label": labels}
        loss_no_entropy = PhaseClassificationLoss(
            key="phase_label", cross_entropy_weight=1.0, entropy_weight=0.0
        )
        loss_with_entropy = PhaseClassificationLoss(
            key="phase_label", cross_entropy_weight=1.0, entropy_weight=1.0
        )
        output_no = loss_no_entropy(predictions, targets)
        output_with = loss_with_entropy(predictions, targets)
        # Entropy term is subtracted, so loss_with < loss_no
        assert output_with.total_loss.item() < output_no.total_loss.item()

    def test_squeezed_trailing_dim_labels(self):
        batch_size, horizon, num_phases = 2, 3, 4
        logits = torch.zeros(batch_size, horizon, num_phases)
        logits[:, :, 0] = 100.0
        labels = torch.zeros(batch_size, horizon, 1, dtype=torch.long)  # (B, T, 1)
        loss = PhaseClassificationLoss(
            key="phase_label",
            cross_entropy_weight=1.0,
            entropy_weight=0.0,
            label_smoothing=0.0,
        )
        output = loss({"phase_label": logits}, {"phase_label": labels})
        assert output.total_loss.item() < 0.01

    def test_metadata_includes_logits_and_labels(self):
        logits = torch.zeros(2, 3, 4)
        labels = torch.zeros(2, 3, dtype=torch.long)
        loss = PhaseClassificationLoss(key="phase_label")
        output = loss({"phase_label": logits}, {"phase_label": labels})
        assert MetadataKey.PHASE_LOGITS.value in output.metadata
        assert MetadataKey.PHASE_LABEL.value in output.metadata

    def test_raises_on_missing_key(self):
        loss = PhaseClassificationLoss(key="phase_label")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions and targets must contain key 'phase_label' for PhaseClassificationLoss."
            ),
        ):
            loss({"wrong": torch.zeros(1)}, {"wrong": torch.zeros(1)})


@pytest.mark.unit
class TestActionTokenLossGetRequiredKeys:
    def test_returns_action_logits_key(self):
        loss = ActionTokenLoss()
        assert loss.get_required_keys() == {DecoderOutputKey.ACTION_LOGITS.value}


@pytest.mark.unit
class TestActionTokenLossForward:
    def test_perfect_predictions_produce_zero_loss(self):
        vocab_size = 10
        batch_size, horizon = 2, 3
        target_tokens = torch.zeros(batch_size, horizon, dtype=torch.long)
        logits = torch.zeros(batch_size, horizon, vocab_size)
        logits[:, :, 0] = 100.0
        predictions = {DecoderOutputKey.ACTION_LOGITS.value: logits}
        targets = {SampleKey.TOKENIZED_ACTIONS.value: target_tokens}
        loss = ActionTokenLoss(label_smoothing=0.0)
        output = loss(predictions, targets)
        assert output.total_loss.item() < 0.01
        assert output.component_losses[
            MetricKey.TOKEN_ACCURACY.value
        ].item() == pytest.approx(1.0)

    def test_random_predictions_have_low_accuracy(self, rng):
        vocab_size = 100
        batch_size, horizon = 4, 10
        logits_data = rng.standard_normal((batch_size, horizon, vocab_size)).astype(
            np.float32
        )
        logits = torch.from_numpy(logits_data)
        target_tokens = torch.zeros(batch_size, horizon, dtype=torch.long)
        predictions = {DecoderOutputKey.ACTION_LOGITS.value: logits}
        targets = {SampleKey.TOKENIZED_ACTIONS.value: target_tokens}
        loss = ActionTokenLoss(label_smoothing=0.0)
        output = loss(predictions, targets)
        assert output.component_losses[MetricKey.TOKEN_ACCURACY.value].item() < 0.5

    def test_perplexity_is_exp_of_cross_entropy(self, rng):
        vocab_size = 10
        batch_size, horizon = 2, 4
        logits_data = rng.standard_normal((batch_size, horizon, vocab_size)).astype(
            np.float32
        )
        logits = torch.from_numpy(logits_data)
        target_tokens = torch.zeros(batch_size, horizon, dtype=torch.long)
        predictions = {DecoderOutputKey.ACTION_LOGITS.value: logits}
        targets = {SampleKey.TOKENIZED_ACTIONS.value: target_tokens}
        loss = ActionTokenLoss(label_smoothing=0.0)
        output = loss(predictions, targets)
        ce = output.component_losses[MetricKey.ACTION_TOKEN_CROSS_ENTROPY.value].item()
        perplexity = output.component_losses[MetricKey.PERPLEXITY.value].item()
        assert perplexity == pytest.approx(math.exp(ce), rel=1e-4)

    def test_raises_on_missing_logits(self):
        loss = ActionTokenLoss()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain keys '{DecoderOutputKey.ACTION_LOGITS.value}' for ActionTokenLoss."
            ),
        ):
            loss({}, {SampleKey.TOKENIZED_ACTIONS.value: torch.zeros(1)})

    def test_action_only_loss_excludes_other_vocabulary_ids_and_eos_position(
        self,
        tokenizer_with_action_ids_factory: Callable[[list[int]], MagicMock],
    ) -> None:
        logits = torch.zeros(1, 3, 10)
        logits[0, 0, 4] = 5.0
        logits[0, 1, 1] = 5.0
        logits[:, :, 9] = 100.0
        logits.requires_grad_()
        target_tokens = torch.tensor([[4, 1, 9]])
        loss = ActionTokenLoss(
            label_smoothing=0.0,
            restrict_to_action_tokens=True,
        )
        loss.set_tokenizer(tokenizer=tokenizer_with_action_ids_factory([4, 1, 7]))

        output = loss(
            predictions={DecoderOutputKey.ACTION_LOGITS.value: logits},
            targets={SampleKey.TOKENIZED_ACTIONS.value: target_tokens},
        )
        expected_loss = -math.log(math.exp(5.0) / (math.exp(5.0) + 2.0))

        assert output.total_loss.item() == pytest.approx(expected_loss)
        assert output.component_losses[
            MetricKey.TOKEN_ACCURACY.value
        ].item() == pytest.approx(1.0)
        output.total_loss.backward()
        torch.testing.assert_close(logits.grad[:, :, 9], torch.zeros(1, 3))
        torch.testing.assert_close(logits.grad[:, 2, :], torch.zeros(1, 10))

    def test_soft_targets_form_gaussian_over_local_action_bins(
        self,
        tokenizer_with_action_ids_factory: Callable[[list[int]], MagicMock],
    ) -> None:
        logits = torch.zeros(1, 1, 8)
        logits[0, 0, [4, 1, 7]] = torch.tensor([1.0, 2.0, 3.0])
        loss = ActionTokenLoss(
            label_smoothing=0.0,
            restrict_to_action_tokens=True,
            soft_target_std=1.0,
        )
        loss.set_tokenizer(tokenizer=tokenizer_with_action_ids_factory([4, 1, 7]))

        output = loss(
            predictions={DecoderOutputKey.ACTION_LOGITS.value: logits},
            targets={SampleKey.TOKENIZED_ACTIONS.value: torch.tensor([[1]])},
        )
        soft_targets = torch.softmax(torch.tensor([-0.5, 0.0, -0.5]), dim=0)
        expected_loss = -(
            soft_targets * torch.log_softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0)
        ).sum()

        torch.testing.assert_close(output.total_loss, expected_loss)

    def test_action_only_loss_requires_tokenizer(self) -> None:
        loss = ActionTokenLoss(restrict_to_action_tokens=True)

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "ActionTokenLoss with restrict_to_action_tokens=True requires "
                "set_tokenizer() before forward()."
            ),
        ):
            loss(
                predictions={
                    DecoderOutputKey.ACTION_LOGITS.value: torch.zeros(1, 1, 3)
                },
                targets={
                    SampleKey.TOKENIZED_ACTIONS.value: torch.zeros(
                        1, 1, dtype=torch.long
                    )
                },
            )


@pytest.mark.unit
class TestActionTokenLossInitialization:
    @pytest.mark.parametrize("soft_target_std", [-1.0, -0.1])
    def test_rejects_negative_soft_target_std(self, soft_target_std: float) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("soft_target_std must be non-negative."),
        ):
            ActionTokenLoss(soft_target_std=soft_target_std)

    def test_soft_targets_require_action_only_loss(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("soft_target_std requires restrict_to_action_tokens=True."),
        ):
            ActionTokenLoss(soft_target_std=1.0)

    def test_soft_targets_cannot_be_combined_with_label_smoothing(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "soft_target_std and label_smoothing cannot both be positive."
            ),
        ):
            ActionTokenLoss(
                label_smoothing=0.1,
                restrict_to_action_tokens=True,
                soft_target_std=1.0,
            )
