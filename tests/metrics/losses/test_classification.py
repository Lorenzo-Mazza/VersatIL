"""Tests for versatil.metrics.losses.classification module."""

import math
import re

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.losses.classification import (
    ActionTokenLoss,
    PhaseClassificationLoss,
)
from versatil.models.decoding.constants import DecoderOutputKey


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
