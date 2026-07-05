"""Tests for versatil.metrics.losses.metadata_passthrough module."""

import pytest
import torch

from versatil.metrics.losses.metadata_passthrough import MetadataPassthrough


@pytest.mark.unit
class TestMetadataPassthroughGetRequiredKeys:
    def test_returns_target_keys(self):
        loss = MetadataPassthrough(
            keys_mapping={"phase_label": "phase_label", "extra": "extra_meta"}
        )
        assert loss.get_required_keys() == {"phase_label", "extra"}


@pytest.mark.unit
class TestMetadataPassthroughForward:
    def test_extracts_targets_into_metadata(self):
        predictions = {"dummy": torch.tensor([1.0])}
        phase_labels = torch.tensor([[0, 1, 2]])
        targets = {"phase_label": phase_labels}
        loss = MetadataPassthrough(keys_mapping={"phase_label": "phase_meta"})
        output = loss(predictions, targets)
        assert torch.equal(output.metadata["phase_meta"], phase_labels)
        assert output.total_loss.item() == pytest.approx(0.0)

    def test_missing_target_key_is_silently_skipped(self):
        predictions = {"dummy": torch.tensor([1.0])}
        targets = {}
        loss = MetadataPassthrough(keys_mapping={"missing_key": "meta"})
        output = loss(predictions, targets)
        assert "meta" not in output.metadata
        assert output.total_loss.item() == pytest.approx(0.0)
