"""Tests for versatil.metrics.base module."""

import numpy as np
import pytest
import torch

from versatil.metrics.base import BaseLoss, LossOutput, reduce_loss_with_padding
from versatil.metrics.constants import MetricKey


@pytest.fixture
def loss_tensor_factory(rng):
    def factory(
        batch_size: int = 2,
        horizon: int = 4,
        extra_dims: tuple[int, ...] = (3,),
    ) -> torch.Tensor:
        shape = (batch_size, horizon) + extra_dims
        data = rng.standard_normal(shape).astype(np.float32)
        return torch.from_numpy(data)

    return factory


@pytest.mark.unit
class TestLossOutputToDict:
    def test_includes_total_loss_key(self, loss_output_factory):
        output = loss_output_factory(total_loss_value=2.5)
        result = output.to_dict()
        assert result[MetricKey.TOTAL_LOSS.value] == pytest.approx(2.5)

    def test_includes_component_losses_as_floats(self, loss_output_factory):
        output = loss_output_factory(
            total_loss_value=3.0,
            component_losses={"mse": 1.0, "l1": 2.0},
        )
        result = output.to_dict()
        assert result["mse"] == pytest.approx(1.0)
        assert result["l1"] == pytest.approx(2.0)

    def test_handles_non_tensor_component_values(self):
        output = LossOutput(
            total_loss=torch.tensor(1.0),
            component_losses={"scalar_loss": 0.75},
        )
        result = output.to_dict()
        assert result["scalar_loss"] == pytest.approx(0.75)


@pytest.mark.unit
class TestLossOutputAdd:
    def test_sums_total_losses(self, loss_output_factory):
        output_a = loss_output_factory(total_loss_value=1.0)
        output_b = loss_output_factory(total_loss_value=2.0)
        combined = output_a + output_b
        assert combined.total_loss.item() == pytest.approx(3.0)

    def test_sums_shared_component_losses(self, loss_output_factory):
        output_a = loss_output_factory(
            total_loss_value=1.0,
            component_losses={"mse": 0.5},
        )
        output_b = loss_output_factory(
            total_loss_value=1.0,
            component_losses={"mse": 0.3},
        )
        combined = output_a + output_b
        assert combined.component_losses["mse"].item() == pytest.approx(0.8)

    def test_handles_disjoint_component_keys(self, loss_output_factory):
        output_a = loss_output_factory(
            total_loss_value=1.0,
            component_losses={"mse": 0.5},
        )
        output_b = loss_output_factory(
            total_loss_value=1.0,
            component_losses={"l1": 0.3},
        )
        combined = output_a + output_b
        assert combined.component_losses["mse"].item() == pytest.approx(0.5)
        assert combined.component_losses["l1"].item() == pytest.approx(0.3)

    def test_merges_metadata_with_later_overriding(self, loss_output_factory):
        output_a = loss_output_factory(
            total_loss_value=1.0,
            metadata={"key_a": "value_a", "shared": "from_a"},
        )
        output_b = loss_output_factory(
            total_loss_value=1.0,
            metadata={"key_b": "value_b", "shared": "from_b"},
        )
        combined = output_a + output_b
        assert combined.metadata["key_a"] == "value_a"
        assert combined.metadata["key_b"] == "value_b"
        assert combined.metadata["shared"] == "from_b"

    def test_raises_type_error_for_non_loss_output(self, loss_output_factory):
        output = loss_output_factory(total_loss_value=1.0)
        with pytest.raises(
            TypeError,
            match=f"Cannot add LossOutput with {int}",
        ):
            output + 42

    def test_result_supports_gradient_flow(self, loss_output_factory):
        tensor_a = torch.tensor(1.0, requires_grad=True)
        tensor_b = torch.tensor(2.0, requires_grad=True)
        output_a = LossOutput(total_loss=tensor_a)
        output_b = LossOutput(total_loss=tensor_b)
        combined = output_a + output_b
        combined.total_loss.backward()
        assert tensor_a.grad is not None
        assert tensor_b.grad is not None


@pytest.mark.unit
class TestLossOutputScale:
    def test_scales_total_loss(self, loss_output_factory):
        output = loss_output_factory(total_loss_value=4.0)
        scaled = output.scale(0.5)
        assert scaled.total_loss.item() == pytest.approx(2.0)

    def test_scales_component_losses(self, loss_output_factory):
        output = loss_output_factory(
            total_loss_value=4.0,
            component_losses={"mse": 2.0, "l1": 1.0},
        )
        scaled = output.scale(0.25)
        assert scaled.component_losses["mse"].item() == pytest.approx(0.5)
        assert scaled.component_losses["l1"].item() == pytest.approx(0.25)

    def test_preserves_metadata_reference(self, loss_output_factory):
        metadata = {"key": "value"}
        output = loss_output_factory(
            total_loss_value=1.0,
            metadata=metadata,
        )
        scaled = output.scale(2.0)
        assert scaled.metadata is metadata

    def test_scale_by_zero_produces_zero_losses(self, loss_output_factory):
        output = loss_output_factory(
            total_loss_value=5.0,
            component_losses={"mse": 3.0},
        )
        scaled = output.scale(0.0)
        assert scaled.total_loss.item() == pytest.approx(0.0)
        assert scaled.component_losses["mse"].item() == pytest.approx(0.0)


@pytest.mark.unit
class TestReduceLossWithPaddingNoMask:
    def test_mean_reduction_without_mask(self, loss_tensor_factory):
        loss = loss_tensor_factory(batch_size=2, horizon=4, extra_dims=(3,))
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=None, reduction="mean"
        )
        assert result.item() == pytest.approx(loss.mean().item())

    def test_sum_reduction_without_mask(self, loss_tensor_factory):
        loss = loss_tensor_factory(batch_size=2, horizon=4, extra_dims=(3,))
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=None, reduction="sum"
        )
        assert result.item() == pytest.approx(loss.sum().item())

    def test_none_reduction_returns_original(self, loss_tensor_factory):
        loss = loss_tensor_factory(batch_size=2, horizon=4, extra_dims=(3,))
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=None, reduction="none"
        )
        assert torch.equal(result, loss)


@pytest.mark.unit
class TestReduceLossWithPaddingMasked:
    def test_mean_reduction_excludes_padded_positions(self, padding_mask_factory):
        batch_size, horizon, dim = 2, 4, 3
        padded_from = 2
        loss = torch.ones(batch_size, horizon, dim)
        is_pad = padding_mask_factory(
            batch_size=batch_size, sequence_length=horizon, padded_from=padded_from
        )
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=is_pad, reduction="mean"
        )
        # masked_loss.sum() = batch_size * padded_from * dim = 2*2*3 = 12
        # pad_mask after unsqueeze is (B, horizon, 1), so pad_mask.sum() = batch_size * padded_from = 4
        # result = 12 / 4 = 3.0 (averages over valid positions, not elements)
        valid_positions = batch_size * padded_from
        valid_elements = valid_positions * dim
        expected = valid_elements / valid_positions
        assert result.item() == pytest.approx(expected, abs=1e-6)

    def test_mean_reduction_with_scalar_loss_per_position(self, padding_mask_factory):
        batch_size, horizon = 2, 4
        padded_from = 2
        # No extra dim: loss is (B, horizon) so mask and loss have same shape
        loss = torch.ones(batch_size, horizon)
        is_pad = padding_mask_factory(
            batch_size=batch_size, sequence_length=horizon, padded_from=padded_from
        )
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=is_pad, reduction="mean"
        )
        # sum = 2*2 = 4, mask_sum = 2*2 = 4, result = 1.0
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_sum_reduction_excludes_padded_positions(self, padding_mask_factory):
        batch_size, horizon, dim = 2, 4, 3
        padded_from = 2
        loss = torch.ones(batch_size, horizon, dim)
        is_pad = padding_mask_factory(
            batch_size=batch_size, sequence_length=horizon, padded_from=padded_from
        )
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=is_pad, reduction="sum"
        )
        expected = batch_size * padded_from * dim
        assert result.item() == pytest.approx(float(expected))

    def test_none_reduction_zeros_out_padded_positions(self, padding_mask_factory):
        batch_size, horizon, dim = 2, 4, 3
        padded_from = 2
        loss = torch.ones(batch_size, horizon, dim)
        is_pad = padding_mask_factory(
            batch_size=batch_size, sequence_length=horizon, padded_from=padded_from
        )
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=is_pad, reduction="none"
        )
        # Valid positions should be 1.0, padded should be 0.0
        assert torch.all(result[:, :padded_from] == 1.0)
        assert torch.all(result[:, padded_from:] == 0.0)

    def test_mask_broadcasts_over_extra_dimensions(self, padding_mask_factory):
        batch_size, horizon = 2, 6
        padded_from = 3
        # Loss has extra dimension (e.g., action_dim)
        loss = torch.ones(batch_size, horizon, 5)
        is_pad = padding_mask_factory(
            batch_size=batch_size, sequence_length=horizon, padded_from=padded_from
        )
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=is_pad, reduction="none"
        )
        # Mask should broadcast: padded positions zeroed across all action dims
        assert result.shape == (batch_size, horizon, 5)
        assert torch.all(result[:, padded_from:, :] == 0.0)

    def test_all_padded_returns_near_zero_mean(self):
        batch_size, horizon, dim = 2, 4, 3
        loss = torch.ones(batch_size, horizon, dim)
        is_pad = torch.ones(batch_size, horizon, dtype=torch.bool)
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=is_pad, reduction="mean"
        )
        # Division by epsilon (1e-8) should give near-zero
        assert result.item() == pytest.approx(0.0, abs=1e-2)

    def test_gradient_flows_through_masked_reduction(self, padding_mask_factory):
        batch_size, horizon, dim = 2, 4, 3
        padded_from = 2
        loss = torch.ones(batch_size, horizon, dim, requires_grad=True)
        is_pad = padding_mask_factory(
            batch_size=batch_size, sequence_length=horizon, padded_from=padded_from
        )
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=is_pad, reduction="mean"
        )
        result.backward()
        assert loss.grad is not None
        # Gradients should be zero for padded positions
        assert torch.all(loss.grad[:, padded_from:] == 0.0)
        # Gradients should be non-zero for valid positions
        assert torch.all(loss.grad[:, :padded_from] != 0.0)


@pytest.mark.unit
class TestBaseLossIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseLoss()

    def test_subclass_must_implement_forward_and_get_required_keys(self):
        class IncompleteLoss(BaseLoss):
            pass

        with pytest.raises(TypeError):
            IncompleteLoss()
