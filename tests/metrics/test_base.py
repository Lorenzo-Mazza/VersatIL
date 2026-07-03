"""Tests for versatil.metrics.base module."""

import numpy as np
import pytest
import torch

from versatil.metrics.base import (
    BaseLoss,
    LossOutput,
    ScalarWeightedLoss,
    _merge_weights,
    reduce_loss_with_padding,
)
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
    def test_mean_reduction_excludes_padded_positions(
        self, loss_tensor_factory, padding_mask_factory
    ):
        batch_size, horizon, dim = 2, 4, 3
        padded_from = 2
        loss = loss_tensor_factory(
            batch_size=batch_size, horizon=horizon, extra_dims=(dim,)
        )
        is_pad = padding_mask_factory(
            batch_size=batch_size, sequence_length=horizon, padded_from=padded_from
        )
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=is_pad, reduction="mean"
        )
        expected = loss[:, :padded_from].mean()
        assert result.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_mean_reduction_matches_unmasked_scale(
        self, loss_tensor_factory, padding_mask_factory
    ):
        batch_size, horizon, dim = 2, 4, 3
        loss = loss_tensor_factory(
            batch_size=batch_size, horizon=horizon, extra_dims=(dim,)
        )
        all_valid = padding_mask_factory(
            batch_size=batch_size, sequence_length=horizon, padded_from=horizon
        )
        result = reduce_loss_with_padding(
            loss_tensor=loss, is_pad=all_valid, reduction="mean"
        )
        assert result.item() == pytest.approx(loss.mean().item(), abs=1e-6)

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


@pytest.mark.unit
class TestMergeWeights:
    def test_flat_override_replaces_existing_values(self):
        existing = {"weight": 1.0, "other_weight": 0.5}
        override = {"weight": 0.2}
        merged = _merge_weights(existing, override)
        assert merged == {"weight": 0.2, "other_weight": 0.5}

    def test_nested_override_deep_merges(self):
        existing = {
            "denoising_prior": {"weight": 0.03},
            "regression_loss": {"mse_weight": 1.0, "l1_weight": 0.0},
        }
        override = {"regression_loss": {"mse_weight": 0.5}}
        merged = _merge_weights(existing, override)
        assert merged == {
            "denoising_prior": {"weight": 0.03},
            "regression_loss": {"mse_weight": 0.5, "l1_weight": 0.0},
        }

    def test_unknown_top_level_key_raises(self):
        with pytest.raises(KeyError, match="Unknown weight key 'bogus'"):
            _merge_weights({"weight": 1.0}, {"bogus": 0.1})

    def test_unknown_nested_key_raises(self):
        with pytest.raises(KeyError, match="Unknown weight key 'bogus_weight'"):
            _merge_weights(
                {"regression_loss": {"mse_weight": 1.0}},
                {"regression_loss": {"bogus_weight": 0.5}},
            )

    def test_scalar_override_into_nested_raises_structure_mismatch(self):
        existing = {"regression_loss": {"mse_weight": 1.0}}
        override = {"regression_loss": 0.0}
        with pytest.raises(
            TypeError,
            match=(
                "Weight override for 'regression_loss' expects a dict subtree, "
                "got float."
            ),
        ):
            _merge_weights(existing, override)

    def test_dict_override_into_scalar_raises_structure_mismatch(self):
        existing = {"weight": 1.0}
        override = {"weight": {"nested": 0.5}}
        with pytest.raises(
            TypeError,
            match=(
                "Weight override for 'weight' expects a scalar, got a dict subtree."
            ),
        ):
            _merge_weights(existing, override)


@pytest.mark.unit
class TestBaseLossWeights:
    class _LeafLoss(BaseLoss):
        def forward(self, predictions, targets, is_pad=None):
            return LossOutput(total_loss=torch.tensor(0.0))

        def get_required_keys(self):
            return set()

    def test_weights_default_is_empty_dict(self):
        loss = self._LeafLoss()
        assert loss.weights == {}

    def test_set_weights_empty_is_noop(self):
        loss = self._LeafLoss()
        loss.set_weights({})

    def test_set_weights_with_content_raises_for_default_subclass(self):
        loss = self._LeafLoss()
        with pytest.raises(
            KeyError,
            match=r"_LeafLoss\.set_weights: missing=\[\], extra=\['weight'\]",
        ):
            loss.set_weights({"weight": 1.0})

    def test_update_weights_empty_is_noop(self):
        loss = self._LeafLoss()
        loss.update_weights({})

    def test_update_weights_unknown_key_raises(self):
        loss = self._LeafLoss()
        with pytest.raises(KeyError, match="Unknown weight key 'weight'"):
            loss.update_weights({"weight": 1.0})


@pytest.mark.unit
class TestScalarWeightedLoss:
    class _ScalarLeaf(ScalarWeightedLoss):
        def __init__(self, weight: float = 1.0):
            super().__init__()
            self.weight = weight

        def forward(self, predictions, targets, is_pad=None):
            return LossOutput(total_loss=torch.tensor(self.weight))

        def get_required_keys(self):
            return set()

    def test_weights_returns_single_weight_dict(self):
        loss = self._ScalarLeaf(weight=0.7)
        assert loss.weights == {"weight": 0.7}

    def test_set_weights_replaces_weight(self):
        loss = self._ScalarLeaf(weight=0.5)
        loss.set_weights({"weight": 0.2})
        assert loss.weight == pytest.approx(0.2)
        assert loss.weights == {"weight": 0.2}

    def test_update_weights_replaces_weight(self):
        loss = self._ScalarLeaf(weight=0.5)
        loss.update_weights({"weight": 0.9})
        assert loss.weight == pytest.approx(0.9)

    def test_set_weights_missing_required_key_raises(self):
        loss = self._ScalarLeaf(weight=0.5)
        with pytest.raises(KeyError):
            loss.set_weights({})

    def test_update_weights_rejects_unknown_key(self):
        loss = self._ScalarLeaf(weight=0.5)
        with pytest.raises(KeyError, match="Unknown weight key 'bogus'"):
            loss.update_weights({"bogus": 0.1})
