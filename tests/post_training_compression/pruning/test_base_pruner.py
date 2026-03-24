"""Tests for versatil.post_training_compression.pruning.base module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from versatil.post_training_compression.pruning.base import BasePruner


@pytest.fixture
def model_with_known_zeros_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    """Factory that creates a Linear model with a known number of zero weights."""

    def factory(
        input_features: int = 8,
        output_features: int = 4,
        zero_count: int = 10,
    ) -> nn.Module:
        model = nn.Linear(
            in_features=input_features,
            out_features=output_features,
            bias=True,
        )
        with torch.no_grad():
            data = rng.standard_normal((output_features, input_features)).astype(
                np.float32
            )
            # Ensure no accidental zeros in the non-zero part
            data = np.where(np.abs(data) < 0.01, 0.01, data)
            flat_data = data.flatten()
            # Set exactly zero_count elements to zero
            indices = rng.choice(len(flat_data), size=zero_count, replace=False)
            flat_data[indices] = 0.0
            model.weight.copy_(
                torch.from_numpy(flat_data.reshape(output_features, input_features))
            )
            # Ensure bias has no zeros
            bias_data = rng.standard_normal(output_features).astype(np.float32)
            bias_data = np.where(np.abs(bias_data) < 0.01, 0.01, bias_data)
            model.bias.copy_(torch.from_numpy(bias_data))
        return model

    return factory


@pytest.mark.unit
class TestBasePruner:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class BasePruner",
        ):
            BasePruner()


@pytest.mark.unit
class TestComputeSparsity:
    @pytest.mark.parametrize(
        "input_features, output_features, zero_count",
        [
            (8, 4, 10),
            (16, 8, 50),
        ],
    )
    def test_counts_zero_parameters_correctly(
        self,
        model_with_known_zeros_factory: Callable[..., nn.Module],
        input_features: int,
        output_features: int,
        zero_count: int,
    ):
        model = model_with_known_zeros_factory(
            input_features=input_features,
            output_features=output_features,
            zero_count=zero_count,
        )
        expected_total = (input_features * output_features) + output_features

        total_parameters, zero_parameters = BasePruner.compute_sparsity(model)

        assert total_parameters == expected_total
        assert zero_parameters == zero_count

    def test_all_nonzero_model_has_zero_sparsity(self, rng: np.random.Generator):
        model = nn.Linear(in_features=4, out_features=2, bias=False)
        with torch.no_grad():
            data = rng.standard_normal((2, 4)).astype(np.float32)
            data = np.where(np.abs(data) < 0.01, 0.01, data)
            model.weight.copy_(torch.from_numpy(data))

        total_parameters, zero_parameters = BasePruner.compute_sparsity(model)

        assert total_parameters == 8
        assert zero_parameters == 0
