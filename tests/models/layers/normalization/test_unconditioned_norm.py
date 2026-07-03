"""Tests for versatil.models.layers.normalization.unconditioned_norm module."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from versatil.models.layers.normalization.rms_norm import RMSNorm
from versatil.models.layers.normalization.unconditioned_norm import UnconditionedNorm


@pytest.fixture
def unconditioned_norm_factory() -> Callable[..., UnconditionedNorm]:
    """Factory for UnconditionedNorm instances."""

    def factory(
        dimension: int = 64,
        base_norm: nn.Module | None = None,
    ) -> UnconditionedNorm:
        if base_norm is None:
            base_norm = nn.LayerNorm(dimension)
        return UnconditionedNorm(norm=base_norm)

    return factory


class TestUnconditionedNormForward:
    def test_output_matches_base_norm(
        self,
        unconditioned_norm_factory: Callable[..., UnconditionedNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        dimension = 64
        base_norm = nn.LayerNorm(dimension)
        norm = unconditioned_norm_factory(base_norm=base_norm)
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=dimension
        )
        output, _ = norm(x=tensor)
        expected = base_norm(tensor)
        assert torch.allclose(output, expected)

    def test_gate_is_always_ones(
        self,
        unconditioned_norm_factory: Callable[..., UnconditionedNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        norm = unconditioned_norm_factory(dimension=32)
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        condition = condition_factory(batch_size=2, conditioning_dimension=16)
        _, gate_no_cond = norm(x=tensor, condition=None)
        _, gate_with_cond = norm(x=tensor, condition=condition)
        assert torch.equal(gate_no_cond, torch.ones(1, dtype=tensor.dtype))
        assert torch.equal(gate_with_cond, torch.ones(1, dtype=tensor.dtype))

    def test_condition_has_no_effect_on_output(
        self,
        unconditioned_norm_factory: Callable[..., UnconditionedNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        norm = unconditioned_norm_factory(dimension=32)
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        condition_a = condition_factory(batch_size=2, conditioning_dimension=16)
        condition_b = condition_a * 10.0
        output_none, _ = norm(x=tensor, condition=None)
        output_a, _ = norm(x=tensor, condition=condition_a)
        output_b, _ = norm(x=tensor, condition=condition_b)
        assert torch.allclose(output_none, output_a)
        assert torch.allclose(output_none, output_b)

    @pytest.mark.parametrize(
        "base_norm_class",
        [
            lambda dim: nn.LayerNorm(dim),
            lambda dim: RMSNorm(normalized_shape=dim),
        ],
        ids=["layernorm", "rmsnorm"],
    )
    def test_works_with_different_base_norms(
        self,
        unconditioned_norm_factory: Callable[..., UnconditionedNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        base_norm_class: Callable,
    ):
        dimension = 64
        base_norm = base_norm_class(dimension)
        norm = unconditioned_norm_factory(base_norm=base_norm)
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=dimension
        )
        output, gate = norm(x=tensor)
        expected = base_norm(tensor)
        assert torch.allclose(output, expected)
        assert torch.equal(gate, torch.ones(1, dtype=tensor.dtype))
