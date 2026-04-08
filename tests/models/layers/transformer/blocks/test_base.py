"""Tests for versatil.models.layers.transformer.blocks.base module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.normalization.unconditioned_norm import UnconditionedNorm
from versatil.models.layers.transformer.blocks.base import TransformerBlock

EMBEDDING_DIMENSION = 32


class _ConcreteBlock(TransformerBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


def test_stores_normalization_and_dropout():
    norm = UnconditionedNorm(torch.nn.LayerNorm(EMBEDDING_DIMENSION))
    block = _ConcreteBlock(normalization=norm, dropout=0.3)
    assert block.normalization is norm
    assert block.residual_dropout.p == 0.3


@pytest.mark.parametrize(
    "gate_value, expected_scale",
    [(1.0, 1.0), (0.0, 0.0), (0.5, 0.5)],
    ids=["full_gate", "zero_gate", "half_gate"],
)
def test_apply_residual_scales_output_by_gate(
    sequence_tensor_factory: Callable[..., torch.Tensor],
    gate_value: float,
    expected_scale: float,
):
    norm = UnconditionedNorm(torch.nn.LayerNorm(EMBEDDING_DIMENSION))
    block = _ConcreteBlock(normalization=norm, dropout=0.0)
    residual = sequence_tensor_factory(
        batch_size=2, sequence_length=4, embedding_dimension=EMBEDDING_DIMENSION
    )
    output = sequence_tensor_factory(
        batch_size=2, sequence_length=4, embedding_dimension=EMBEDDING_DIMENSION
    )
    gate = torch.full((1,), gate_value)
    result = block.apply_residual(residual=residual, output=output, gate=gate)
    expected = residual + expected_scale * output
    assert torch.allclose(result, expected)
