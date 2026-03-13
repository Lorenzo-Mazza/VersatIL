"""Tests for versatil.models.layers.normalization.rms_norm module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from versatil.models.layers.normalization.rms_norm import RMSNorm


@pytest.fixture
def rms_norm_factory() -> Callable[..., RMSNorm]:
    """Factory for RMSNorm instances with configurable parameters."""
    def factory(
        normalized_shape: int = 64,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> RMSNorm:
        return RMSNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )
    return factory


@pytest.fixture
def input_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for input tensors with configurable shape."""
    def factory(
        batch_size: int = 2,
        sequence_length: int | None = None,
        feature_dim: int = 64,
    ) -> torch.Tensor:
        if sequence_length is not None:
            shape = (batch_size, sequence_length, feature_dim)
        else:
            shape = (batch_size, feature_dim)
        return torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32)
        )
    return factory


class TestRMSNormInitialization:

    @pytest.mark.parametrize("normalized_shape", [32, 128])
    @pytest.mark.parametrize("elementwise_affine", [True, False])
    @pytest.mark.parametrize("eps", [1e-6, 1e-8])
    def test_stores_configuration(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        normalized_shape: int,
        elementwise_affine: bool,
        eps: float,
    ):
        norm = rms_norm_factory(
            normalized_shape=normalized_shape,
            elementwise_affine=elementwise_affine,
            eps=eps,
        )
        assert norm.elementwise_affine == elementwise_affine
        assert norm.eps == eps

    def test_weight_is_parameter_when_affine(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
    ):
        norm = rms_norm_factory(
            normalized_shape=64,
            elementwise_affine=True,
        )
        assert isinstance(norm.weight, nn.Parameter)
        assert norm.weight.shape == (64,)
        assert torch.allclose(norm.weight.data, torch.ones(64))

    def test_weight_is_none_when_not_affine(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
    ):
        norm = rms_norm_factory(
            normalized_shape=64,
            elementwise_affine=False,
        )
        assert norm.weight is None

    def test_inherits_from_nn_module(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
    ):
        norm = rms_norm_factory(normalized_shape=64, elementwise_affine=True)
        assert isinstance(norm, nn.Module)


class TestRMSNormForward:

    @pytest.mark.parametrize("batch_size, sequence_length, feature_dim", [
        (2, None, 64),
        (4, 10, 32),
        (1, None, 128),
    ])
    def test_output_shape_matches_input(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        input_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int | None,
        feature_dim: int,
    ):
        norm = rms_norm_factory(
            normalized_shape=feature_dim,
            elementwise_affine=True,
        )
        tensor = input_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dim=feature_dim,
        )
        output = norm(tensor)
        assert output.shape == tensor.shape

    def test_output_rms_approximately_one_without_affine(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 256
        norm = rms_norm_factory(
            normalized_shape=feature_dim,
            elementwise_affine=False,
        )
        tensor = input_tensor_factory(
            batch_size=4,
            feature_dim=feature_dim,
        )
        output = norm(tensor)
        rms_values = torch.sqrt(torch.mean(output ** 2, dim=-1))
        assert torch.allclose(rms_values, torch.ones_like(rms_values), atol=1e-4)

    @pytest.mark.parametrize("elementwise_affine", [True, False])
    def test_output_is_differentiable(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        input_tensor_factory: Callable[..., torch.Tensor],
        elementwise_affine: bool,
    ):
        feature_dim = 64
        norm = rms_norm_factory(
            normalized_shape=feature_dim,
            elementwise_affine=elementwise_affine,
        )
        tensor = input_tensor_factory(
            batch_size=2,
            feature_dim=feature_dim,
        )
        tensor.requires_grad_(True)
        output = norm(tensor)
        loss = output.sum()
        loss.backward()
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape

    def test_affine_modifies_output(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 64
        norm_affine = rms_norm_factory(
            normalized_shape=feature_dim,
            elementwise_affine=True,
        )
        norm_no_affine = rms_norm_factory(
            normalized_shape=feature_dim,
            elementwise_affine=False,
        )
        tensor = input_tensor_factory(
            batch_size=2,
            feature_dim=feature_dim,
        )
        output_affine = norm_affine(tensor)
        output_no_affine = norm_no_affine(tensor)
        # With default weight=ones, both should produce identical outputs
        assert torch.allclose(output_affine, output_no_affine, atol=1e-6)
        # After modifying weight, outputs should differ
        norm_affine.weight.data.fill_(2.0)
        output_scaled = norm_affine(tensor)
        assert not torch.allclose(output_scaled, output_no_affine)
