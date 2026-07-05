"""Tests for versatil.models.layers.normalization.rms_norm module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.normalization.rms_norm import RMSNorm


@pytest.fixture
def rms_norm_factory() -> Callable[..., RMSNorm]:
    """Factory for RMSNorm instances with configurable parameters."""

    def factory(
        normalized_shape: int = 64,
        epsilon: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> RMSNorm:
        return RMSNorm(
            normalized_shape=normalized_shape,
            epsilon=epsilon,
            elementwise_affine=elementwise_affine,
        )

    return factory


class TestRMSNormInitialization:
    @pytest.mark.parametrize("normalized_shape", [32, 128])
    @pytest.mark.parametrize("elementwise_affine", [True, False])
    @pytest.mark.parametrize("epsilon", [1e-6, 1e-8])
    def test_stores_configuration(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        normalized_shape: int,
        elementwise_affine: bool,
        epsilon: float,
    ):
        norm = rms_norm_factory(
            normalized_shape=normalized_shape,
            elementwise_affine=elementwise_affine,
            epsilon=epsilon,
        )
        assert norm.elementwise_affine == elementwise_affine
        assert norm.epsilon == epsilon

    def test_affine_weight_is_learnable_and_initialized_to_ones(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
    ):
        norm = rms_norm_factory(normalized_shape=64, elementwise_affine=True)
        assert torch.allclose(norm.weight.data, torch.ones(64))
        assert "weight" in dict(norm.named_parameters())

    def test_non_affine_has_no_learnable_parameters(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
    ):
        norm = rms_norm_factory(normalized_shape=64, elementwise_affine=False)
        assert len(list(norm.parameters())) == 0


class TestRMSNormForward:
    @pytest.mark.parametrize("feature_dimension", [64, 128])
    def test_output_shape_matches_flat_input(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        flat_tensor_factory: Callable[..., torch.Tensor],
        feature_dimension: int,
    ):
        norm = rms_norm_factory(
            normalized_shape=feature_dimension,
            elementwise_affine=True,
        )
        tensor = flat_tensor_factory(
            batch_size=2,
            feature_dimension=feature_dimension,
        )
        output = norm(tensor)
        assert output.shape == tensor.shape

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    def test_output_shape_matches_sequence_input(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        embedding_dimension: int,
    ):
        norm = rms_norm_factory(
            normalized_shape=embedding_dimension,
            elementwise_affine=True,
        )
        tensor = sequence_tensor_factory(
            batch_size=4,
            sequence_length=10,
            embedding_dimension=embedding_dimension,
        )
        output = norm(tensor)
        assert output.shape == tensor.shape

    def test_output_rms_approximately_one_without_affine(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        feature_dimension = 256
        norm = rms_norm_factory(
            normalized_shape=feature_dimension,
            elementwise_affine=False,
        )
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=feature_dimension)
        output = norm(tensor)
        rms_values = torch.sqrt(torch.mean(output**2, dim=-1))
        assert torch.allclose(rms_values, torch.ones_like(rms_values), atol=1e-4)

    def test_affine_weight_scales_rms(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        feature_dimension = 256
        norm = rms_norm_factory(
            normalized_shape=feature_dimension,
            elementwise_affine=True,
        )
        norm.weight.data.fill_(2.0)
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=feature_dimension)
        output = norm(tensor)
        # RMS-normalized input has RMS ≈ 1, multiplied by 2 gives RMS ≈ 2
        rms_values = torch.sqrt(torch.mean(output**2, dim=-1))
        assert torch.allclose(rms_values, torch.full_like(rms_values, 2.0), atol=1e-2)

    def test_different_eps_affects_near_zero_input(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
    ):
        feature_dimension = 64
        norm_small_eps = rms_norm_factory(
            normalized_shape=feature_dimension,
            epsilon=1e-6,
            elementwise_affine=False,
        )
        norm_large_eps = rms_norm_factory(
            normalized_shape=feature_dimension,
            epsilon=1.0,
            elementwise_affine=False,
        )
        # Near-zero input where eps dominates the denominator
        tensor = torch.full((2, feature_dimension), 1e-8)
        output_small = norm_small_eps(tensor)
        output_large = norm_large_eps(tensor)
        assert not torch.allclose(output_small, output_large)

    def test_gradient_flows_through_normalization(
        self,
        rms_norm_factory: Callable[..., RMSNorm],
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        feature_dimension = 64
        norm = rms_norm_factory(
            normalized_shape=feature_dimension,
            elementwise_affine=True,
        )
        tensor = flat_tensor_factory(batch_size=2, feature_dimension=feature_dimension)
        tensor.requires_grad_(True)
        output = norm(tensor)
        output.sum().backward()
        assert tensor.grad.shape == tensor.shape
        assert torch.all(torch.isfinite(tensor.grad))
