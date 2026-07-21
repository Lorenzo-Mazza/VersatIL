"""Tests for versatil.models.decoding.action_heads.gaussian module."""

import re
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch

from versatil.models.decoding.action_heads.blocks import MLPBlock
from versatil.models.decoding.action_heads.gaussian import GaussianHead
from versatil.models.decoding.constants import DecoderOutputKey


@pytest.mark.unit
@pytest.mark.parametrize("min_logvar", [-10.0, -5.0])
@pytest.mark.parametrize("max_logvar", [4.0, 2.0])
def test_gaussian_head_stores_configuration(
    gaussian_head_factory: Callable[..., GaussianHead],
    min_logvar: float,
    max_logvar: float,
):
    head = gaussian_head_factory(min_logvar=min_logvar, max_logvar=max_logvar)
    assert head.min_logvar == min_logvar
    assert head.max_logvar == max_logvar


@pytest.mark.unit
class TestGaussianHeadSetOutputDim:
    def test_creates_mean_projection(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
    ):
        head = gaussian_head_factory(input_dimension=64)
        head.set_output_dim(3)
        assert head.output_proj is not None
        assert head.output_proj.out_features == 3

    def test_creates_logvar_projection(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
    ):
        head = gaussian_head_factory(input_dimension=64)
        head.set_output_dim(3)
        assert head._logvar_proj is not None
        assert head._logvar_proj.out_features == 3

    def test_projection_uses_last_block_dim(self):
        blocks = [MLPBlock(input_dimension=64, hidden_dimensions=[32])]
        head = GaussianHead(input_dimension=64, blocks=blocks)
        head.set_output_dim(3)
        assert head.output_proj.in_features == 32
        assert head._logvar_proj.in_features == 32


@pytest.mark.unit
class TestGaussianHeadForward:
    def test_raises_if_output_dim_not_set(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        head = gaussian_head_factory()
        embedding = embedding_tensor_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape("output_dim not set. Call set_output_dim() first."),
        ):
            head(embedding)

    def test_returns_dict_with_mean_and_logvar(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        head = gaussian_head_factory(input_dimension=64)
        head.set_output_dim(3)
        embedding = embedding_tensor_factory(embedding_dimension=64)
        result = head(embedding)
        assert isinstance(result, dict)
        assert DecoderOutputKey.MEAN.value in result
        assert DecoderOutputKey.LOGVAR.value in result

    @pytest.mark.parametrize("output_dim", [3, 7])
    def test_output_shapes(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
        output_dim: int,
    ):
        head = gaussian_head_factory(input_dimension=64)
        head.set_output_dim(output_dim)
        embedding = embedding_tensor_factory(embedding_dimension=64)
        result = head(embedding)
        assert result[DecoderOutputKey.MEAN.value].shape == (2, 8, output_dim)
        assert result[DecoderOutputKey.LOGVAR.value].shape == (2, 8, output_dim)

    def test_logvar_clamped_within_bounds(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        min_logvar = -5.0
        max_logvar = 2.0
        head = gaussian_head_factory(
            input_dimension=64,
            min_logvar=min_logvar,
            max_logvar=max_logvar,
        )
        head.set_output_dim(3)
        # Use large embeddings to push logvar towards extremes
        embedding = embedding_tensor_factory(embedding_dimension=64) * 100
        result = head(embedding)
        logvar = result[DecoderOutputKey.LOGVAR.value]
        assert logvar.min().item() >= min_logvar
        assert logvar.max().item() <= max_logvar

    def test_mean_equals_mean_projection_of_applied_blocks(
        self,
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        blocks = [MLPBlock(input_dimension=64, hidden_dimensions=[32])]
        head = GaussianHead(input_dimension=64, blocks=blocks)
        head.set_output_dim(3)
        head.eval()
        embedding = embedding_tensor_factory(embedding_dimension=64)
        expected_mean = head.output_proj(head._apply_blocks(embedding))
        result = head(embedding)
        torch.testing.assert_close(result[DecoderOutputKey.MEAN.value], expected_mean)

    def test_mean_and_logvar_use_independent_projections(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ):
        head = gaussian_head_factory(
            input_dimension=64, max_logvar=100.0, min_logvar=-100.0
        )
        head.set_output_dim(3)
        head.eval()
        embedding = embedding_tensor_factory(embedding_dimension=64)
        baseline = head(embedding)[DecoderOutputKey.MEAN.value].clone()
        # Mutating only the logvar projection must leave the mean untouched.
        with torch.no_grad():
            head._logvar_proj.weight.add_(5.0)
            head._logvar_proj.bias.add_(5.0)
        mutated = head(embedding)
        torch.testing.assert_close(mutated[DecoderOutputKey.MEAN.value], baseline)
        assert not torch.allclose(
            mutated[DecoderOutputKey.LOGVAR.value],
            head.output_proj(embedding),
        )


class TestGaussianHeadTemporalBias:
    @pytest.mark.unit
    def test_enable_before_set_output_dim_raises(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
    ) -> None:
        head = gaussian_head_factory(input_dimension=64)
        with pytest.raises(
            RuntimeError,
            match=re.escape("output_dim not set. Call set_output_dim() first."),
        ):
            head.enable_temporal_bias(horizon=8)

    @pytest.mark.parametrize("horizon", [1, 8])
    @pytest.mark.parametrize("output_dim", [3, 7])
    @pytest.mark.unit
    def test_creates_zero_bias_with_horizon_shape(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
        horizon: int,
        output_dim: int,
    ) -> None:
        head = gaussian_head_factory(input_dimension=64, output_dim=output_dim)
        head.enable_temporal_bias(horizon=horizon)
        assert head.temporal_bias.shape == (horizon, output_dim)
        torch.testing.assert_close(
            head.temporal_bias,
            torch.zeros(horizon, output_dim),
            atol=0,
            rtol=0,
        )
        assert head.temporal_bias.requires_grad is True

    @pytest.mark.unit
    def test_forward_adds_temporal_bias_after_mean_projection(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ) -> None:
        horizon = 8
        output_dimension = 3
        head = gaussian_head_factory(input_dimension=64, output_dim=output_dimension)
        head.enable_temporal_bias(horizon=horizon)
        embedding = embedding_tensor_factory(
            embedding_dimension=64,
            prediction_horizon=horizon,
        )
        projected_mean = torch.zeros(2, horizon, output_dimension)
        projected_logvar = torch.zeros_like(projected_mean)
        temporal_bias = torch.arange(
            horizon * output_dimension,
            dtype=torch.float32,
        ).view(horizon, output_dimension)
        with (
            torch.no_grad(),
            patch.object(head, "_apply_blocks", return_value=embedding) as apply_blocks,
            patch.object(
                head.output_proj,
                "forward",
                return_value=projected_mean,
            ) as mean_projection,
            patch.object(
                head._logvar_proj,
                "forward",
                return_value=projected_logvar,
            ) as logvar_projection,
        ):
            head.temporal_bias.copy_(temporal_bias)
            result = head(embedding)

        apply_blocks.assert_called_once_with(embedding)
        mean_projection.assert_called_once_with(embedding)
        logvar_projection.assert_called_once_with(embedding)
        torch.testing.assert_close(
            result[DecoderOutputKey.MEAN.value],
            projected_mean + temporal_bias,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            result[DecoderOutputKey.LOGVAR.value],
            projected_logvar,
            atol=0,
            rtol=0,
        )

    @pytest.mark.integration
    def test_zero_bias_keeps_mean_unchanged(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ) -> None:
        head = gaussian_head_factory(input_dimension=64, output_dim=3)
        head.eval()
        embedding = embedding_tensor_factory(
            embedding_dimension=64, prediction_horizon=8
        )
        baseline = head(embedding)[DecoderOutputKey.MEAN.value].clone()
        head.enable_temporal_bias(horizon=8)
        result = head(embedding)
        torch.testing.assert_close(result[DecoderOutputKey.MEAN.value], baseline)

    @pytest.mark.integration
    def test_bias_shifts_mean_per_timestep(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
    ) -> None:
        horizon = 8
        output_dim = 3
        head = gaussian_head_factory(input_dimension=64, output_dim=output_dim)
        head.eval()
        embedding = embedding_tensor_factory(
            embedding_dimension=64, prediction_horizon=horizon
        )
        baseline = head(embedding)[DecoderOutputKey.MEAN.value].clone()
        baseline_logvar = head(embedding)[DecoderOutputKey.LOGVAR.value].clone()
        head.enable_temporal_bias(horizon=horizon)
        trajectory = torch.arange(horizon * output_dim, dtype=torch.float32).view(
            horizon, output_dim
        )
        with torch.no_grad():
            head.temporal_bias.copy_(trajectory)
        result = head(embedding)
        torch.testing.assert_close(
            result[DecoderOutputKey.MEAN.value], baseline + trajectory
        )
        torch.testing.assert_close(
            result[DecoderOutputKey.LOGVAR.value], baseline_logvar
        )
