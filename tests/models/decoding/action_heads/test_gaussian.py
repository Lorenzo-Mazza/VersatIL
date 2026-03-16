"""Tests for versatil.models.decoding.action_heads.gaussian module."""
import re
from collections.abc import Callable

import pytest
import torch

from versatil.models.decoding.action_heads.blocks import MLPBlock
from versatil.models.decoding.action_heads.gaussian import GaussianHead
from versatil.models.decoding.constants import DecoderOutputKey


class TestGaussianHeadInitialization:

    @pytest.mark.parametrize("min_logvar", [-10.0, -5.0])
    @pytest.mark.parametrize("max_logvar", [4.0, 2.0])
    def test_stores_configuration(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
        min_logvar: float,
        max_logvar: float,
    ):
        head = gaussian_head_factory(min_logvar=min_logvar, max_logvar=max_logvar)
        assert head.min_logvar == min_logvar
        assert head.max_logvar == max_logvar


class TestGaussianHeadSetOutputDim:

    def test_creates_mean_projection(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
    ):
        head = gaussian_head_factory(input_dim=64)
        head.set_output_dim(3)
        assert head.output_proj is not None
        assert head.output_proj.out_features == 3

    def test_creates_logvar_projection(
        self,
        gaussian_head_factory: Callable[..., GaussianHead],
    ):
        head = gaussian_head_factory(input_dim=64)
        head.set_output_dim(3)
        assert head._logvar_proj is not None
        assert head._logvar_proj.out_features == 3

    def test_projection_uses_last_block_dim(self):
        blocks = [MLPBlock(input_dim=64, hidden_dims=[32])]
        head = GaussianHead(input_dim=64, blocks=blocks)
        head.set_output_dim(3)
        assert head.output_proj.in_features == 32
        assert head._logvar_proj.in_features == 32


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
        head = gaussian_head_factory(input_dim=64)
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
        head = gaussian_head_factory(input_dim=64)
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
            input_dim=64,
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
