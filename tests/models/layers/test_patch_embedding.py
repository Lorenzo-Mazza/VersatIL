"""Tests for versatil.models.layers.patch_embedding module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from versatil.models.layers.frozen_batchnorm import FrozenBatchNorm2d
from versatil.models.layers.patch_embedding import (
    PatchEmbedding,
    PatchEmbedType,
    PatchMerging,
)


@pytest.fixture
def patch_embedding_factory() -> Callable[..., PatchEmbedding]:
    """Factory for PatchEmbedding instances."""

    def factory(
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        embed_type: str = PatchEmbedType.STANDARD.value,
        norm_layer: type | None = None,
    ) -> PatchEmbedding:
        return PatchEmbedding(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            embed_type=embed_type,
            norm_layer=norm_layer,
        )

    return factory


@pytest.fixture
def patch_merging_factory() -> Callable[..., PatchMerging]:
    """Factory for PatchMerging instances."""

    def factory(
        dim: int = 64,
        out_dim: int = 128,
    ) -> PatchMerging:
        return PatchMerging(
            dim=dim,
            out_dim=out_dim,
        )

    return factory


class TestPatchEmbeddingInitialization:
    @pytest.mark.parametrize("patch_size", [8, 16])
    @pytest.mark.parametrize("in_chans", [1, 3])
    @pytest.mark.parametrize("embed_dim", [256, 768])
    @pytest.mark.parametrize(
        "embed_type",
        [
            PatchEmbedType.STANDARD.value,
            PatchEmbedType.OVERLAPPING.value,
        ],
    )
    def test_stores_configuration(
        self,
        patch_embedding_factory: Callable[..., PatchEmbedding],
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        embed_type: str,
    ):
        module = patch_embedding_factory(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            embed_type=embed_type,
        )
        assert module.patch_size == patch_size
        assert module.in_chans == in_chans
        assert module.embed_dim == embed_dim
        assert module.embed_type == embed_type

    @pytest.mark.parametrize(
        "embed_type, expectation",
        [
            (PatchEmbedType.STANDARD.value, does_not_raise()),
            (PatchEmbedType.PROGRESSIVE.value, does_not_raise()),
            (PatchEmbedType.OVERLAPPING.value, does_not_raise()),
            (
                "unknown_type",
                pytest.raises(
                    ValueError, match=re.escape("Unknown embed_type: unknown_type")
                ),
            ),
        ],
    )
    def test_embed_type_validation(
        self,
        patch_embedding_factory: Callable[..., PatchEmbedding],
        embed_type: str,
        expectation,
    ):
        with expectation:
            patch_embedding_factory(embed_type=embed_type)

    @pytest.mark.parametrize(
        "norm_layer, expectation",
        [
            (torch.nn.BatchNorm2d, does_not_raise()),
            (torch.nn.SyncBatchNorm, does_not_raise()),
            (FrozenBatchNorm2d, does_not_raise()),
            (
                torch.nn.LayerNorm,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "LayerNorm is not supported for progressive embedding. "
                        "Use a BatchNorm variant (e.g. nn.BatchNorm2d, FrozenBatchNorm2d)."
                    ),
                ),
            ),
            (
                torch.nn.GroupNorm,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "GroupNorm is not supported for progressive embedding. "
                        "Use a BatchNorm variant (e.g. nn.BatchNorm2d, FrozenBatchNorm2d)."
                    ),
                ),
            ),
        ],
    )
    def test_progressive_norm_layer_validation(
        self,
        norm_layer: type,
        expectation,
    ):
        with expectation:
            PatchEmbedding(
                patch_size=4,
                in_chans=3,
                embed_dim=64,
                embed_type=PatchEmbedType.PROGRESSIVE.value,
                norm_layer=norm_layer,
            )


class TestPatchEmbeddingForward:
    def test_standard_output_shape(
        self,
        patch_embedding_factory: Callable[..., PatchEmbedding],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        embed_dim = 256
        patch_size = 16
        height, width = 64, 64
        module = patch_embedding_factory(
            embed_dim=embed_dim,
            patch_size=patch_size,
            embed_type=PatchEmbedType.STANDARD.value,
        )
        x = nchw_tensor_factory(channels=3, height=height, width=width)
        output = module(x)
        expected_num_patches = (height // patch_size) * (width // patch_size)
        assert output.shape == (2, expected_num_patches, embed_dim)

    def test_progressive_output_shape(
        self,
        patch_embedding_factory: Callable[..., PatchEmbedding],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        embed_dim = 256
        patch_size = 4
        height, width = 64, 64
        module = patch_embedding_factory(
            embed_dim=embed_dim,
            patch_size=patch_size,
            embed_type=PatchEmbedType.PROGRESSIVE.value,
        )
        x = nchw_tensor_factory(channels=3, height=height, width=width)
        output = module(x)
        # Progressive downsamples by stride 2 per stage; patch_size=4 means 2 stages -> total stride 4
        expected_height = height // patch_size
        expected_width = width // patch_size
        assert output.shape == (2, expected_height, expected_width, embed_dim)

    def test_overlapping_output_shape(
        self,
        patch_embedding_factory: Callable[..., PatchEmbedding],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        embed_dim = 256
        patch_size = 16
        height, width = 64, 64
        module = patch_embedding_factory(
            embed_dim=embed_dim,
            patch_size=patch_size,
            embed_type=PatchEmbedType.OVERLAPPING.value,
        )
        x = nchw_tensor_factory(channels=3, height=height, width=width)
        output = module(x)
        # Overlapping: stride = patch_size // 2, padding = patch_size // 4
        stride = patch_size // 2
        padding = patch_size // 4
        expected_h = (height + 2 * padding - patch_size) // stride + 1
        expected_w = (width + 2 * padding - patch_size) // stride + 1
        expected_num_patches = expected_h * expected_w
        assert output.shape == (2, expected_num_patches, embed_dim)

    def test_return_patch_size_standard(
        self,
        patch_embedding_factory: Callable[..., PatchEmbedding],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        patch_size = 16
        height, width = 64, 64
        embed_dim = 768
        module = patch_embedding_factory(
            patch_size=patch_size,
            embed_dim=embed_dim,
            embed_type=PatchEmbedType.STANDARD.value,
        )
        x = nchw_tensor_factory(channels=3, height=height, width=width)
        tensor, h, w = module(x, return_patch_size=True)
        expected_h = height // patch_size
        expected_w = width // patch_size
        assert h == expected_h
        assert w == expected_w
        assert tensor.shape == (2, expected_h * expected_w, embed_dim)

    def test_return_patch_size_progressive(
        self,
        patch_embedding_factory: Callable[..., PatchEmbedding],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        patch_size = 4
        height, width = 64, 64
        embed_dim = 768
        module = patch_embedding_factory(
            patch_size=patch_size,
            embed_dim=embed_dim,
            embed_type=PatchEmbedType.PROGRESSIVE.value,
        )
        x = nchw_tensor_factory(channels=3, height=height, width=width)
        tensor, h, w = module(x, return_patch_size=True)
        expected_h = height // patch_size
        expected_w = width // patch_size
        assert h == expected_h
        assert w == expected_w
        assert tensor.shape == (2, expected_h, expected_w, embed_dim)


class TestPatchMergingInitialization:
    @pytest.mark.parametrize("dim", [32, 64])
    @pytest.mark.parametrize("out_dim", [64, 128])
    def test_stores_configuration(
        self,
        patch_merging_factory: Callable[..., PatchMerging],
        dim: int,
        out_dim: int,
    ):
        module = patch_merging_factory(dim=dim, out_dim=out_dim)
        assert module.dim == dim
        assert module.out_dim == out_dim


class TestPatchMergingForward:
    @pytest.mark.parametrize(
        "height, width",
        [
            (16, 16),
            (8, 12),
        ],
    )
    def test_halves_spatial_dimensions(
        self,
        patch_merging_factory: Callable[..., PatchMerging],
        nhwc_tensor_factory: Callable[..., torch.Tensor],
        height: int,
        width: int,
    ):
        dim = 64
        out_dim = 128
        module = patch_merging_factory(dim=dim, out_dim=out_dim)
        x = nhwc_tensor_factory(height=height, width=width, channels=dim)
        output = module(x)
        assert output.shape == (2, height // 2, width // 2, out_dim)

    def test_works_with_batchnorm(
        self,
        nhwc_tensor_factory: Callable[..., torch.Tensor],
    ):
        dim = 64
        out_dim = 128
        height, width = 16, 16
        module = PatchMerging(dim=dim, out_dim=out_dim, norm_layer=torch.nn.BatchNorm2d)
        x = nhwc_tensor_factory(height=height, width=width, channels=dim)
        output = module(x)
        assert output.shape == (2, height // 2, width // 2, out_dim)
