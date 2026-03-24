"""Tests for versatil.post_training_compression.preparation.fusion module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from versatil.models.layers.normalization.frozen_batchnorm import FrozenBatchNorm2d
from versatil.post_training_compression.preparation.fusion import (
    fuse_all_conv_batchnorm_pairs,
    fuse_conv_batchnorm,
)


@pytest.fixture
def conv_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Conv2d]:
    """Factory for Conv2d with deterministic weights."""

    def factory(
        in_channels: int = 3,
        out_channels: int = 16,
        kernel_size: int = 3,
        bias: bool = False,
        groups: int = 1,
        dilation: int = 1,
    ) -> nn.Conv2d:
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
            groups=groups,
            dilation=dilation,
        )
        conv.weight.data = torch.from_numpy(
            rng.standard_normal(conv.weight.shape).astype(np.float32)
        )
        if bias:
            conv.bias.data = torch.from_numpy(
                rng.standard_normal(conv.bias.shape).astype(np.float32)
            )
        return conv

    return factory


@pytest.fixture
def batchnorm_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.BatchNorm2d]:
    """Factory for BatchNorm2d with randomized statistics."""

    def factory(
        num_features: int = 16,
    ) -> nn.BatchNorm2d:
        batchnorm = nn.BatchNorm2d(num_features=num_features)
        batchnorm.weight.data = torch.from_numpy(
            rng.standard_normal(num_features).astype(np.float32)
        )
        batchnorm.bias.data = torch.from_numpy(
            rng.standard_normal(num_features).astype(np.float32)
        )
        batchnorm.running_mean.data = torch.from_numpy(
            rng.standard_normal(num_features).astype(np.float32)
        )
        batchnorm.running_var.data = torch.from_numpy(
            rng.uniform(low=0.1, high=2.0, size=num_features).astype(np.float32)
        )
        batchnorm.eval()
        return batchnorm

    return factory


@pytest.mark.unit
class TestFuseConvBatchnorm:
    @pytest.mark.parametrize("conv_bias", [False, True])
    def test_fused_output_matches_sequential(
        self,
        conv_factory: Callable[..., nn.Conv2d],
        batchnorm_factory: Callable[..., nn.BatchNorm2d],
        spatial_input_factory: Callable[..., torch.Tensor],
        conv_bias: bool,
    ):
        in_channels = 3
        out_channels = 16
        conv = conv_factory(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=conv_bias,
        )
        batchnorm = batchnorm_factory(num_features=out_channels)
        input_data = spatial_input_factory(channels=in_channels)

        conv.eval()
        with torch.no_grad():
            expected = batchnorm(conv(input_data))

        fused = fuse_conv_batchnorm(conv=conv, batchnorm=batchnorm)
        fused.eval()
        with torch.no_grad():
            actual = fused(input_data)

        assert torch.allclose(actual, expected, atol=1e-5)

    def test_fused_conv_always_has_bias(
        self,
        conv_factory: Callable[..., nn.Conv2d],
        batchnorm_factory: Callable[..., nn.BatchNorm2d],
    ):
        conv = conv_factory(out_channels=16, bias=False)
        fused = fuse_conv_batchnorm(
            conv=conv, batchnorm=batchnorm_factory(num_features=16)
        )
        assert fused.bias is not None

    def test_raises_for_module_without_bn_buffers(self):
        with pytest.raises(
            ValueError,
            match="Module Linear does not have the required BatchNorm buffers",
        ):
            fuse_conv_batchnorm(
                conv=nn.Conv2d(3, 16, 3),
                batchnorm=nn.Linear(16, 32),
            )

    def test_fused_preserves_all_conv_spatial_properties(
        self,
        conv_factory: Callable[..., nn.Conv2d],
        batchnorm_factory: Callable[..., nn.BatchNorm2d],
    ):
        conv = conv_factory(
            out_channels=16,
            kernel_size=5,
            dilation=2,
        )
        fused = fuse_conv_batchnorm(
            conv=conv, batchnorm=batchnorm_factory(num_features=16)
        )

        assert fused.kernel_size == conv.kernel_size
        assert fused.stride == conv.stride
        assert fused.padding == conv.padding
        assert fused.groups == conv.groups
        assert fused.dilation == conv.dilation
        assert fused.padding_mode == conv.padding_mode

    def test_fused_with_frozen_batchnorm(
        self,
        conv_factory: Callable[..., nn.Conv2d],
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
        spatial_input_factory: Callable[..., torch.Tensor],
    ):
        in_channels = 3
        out_channels = 8
        conv = conv_factory(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
        )
        frozen_bn = frozen_batchnorm_factory(num_features=out_channels)
        input_data = spatial_input_factory(channels=in_channels)

        conv.eval()
        with torch.no_grad():
            expected = frozen_bn(conv(input_data))

        fused = fuse_conv_batchnorm(conv=conv, batchnorm=frozen_bn)
        fused.eval()
        with torch.no_grad():
            actual = fused(input_data)

        assert torch.allclose(actual, expected, atol=1e-5)

    def test_fused_with_grouped_convolution(
        self,
        conv_factory: Callable[..., nn.Conv2d],
        batchnorm_factory: Callable[..., nn.BatchNorm2d],
        spatial_input_factory: Callable[..., torch.Tensor],
    ):
        in_channels = 16
        out_channels = 16
        groups = 4
        conv = conv_factory(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
        )
        batchnorm = batchnorm_factory(num_features=out_channels)
        input_data = spatial_input_factory(channels=in_channels)

        conv.eval()
        with torch.no_grad():
            expected = batchnorm(conv(input_data))

        fused = fuse_conv_batchnorm(conv=conv, batchnorm=batchnorm)
        fused.eval()
        with torch.no_grad():
            actual = fused(input_data)

        assert torch.allclose(actual, expected, atol=1e-5)
        assert fused.groups == groups


@pytest.mark.unit
class TestFuseAllConvBatchnormPairs:
    def test_fuses_all_pairs_and_returns_count(
        self,
        conv_bn_model_factory: Callable[..., nn.Module],
    ):
        model = conv_bn_model_factory(num_pairs=3)
        assert fuse_all_conv_batchnorm_pairs(model) == 3

    def test_bn_replaced_with_identity(
        self,
        conv_bn_model_factory: Callable[..., nn.Module],
    ):
        model = conv_bn_model_factory(num_pairs=2)
        fuse_all_conv_batchnorm_pairs(model)

        assert isinstance(model[1], nn.Identity)
        assert isinstance(model[3], nn.Identity)

    def test_fused_output_matches_original(
        self,
        conv_bn_model_factory: Callable[..., nn.Module],
        spatial_input_factory: Callable[..., torch.Tensor],
    ):
        in_channels = 3
        model = conv_bn_model_factory(
            in_channels=in_channels,
            num_pairs=2,
            conv_bias=True,
        )
        model.eval()
        input_data = spatial_input_factory(channels=in_channels)

        with torch.no_grad():
            expected = model(input_data)

        fuse_all_conv_batchnorm_pairs(model)

        with torch.no_grad():
            actual = model(input_data)

        assert torch.allclose(actual, expected, atol=1e-5)

    def test_returns_zero_when_no_conv_bn_pairs(self):
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
        )
        assert fuse_all_conv_batchnorm_pairs(model) == 0

    def test_skips_non_consecutive_conv_bn(self):
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        assert fuse_all_conv_batchnorm_pairs(model) == 0

    def test_handles_nested_modules(
        self,
        spatial_input_factory: Callable[..., torch.Tensor],
    ):
        inner = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
        )
        outer = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            inner,
        )
        outer.eval()
        input_data = spatial_input_factory(channels=3)

        with torch.no_grad():
            expected = outer(input_data)

        count = fuse_all_conv_batchnorm_pairs(outer)

        assert count == 2
        with torch.no_grad():
            actual = outer(input_data)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_preserves_activation_with_output_equivalence(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
        spatial_input_factory: Callable[..., torch.Tensor],
    ):
        num_features = 8
        frozen_bn = frozen_batchnorm_factory(num_features=num_features)
        frozen_bn.act = nn.ReLU()
        activation = nn.ReLU()

        conv = nn.Conv2d(3, num_features, 3, padding=1)
        conv.eval()
        input_data = spatial_input_factory(channels=3)

        # Expected: conv → frozen_bn → relu (frozen_bn.forward doesn't call .act)
        with torch.no_grad():
            expected = activation(frozen_bn(conv(input_data)))

        model = nn.Sequential(conv, frozen_bn)
        fuse_all_conv_batchnorm_pairs(model)

        assert isinstance(model[1], nn.ReLU)
        with torch.no_grad():
            actual = nn.Sequential(model[0], model[1])(input_data)
        assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFusionOnCuda:
    def test_fuse_conv_batchnorm_on_cuda(
        self,
        conv_factory: Callable[..., nn.Conv2d],
        batchnorm_factory: Callable[..., nn.BatchNorm2d],
        spatial_input_factory: Callable[..., torch.Tensor],
    ):
        conv = conv_factory(in_channels=3, out_channels=16).cuda()
        batchnorm = batchnorm_factory(num_features=16).cuda()
        input_data = spatial_input_factory(channels=3).cuda()
        conv.eval()
        with torch.no_grad():
            expected = batchnorm(conv(input_data))
        fused = fuse_conv_batchnorm(conv=conv, batchnorm=batchnorm)
        fused.eval()
        assert fused.weight.device.type == "cuda"
        assert fused.bias.device.type == "cuda"
        with torch.no_grad():
            actual = fused(input_data)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_fuse_all_pairs_on_cuda(
        self,
        conv_bn_model_factory: Callable[..., nn.Module],
        spatial_input_factory: Callable[..., torch.Tensor],
    ):
        model = conv_bn_model_factory(in_channels=3, num_pairs=2, conv_bias=True).cuda()
        model.eval()
        input_data = spatial_input_factory(channels=3).cuda()
        with torch.no_grad():
            expected = model(input_data)
        count = fuse_all_conv_batchnorm_pairs(model)
        assert count == 2
        with torch.no_grad():
            actual = model(input_data)
        assert torch.allclose(actual, expected, atol=1e-5)
