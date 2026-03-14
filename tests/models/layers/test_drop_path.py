"""Tests for versatil.models.layers.drop_path module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.drop_path import DropPath


@pytest.fixture
def drop_path_factory() -> Callable[..., DropPath]:
    """Factory for DropPath instances."""
    def factory(
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> DropPath:
        return DropPath(
            drop_prob=drop_prob,
            scale_by_keep=scale_by_keep,
        )
    return factory


class TestDropPathInitialization:

    @pytest.mark.parametrize("drop_prob", [0.0, 0.5])
    @pytest.mark.parametrize("scale_by_keep", [True, False])
    def test_stores_configuration(
        self,
        drop_path_factory: Callable[..., DropPath],
        drop_prob: float,
        scale_by_keep: bool,
    ):
        module = drop_path_factory(drop_prob=drop_prob, scale_by_keep=scale_by_keep)
        assert module.drop_prob == drop_prob
        assert module.scale_by_keep == scale_by_keep


class TestDropPathForward:

    def test_returns_unchanged_when_drop_prob_zero(
        self,
        drop_path_factory: Callable[..., DropPath],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = drop_path_factory(drop_prob=0.0)
        module.train()
        x = nchw_tensor_factory()
        output = module(x)
        assert torch.equal(output, x)

    def test_returns_unchanged_when_not_training(
        self,
        drop_path_factory: Callable[..., DropPath],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = drop_path_factory(drop_prob=0.9)
        module.eval()
        x = nchw_tensor_factory()
        output = module(x)
        assert torch.equal(output, x)

    def test_drops_samples_when_training(
        self,
        drop_path_factory: Callable[..., DropPath],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        torch.manual_seed(0)
        module = drop_path_factory(drop_prob=0.99)
        module.train()
        x = nchw_tensor_factory(batch_size=32)
        output = module(x)
        # With drop_prob=0.99 and 32 samples, most samples should be zeroed
        sample_norms = output.flatten(1).norm(dim=1)
        zeroed_count = (sample_norms == 0.0).sum().item()
        assert zeroed_count > 0

    def test_preserves_output_shape(
        self,
        drop_path_factory: Callable[..., DropPath],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = drop_path_factory(drop_prob=0.5)
        module.train()
        x = nchw_tensor_factory(batch_size=8, channels=16)
        output = module(x)
        assert output.shape == x.shape

    def test_drops_all_samples_when_drop_prob_one(
        self,
        drop_path_factory: Callable[..., DropPath],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = drop_path_factory(drop_prob=1.0, scale_by_keep=True)
        module.train()
        x = nchw_tensor_factory(batch_size=8)
        output = module(x)
        assert torch.all(output == 0.0), (
            "With drop_prob=1.0, all samples should be zeroed"
        )

    def test_scale_by_keep_increases_non_dropped_values(
        self,
        drop_path_factory: Callable[..., DropPath],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        torch.manual_seed(42)
        module_scaled = drop_path_factory(drop_prob=0.5, scale_by_keep=True)
        module_scaled.train()
        x = nchw_tensor_factory(batch_size=16)

        torch.manual_seed(42)
        output_scaled = module_scaled(x)

        torch.manual_seed(42)
        module_unscaled = drop_path_factory(drop_prob=0.5, scale_by_keep=False)
        module_unscaled.train()

        torch.manual_seed(42)
        output_unscaled = module_unscaled(x)

        # Non-zero samples in scaled output should be larger than in unscaled
        non_zero_mask = output_scaled.flatten(1).norm(dim=1) > 0
        assert non_zero_mask.any(), (
            "With drop_prob=0.5 and batch_size=16, at least some samples must survive"
        )
        scaled_norms = output_scaled[non_zero_mask].abs().mean()
        unscaled_norms = output_unscaled[non_zero_mask].abs().mean()
        assert scaled_norms > unscaled_norms
