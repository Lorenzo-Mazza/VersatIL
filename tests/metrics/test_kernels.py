"""Tests for versatil.metrics.kernels module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.metrics.kernels import (
    IMQKernel,
    KernelType,
    MMDKernel,
    RBFKernel,
)


@pytest.fixture
def point_set_factory(rng):
    def factory(
        num_points: int = 8,
        dimension: int = 4,
    ) -> torch.Tensor:
        data = rng.standard_normal((num_points, dimension)).astype(np.float32)
        return torch.from_numpy(data)

    return factory


@pytest.mark.unit
class TestMMDKernelPairwiseSquaredDistances:
    def test_identical_points_return_near_zero_distances(self):
        point = torch.tensor([[1.0, 2.0, 3.0]])
        kernel = RBFKernel()
        distances = kernel.compute_pairwise_squared_distances(x=point, y=point)
        # Clamped to 1e-10 minimum
        assert distances.item() == pytest.approx(1e-10, abs=1e-12)

    def test_known_distance_two_points(self):
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[3.0, 4.0]])
        kernel = RBFKernel()
        distances = kernel.compute_pairwise_squared_distances(x=x, y=y)
        # ||[0,0] - [3,4]||^2 = 9 + 16 = 25
        assert distances.item() == pytest.approx(25.0, abs=1e-4)

    def test_output_shape(self, point_set_factory):
        x = point_set_factory(num_points=5, dimension=3)
        y = point_set_factory(num_points=7, dimension=3)
        kernel = RBFKernel()
        distances = kernel.compute_pairwise_squared_distances(x=x, y=y)
        assert distances.shape == (5, 7)

    def test_symmetry_property(self, point_set_factory):
        x = point_set_factory(num_points=4, dimension=3)
        y = point_set_factory(num_points=6, dimension=3)
        kernel = RBFKernel()
        dist_xy = kernel.compute_pairwise_squared_distances(x=x, y=y)
        dist_yx = kernel.compute_pairwise_squared_distances(x=y, y=x)
        assert torch.allclose(dist_xy, dist_yx.t(), atol=1e-5)

    def test_all_distances_non_negative(self, point_set_factory):
        x = point_set_factory(num_points=10, dimension=5)
        y = point_set_factory(num_points=8, dimension=5)
        kernel = RBFKernel()
        distances = kernel.compute_pairwise_squared_distances(x=x, y=y)
        assert torch.all(distances >= 0)

    def test_flattens_higher_dimensional_input(self, point_set_factory):
        x = point_set_factory(num_points=6, dimension=4).reshape(2, 3, 4)
        y = point_set_factory(num_points=9, dimension=4).reshape(3, 3, 4)
        kernel = RBFKernel()
        distances = kernel.compute_pairwise_squared_distances(x=x, y=y)
        assert distances.shape == (6, 9)


@pytest.mark.unit
class TestMMDKernelMedianSquaredDistance:
    def test_returns_float(self, point_set_factory):
        points = point_set_factory(num_points=10, dimension=3)
        kernel = RBFKernel()
        result = kernel.compute_median_squared_distance(points=points)
        assert isinstance(result, float)

    def test_single_point_returns_default(self):
        point = torch.tensor([[1.0, 2.0]])
        kernel = RBFKernel()
        result = kernel.compute_median_squared_distance(points=point)
        # Single point has no pairwise distances, returns 1.0
        assert result == 1.0

    def test_identical_points_return_default(self):
        points = torch.ones(5, 3)
        kernel = RBFKernel()
        result = kernel.compute_median_squared_distance(points=points)
        # All distances zero (below threshold), falls back to 1.0
        assert result == 1.0

    def test_known_median_for_three_equidistant_points(self):
        # Three points: [0,0], [1,0], [0,1]
        # Pairwise squared distances: 1.0, 1.0, 2.0
        # Median = 1.0
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        kernel = RBFKernel()
        result = kernel.compute_median_squared_distance(points=points)
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_positive_result_for_spread_points(self, point_set_factory):
        points = point_set_factory(num_points=20, dimension=3)
        kernel = RBFKernel()
        result = kernel.compute_median_squared_distance(points=points)
        assert result > 0

    def test_detaches_input(self, point_set_factory):
        points = point_set_factory(num_points=5, dimension=3).requires_grad_(True)
        kernel = RBFKernel()
        result = kernel.compute_median_squared_distance(points=points)
        # Should not raise; computation detaches from grad
        assert isinstance(result, float)


@pytest.mark.unit
class TestRBFKernel:
    def test_default_bandwidth_multipliers(self):
        kernel = RBFKernel()
        assert kernel.bandwidth_multipliers == [0.2, 0.5, 1.0, 2.0, 5.0]

    def test_custom_bandwidth_multipliers(self):
        kernel = RBFKernel(bandwidth_multipliers=[1.0, 10.0])
        assert kernel.bandwidth_multipliers == [1.0, 10.0]

    def test_output_shape(self, point_set_factory):
        x = point_set_factory(num_points=5, dimension=3)
        y = point_set_factory(num_points=7, dimension=3)
        kernel = RBFKernel()
        result = kernel(x, y)
        assert result.shape == (5, 7)

    def test_identical_points_produce_high_kernel_values(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        kernel = RBFKernel()
        result = kernel(x, x)
        # K(x, x) should be close to 1.0 (exp of near-zero)
        assert result.item() == pytest.approx(1.0, abs=0.01)

    def test_close_points_higher_than_far_points(self, point_set_factory):
        origin = torch.tensor([[0.0, 0.0, 0.0]])
        close = torch.tensor([[0.1, 0.0, 0.0]])
        far = torch.tensor([[5.0, 5.0, 5.0]])
        # Use many background points so the median isn't dominated by the query pair
        background = point_set_factory(num_points=20, dimension=3)
        x_close = torch.cat([origin, close, background], dim=0)
        x_far = torch.cat([origin, far, background], dim=0)
        kernel = RBFKernel()
        # Compute full kernel matrices, extract the (0,1) entry
        k_close = kernel(x_close, x_close)[0, 1]
        k_far = kernel(x_far, x_far)[0, 1]
        assert k_close.item() > k_far.item()

    def test_symmetry(self, point_set_factory):
        x = point_set_factory(num_points=5, dimension=3)
        y = point_set_factory(num_points=5, dimension=3)
        kernel = RBFKernel()
        k_xy = kernel(x, y)
        k_yx = kernel(y, x)
        assert torch.allclose(k_xy, k_yx.t(), atol=1e-5)

    def test_all_values_positive(self, point_set_factory):
        x = point_set_factory(num_points=8, dimension=4)
        y = point_set_factory(num_points=6, dimension=4)
        kernel = RBFKernel()
        result = kernel(x, y)
        assert torch.all(result > 0)

    def test_all_values_at_most_one(self, point_set_factory):
        x = point_set_factory(num_points=8, dimension=4)
        y = point_set_factory(num_points=6, dimension=4)
        kernel = RBFKernel()
        result = kernel(x, y)
        # Each RBF component is exp(-d^2 / bandwidth) <= 1, average <= 1
        assert torch.all(result <= 1.0 + 1e-5)

    def test_gradient_flows(self, point_set_factory):
        x = point_set_factory(num_points=4, dimension=3).requires_grad_(True)
        y = point_set_factory(num_points=4, dimension=3)
        kernel = RBFKernel()
        result = kernel(x, y)
        result.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_mathematical_correctness_single_bandwidth(self):
        # Manual calculation: K(x,y) = exp(-||x-y||^2 / (2 * mult * median_dist^2))
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[1.0, 0.0]])
        kernel = RBFKernel(bandwidth_multipliers=[1.0])
        result = kernel(x, y)
        # Combined = [[0,0],[1,0]], median pairwise dist^2 = 1.0
        # bandwidth = 2 * 1.0 * 1.0 = 2.0
        # K = exp(-1.0 / 2.0) = exp(-0.5)
        expected = np.exp(-0.5)
        assert result.item() == pytest.approx(expected, abs=1e-4)


@pytest.mark.unit
class TestIMQKernel:
    def test_default_bandwidth_multipliers(self):
        kernel = IMQKernel()
        assert kernel.bandwidth_multipliers == [0.2, 0.5, 1.0, 2.0, 5.0]

    def test_custom_bandwidth_multipliers(self):
        kernel = IMQKernel(bandwidth_multipliers=[0.5, 2.0])
        assert kernel.bandwidth_multipliers == [0.5, 2.0]

    def test_output_shape(self, point_set_factory):
        x = point_set_factory(num_points=5, dimension=3)
        y = point_set_factory(num_points=7, dimension=3)
        kernel = IMQKernel()
        result = kernel(x, y)
        assert result.shape == (5, 7)

    def test_identical_points_produce_high_kernel_values(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        kernel = IMQKernel()
        result = kernel(x, x)
        # C / (C + ~0) should be close to 1.0
        assert result.item() == pytest.approx(1.0, abs=0.01)

    def test_close_points_higher_than_far_points(self, point_set_factory):
        origin = torch.tensor([[0.0, 0.0, 0.0]])
        close = torch.tensor([[0.1, 0.0, 0.0]])
        far = torch.tensor([[5.0, 5.0, 5.0]])
        background = point_set_factory(num_points=20, dimension=3)
        x_close = torch.cat([origin, close, background], dim=0)
        x_far = torch.cat([origin, far, background], dim=0)
        kernel = IMQKernel()
        k_close = kernel(x_close, x_close)[0, 1]
        k_far = kernel(x_far, x_far)[0, 1]
        assert k_close.item() > k_far.item()

    def test_symmetry(self, point_set_factory):
        x = point_set_factory(num_points=5, dimension=3)
        y = point_set_factory(num_points=5, dimension=3)
        kernel = IMQKernel()
        k_xy = kernel(x, y)
        k_yx = kernel(y, x)
        assert torch.allclose(k_xy, k_yx.t(), atol=1e-5)

    def test_all_values_positive(self, point_set_factory):
        x = point_set_factory(num_points=8, dimension=4)
        y = point_set_factory(num_points=6, dimension=4)
        kernel = IMQKernel()
        result = kernel(x, y)
        assert torch.all(result > 0)

    def test_all_values_at_most_one(self, point_set_factory):
        x = point_set_factory(num_points=8, dimension=4)
        y = point_set_factory(num_points=6, dimension=4)
        kernel = IMQKernel()
        result = kernel(x, y)
        # C / (C + d^2) <= 1 for all d^2 >= 0
        assert torch.all(result <= 1.0 + 1e-5)

    def test_heavier_tails_than_rbf(self, point_set_factory):
        # Use many points so the median bandwidth stabilizes, then compare
        # a pair of moderately distant points
        background = point_set_factory(num_points=20, dimension=2)
        x = torch.cat([torch.tensor([[0.0, 0.0]]), background], dim=0)
        y = torch.cat([torch.tensor([[10.0, 10.0]]), background], dim=0)
        rbf = RBFKernel(bandwidth_multipliers=[1.0])
        imq = IMQKernel(bandwidth_multipliers=[1.0])
        # Both use same combined set for median, so bandwidth is comparable
        rbf_val = rbf(x, y)[0, 0].item()
        imq_val = imq(x, y)[0, 0].item()
        # IMQ has heavier tails, so for distant points it should be larger
        assert imq_val > rbf_val

    def test_gradient_flows(self, point_set_factory):
        x = point_set_factory(num_points=4, dimension=3).requires_grad_(True)
        y = point_set_factory(num_points=4, dimension=3)
        kernel = IMQKernel()
        result = kernel(x, y)
        result.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_mathematical_correctness_single_bandwidth(self):
        # Manual: K(x,y) = C / (C + ||x-y||^2), C = 2 * mult * median_dist^2
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[1.0, 0.0]])
        kernel = IMQKernel(bandwidth_multipliers=[1.0])
        result = kernel(x, y)
        # Combined = [[0,0],[1,0]], median dist^2 = 1.0
        # C = 2 * 1.0 * 1.0 = 2.0
        # d^2 = 1.0
        # K = 2.0 / (2.0 + 1.0) = 2/3
        expected = 2.0 / 3.0
        assert result.item() == pytest.approx(expected, abs=1e-4)


@pytest.mark.unit
class TestKernelType:
    def test_rbf_value(self):
        assert KernelType.RBF.value == "rbf"

    def test_imq_value(self):
        assert KernelType.IMQ.value == "imq"

    def test_to_kernel_creates_rbf(self):
        kernel = KernelType.RBF.to_kernel()
        assert isinstance(kernel, RBFKernel)

    def test_to_kernel_creates_imq(self):
        kernel = KernelType.IMQ.to_kernel()
        assert isinstance(kernel, IMQKernel)

    def test_to_kernel_passes_bandwidth_multipliers(self):
        multipliers = [0.1, 1.0]
        kernel = KernelType.RBF.to_kernel(bandwidth_multipliers=multipliers)
        assert kernel.bandwidth_multipliers == multipliers

    def test_to_kernel_imq_passes_bandwidth_multipliers(self):
        multipliers = [0.5, 5.0]
        kernel = KernelType.IMQ.to_kernel(bandwidth_multipliers=multipliers)
        assert kernel.bandwidth_multipliers == multipliers

    def test_to_kernel_default_bandwidth(self):
        kernel = KernelType.RBF.to_kernel()
        assert kernel.bandwidth_multipliers == [0.2, 0.5, 1.0, 2.0, 5.0]

    def test_to_kernel_passes_use_median_heuristic(self):
        kernel = KernelType.IMQ.to_kernel(use_median_heuristic=False)
        assert kernel.use_median_heuristic is False

    def test_to_kernel_default_uses_median_heuristic(self):
        kernel = KernelType.RBF.to_kernel()
        assert kernel.use_median_heuristic is True

    def test_lookup_by_value(self):
        assert KernelType("rbf") is KernelType.RBF
        assert KernelType("imq") is KernelType.IMQ


@pytest.mark.unit
class TestResolveBandwidth:
    def test_median_heuristic_returns_scaled_median(self, point_set_factory: Callable):
        x = point_set_factory(num_points=20, dimension=4)
        y = point_set_factory(num_points=20, dimension=4)
        kernel = RBFKernel(use_median_heuristic=True)

        base = kernel._resolve_base_bandwidth(x, y)
        combined = torch.cat([x, y], dim=0)
        expected = 2.0 * kernel.compute_median_squared_distance(combined)

        assert base == pytest.approx(expected, rel=1e-5)

    def test_fixed_bandwidth_returns_one(self, point_set_factory: Callable):
        x = point_set_factory(num_points=20, dimension=4)
        y = point_set_factory(num_points=20, dimension=4)
        kernel = RBFKernel(use_median_heuristic=False)

        base = kernel._resolve_base_bandwidth(x, y)

        assert base == 1.0


@pytest.mark.unit
class TestFixedVsAdaptiveBandwidth:
    def test_rbf_fixed_bandwidth_differs_from_adaptive(
        self, point_set_factory: Callable
    ):
        x = point_set_factory(num_points=30, dimension=4)
        y = point_set_factory(num_points=30, dimension=4)
        adaptive = RBFKernel(use_median_heuristic=True)
        fixed = RBFKernel(use_median_heuristic=False)

        k_adaptive = adaptive(x, y)
        k_fixed = fixed(x, y)

        assert not torch.allclose(k_adaptive, k_fixed)

    def test_imq_fixed_bandwidth_differs_from_adaptive(
        self, point_set_factory: Callable
    ):
        x = point_set_factory(num_points=30, dimension=4)
        y = point_set_factory(num_points=30, dimension=4)
        adaptive = IMQKernel(use_median_heuristic=True)
        fixed = IMQKernel(use_median_heuristic=False)

        k_adaptive = adaptive(x, y)
        k_fixed = fixed(x, y)

        assert not torch.allclose(k_adaptive, k_fixed)

    def test_imq_fixed_bandwidth_with_wae_scale(self):
        latent_dim = 8
        x = torch.randn(50, latent_dim)
        y = torch.randn(50, latent_dim)
        kernel = IMQKernel(
            bandwidth_multipliers=[2.0 * latent_dim],
            use_median_heuristic=False,
        )

        result = kernel(x, y)

        assert result.shape == (50, 50)
        assert (result > 0).all()


@pytest.mark.unit
class TestMMDKernelIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            MMDKernel()
