"""Tests for versatil.data.tokenization.binned_value_discretizer."""

import re
from unittest.mock import patch

import numpy as np
import pytest
import torch

from versatil.data.constants import BinningStrategy
from versatil.data.tokenization.binned_value_discretizer import BinnedValueDiscretizer


class TestBinnedValueDiscretizerInit:
    @pytest.mark.parametrize("num_bins", [8, 16, 32, 256])
    def test_stores_num_bins(self, binned_value_discretizer_factory, num_bins):
        tokenizer = binned_value_discretizer_factory(num_bins=num_bins)
        assert tokenizer.num_bins == num_bins

    def test_stores_device(self, binned_value_discretizer_factory, device):
        tokenizer = binned_value_discretizer_factory(device=device)
        assert tokenizer.device == device

    def test_default_device_is_cpu(self):
        tokenizer = BinnedValueDiscretizer(num_bins=16)
        assert tokenizer.device == torch.device("cpu")

    def test_bin_edges_none_before_fitting(self, binned_value_discretizer_factory):
        tokenizer = binned_value_discretizer_factory()
        assert tokenizer.bin_edges is None

    def test_is_fitted_false_before_fitting(self, binned_value_discretizer_factory):
        tokenizer = binned_value_discretizer_factory()
        assert tokenizer._is_fitted is False


class TestBinnedValueDiscretizerFit:
    def test_sets_is_fitted_true(self, fitted_binned_value_discretizer_factory):
        tokenizer = fitted_binned_value_discretizer_factory()
        assert tokenizer._is_fitted is True

    @pytest.mark.parametrize(
        "num_bins, num_dimensions",
        [(8, 5), (16, 3), (32, 7)],
    )
    def test_bin_edges_shape_matches_dimensions_and_bins(
        self, rng, binned_value_discretizer_factory, num_bins, num_dimensions
    ):
        tokenizer = binned_value_discretizer_factory(num_bins=num_bins)
        data = rng.standard_normal((50, num_dimensions)).astype(np.float32)
        tokenizer.fit(data)
        assert tokenizer.bin_edges.shape == (num_dimensions, num_bins - 1)

    def test_bin_edges_are_sorted_per_dimension(
        self, rng, binned_value_discretizer_factory
    ):
        tokenizer = binned_value_discretizer_factory(num_bins=16)
        data = rng.standard_normal((100, 3)).astype(np.float32)
        tokenizer.fit(data)
        for dimension in range(3):
            edges = tokenizer.bin_edges[dimension].cpu().numpy()
            assert np.all(edges[:-1] <= edges[1:])

    def test_fit_with_3d_input_reshapes_to_2d(
        self, rng, binned_value_discretizer_factory
    ):
        tokenizer = binned_value_discretizer_factory(num_bins=8)
        data_3d = rng.standard_normal((10, 5, 4)).astype(np.float32)
        tokenizer.fit(data_3d)
        assert tokenizer.bin_edges.shape == (4, 7)

    def test_bin_edges_on_correct_device(
        self, binned_value_discretizer_factory, rng, device
    ):
        tokenizer = binned_value_discretizer_factory(num_bins=8, device=device)
        data = rng.standard_normal((50, 3)).astype(np.float32)
        tokenizer.fit(data)
        assert tokenizer.bin_edges.device.type == device.type

    def test_fit_logs_info(self, rng, binned_value_discretizer_factory):
        tokenizer = binned_value_discretizer_factory(num_bins=8)
        data = rng.standard_normal((50, 3)).astype(np.float32)
        with patch(
            "versatil.data.tokenization.binned_value_discretizer.logging"
        ) as mock_logging:
            tokenizer.fit(data)
            mock_logging.info.assert_called_once()
            log_message = mock_logging.info.call_args[0][0]
            assert "8 bins" in log_message
            assert "50 samples" in log_message
            assert "3 dimensions" in log_message


class TestBinnedValueDiscretizerEncode:
    def test_encode_raises_when_not_fitted(self, binned_value_discretizer_factory):
        tokenizer = binned_value_discretizer_factory()
        data = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(
            RuntimeError,
            match=re.escape("Discretizer must be fitted before encoding"),
        ):
            tokenizer.encode(data)

    def test_encode_numpy_returns_long_tensor(
        self, fitted_binned_value_discretizer_factory, rng
    ):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=16, num_dimensions=3
        )
        data = rng.standard_normal((5, 3)).astype(np.float32)
        tokens = tokenizer.encode(data)
        assert tokens.dtype == torch.long

    @pytest.mark.parametrize(
        "num_samples, num_dimensions",
        [(10, 4), (1, 7), (50, 2)],
    )
    def test_encode_preserves_shape(
        self, fitted_binned_value_discretizer_factory, rng, num_samples, num_dimensions
    ):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=16, num_dimensions=num_dimensions
        )
        data = rng.standard_normal((num_samples, num_dimensions)).astype(np.float32)
        tokens = tokenizer.encode(data)
        assert tokens.shape == (num_samples, num_dimensions)

    @pytest.mark.parametrize("num_bins", [8, 16, 64, 256])
    def test_encode_values_in_valid_range(
        self, fitted_binned_value_discretizer_factory, rng, num_bins
    ):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=num_bins, num_dimensions=3
        )
        data = rng.standard_normal((20, 3)).astype(np.float32)
        tokens = tokenizer.encode(data)
        assert tokens.min() >= 0
        assert tokens.max() < num_bins

    def test_encode_torch_tensor_input(
        self, fitted_binned_value_discretizer_factory, rng
    ):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=16, num_dimensions=3
        )
        data = torch.from_numpy(rng.standard_normal((5, 3)).astype(np.float32))
        tokens = tokenizer.encode(data)
        assert tokens.dtype == torch.long
        assert tokens.shape == (5, 3)

    def test_encode_3d_input_preserves_shape(
        self, fitted_binned_value_discretizer_factory, rng
    ):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=8, num_dimensions=4
        )
        data = rng.standard_normal((2, 3, 4)).astype(np.float32)
        tokens = tokenizer.encode(data)
        assert tokens.shape == (2, 3, 4)


class TestBinnedValueDiscretizerDecode:
    def test_decode_raises_when_not_fitted(self, binned_value_discretizer_factory):
        tokenizer = binned_value_discretizer_factory()
        tokens = torch.zeros((5, 3), dtype=torch.long)
        with pytest.raises(
            RuntimeError,
            match=re.escape("Discretizer must be fitted before decoding"),
        ):
            tokenizer.decode(tokens)

    def test_decode_returns_float32(self, fitted_binned_value_discretizer_factory, rng):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=16, num_dimensions=3
        )
        data = rng.standard_normal((5, 3)).astype(np.float32)
        tokens = tokenizer.encode(data)
        decoded = tokenizer.decode(tokens)
        assert decoded.dtype == torch.float32

    @pytest.mark.parametrize(
        "num_samples, num_dimensions",
        [(10, 4), (1, 7), (50, 2)],
    )
    def test_decode_preserves_shape(
        self, fitted_binned_value_discretizer_factory, rng, num_samples, num_dimensions
    ):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=16, num_dimensions=num_dimensions
        )
        data = rng.standard_normal((num_samples, num_dimensions)).astype(np.float32)
        tokens = tokenizer.encode(data)
        decoded = tokenizer.decode(tokens)
        assert decoded.shape == (num_samples, num_dimensions)

    def test_decode_numpy_input(self, fitted_binned_value_discretizer_factory):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=16, num_dimensions=3
        )
        tokens = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        decoded = tokenizer.decode(tokens)
        assert decoded.dtype == torch.float32
        assert decoded.shape == (2, 3)

    def test_encode_decode_roundtrip_approximate(
        self, fitted_binned_value_discretizer_factory, rng
    ):
        num_bins = 256
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=num_bins, num_dimensions=3, num_samples=1000
        )
        data = rng.uniform(-1, 1, (50, 3)).astype(np.float32)
        tokens = tokenizer.encode(data)
        decoded = tokenizer.decode(tokens)
        max_error = torch.abs(decoded - torch.tensor(data, dtype=torch.float32)).max()
        assert max_error < 0.5


class TestBinnedValueDiscretizerGetBinCenters:
    def test_first_bin_center_extrapolated_below_first_edge(
        self, fitted_binned_value_discretizer_factory
    ):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=8, num_dimensions=1
        )
        centers = tokenizer._geometric_bin_centers(dim=0)
        edges = tokenizer.bin_edges[0]
        expected = edges[0] - (edges[1] - edges[0]) / 2
        assert torch.isclose(centers[0], expected)

    def test_last_bin_center_extrapolated_above_last_edge(
        self, fitted_binned_value_discretizer_factory
    ):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=8, num_dimensions=1
        )
        centers = tokenizer._geometric_bin_centers(dim=0)
        edges = tokenizer.bin_edges[0]
        expected = edges[-1] + (edges[-1] - edges[-2]) / 2
        assert torch.isclose(centers[-1], expected)

    def test_middle_bin_centers_are_averages_of_adjacent_edges(
        self, fitted_binned_value_discretizer_factory
    ):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=8, num_dimensions=1
        )
        centers = tokenizer._geometric_bin_centers(dim=0)
        edges = tokenizer.bin_edges[0]
        for i in range(1, 7):
            expected = (edges[i - 1] + edges[i]) / 2
            assert torch.isclose(centers[i], expected)

    @pytest.mark.parametrize("num_bins", [8, 16, 64])
    def test_bin_centers_length_equals_num_bins(
        self, fitted_binned_value_discretizer_factory, num_bins
    ):
        tokenizer = fitted_binned_value_discretizer_factory(
            num_bins=num_bins, num_dimensions=2
        )
        centers = tokenizer._geometric_bin_centers(dim=0)
        assert centers.shape == (num_bins,)


class TestBinnedValueDiscretizerTo:
    def test_to_updates_device(self, fitted_binned_value_discretizer_factory, device):
        tokenizer = fitted_binned_value_discretizer_factory()
        tokenizer.to(device)
        assert tokenizer.device == device

    def test_to_moves_bin_edges(self, fitted_binned_value_discretizer_factory, device):
        tokenizer = fitted_binned_value_discretizer_factory()
        tokenizer.to(device)
        assert tokenizer.bin_edges.device.type == device.type

    def test_to_returns_self(self, fitted_binned_value_discretizer_factory, device):
        tokenizer = fitted_binned_value_discretizer_factory()
        result = tokenizer.to(device)
        assert result is tokenizer

    def test_to_handles_none_bin_edges(self, binned_value_discretizer_factory, device):
        tokenizer = binned_value_discretizer_factory()
        tokenizer.to(device)
        assert tokenizer.bin_edges is None


class TestBinnedValueDiscretizerStateDict:
    def test_state_dict_keys(self, fitted_binned_value_discretizer_factory):
        tokenizer = fitted_binned_value_discretizer_factory()
        state = tokenizer.state_dict()
        assert set(state.keys()) == {
            "num_bins",
            "binning_strategy",
            "min_value",
            "max_value",
            "bin_edges",
            "bin_values",
            "is_fitted",
        }

    @pytest.mark.parametrize("num_bins", [8, 32, 128])
    def test_state_dict_contains_num_bins(
        self, fitted_binned_value_discretizer_factory, num_bins
    ):
        tokenizer = fitted_binned_value_discretizer_factory(num_bins=num_bins)
        state = tokenizer.state_dict()
        assert state["num_bins"] == num_bins

    def test_state_dict_contains_bin_edges(
        self, fitted_binned_value_discretizer_factory
    ):
        tokenizer = fitted_binned_value_discretizer_factory()
        state = tokenizer.state_dict()
        assert isinstance(state["bin_edges"], torch.Tensor)

    def test_state_dict_contains_is_fitted(
        self, fitted_binned_value_discretizer_factory
    ):
        tokenizer = fitted_binned_value_discretizer_factory()
        state = tokenizer.state_dict()
        assert state["is_fitted"] is True

    def test_state_dict_bin_edges_always_on_cpu(
        self, fitted_binned_value_discretizer_factory, device
    ):
        tokenizer = fitted_binned_value_discretizer_factory(device=device)
        state = tokenizer.state_dict()
        assert state["bin_edges"].device.type == "cpu"

    def test_state_dict_unfitted_has_none_bin_edges(
        self, binned_value_discretizer_factory
    ):
        tokenizer = binned_value_discretizer_factory()
        state = tokenizer.state_dict()
        assert state["bin_edges"] is None
        assert state["is_fitted"] is False


class TestBinnedValueDiscretizerLoadStateDict:
    def test_load_state_dict_restores_num_bins(
        self, fitted_binned_value_discretizer_factory
    ):
        original = fitted_binned_value_discretizer_factory(
            num_bins=32, num_dimensions=3
        )
        state = original.state_dict()
        restored = BinnedValueDiscretizer(num_bins=8)
        restored.load_state_dict(state)
        assert restored.num_bins == 32

    def test_load_state_dict_restores_is_fitted(
        self, fitted_binned_value_discretizer_factory
    ):
        original = fitted_binned_value_discretizer_factory()
        state = original.state_dict()
        restored = BinnedValueDiscretizer(num_bins=8)
        restored.load_state_dict(state)
        assert restored._is_fitted is True

    def test_load_state_dict_restores_bin_edges(
        self, fitted_binned_value_discretizer_factory
    ):
        original = fitted_binned_value_discretizer_factory(
            num_bins=16, num_dimensions=3
        )
        state = original.state_dict()
        restored = BinnedValueDiscretizer(num_bins=8)
        restored.load_state_dict(state)
        assert torch.allclose(restored.bin_edges, original.bin_edges)

    def test_load_state_dict_with_none_bin_edges(
        self, binned_value_discretizer_factory
    ):
        original = binned_value_discretizer_factory()
        state = original.state_dict()
        restored = BinnedValueDiscretizer(num_bins=8)
        restored.load_state_dict(state)
        assert restored.bin_edges is None
        assert restored._is_fitted is False

    def test_load_state_dict_with_none_bin_edges_clears_existing_edges(
        self,
        fitted_binned_value_discretizer_factory,
        binned_value_discretizer_factory,
    ):
        restored = fitted_binned_value_discretizer_factory()
        original = binned_value_discretizer_factory()
        state = original.state_dict()
        restored.load_state_dict(state)
        assert restored.bin_edges is None
        assert restored._is_fitted is False

    def test_loaded_tokenizer_can_encode(
        self, fitted_binned_value_discretizer_factory, rng
    ):
        original = fitted_binned_value_discretizer_factory(
            num_bins=16, num_dimensions=3
        )
        state = original.state_dict()
        restored = BinnedValueDiscretizer(num_bins=8)
        restored.load_state_dict(state)
        data = rng.standard_normal((5, 3)).astype(np.float32)
        original_tokens = original.encode(data)
        restored_tokens = restored.encode(data)
        assert torch.equal(original_tokens, restored_tokens)


class TestBinningStrategies:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown binning_strategy"):
            BinnedValueDiscretizer(num_bins=16, binning_strategy="log")

    def test_uniform_roundtrip_error_bounded_by_bin_width(
        self, binned_value_discretizer_factory, rng
    ):
        num_bins = 256
        tokenizer = binned_value_discretizer_factory(
            num_bins=num_bins,
            binning_strategy=BinningStrategy.UNIFORM.value,
        )
        tokenizer.fit(rng.uniform(-1, 1, (100, 2)).astype(np.float32))

        probe = torch.tensor([[-1.0, 1.0], [0.0, 0.5]])
        decoded = tokenizer.decode(tokenizer.encode(probe))
        bin_width = 2.0 / num_bins
        assert torch.abs(decoded - probe).max() <= bin_width

    def test_quantile_decodes_repeated_values_exactly(
        self, binned_value_discretizer_factory, rng
    ):
        # Regression: duplicate quantile edges on bimodal data (e.g. a binary
        # gripper) used to decode the minority mode to the bin midpoint
        # (+1 -> 0.0). Per-bin data means keep repeated values exact.
        tokenizer = binned_value_discretizer_factory(
            num_bins=256,
            binning_strategy=BinningStrategy.QUANTILE.value,
        )
        gripper = np.where(rng.random(5000) < 0.7, -1.0, 1.0)
        tokenizer.fit(gripper.reshape(-1, 1).astype(np.float32))

        probe = torch.tensor([[-1.0], [1.0]])
        decoded = tokenizer.decode(tokenizer.encode(probe))
        torch.testing.assert_close(decoded, probe)

    def test_legacy_state_dict_loads_as_quantile_with_geometric_centers(
        self, binned_value_discretizer_factory, rng
    ):
        tokenizer = binned_value_discretizer_factory(
            num_bins=16,
            binning_strategy=BinningStrategy.QUANTILE.value,
        )
        tokenizer.fit(rng.standard_normal((200, 2)).astype(np.float32))
        state = tokenizer.state_dict()
        del state["binning_strategy"]
        del state["bin_values"]

        restored = BinnedValueDiscretizer(num_bins=16)
        restored.load_state_dict(state)

        assert restored.binning_strategy == BinningStrategy.QUANTILE.value
        assert restored.bin_values is None
        probe = torch.tensor([[0.0, 0.0]])
        decoded = restored.decode(restored.encode(probe))
        assert decoded.shape == probe.shape
