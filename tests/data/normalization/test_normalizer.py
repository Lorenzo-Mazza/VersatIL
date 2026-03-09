"""Tests for versatil.data.normalization.normalizer module."""
import numpy as np
import pytest
import torch

from versatil.data.constants import KinematicsNormalizationType
from versatil.data.normalization.normalizer import (
    LinearNormalizer,
    SequentialNormalizer,
    SingleFieldLinearNormalizer,
)

ALL_MODES = [member.value for member in KinematicsNormalizationType]


@pytest.fixture
def gaussian_then_minmax_sequential(rng: np.random.Generator) -> SequentialNormalizer:
    """Two-stage sequential normalizer: gaussian → min_max."""
    data = torch.from_numpy(rng.standard_normal((100, 3)).astype(np.float32)) * 10.0 + 50.0
    first = SingleFieldLinearNormalizer.create_fit(
        data=data,
        mode=KinematicsNormalizationType.GAUSSIAN.value,
    )
    second = SingleFieldLinearNormalizer.create_fit(
        data=first.normalize(data),
        mode=KinematicsNormalizationType.MIN_MAX.value,
    )
    return SequentialNormalizer(normalizers=[first, second])


class TestSingleFieldLinearNormalizerFit:

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_fit_produces_scale_and_offset(self, rng: np.random.Generator, mode: str):
        data = torch.from_numpy(rng.standard_normal((100, 3)).astype(np.float32))
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data=data, mode=mode)

        assert "scale" in normalizer.params_dict
        assert "offset" in normalizer.params_dict
        assert "input_stats" in normalizer.params_dict

    def test_fit_stores_correct_input_stats(self):
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        stats = normalizer.get_input_stats()
        torch.testing.assert_close(stats["min"], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(stats["max"], torch.tensor([5.0, 6.0]))

    def test_fit_accepts_numpy_input(self, rng: np.random.Generator):
        data = rng.standard_normal((50, 3)).astype(np.float32)
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        assert "scale" in normalizer.params_dict

    def test_fit_parameters_have_no_gradients(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32))
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        for parameter in normalizer.parameters():
            assert not parameter.requires_grad


class TestSingleFieldLinearNormalizerNormalize:

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_unnormalize_inverts_normalize(self, rng: np.random.Generator, mode: str):
        data = torch.from_numpy(rng.standard_normal((100, 4)).astype(np.float32)) * 5.0 + 10.0
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data=data, mode=mode)

        recovered = normalizer.unnormalize(normalizer.normalize(data))

        torch.testing.assert_close(recovered, data, atol=1e-5, rtol=1e-5)

    def test_normalize_accepts_numpy_input(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32))
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        numpy_input = rng.standard_normal((10, 3)).astype(np.float32)
        result = normalizer.normalize(numpy_input)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 3)

    def test_normalize_preserves_higher_dimensional_shapes(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((20, 3)).astype(np.float32))
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        # 3D input: (batch, time, features)
        input_3d = torch.from_numpy(rng.standard_normal((4, 5, 3)).astype(np.float32))
        result = normalizer.normalize(input_3d)

        assert result.shape == (4, 5, 3)

    def test_callable_invokes_normalize(self):
        normalizer = SingleFieldLinearNormalizer.create_identity()
        data = torch.tensor([5.0])

        result = normalizer(data)

        torch.testing.assert_close(result, data)


class TestSingleFieldLinearNormalizerMinMax:

    def test_normalize_maps_to_output_range(self):
        data = torch.tensor([[0.0], [5.0], [10.0]])
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            output_min=-1.0,
            output_max=1.0,
        )

        result = normalizer.normalize(data)

        torch.testing.assert_close(result[0], torch.tensor([-1.0]))
        torch.testing.assert_close(result[2], torch.tensor([1.0]))

    def test_constant_dimension_handled(self):
        """Dimensions with zero range should not produce NaN or Inf."""
        data = torch.tensor([[1.0, 5.0], [1.0, 10.0], [1.0, 15.0]])
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
        )

        result = normalizer.normalize(data)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_clamp_range_prevents_extreme_scales(self):
        """Very small ranges should be clamped to min_range."""
        data = torch.tensor([[0.0, 0.0], [0.001, 0.001]])
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            clamp_range=True,
            min_range=0.04,
        )

        scale = normalizer.params_dict["scale"]
        # scale = (output_max - output_min) / clamped_range = 2.0 / 0.04 = 50.0
        assert torch.all(scale <= 50.0 + 1e-5)

    def test_no_offset_when_fit_offset_false(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((100, 3)).astype(np.float32))
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            fit_offset=False,
        )

        offset = normalizer.params_dict["offset"]
        torch.testing.assert_close(offset, torch.zeros_like(offset))


class TestSingleFieldLinearNormalizerGaussian:

    def test_normalize_centers_and_scales(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((1000, 2)).astype(np.float32)) * 5.0 + 10.0
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.GAUSSIAN.value,
        )

        result = normalizer.normalize(data)

        assert result.mean(dim=0).abs().max() < 0.1
        assert (result.std(dim=0) - 1.0).abs().max() < 0.1

    def test_clamp_std_prevents_extreme_scales(self):
        """Very small std should be clamped to min_std."""
        data = torch.tensor([[1.0], [1.0001], [1.0002]])
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.GAUSSIAN.value,
            clamp_range=True,
            min_std=0.02,
        )

        scale = normalizer.params_dict["scale"]
        # scale = 1 / clamped_std = 1 / 0.02 = 50.0
        assert torch.all(scale <= 50.0 + 1e-5)

    def test_no_offset_when_fit_offset_false(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((100, 3)).astype(np.float32)) * 5.0 + 10.0
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.GAUSSIAN.value,
            fit_offset=False,
        )

        offset = normalizer.params_dict["offset"]
        torch.testing.assert_close(offset, torch.zeros_like(offset))


class TestSingleFieldLinearNormalizerDemean:

    def test_demean_subtracts_mean_without_scaling(self):
        data = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.DEMEAN.value,
        )

        result = normalizer.normalize(data)

        torch.testing.assert_close(
            normalizer.params_dict["scale"],
            torch.ones(2),
        )
        assert result.mean(dim=0).abs().max() < 1e-5

    def test_no_offset_when_fit_offset_false(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((100, 3)).astype(np.float32)) + 50.0
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.DEMEAN.value,
            fit_offset=False,
        )

        offset = normalizer.params_dict["offset"]
        torch.testing.assert_close(offset, torch.zeros_like(offset))


class TestSingleFieldLinearNormalizerFactoryMethods:

    def test_create_fit_returns_fitted_normalizer(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32))
        normalizer = SingleFieldLinearNormalizer.create_fit(
            data=data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
        )

        assert "scale" in normalizer.params_dict
        result = normalizer.normalize(data)
        assert result.shape == data.shape

    def test_create_manual_sets_scale_and_offset(self):
        scale = torch.tensor([2.0, 3.0])
        offset = torch.tensor([1.0, -1.0])
        stats = {
            "min": torch.tensor([0.0, 0.0]),
            "max": torch.tensor([10.0, 10.0]),
        }

        normalizer = SingleFieldLinearNormalizer.create_manual(
            scale=scale, offset=offset, input_stats_dict=stats,
        )

        torch.testing.assert_close(normalizer.params_dict["scale"], scale)
        torch.testing.assert_close(normalizer.params_dict["offset"], offset)

    def test_create_identity_is_passthrough(self):
        normalizer = SingleFieldLinearNormalizer.create_identity()
        data = torch.tensor([1.0, 2.0, 3.0])

        result = normalizer.normalize(data)

        torch.testing.assert_close(result, data)


class TestSingleFieldLinearNormalizerOutputStats:

    def test_get_output_stats_returns_normalized_stats(self):
        data = torch.tensor([[0.0], [5.0], [10.0]])
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            output_min=-1.0,
            output_max=1.0,
        )

        output_stats = normalizer.get_output_stats()

        torch.testing.assert_close(output_stats["min"], torch.tensor([-1.0]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(output_stats["max"], torch.tensor([1.0]), atol=1e-5, rtol=1e-5)


class TestSingleFieldLinearNormalizerLastNDims:

    def test_last_n_dims_two_flattens_spatial(self, rng: np.random.Generator):
        """With last_n_dims=2, a (B, H, W) tensor treats H*W as feature dim."""
        data = torch.from_numpy(rng.standard_normal((10, 4, 4)).astype(np.float32))
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            last_n_dims=2,
            mode=KinematicsNormalizationType.MIN_MAX.value,
        )

        assert normalizer.params_dict["scale"].shape == (16,)

        result = normalizer.normalize(data)
        assert result.shape == data.shape

    def test_last_n_dims_zero_treats_all_as_batch(self, rng: np.random.Generator):
        """With last_n_dims=0, all dimensions are batch, feature dim is 1."""
        data = torch.from_numpy(rng.standard_normal((10, 3)).astype(np.float32))
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data=data,
            last_n_dims=0,
            mode=KinematicsNormalizationType.MIN_MAX.value,
        )

        assert normalizer.params_dict["scale"].shape == (1,)


class TestLinearNormalizerDictFit:

    def test_fit_with_dict_creates_per_key_params(self, rng: np.random.Generator):
        data = {
            "position": torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32)),
            "orientation": torch.from_numpy(rng.standard_normal((50, 4)).astype(np.float32)),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        assert "position" in normalizer.params_dict
        assert "orientation" in normalizer.params_dict

    def test_fit_with_tensor_uses_default_key(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32))
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        assert "_default" in normalizer.params_dict


class TestLinearNormalizerNormalize:

    def test_normalize_dict_applies_per_key(self, rng: np.random.Generator):
        data = {
            "position": torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32)),
            "orientation": torch.from_numpy(rng.standard_normal((50, 4)).astype(np.float32)),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        result = normalizer.normalize(data)

        assert "position" in result
        assert "orientation" in result
        assert result["position"].shape == (50, 3)
        assert result["orientation"].shape == (50, 4)

    def test_normalize_dict_skips_keys_without_params(self, rng: np.random.Generator):
        """Keys not in params_dict should pass through unchanged."""
        data = {"position": torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32))}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        language_tensor = torch.from_numpy(rng.integers(0, 100, (50, 10)).astype(np.int64))
        input_with_extra = {
            "position": data["position"],
            "language": language_tensor,
        }
        result = normalizer.normalize(input_with_extra)

        assert torch.equal(result["language"], language_tensor)

    def test_normalize_tensor_uses_default_params(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32))
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        result = normalizer.normalize(data)

        assert result.shape == data.shape

    def test_normalize_raises_when_not_initialized(self, rng: np.random.Generator):
        normalizer = LinearNormalizer()

        with pytest.raises(RuntimeError, match="Not initialized"):
            normalizer.normalize(torch.from_numpy(rng.standard_normal((10, 3)).astype(np.float32)))

    def test_unnormalize_dict_inverts_normalize(self, rng: np.random.Generator):
        data = {
            "position": torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32)),
            "orientation": torch.from_numpy(rng.standard_normal((50, 4)).astype(np.float32)),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        normalized = normalizer.normalize(data)
        recovered = normalizer.unnormalize(normalized)

        torch.testing.assert_close(
            recovered["position"], data["position"], atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            recovered["orientation"], data["orientation"], atol=1e-5, rtol=1e-5,
        )

    def test_callable_invokes_normalize(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32))
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        result = normalizer(data)

        torch.testing.assert_close(result, normalizer.normalize(data))


class TestLinearNormalizerSubscriptAccess:

    def test_getitem_returns_single_field_normalizer(self, rng: np.random.Generator):
        data = {"position": torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32))}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        single = normalizer["position"]

        assert isinstance(single, SingleFieldLinearNormalizer)

    def test_setitem_stores_single_field_normalizer(self):
        normalizer = LinearNormalizer()
        single = SingleFieldLinearNormalizer.create_identity()

        normalizer["custom_key"] = single

        retrieved = normalizer["custom_key"]
        assert isinstance(retrieved, SingleFieldLinearNormalizer)


class TestLinearNormalizerGetInputStats:

    def test_get_input_stats_dict_mode(self, rng: np.random.Generator):
        data = {
            "position": torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32)),
            "orientation": torch.from_numpy(rng.standard_normal((50, 4)).astype(np.float32)),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        stats = normalizer.get_input_stats()

        assert "position" in stats
        assert "orientation" in stats
        assert "min" in stats["position"]
        assert "max" in stats["position"]

    def test_get_input_stats_tensor_mode(self, rng: np.random.Generator):
        data = torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32))
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        stats = normalizer.get_input_stats()

        assert "min" in stats
        assert "max" in stats

    def test_get_input_stats_raises_when_not_initialized(self):
        normalizer = LinearNormalizer()

        with pytest.raises(RuntimeError, match="Not initialized"):
            normalizer.get_input_stats()


class TestLinearNormalizerGetOutputStats:

    def test_get_output_stats_tensor_fitted(self):
        data = torch.tensor([[0.0], [5.0], [10.0]])
        normalizer = LinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            output_min=-1.0,
            output_max=1.0,
        )

        output_stats = normalizer.get_output_stats(key="_default")

        torch.testing.assert_close(output_stats["min"], torch.tensor([-1.0]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(output_stats["max"], torch.tensor([1.0]), atol=1e-5, rtol=1e-5)

    def test_get_output_stats_dict_fitted(self):
        data = {
            "position": torch.tensor([[0.0, 0.0], [10.0, 10.0]]),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(
            data=data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            output_min=-1.0,
            output_max=1.0,
        )

        output_stats = normalizer.get_output_stats(key="position")

        torch.testing.assert_close(output_stats["min"], torch.tensor([-1.0, -1.0]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(output_stats["max"], torch.tensor([1.0, 1.0]), atol=1e-5, rtol=1e-5)


class TestLinearNormalizerStateDictPersistence:

    def test_state_dict_roundtrip(self, rng: np.random.Generator):
        data = {"position": torch.from_numpy(rng.standard_normal((50, 3)).astype(np.float32))}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, mode=KinematicsNormalizationType.MIN_MAX.value)

        state = normalizer.state_dict()
        loaded = LinearNormalizer()
        loaded.load_state_dict(state)

        test_input = {"position": torch.from_numpy(rng.standard_normal((10, 3)).astype(np.float32))}
        original_result = normalizer.normalize(test_input)
        loaded_result = loaded.normalize(test_input)

        torch.testing.assert_close(
            original_result["position"],
            loaded_result["position"],
        )


class TestSequentialNormalizerComposition:

    def test_applies_normalizers_in_order(
        self,
        rng: np.random.Generator,
        gaussian_then_minmax_sequential: SequentialNormalizer,
    ):
        data = torch.from_numpy(rng.standard_normal((20, 3)).astype(np.float32)) * 10.0 + 50.0
        result = gaussian_then_minmax_sequential.normalize(data)

        first, second = gaussian_then_minmax_sequential.normalizers
        expected = second.normalize(first.normalize(data))
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_unnormalize_inverts_normalize(
        self,
        rng: np.random.Generator,
        gaussian_then_minmax_sequential: SequentialNormalizer,
    ):
        data = torch.from_numpy(rng.standard_normal((20, 3)).astype(np.float32)) * 10.0 + 50.0
        recovered = gaussian_then_minmax_sequential.unnormalize(
            gaussian_then_minmax_sequential.normalize(data),
        )

        torch.testing.assert_close(recovered, data, atol=1e-4, rtol=1e-4)

    def test_empty_normalizers_raises(self):
        with pytest.raises(ValueError, match="At least one normalizer required"):
            SequentialNormalizer(normalizers=[])

    def test_fit_raises_not_implemented(self, rng: np.random.Generator):
        normalizer = SequentialNormalizer(
            normalizers=[SingleFieldLinearNormalizer.create_identity()],
        )

        with pytest.raises(NotImplementedError):
            normalizer.fit(data=torch.from_numpy(rng.standard_normal((10, 3)).astype(np.float32)))

    def test_get_input_stats_returns_first_normalizer_stats(
        self,
        gaussian_then_minmax_sequential: SequentialNormalizer,
    ):
        first = gaussian_then_minmax_sequential.normalizers[0]

        torch.testing.assert_close(
            gaussian_then_minmax_sequential.get_input_stats()["mean"],
            first.get_input_stats()["mean"],
        )

    def test_get_output_stats_propagates_through_all_normalizers(
        self,
        gaussian_then_minmax_sequential: SequentialNormalizer,
    ):
        output_stats = gaussian_then_minmax_sequential.get_output_stats()

        assert "min" in output_stats
        assert "max" in output_stats
        assert "mean" in output_stats
        assert "std" in output_stats


class TestSequentialNormalizerLinearNormalizerIntegration:

    def test_setitem_and_getitem_roundtrip(
        self,
        gaussian_then_minmax_sequential: SequentialNormalizer,
    ):
        linear = LinearNormalizer()
        linear["position"] = gaussian_then_minmax_sequential

        retrieved = linear["position"]

        assert isinstance(retrieved, SequentialNormalizer)

    def test_sequential_stored_in_linear_normalizer_produces_same_results(
        self,
        rng: np.random.Generator,
        gaussian_then_minmax_sequential: SequentialNormalizer,
    ):
        linear = LinearNormalizer()
        linear["position"] = gaussian_then_minmax_sequential

        test_input = torch.from_numpy(rng.standard_normal((10, 3)).astype(np.float32)) * 10.0
        direct_result = gaussian_then_minmax_sequential.normalize(test_input)

        retrieved = linear["position"]
        retrieved_result = retrieved.normalize(test_input)

        torch.testing.assert_close(
            retrieved_result, direct_result, atol=1e-5, rtol=1e-5,
        )

    def test_state_dict_roundtrip_with_sequential(
        self,
        rng: np.random.Generator,
        gaussian_then_minmax_sequential: SequentialNormalizer,
    ):
        linear = LinearNormalizer()
        linear["position"] = gaussian_then_minmax_sequential

        state = linear.state_dict()
        loaded = LinearNormalizer()
        loaded.load_state_dict(state)

        test_input = torch.from_numpy(rng.standard_normal((10, 3)).astype(np.float32)) * 10.0
        original_result = linear["position"].normalize(test_input)
        loaded_result = loaded["position"].normalize(test_input)

        torch.testing.assert_close(
            loaded_result, original_result, atol=1e-5, rtol=1e-5,
        )