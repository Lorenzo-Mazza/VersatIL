"""Tests for versatil.data.normalization.image_normalizer module."""

import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.data.constants import (
    CLIP_RGB_MEAN,
    CLIP_RGB_STD,
    ImageNormalizationType,
)
from versatil.data.normalization.image_normalizer import (
    _to_tensor,
    array_to_stats,
    create_image_normalizer,
    get_depth_image_normalizer,
    get_range_normalizer_from_stat,
    get_rgb_image_normalizer,
)
from versatil.data.normalization.normalizer import (
    SequentialNormalizer,
    SingleFieldLinearNormalizer,
)

ALL_IMAGE_NORM_TYPES = [member.value for member in ImageNormalizationType]
RGB_ONLY_IMAGE_NORM_TYPES = [
    ImageNormalizationType.CLIP.value,
]
DEPTH_IMAGE_NORM_TYPES = [
    norm_type
    for norm_type in ALL_IMAGE_NORM_TYPES
    if norm_type not in RGB_ONLY_IMAGE_NORM_TYPES
]


@pytest.fixture
def rgb_image_tensor(rng: np.random.Generator) -> Callable[..., torch.Tensor]:
    """Factory for creating RGB image tensors in [0, 1] range."""

    def factory(
        batch_size: int = 2,
        channels: int = 3,
    ) -> torch.Tensor:
        return torch.from_numpy(rng.random((batch_size, channels)).astype(np.float32))

    return factory


@pytest.fixture
def depth_stats() -> dict[str, float]:
    """Standard depth image statistics for testing."""
    return {
        "input_min": 0.5,
        "input_max": 5.0,
        "input_mean": 2.75,
        "input_std": 1.3,
    }


class TestToTensor:
    def test_converts_numpy_array(self):
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _to_tensor(array)

        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_converts_zero_dimensional_numpy_array(self):
        scalar = np.float32(5.0)
        result = _to_tensor(scalar)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)
        torch.testing.assert_close(result, torch.tensor([5.0]))

    def test_converts_python_scalar(self):
        result = _to_tensor(3.14)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    def test_converts_numpy_scalar_type(self):
        result = _to_tensor(np.float64(2.0))

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    def test_respects_device_parameter(self):
        result = _to_tensor(1.0, device=torch.device("cpu"))

        assert result.device == torch.device("cpu")


class TestGetRgbImageNormalizer:
    @pytest.mark.parametrize("norm_type", ALL_IMAGE_NORM_TYPES)
    def test_returns_normalizer_for_all_types(self, norm_type: str):
        normalizer = get_rgb_image_normalizer(norm_type=norm_type)

        assert isinstance(
            normalizer, (SingleFieldLinearNormalizer, SequentialNormalizer)
        )

    def test_zero_to_one_maps_unit_range_to_unit_range(
        self,
        rgb_image_tensor: Callable[..., torch.Tensor],
    ):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
        )

        zeros = torch.zeros(1, 3)
        ones = torch.ones(1, 3)

        torch.testing.assert_close(
            normalizer.normalize(zeros), zeros, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            normalizer.normalize(ones), ones, atol=1e-5, rtol=1e-5
        )

    def test_minus_one_to_one_maps_to_symmetric_range(self):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.MINUS_ONE_TO_ONE.value,
        )

        torch.testing.assert_close(
            normalizer.normalize(torch.zeros(1, 3)),
            torch.full((1, 3), -1.0),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            normalizer.normalize(torch.ones(1, 3)),
            torch.ones(1, 3),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_imagenet_returns_standardization_normalizer(self):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.IMAGENET.value,
        )

        assert isinstance(normalizer, SingleFieldLinearNormalizer)

    def test_clip_uses_clip_processor_statistics(self):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.CLIP.value,
        )
        clip_mean = torch.tensor([[0.48145466, 0.4578275, 0.40821073]])

        normalized = normalizer.normalize(clip_mean)

        torch.testing.assert_close(
            normalized,
            torch.zeros_like(clip_mean),
            atol=1e-6,
            rtol=1e-6,
        )

    @pytest.mark.parametrize("norm_type", ALL_IMAGE_NORM_TYPES)
    def test_unnormalize_inverts_normalize(
        self,
        norm_type: str,
        rgb_image_tensor: Callable[..., torch.Tensor],
    ):
        normalizer = get_rgb_image_normalizer(norm_type=norm_type)

        image = rgb_image_tensor()
        recovered = normalizer.unnormalize(normalizer.normalize(image))

        torch.testing.assert_close(recovered, image, atol=1e-4, rtol=1e-4)


class TestGetDepthImageNormalizer:
    @pytest.mark.parametrize("norm_type", DEPTH_IMAGE_NORM_TYPES)
    def test_returns_normalizer_for_all_types(
        self,
        norm_type: str,
        depth_stats: dict[str, float],
    ):
        normalizer = get_depth_image_normalizer(
            **depth_stats,
            norm_type=norm_type,
        )

        assert isinstance(
            normalizer, (SingleFieldLinearNormalizer, SequentialNormalizer)
        )

    @pytest.mark.parametrize("norm_type", RGB_ONLY_IMAGE_NORM_TYPES)
    def test_rgb_only_normalization_types_raise_for_depth(
        self,
        norm_type: str,
        depth_stats: dict[str, float],
    ):
        expected_error = (
            f"Depth normalization type '{norm_type}' is RGB-only. "
            "Use one of: ['zero_to_one', 'minus_one_to_one', 'imagenet']"
        )
        with pytest.raises(ValueError, match=re.escape(expected_error)):
            get_depth_image_normalizer(
                **depth_stats,
                norm_type=norm_type,
            )

    def test_zero_to_one_maps_depth_range_to_unit(
        self,
        depth_stats: dict[str, float],
    ):
        normalizer = get_depth_image_normalizer(
            **depth_stats,
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
        )

        torch.testing.assert_close(
            normalizer.normalize(torch.tensor([depth_stats["input_min"]])),
            torch.tensor([0.0]),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            normalizer.normalize(torch.tensor([depth_stats["input_max"]])),
            torch.tensor([1.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_imagenet_returns_sequential_normalizer(
        self,
        depth_stats: dict[str, float],
    ):
        normalizer = get_depth_image_normalizer(
            **depth_stats,
            norm_type=ImageNormalizationType.IMAGENET.value,
        )

        assert isinstance(normalizer, SequentialNormalizer)

    @pytest.mark.parametrize("norm_type", DEPTH_IMAGE_NORM_TYPES)
    def test_unnormalize_inverts_normalize(
        self,
        norm_type: str,
        depth_stats: dict[str, float],
    ):
        normalizer = get_depth_image_normalizer(
            **depth_stats,
            norm_type=norm_type,
        )

        depth_values = torch.tensor([1.0, 2.5, 4.0])
        recovered = normalizer.unnormalize(normalizer.normalize(depth_values))

        torch.testing.assert_close(recovered, depth_values, atol=1e-4, rtol=1e-4)


class TestCreateImageNormalizer:
    def test_without_standardization_returns_single_field(self):
        normalizer = create_image_normalizer(
            input_min=0.0,
            input_max=255.0,
            input_mean=127.5,
            input_std=73.9,
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
        )

        assert isinstance(normalizer, SingleFieldLinearNormalizer)

    def test_with_standardization_returns_sequential(self):
        normalizer = create_image_normalizer(
            input_min=0.0,
            input_max=255.0,
            input_mean=127.5,
            input_std=73.9,
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
            standardization_mean=np.array([0.485], dtype=np.float32),
            standardization_std=np.array([0.229], dtype=np.float32),
        )

        assert isinstance(normalizer, SequentialNormalizer)

    @pytest.mark.parametrize(
        ("norm_type", "mean", "std"),
        [
            (
                ImageNormalizationType.CLIP.value,
                torch.tensor(CLIP_RGB_MEAN),
                torch.tensor(CLIP_RGB_STD),
            ),
        ],
    )
    def test_rgb_only_pretrained_types_use_default_standardization(
        self,
        norm_type: str,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        normalizer = create_image_normalizer(
            input_min=0.0,
            input_max=255.0,
            input_mean=127.5,
            input_std=np.sqrt((255.0**2) / 12.0),
            norm_type=norm_type,
        )

        assert isinstance(normalizer, SequentialNormalizer)
        torch.testing.assert_close(
            normalizer.normalize(mean * 255.0),
            torch.zeros_like(mean),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            normalizer.normalize(torch.zeros_like(mean)),
            -mean / std,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_requires_complete_standardization_stats(self):
        with pytest.raises(
            ValueError,
            match="standardization_mean and standardization_std must be provided together",
        ):
            create_image_normalizer(
                input_min=0.0,
                input_max=255.0,
                input_mean=127.5,
                input_std=73.9,
                norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
                standardization_mean=np.array([0.485], dtype=np.float32),
            )

    def test_unsupported_norm_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported normalization type"):
            create_image_normalizer(
                input_min=0.0,
                input_max=1.0,
                input_mean=0.5,
                input_std=0.3,
                norm_type="unsupported",
            )

    def test_multi_channel_input(
        self,
        rgb_image_tensor: Callable[..., torch.Tensor],
    ):
        normalizer = create_image_normalizer(
            input_min=np.zeros(3, dtype=np.float32),
            input_max=np.ones(3, dtype=np.float32),
            input_mean=np.full(3, 0.5, dtype=np.float32),
            input_std=np.full(3, 0.3, dtype=np.float32),
            norm_type=ImageNormalizationType.MINUS_ONE_TO_ONE.value,
        )

        image = rgb_image_tensor()
        result = normalizer.normalize(image)
        assert result.shape == image.shape

    def test_with_device_parameter(self):
        normalizer = create_image_normalizer(
            input_min=0.0,
            input_max=1.0,
            input_mean=0.5,
            input_std=0.3,
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value,
            device=torch.device("cpu"),
        )

        result = normalizer.normalize(torch.tensor([0.5]))
        assert result.device == torch.device("cpu")


class TestGetRangeNormalizerFromStat:
    def test_creates_normalizer_from_precomputed_stats(self):
        stat = {
            "min": torch.tensor([0.0, 0.0]),
            "max": torch.tensor([10.0, 10.0]),
            "mean": torch.tensor([5.0, 5.0]),
            "std": torch.tensor([2.9, 2.9]),
        }

        normalizer = get_range_normalizer_from_stat(
            stat=stat,
            output_min=-1.0,
            output_max=1.0,
        )

        assert isinstance(normalizer, SingleFieldLinearNormalizer)
        torch.testing.assert_close(
            normalizer.normalize(stat["min"]),
            torch.tensor([-1.0, -1.0]),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            normalizer.normalize(stat["max"]),
            torch.tensor([1.0, 1.0]),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_handles_constant_dimensions(self):
        stat = {
            "min": torch.tensor([5.0, 0.0]),
            "max": torch.tensor([5.0, 10.0]),
            "mean": torch.tensor([5.0, 5.0]),
            "std": torch.tensor([0.0, 2.9]),
        }

        normalizer = get_range_normalizer_from_stat(stat=stat)

        result = normalizer.normalize(stat["min"])
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))


class TestArrayToStats:
    def test_computes_correct_statistics(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        stats = array_to_stats(data)

        np.testing.assert_allclose(stats["min"], [1.0, 2.0])
        np.testing.assert_allclose(stats["max"], [5.0, 6.0])
        np.testing.assert_allclose(stats["mean"], [3.0, 4.0])

    def test_single_column_array(self):
        data = np.array([[1.0], [2.0], [3.0]])

        stats = array_to_stats(data)

        np.testing.assert_allclose(stats["min"], [1.0])
        np.testing.assert_allclose(stats["max"], [3.0])
