"""Tests for versatil.data.preprocessing.codecs module."""

import asyncio
import re
from collections.abc import Callable

import cv2
import numpy as np
import pytest
from numcodecs import Blosc
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import NDBuffer, default_buffer_prototype
from zarr.core.dtype import UInt8

from versatil.data.preprocessing.codecs import (
    WEBP_CODEC_NAME,
    WebPCodec,
    _protective_squeeze,
)


@pytest.fixture
def array_spec_factory() -> Callable[..., ArraySpec]:
    """Factory for creating zarr ArraySpec with configurable shape."""

    def factory(
        shape: tuple[int, ...] = (1, 64, 64, 3),
    ) -> ArraySpec:
        return ArraySpec(
            shape=shape,
            dtype=UInt8(),
            fill_value=0,
            config=ArrayConfig(order="C", write_empty_chunks=True),
            prototype=default_buffer_prototype(),
        )

    return factory


@pytest.fixture
def noisy_image_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    """Factory for random uint8 images shaped (N, H, W, C)."""

    def factory(
        batch_size: int = 1,
        image_height: int = 32,
        image_width: int = 32,
        channels: int = 3,
    ) -> np.ndarray:
        return rng.integers(
            0,
            255,
            size=(batch_size, image_height, image_width, channels),
            dtype=np.uint8,
        )

    return factory


@pytest.fixture
def gradient_image_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    """Factory for a smooth gradient plus sensor noise resembling camera frames."""

    def factory(
        image_height: int = 128,
        image_width: int = 128,
        noise_amplitude: int = 10,
    ) -> np.ndarray:
        rows = np.linspace(0, 255, image_height).astype(np.uint8)
        columns = np.linspace(0, 200, image_width).astype(np.uint8)
        image = np.zeros((1, image_height, image_width, 3), dtype=np.uint8)
        image[0, :, :, 0] = rows[:, None]
        image[0, :, :, 1] = columns[None, :]
        image[0, :, :, 2] = 128
        noise = rng.integers(
            -noise_amplitude,
            noise_amplitude,
            size=(image_height, image_width, 3),
            dtype=np.int16,
        )
        image[0] = np.clip(image[0].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return image

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_shape, expected_shape",
    [
        ((32, 32, 3), (32, 32, 3)),
        ((1, 32, 32, 3), (32, 32, 3)),
        ((1, 1, 32, 32, 3), (32, 32, 3)),
        ((4, 32, 32, 3), (4, 32, 32, 3)),
        ((2, 3, 16, 16, 3), (6, 16, 16, 3)),
    ],
    ids=[
        "3d_hwc_unchanged",
        "single_batch_squeezed",
        "nested_unit_batch_squeezed",
        "multi_image_batch_preserved",
        "multi_dim_batch_flattened",
    ],
)
def test_protective_squeeze_output_shape(
    rng: np.random.Generator,
    input_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
):
    array = rng.integers(0, 255, size=input_shape, dtype=np.uint8)

    result = _protective_squeeze(array)

    assert result.shape == expected_shape


class TestWebPCodecInit:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "level",
        [1, 50, 99, 100],
        ids=["minimum", "medium", "high", "maximum"],
    )
    def test_stores_explicit_quality_level(self, level: int):
        codec = WebPCodec(level=level)

        assert codec.level == level

    @pytest.mark.unit
    def test_default_quality_level(self):
        codec = WebPCodec()

        assert codec.level == 99

    @pytest.mark.unit
    def test_frozen_dataclass_prevents_mutation(self):
        codec = WebPCodec(level=80)

        with pytest.raises(
            AttributeError, match=re.escape("cannot assign to field 'level'")
        ):
            codec.level = 50


class TestWebPCodecSerialization:
    @pytest.mark.unit
    def test_to_dict_includes_name_and_level(self):
        codec = WebPCodec(level=75)

        result = codec.to_dict()

        assert result == {
            "name": WEBP_CODEC_NAME,
            "configuration": {"level": 75},
        }

    @pytest.mark.unit
    def test_from_dict_restores_level(self):
        serialized = {
            "name": WEBP_CODEC_NAME,
            "configuration": {"level": 85},
        }

        codec = WebPCodec.from_dict(serialized)

        assert codec.level == 85

    @pytest.mark.unit
    def test_from_dict_without_configuration_uses_default(self):
        serialized = {"name": WEBP_CODEC_NAME}

        codec = WebPCodec.from_dict(serialized)

        assert codec.level == 99

    @pytest.mark.unit
    @pytest.mark.parametrize("level", [1, 42, 100])
    def test_roundtrip_preserves_level(self, level: int):
        original = WebPCodec(level=level)

        restored = WebPCodec.from_dict(original.to_dict())

        assert restored.level == level


class TestWebPCodecEncode:
    @pytest.mark.integration
    def test_encode_produces_non_empty_buffer(
        self,
        array_spec_factory: Callable[..., ArraySpec],
        noisy_image_factory: Callable[..., np.ndarray],
    ):
        spec = array_spec_factory(shape=(1, 32, 32, 3))
        image = noisy_image_factory(image_height=32, image_width=32)
        nd_buffer = NDBuffer.from_ndarray_like(image)
        codec = WebPCodec(level=99)

        result = asyncio.run(codec._encode_single(nd_buffer, spec))

        assert result is not None
        encoded_bytes = np.asarray(result.as_array_like())
        assert encoded_bytes.nbytes > 0

    @pytest.mark.integration
    def test_encode_rgb_converts_to_bgr_for_opencv(
        self,
        array_spec_factory: Callable[..., ArraySpec],
    ):
        # Pure red image: RGB (255, 0, 0) should become BGR (0, 0, 255)
        spec = array_spec_factory(shape=(1, 8, 8, 3))
        red_image = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        red_image[..., 0] = 255
        nd_buffer = NDBuffer.from_ndarray_like(red_image)
        codec = WebPCodec(level=100)

        result = asyncio.run(codec._encode_single(nd_buffer, spec))

        # Decode with raw OpenCV to verify BGR storage
        encoded_bytes = np.asarray(result.as_array_like())
        bgr_decoded = cv2.imdecode(encoded_bytes, cv2.IMREAD_UNCHANGED)
        # In BGR ordering, index 2 is the red channel
        assert bgr_decoded[0, 0, 2] == 255
        assert bgr_decoded[0, 0, 0] == 0

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "level",
        [50, 99],
        ids=["medium_quality", "high_quality"],
    )
    def test_higher_quality_produces_larger_encoded_output(
        self,
        array_spec_factory: Callable[..., ArraySpec],
        noisy_image_factory: Callable[..., np.ndarray],
        level: int,
    ):
        spec = array_spec_factory(shape=(1, 32, 32, 3))
        image = noisy_image_factory(image_height=32, image_width=32)
        nd_buffer = NDBuffer.from_ndarray_like(image)
        low_codec = WebPCodec(level=1)
        high_codec = WebPCodec(level=level)

        low_result = asyncio.run(low_codec._encode_single(nd_buffer, spec))
        high_result = asyncio.run(high_codec._encode_single(nd_buffer, spec))

        low_size = np.asarray(low_result.as_array_like()).nbytes
        high_size = np.asarray(high_result.as_array_like()).nbytes
        assert high_size >= low_size


class TestWebPCodecDecode:
    @pytest.mark.integration
    def test_decode_restores_original_shape(
        self,
        array_spec_factory: Callable[..., ArraySpec],
        noisy_image_factory: Callable[..., np.ndarray],
    ):
        spec = array_spec_factory(shape=(1, 32, 32, 3))
        image = noisy_image_factory(image_height=32, image_width=32)
        nd_buffer = NDBuffer.from_ndarray_like(image)
        codec = WebPCodec(level=99)

        encoded = asyncio.run(codec._encode_single(nd_buffer, spec))
        decoded = asyncio.run(codec._decode_single(encoded, spec))

        decoded_array = np.asarray(decoded.as_ndarray_like())
        assert decoded_array.shape == (1, 32, 32, 3)
        assert decoded_array.dtype == np.uint8

    @pytest.mark.integration
    def test_decode_bgr_to_rgb_preserves_dominant_channel(
        self,
        array_spec_factory: Callable[..., ArraySpec],
    ):
        # Encode a pure-red image and verify red channel remains dominant
        spec = array_spec_factory(shape=(1, 8, 8, 3))
        red_image = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        red_image[..., 0] = 255
        nd_buffer = NDBuffer.from_ndarray_like(red_image)
        codec = WebPCodec(level=100)

        encoded = asyncio.run(codec._encode_single(nd_buffer, spec))
        decoded = asyncio.run(codec._decode_single(encoded, spec))

        decoded_array = np.asarray(decoded.as_ndarray_like())
        # Red channel should be dominant after round-trip
        assert decoded_array[0, 0, 0, 0] > 250
        assert decoded_array[0, 0, 0, 1] < 5
        assert decoded_array[0, 0, 0, 2] < 5

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "level, max_compression_ratio",
        [
            (1, 0.10),
            (50, 0.25),
            (99, 0.50),
        ],
        ids=["low_quality", "medium_quality", "high_quality"],
    )
    def test_encoded_size_below_expected_ratio(
        self,
        array_spec_factory: Callable[..., ArraySpec],
        noisy_image_factory: Callable[..., np.ndarray],
        level: int,
        max_compression_ratio: float,
    ):
        spec = array_spec_factory(shape=(1, 64, 64, 3))
        image = noisy_image_factory(image_height=64, image_width=64)
        nd_buffer = NDBuffer.from_ndarray_like(image)
        codec = WebPCodec(level=level)

        encoded = asyncio.run(codec._encode_single(nd_buffer, spec))

        encoded_size = np.asarray(encoded.as_array_like()).nbytes
        compression_ratio = encoded_size / image.nbytes
        assert compression_ratio < max_compression_ratio

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "level, max_compression_ratio, max_mean_absolute_error",
        [
            (1, 0.05, 3.0),
            (50, 0.05, 1.0),
            (99, 0.05, 0.1),
        ],
        ids=["low_quality", "medium_quality", "high_quality"],
    )
    def test_correlated_image_roundtrip_compression_and_error(
        self,
        array_spec_factory: Callable[..., ArraySpec],
        level: int,
        max_compression_ratio: float,
        max_mean_absolute_error: float,
    ):
        # Circle on black background: spatially correlated like a natural image
        spec = array_spec_factory(shape=(1, 64, 64, 3))
        y_grid, x_grid = np.mgrid[:64, :64]
        circle_mask = ((y_grid - 32) ** 2 + (x_grid - 32) ** 2) < 20**2
        image = np.zeros((1, 64, 64, 3), dtype=np.uint8)
        image[0, circle_mask] = 255
        nd_buffer = NDBuffer.from_ndarray_like(image)
        codec = WebPCodec(level=level)

        encoded = asyncio.run(codec._encode_single(nd_buffer, spec))
        decoded = asyncio.run(codec._decode_single(encoded, spec))

        encoded_size = np.asarray(encoded.as_array_like()).nbytes
        compression_ratio = encoded_size / image.nbytes
        assert compression_ratio < max_compression_ratio

        decoded_array = np.asarray(decoded.as_ndarray_like())
        mean_absolute_error = np.mean(
            np.abs(image.astype(np.float32) - decoded_array.astype(np.float32))
        )
        assert mean_absolute_error < max_mean_absolute_error


@pytest.mark.integration
def test_webp_compresses_much_better_than_blosc_with_low_error(
    array_spec_factory: Callable[..., ArraySpec],
    gradient_image_factory: Callable[..., np.ndarray],
):
    spec = array_spec_factory(shape=(1, 128, 128, 3))
    image = gradient_image_factory(image_height=128, image_width=128)
    nd_buffer = NDBuffer.from_ndarray_like(image)
    webp_codec = WebPCodec(level=99)

    webp_encoded = asyncio.run(webp_codec._encode_single(nd_buffer, spec))
    webp_decoded = asyncio.run(webp_codec._decode_single(webp_encoded, spec))
    webp_size = np.asarray(webp_encoded.as_array_like()).nbytes

    blosc_codec = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    blosc_size = len(blosc_codec.encode(image.tobytes()))

    assert webp_size < blosc_size * 0.25

    decoded_array = np.asarray(webp_decoded.as_ndarray_like())
    mean_absolute_error = np.mean(
        np.abs(image.astype(np.float32) - decoded_array.astype(np.float32))
    )
    assert mean_absolute_error < 5.0


@pytest.mark.unit
def test_compute_encoded_size_raises_not_implemented(
    array_spec_factory: Callable[..., ArraySpec],
):
    codec = WebPCodec(level=99)
    spec = array_spec_factory(shape=(1, 64, 64, 3))

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "WebP encoded size is data-dependent and cannot be precomputed."
        ),
    ):
        codec.compute_encoded_size(input_byte_length=1024, _chunk_spec=spec)
