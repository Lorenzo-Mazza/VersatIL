"""Tests for versatil.data.preprocessing.sharding module."""

import re
from collections.abc import Callable

import pytest

from versatil.data.preprocessing.sharding import (
    get_image_shard_shape,
    is_uint8_image_spec,
    validate_image_frames_per_shard,
)


class TestIsUint8ImageSpec:
    @pytest.mark.parametrize(
        "shape,dtype,expected",
        [
            ((0, 64, 64, 3), "uint8", True),
            ((0, 7), "float32", False),
            ((0, 64, 64, 3), "float32", False),
            ((0,), "str", False),
            ((0, 7), "uint8", False),
        ],
        ids=[
            "4d_uint8_image",
            "2d_float32_numerical",
            "4d_float32_not_uint8",
            "1d_string",
            "2d_uint8_not_image",
        ],
    )
    def test_returns_expected_for_various_specs(
        self,
        spec_factory: Callable[..., dict],
        shape: tuple,
        dtype: str,
        expected: bool,
    ):
        spec = spec_factory(shape=shape, dtype=dtype)

        assert is_uint8_image_spec(spec) is expected


class TestGetImageShardShape:
    @pytest.mark.parametrize(
        "image_chunks,image_frames_per_shard,expected",
        [
            ((1, 64, 64, 3), 64, (64, 64, 64, 3)),
            ((16, 32, 32, 3), 64, (64, 32, 32, 3)),
            ((1, 64, 64, 3), None, None),
        ],
    )
    def test_returns_requested_shape(
        self,
        image_chunks: tuple[int, ...],
        image_frames_per_shard: int | None,
        expected: tuple[int, ...] | None,
    ):
        result = get_image_shard_shape(
            image_chunks=image_chunks,
            image_frames_per_shard=image_frames_per_shard,
        )

        assert result == expected

    def test_rejects_shard_length_not_aligned_with_chunks(self):
        image_frames_per_shard = 50
        image_chunk_length = 16
        error_message = (
            "image_frames_per_shard must be a multiple of image chunk "
            f"length {image_chunk_length}, got {image_frames_per_shard}."
        )

        with pytest.raises(ValueError, match=re.escape(error_message)):
            get_image_shard_shape(
                image_chunks=(image_chunk_length, 64, 64, 3),
                image_frames_per_shard=image_frames_per_shard,
            )


class TestValidateImageFramesPerShard:
    @pytest.mark.parametrize("image_frames_per_shard", [0, -1])
    def test_rejects_non_positive_shard_lengths(
        self,
        image_frames_per_shard: int,
    ):
        error_message = (
            f"image_frames_per_shard must be positive, got {image_frames_per_shard}."
        )

        with pytest.raises(ValueError, match=re.escape(error_message)):
            validate_image_frames_per_shard(
                image_frames_per_shard=image_frames_per_shard,
            )

    @pytest.mark.parametrize("image_frames_per_shard", [64.5, True])
    def test_rejects_non_integer_shard_length(
        self,
        image_frames_per_shard: float | bool,
    ):
        error_message = (
            "image_frames_per_shard must be an integer or None, "
            f"got {type(image_frames_per_shard)}."
        )

        with pytest.raises(TypeError, match=re.escape(error_message)):
            validate_image_frames_per_shard(
                image_frames_per_shard=image_frames_per_shard,
            )

    @pytest.mark.parametrize("image_frames_per_shard", [None, 1, 64])
    def test_accepts_disabled_or_positive_integer_shard_lengths(
        self,
        image_frames_per_shard: int | None,
    ):
        validate_image_frames_per_shard(
            image_frames_per_shard=image_frames_per_shard,
        )
