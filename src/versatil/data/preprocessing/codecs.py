"""Zarr v3 codecs for image compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import cv2
import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

WEBP_CODEC_NAME = "versatil_webp"


def _protective_squeeze(x: np.ndarray) -> np.ndarray:
    """Squeeze batch dims while preserving the last 3 image dims (H, W, C).

    Args:
        x: Array with shape (..., H, W, C).

    Returns:
        Array with shape (H, W, C) or (N, H, W, C) if multiple images.
    """
    img_shape = x.shape[-3:]
    if len(x.shape) > 3:
        n_imgs = int(np.prod(x.shape[:-3]))
        if n_imgs > 1:
            img_shape = (-1,) + img_shape
    return x.reshape(img_shape)


@dataclass(frozen=True)
class WebPCodec(ArrayBytesCodec):
    """WebP image codec for zarr v3 using OpenCV.

    Compresses uint8 image array chunks using WebP encoding.
    Chunks should contain a single image with shape (1, H, W, C)
    for optimal compression.

    Args:
        level: WebP quality level (1-100). Higher is better quality
            but larger files. Default 99 provides near-lossless quality
            at ~35% less space than JPEG at the same quality.
    """

    is_fixed_size = False
    level: int

    def __init__(self, level: int = 99) -> None:
        object.__setattr__(self, "level", level)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Build the codec from a Zarr named-configuration dict."""
        _, config = parse_named_configuration(
            data, WEBP_CODEC_NAME, require_configuration=False
        )
        config = config or {}
        return cls(**config)

    def to_dict(self) -> dict[str, JSON]:
        """Serialize the codec as a Zarr named-configuration dict."""
        return {
            "name": WEBP_CODEC_NAME,
            "configuration": {"level": self.level},
        }

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        buf = np.asarray(chunk_bytes.as_array_like())
        image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image[..., ::-1]  # BGR -> RGB
        image = np.ascontiguousarray(image.reshape(chunk_spec.shape))
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(image)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = np.asarray(chunk_array.as_ndarray_like())
        arr = _protective_squeeze(arr)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.ascontiguousarray(arr[..., ::-1])  # RGB -> BGR
        _, encoded = cv2.imencode(".webp", arr, [cv2.IMWRITE_WEBP_QUALITY, self.level])
        return chunk_spec.prototype.buffer.from_array_like(encoded.ravel())

    def compute_encoded_size(
        self, input_byte_length: int, _chunk_spec: ArraySpec
    ) -> int:
        """Encoded size is data-dependent for WebP, so it cannot be precomputed."""
        raise NotImplementedError(
            "WebP encoded size is data-dependent and cannot be precomputed."
        )


register_codec(key=WEBP_CODEC_NAME, codec_cls=WebPCodec)
