"""Tests for versatil.explainability.visualization module."""

import re
from collections.abc import Callable

import numpy as np
import pytest

from versatil.explainability.visualization import show_cam_on_image


@pytest.fixture
def unit_image_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    def factory(height: int = 8, width: int = 8) -> np.ndarray:
        return rng.random((height, width, 3)).astype(np.float32)

    return factory


@pytest.fixture
def unit_mask_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    def factory(height: int = 8, width: int = 8) -> np.ndarray:
        return rng.random((height, width)).astype(np.float32)

    return factory


class TestShowCamOnImage:
    def test_returns_uint8_overlay_with_image_shape(
        self,
        unit_image_factory: Callable[..., np.ndarray],
        unit_mask_factory: Callable[..., np.ndarray],
    ):
        image = unit_image_factory()
        mask = unit_mask_factory()

        overlay = show_cam_on_image(image=image, mask=mask, image_weight=0.5)

        assert overlay.shape == image.shape
        assert overlay.dtype == np.uint8

    def test_full_image_weight_returns_scaled_image(
        self,
        unit_image_factory: Callable[..., np.ndarray],
    ):
        image = unit_image_factory()
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        overlay = show_cam_on_image(image=image, mask=mask, image_weight=1.0)

        np.testing.assert_array_equal(overlay, (255 * image).astype(np.uint8))

    def test_use_rgb_reverses_heatmap_channels(
        self,
        unit_mask_factory: Callable[..., np.ndarray],
    ):
        image = np.zeros((8, 8, 3), dtype=np.float32)
        mask = unit_mask_factory()

        bgr_overlay = show_cam_on_image(
            image=image, mask=mask, use_rgb=False, image_weight=0.0
        )
        rgb_overlay = show_cam_on_image(
            image=image, mask=mask, use_rgb=True, image_weight=0.0
        )

        np.testing.assert_array_equal(rgb_overlay, bgr_overlay[:, :, ::-1])

    def test_raises_when_image_exceeds_unit_range(
        self,
        unit_mask_factory: Callable[..., np.ndarray],
    ):
        image = np.full((8, 8, 3), 255.0, dtype=np.float32)
        mask = unit_mask_factory()

        with pytest.raises(
            ValueError,
            match=re.escape("The input image should be np.float32 in the range [0, 1]"),
        ):
            show_cam_on_image(image=image, mask=mask)

    def test_raises_when_image_weight_out_of_range(
        self,
        unit_image_factory: Callable[..., np.ndarray],
        unit_mask_factory: Callable[..., np.ndarray],
    ):
        image = unit_image_factory()
        mask = unit_mask_factory()

        with pytest.raises(
            ValueError,
            match=re.escape("image_weight should be in the range [0, 1]. Got: 1.5"),
        ):
            show_cam_on_image(image=image, mask=mask, image_weight=1.5)
