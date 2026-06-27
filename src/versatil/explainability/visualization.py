"""Image rendering helpers for explainability files."""

import cv2
import numpy as np


def show_cam_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.5,
) -> np.ndarray:
    """Overlay a heatmap on an image.

    Args:
        image: Base image in RGB or BGR format, stored as float values in
            ``[0, 1]``.
        mask: CAM heatmap with values expected in ``[0, 1]``.
        use_rgb: Whether to convert the OpenCV heatmap from BGR to RGB.
        colormap: OpenCV colormap identifier.
        image_weight: Blend weight for the original image.

    Returns:
        Heatmap overlay as a ``uint8`` image.

    Raises:
        ValueError: If ``image`` is not normalized to ``[0, 1]``.
        ValueError: If ``image_weight`` is outside ``[0, 1]``.
    """
    heatmap = cv2.applyColorMap((255 * mask).astype(np.uint8), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    if np.max(image) > 1:
        raise ValueError("The input image should be np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise ValueError(
            f"image_weight should be in the range [0, 1]. Got: {image_weight}"
        )

    overlay = (1 - image_weight) * heatmap + image_weight * image
    overlay = overlay / np.max(overlay)
    result: np.ndarray = (255 * overlay).astype(np.uint8)
    return result
