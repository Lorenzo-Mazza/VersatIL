"""Image processor for per-camera resize, augmentation, and normalization."""

import albumentations as A
import cv2
import numpy as np
import torch

from versatil.data.metadata import CameraMetadata


class ImageProcessor:
    """Per-camera image processing: resize, augmentation, normalization, channel reorder.

    Note:
        At training time: resize → color augmentation (RGB only) → spatial augmentation → normalize → reorder.
        At inference time: resize → normalize → reorder.
    """

    def __init__(
        self,
        color_augmentation: A.Compose | None = None,
        spatial_augmentation: A.Compose | None = None,
        camera_metadata: dict[str, CameraMetadata] | None = None,
        train: bool = True,
    ):
        """
        Args:
            color_augmentation: Albumentations compose for color augmentations.
            spatial_augmentation: Albumentations compose for spatial augmentations.
            camera_metadata: Per-camera metadata from the observation space.
                Each camera's image_height/image_width defines the resize target.
                Single-channel cameras use nearest-neighbor interpolation.
            train: Whether in training mode. Augmentations are only applied during training.
        """
        self.train = train
        self.use_color = (
            color_augmentation is not None and color_augmentation.transforms and train
        )
        self.use_spatial = (
            spatial_augmentation is not None
            and spatial_augmentation.transforms
            and train
        )

        self.photometric_transform = None
        self.spatial_transform = None
        if self.use_color:
            self.photometric_transform = color_augmentation
        if self.use_spatial:
            self.spatial_transform = spatial_augmentation

        self._camera_resize: dict[str, A.Resize] = {}
        self._rgb_cameras: set[str] = set()
        if camera_metadata is not None:
            for camera_key, metadata in camera_metadata.items():
                if metadata.is_rgb:
                    self._rgb_cameras.add(camera_key)
                interpolation = (
                    cv2.INTER_NEAREST
                    if metadata.is_single_channel
                    else cv2.INTER_LINEAR
                )
                self._camera_resize[camera_key] = A.Resize(
                    height=metadata.image_height,
                    width=metadata.image_width,
                    interpolation=interpolation,
                    p=1.0,
                )

    def process(self, images: np.ndarray, camera_key: str) -> torch.Tensor:
        """Process camera images: resize, augment, normalize, reorder channels.

        Args:
            images: Array of images (T, H, W, C) as uint8 or float.
            camera_key: Camera key for per-camera processing.

        Returns:
            Processed images as (T, C, H, W) float32 tensor.
        """
        resize = self._camera_resize.get(camera_key)
        if resize is not None:
            images = np.stack([resize(image=frame)["image"] for frame in images])

        is_rgb = camera_key in self._rgb_cameras
        if is_rgb and self.photometric_transform is not None:
            images = np.stack(
                [self.photometric_transform(image=frame)["image"] for frame in images]
            )
        if self.spatial_transform is not None:
            images = np.stack(
                [self.spatial_transform(image=frame)["image"] for frame in images]
            )

        if is_rgb:
            images = images.astype(np.float32) / 255.0
            images = np.moveaxis(images, -1, 1)  # (T, H, W, C) → (T, C, H, W)
        else:
            images = images.astype(np.float32)
            # (T, 1, H, W) or  (T, C, H, W)
            images = images[:, None] if images.ndim == 3 else np.moveaxis(images, -1, 1)

        return torch.from_numpy(images)
