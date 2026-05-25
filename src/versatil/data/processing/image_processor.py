"""Image processor for per-camera resize, augmentation, and normalization."""

import albumentations as A
import cv2
import numpy as np
import torch

from versatil.data.metadata import CameraMetadata


class ImageProcessor:
    """Per-camera image processing: resize, augmentation, normalization, channel reorder.

    Note:
        At training time: resize → color augmentation (RGB only) → spatial
        augmentation → channel reorder → dtype normalization.
        At inference time: resize → channel reorder → dtype normalization.
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
        self._camera_max_pixel_values: dict[str, float | None] = {}
        if camera_metadata is not None:
            for camera_key, metadata in camera_metadata.items():
                if metadata.is_rgb:
                    self._rgb_cameras.add(camera_key)
                self._camera_max_pixel_values[camera_key] = metadata.max_pixel_value
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

    @staticmethod
    def normalize_image_tensor(
        image: torch.Tensor,
        max_pixel_value: float | None = None,
    ) -> torch.Tensor:
        """Normalize image tensors to floating point.

        Args:
            image: Image tensor.
            max_pixel_value: Value used to scale the image tensor. ``None``
                disables scaling.

        Returns:
            Float image tensor.
        """
        image = image.float()
        if max_pixel_value is not None:
            return image / max_pixel_value
        return image

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

        images = images[:, None] if images.ndim == 3 else np.moveaxis(images, -1, 1)
        image_tensor = torch.from_numpy(images)  # (T, C, H, W)
        return self.normalize_image_tensor(
            image=image_tensor,
            max_pixel_value=self._camera_max_pixel_values.get(camera_key),
        )
