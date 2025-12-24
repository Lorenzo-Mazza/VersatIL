"""Augmentation pipeline for episodic dataset."""

import albumentations as A
import cv2
import numpy as np



class AugmentationPipeline:
    """Manages data augmentation pipelines for training."""

    def __init__(
        self,
        color_augmentation: A.Compose | None = None,
        spatial_augmentation: A.Compose | None = None,
        target_height: int | None = None,
        target_width: int | None = None,
        train: bool = True,
    ):
        """
        Args:
            color_augmentation: Albumentations compose object for color augmentations
            spatial_augmentation: Albumentations compose object for spatial augmentations
            target_height: Target height for resizing images
            target_width: Target width for resizing images
            train: Training mode (augmentations only applied during training)
        """
        self.use_color = color_augmentation is not None and color_augmentation.transforms and train
        self.use_spatial = spatial_augmentation is not None and spatial_augmentation.transforms and train
        self.use_resize = target_height is not None and target_width is not None

        self.photometric_transform = None
        self.spatial_transform = None
        self.resize_transform_rgb = None
        self.resize_transform_depth = None

        if self.use_color:
            self.photometric_transform = color_augmentation
        if self.use_spatial:
            self.spatial_transform = spatial_augmentation

        if self.use_resize:
            self.resize_transform_rgb = A.Resize(
                height=target_height,
                width=target_width,
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            )
            self.resize_transform_depth = A.Resize(
                height=target_height,
                width=target_width,
                interpolation=cv2.INTER_NEAREST, # Nearest neighbor interpolation for depth to preserve values
                p=1.0
            )


    def apply_rgb_augmentations(
        self, images: np.ndarray
    ) -> np.ndarray:
        """Apply color and spatial augmentations to RGB images.

        Args:
            images: Array of images (T, H, W, 3)

        Returns:
            Augmented images
        """
        if self.resize_transform_rgb is not None:
            images = np.stack([self.resize_transform_rgb(image=frame)["image"] for frame in images])
        if self.photometric_transform is not None:
            images = np.stack([self.photometric_transform(image=frame)["image"] for frame in images])
        if self.spatial_transform is not None:
            images = np.stack([self.spatial_transform(image=frame)["image"] for frame in images])
        return images


    def apply_depth_augmentations(
        self, images: np.ndarray
    ) -> np.ndarray:
        """Apply spatial augmentations to depth images.

        Args:
            images: Array of depth images (T, H, W) or (T, H, W, 1)

        Returns:
            Augmented depth images
        """
        if self.resize_transform_depth is not None:
            images = np.stack([self.resize_transform_depth(image=frame)["image"] for frame in images])
        if self.spatial_transform is not None:
            images = np.stack([self.spatial_transform(image=frame)["image"] for frame in images])
        return images
