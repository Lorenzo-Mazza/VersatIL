"""Augmentation pipeline for episodic dataset.

Handles:
- Color and spatial augmentation setup
- Rotation augmentation
- Image-specific transformations
"""

import random

import albumentations as A
import cv2
import numpy as np



class AugmentationPipeline:
    """Manages data augmentation pipelines for training."""

    def __init__(
        self,
        color_augmentation: A.Compose | None = None,
        spatial_augmentation: A.Compose | None = None,
        rotation_augmentation: A.Compose | None = None,
        target_height: int | None = None,
        target_width: int | None = None,
        train: bool = True,
    ):
        """
        Args:
            color_augmentation: Hydra config for color augmentations
            spatial_augmentation: Hydra config for spatial augmentations
            rotation_augmentation: Hydra config for rotation augmentations
            train: Training mode (augmentations only applied during training)
        """
        self.use_color = color_augmentation is not None and color_augmentation.transforms and train
        self.use_spatial = spatial_augmentation is not None and spatial_augmentation.transforms and train
        self.use_rotation = rotation_augmentation is not None and rotation_augmentation.transforms and train
        self.use_resize = target_height is not None and target_width is not None

        self.photometric_transform = None
        self.spatial_transform = None
        self.rotation_transform = None
        self.resize_transform_rgb = None
        self.resize_transform_depth = None

        if self.use_color:
            self.photometric_transform = color_augmentation
        if self.use_spatial:
            self.spatial_transform = spatial_augmentation
        if self.use_rotation:
            self.rotation_transform = rotation_augmentation

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

    def setup_rotation(self) -> tuple[float, np.ndarray | None]:
        """Sample a rotation angle and compute rotation matrix.

        Returns:
            Tuple of (angle in degrees, 3x3 rotation matrix or None)
        """
        if (
            self.use_rotation
            and self.rotation_transform is not None
            and random.random() < self.rotation_transform.p
        ):
            self.rotation_transform: A.Rotate = self.rotation_transform.transforms[0]  # Assuming single transform
            angle = random.uniform(
                self.rotation_transform.limit[0], self.rotation_transform.limit[1]
            )
            if angle != 0:
                theta_rad = np.deg2rad(angle)
                cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
                R = np.array(
                    [[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]]
                )
                return angle, R
        return 0, None

    def apply_rgb_augmentations(
        self, images: np.ndarray, angle: float = 0
    ) -> np.ndarray:
        """Apply color and spatial augmentations to RGB images.

        Args:
            images: Array of images (T, H, W, 3)
            angle: Rotation angle in degrees

        Returns:
            Augmented images
        """
        if self.resize_transform_rgb is not None:
            images = np.stack([self.resize_transform_rgb(image=frame)["image"] for frame in images])
        if self.photometric_transform is not None:
            images = np.stack([self.photometric_transform(image=frame)["image"] for frame in images])
        if self.spatial_transform is not None:
            images = np.stack([self.spatial_transform(image=frame)["image"] for frame in images])
        if angle != 0:
            rotate_transform = A.Rotate(limit=(angle, angle), p=1.0, interpolation=cv2.INTER_LINEAR)
            images = np.stack([rotate_transform(image=frame)["image"] for frame in images])
        return images

    def apply_depth_augmentations(
        self, images: np.ndarray, angle: float = 0
    ) -> np.ndarray:
        """Apply spatial augmentations to depth images.

        Note: Color augmentations are NOT applied to depth.

        Args:
            images: Array of depth images (T, H, W) or (T, H, W, 1)
            angle: Rotation angle in degrees

        Returns:
            Augmented depth images
        """
        if self.resize_transform_depth is not None:
            images = np.stack([self.resize_transform_depth(image=frame)["image"] for frame in images])
        if self.spatial_transform is not None:
            images = np.stack([self.spatial_transform(image=frame)["image"] for frame in images])
        if angle != 0:
            rotate_transform = A.Rotate(limit=(angle, angle), p=1.0, interpolation=cv2.INTER_NEAREST)
            images = np.stack([rotate_transform(image=frame)["image"] for frame in images])
        return images


    def rotate_proprio_data(
        self,
        camera_frame: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> np.ndarray:
        """Rotate proprioceptive robot observations.

        Args:
            camera_frame: Camera-frame proprioceptive array
            rotation_matrix: 3x3 rotation matrix
        Returns:
            Rotated robot state
        """
        camera_frame = camera_frame.copy()
        camera_frame[..., :3] = (rotation_matrix @ camera_frame[..., :3].T).T
        return camera_frame
