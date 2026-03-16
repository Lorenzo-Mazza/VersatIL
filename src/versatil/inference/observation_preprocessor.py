"""Observation preprocessing for the inference pipeline."""

import logging

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from tso_robotics_sockets import CompressionType, decompress_array

from versatil.data.constants import Cameras
from versatil_constants.shared import ObsKey


class ObservationPreprocessor:
    """Parses server responses and transforms observations into model-ready tensors.

    Handles single and multi-environment responses, RGB normalization,
    depth clamping, and albumentations transforms.
    """

    def __init__(
        self,
        camera_keys: list[str],
        proprioceptive_keys: list[str],
        has_language: bool,
        image_height: int,
        image_width: int,
        compression_type: str = CompressionType.RAW.value,
        rotate_images: bool = False,
        depth_clamp_range: tuple[float, float] | None = None,
    ):
        """Initialize the observation preprocessor.

        Args:
            camera_keys: Camera observation keys (RGB + optional depth).
            proprioceptive_keys: Proprioceptive observation keys.
            has_language: Whether language instructions are expected.
            image_height: Target image height for resizing.
            image_width: Target image width for resizing.
            compression_type: Compression format used by the server for images.
            rotate_images: Whether to flip images 180 degrees.
            depth_clamp_range: Optional (min, max) for depth clamping.
        """
        self.camera_keys = camera_keys
        self.proprioceptive_keys = proprioceptive_keys
        self.has_language = has_language
        self.compression_type = compression_type
        self.rotate_images = rotate_images
        self.depth_clamp_range = depth_clamp_range

        self.depth_key = Cameras.DEPTH.value
        self.has_depth = self.depth_key in self.camera_keys
        self.rgb_camera_keys = [
            key for key in self.camera_keys if key != self.depth_key
        ]

        additional_targets = {}
        for camera_key in self.rgb_camera_keys[1:]:
            additional_targets[camera_key] = "image"
        if self.has_depth:
            additional_targets[self.depth_key] = "mask"
        self.image_transform = A.Compose(
            [
                A.Resize(height=image_height, width=image_width),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )

    def parse_response(
        self, response: dict
    ) -> dict[int, dict[str, np.ndarray | str]]:
        """Parse server response into per-environment observation dicts.

        Args:
            response: Raw server response.

        Returns:
            Dict mapping environment index to observation dict.
        """
        first_camera = self.camera_keys[0] if self.camera_keys else None
        is_multi_environment = first_camera is not None and isinstance(
            response.get(first_camera), dict
        )
        if is_multi_environment:
            return self._parse_multi_environment(response=response)
        return self._parse_single_environment(response=response)

    def _parse_single_environment(
        self, response: dict
    ) -> dict[int, dict[str, np.ndarray | str]]:
        """Parse single-environment response, wrapped as environment 0.

        Args:
            response: Raw server response.

        Returns:
            Dict with environment index 0 mapping to observation dict.
        """
        observations: dict[str, np.ndarray | str] = {}
        for camera_key in self.camera_keys:
            if camera_key in response:
                image = decompress_array(
                    response[camera_key], method=self.compression_type
                )
                if self.rotate_images:
                    image = np.ascontiguousarray(image[::-1, ::-1])
                observations[camera_key] = image
        for key in self.proprioceptive_keys:
            if key in response:
                observations[key] = np.array(
                    response[key], dtype=np.float32
                )
        if self.has_language and ObsKey.LANGUAGE.value in response:
            observations[ObsKey.LANGUAGE.value] = response[
                ObsKey.LANGUAGE.value
            ]
        return {0: observations}

    def _parse_multi_environment(
        self, response: dict
    ) -> dict[int, dict[str, np.ndarray | str]]:
        """Parse multi-environment response keyed by environment index.

        Args:
            response: Raw server response with dict-valued observation data.

        Returns:
            Dict mapping each environment index to observation dict.
        """
        first_camera = self.camera_keys[0]
        environment_indices = [
            int(key) for key in response[first_camera].keys()
        ]
        per_environment: dict[int, dict[str, np.ndarray | str]] = {}
        for environment_index in environment_indices:
            index_string = str(environment_index)
            observations: dict[str, np.ndarray | str] = {}
            for camera_key in self.camera_keys:
                image = decompress_array(
                    response[camera_key][index_string],
                    method=self.compression_type,
                )
                if self.rotate_images:
                    image = np.ascontiguousarray(image[::-1, ::-1])
                observations[camera_key] = image
            for key in self.proprioceptive_keys:
                if key in response:
                    observations[key] = np.array(
                        response[key][index_string], dtype=np.float32
                    )
            if self.has_language and ObsKey.LANGUAGE.value in response:
                observations[ObsKey.LANGUAGE.value] = response[
                    ObsKey.LANGUAGE.value
                ][index_string]
            per_environment[environment_index] = observations
        return per_environment

    def transform_camera_observations(
        self, recent_observations: dict[str, list]
    ) -> dict[str, torch.Tensor]:
        """Transform a temporal sequence of camera images into model-ready tensors.

        All RGB cameras are transformed together per timestep for consistent
        spatial augmentation. Depth is resized and clamped separately.

        Args:
            recent_observations: Dict mapping key to list of images.

        Returns:
            Dict mapping camera key to tensor (observation_horizon, C, H, W).
        """
        if not self.camera_keys:
            return {}
        camera_tensors: dict[str, list[torch.Tensor]] = {
            key: [] for key in self.camera_keys
        }
        first_rgb_key = self.rgb_camera_keys[0] if self.rgb_camera_keys else None
        reference_key = first_rgb_key or self.depth_key
        observation_count = len(recent_observations[reference_key])

        depth_only = first_rgb_key is None and self.has_depth

        for timestep in range(observation_count):
            transform_kwargs: dict[str, np.ndarray] = {}
            if first_rgb_key is not None:
                transform_kwargs["image"] = recent_observations[first_rgb_key][timestep]
                for rgb_key in self.rgb_camera_keys[1:]:
                    transform_kwargs[rgb_key] = recent_observations[rgb_key][timestep]
            if self.has_depth:
                depth_image = recent_observations[self.depth_key][timestep]
                if depth_only:
                    transform_kwargs["image"] = depth_image
                else:
                    transform_kwargs[self.depth_key] = depth_image

            transformed = self.image_transform(**transform_kwargs)

            if first_rgb_key is not None:
                camera_tensors[first_rgb_key].append(
                    self._normalize_image_tensor(image=transformed["image"])
                )
                for rgb_key in self.rgb_camera_keys[1:]:
                    camera_tensors[rgb_key].append(
                        self._normalize_image_tensor(image=transformed[rgb_key])
                    )
            if self.has_depth:
                depth_result_key = "image" if depth_only else self.depth_key
                depth_tensor = transformed[depth_result_key].float()
                if depth_tensor.dim() == 2:
                    depth_tensor = depth_tensor.unsqueeze(0)
                if self.depth_clamp_range is not None:
                    depth_min, depth_max = self.depth_clamp_range
                    depth_tensor = torch.clamp(depth_tensor, min=depth_min, max=depth_max)
                camera_tensors[self.depth_key].append(depth_tensor)

        return {
            key: torch.stack(tensors)
            for key, tensors in camera_tensors.items()
        }

    @staticmethod
    def _normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor to [0, 1] range.

        Handles both uint8 [0, 255] and float [0, 1] inputs.

        Args:
            image: Image tensor from albumentations transform.

        Returns:
            Float tensor in [0, 1] range.
        """
        if image.dtype == torch.uint8:
            return image.float() / 255.0
        if image.max() > 1.0:
            logging.warning(
                "Received float image with max %.1f > 1.0, dividing by 255.",
                image.max().item(),
            )
            return image / 255.0
        logging.warning(
            "Received float image already in [0, 1] range, skipping normalization."
        )
        return image