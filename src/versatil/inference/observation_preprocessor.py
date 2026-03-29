"""Observation preprocessing for the inference pipeline."""

import logging

import numpy as np
import torch
from tso_robotics_sockets import CompressionType, decompress_array
from versatil_constants.shared import ObsKey

from versatil.data.constants import Cameras
from versatil.data.metadata import CameraMetadata
from versatil.data.processing.image_processor import ImageProcessor


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
        camera_metadata: dict[str, CameraMetadata],
        compression_type: str = CompressionType.RAW.value,
        rotate_images: bool = False,
        depth_clamp_range: tuple[float, float] | None = None,
    ):
        """Initialize the observation preprocessor.

        Args:
            camera_keys: Camera observation keys (RGB + optional depth).
            proprioceptive_keys: Proprioceptive observation keys.
            has_language: Whether language instructions are expected.
            camera_metadata: Per-camera metadata with training-time image dimensions.
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

        self.image_processor = ImageProcessor(
            camera_metadata=camera_metadata,
            train=False,
        )

    def parse_response(self, response: dict) -> dict[int, dict[str, np.ndarray | str]]:
        """Parse server response into per-environment observation dicts.

        Args:
            response: Raw server response.

        Returns:
            Dict mapping environment index to observation dict.
        """
        first_key = next(iter(response), None)
        is_multi_environment = first_key is not None and isinstance(
            response.get(first_key), dict
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
                observations[key] = np.array(response[key], dtype=np.float32)
        if self.has_language and ObsKey.LANGUAGE.value in response:
            observations[ObsKey.LANGUAGE.value] = response[ObsKey.LANGUAGE.value]
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
        first_key = next(iter(response))
        environment_indices = [int(key) for key in response[first_key]]
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
                observations[ObsKey.LANGUAGE.value] = response[ObsKey.LANGUAGE.value][
                    index_string
                ]
            per_environment[environment_index] = observations
        return per_environment

    def transform_camera_observations(
        self, recent_observations: dict[str, list]
    ) -> dict[str, torch.Tensor]:
        """Transform a temporal sequence of camera images into model-ready tensors.

        Note:
            Uses ImageProcessor for per-camera resize and normalization.
            Depth images are additionally clamped if depth_clamp_range is set.

        Args:
            recent_observations: Dict mapping key to list of images per timestep.

        Returns:
            Dict mapping camera key to tensor (observation_horizon, C, H, W).
        """
        if not self.camera_keys:
            return {}
        result = {}
        for camera_key in self.camera_keys:
            if camera_key not in recent_observations:
                raise ValueError(
                    f"Missing camera key '{camera_key}' in the server observation data."
                )
            images = np.stack(recent_observations[camera_key])  # (T, H, W, C)
            processed = self.image_processor.process(
                images=images, camera_key=camera_key
            )
            # TODO: this currently assumes that only a camera with key "depth" is a depth camera - should ideally be specified in metadata
            if camera_key == self.depth_key and self.depth_clamp_range is not None:
                depth_min, depth_max = self.depth_clamp_range
                processed = torch.clamp(processed, min=depth_min, max=depth_max)
            result[camera_key] = processed

        return result

    @staticmethod
    def _normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor to [0, 1] range.

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
