"""Observation preprocessing for the inference pipeline."""

import numpy as np
import torch
from tso_robotics_sockets import CompressionType, decompress_array
from versatil_constants.shared import ObsKey

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
        state_keys: list[str],
        has_language: bool,
        camera_metadata: dict[str, CameraMetadata],
        compression_type: str = CompressionType.RAW.value,
        rotate_images: bool = False,
        depth_clamp_ranges: dict[str, tuple[float, float]] | None = None,
        state_dtypes: dict[str, str] | None = None,
    ):
        """Initialize the observation preprocessor.

        Args:
            camera_keys: Camera observation keys (RGB + optional depth).
            state_keys: Numerical non-image observation keys.
            has_language: Whether language instructions are expected.
            camera_metadata: Per-camera metadata with training-time image dimensions.
            compression_type: Compression format used by the server for images.
            rotate_images: Whether to flip images 180 degrees.
            depth_clamp_ranges: Optional per-camera (min, max) depth clamps.
            state_dtypes: Numpy dtype names per state key from the training
                observation metadata; unlisted keys parse as float32.
        """
        self.camera_keys = camera_keys
        self.state_keys = state_keys
        self.has_language = has_language
        self.compression_type = compression_type
        self.rotate_images = rotate_images
        self.depth_clamp_ranges = depth_clamp_ranges or {}
        self.state_dtypes = {
            key: np.dtype(dtype_name)
            for key, dtype_name in (state_dtypes or {}).items()
        }
        self.camera_metadata = camera_metadata

        self.depth_camera_keys = [
            key for key in self.camera_keys if self.camera_metadata[key].is_depth
        ]
        self.has_depth = len(self.depth_camera_keys) > 0
        self.rgb_camera_keys = [
            key for key in self.camera_keys if self.camera_metadata[key].is_rgb
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
        observation_keys = self.camera_keys + self.state_keys
        if self.has_language:
            observation_keys = observation_keys + [ObsKey.LANGUAGE.value]
        missing_keys = [key for key in observation_keys if key not in response]
        if missing_keys:
            raise KeyError(
                f"Server response missing requested keys: {missing_keys}. "
                f"Available keys: {list(response.keys())}"
            )
        is_multi_environment = any(
            isinstance(response.get(key), dict) for key in observation_keys
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
            image = decompress_array(response[camera_key], method=self.compression_type)
            if self.rotate_images:
                image = np.ascontiguousarray(image[::-1, ::-1])
            observations[camera_key] = image
        for key in self.state_keys:
            observations[key] = np.array(
                response[key], dtype=self.state_dtypes.get(key, np.float32)
            )
        if self.has_language:
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
        observation_keys = self.camera_keys + self.state_keys
        if self.has_language:
            observation_keys = observation_keys + [ObsKey.LANGUAGE.value]
        first_observation_key = next(
            key for key in observation_keys if isinstance(response.get(key), dict)
        )
        environment_indices = [int(key) for key in response[first_observation_key]]
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
            for key in self.state_keys:
                observations[key] = np.array(
                    response[key][index_string],
                    dtype=self.state_dtypes.get(key, np.float32),
                )
            if self.has_language:
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
            Depth images are clamped to their camera's configured range.

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
            if (
                self.camera_metadata[camera_key].is_depth
                and camera_key in self.depth_clamp_ranges
            ):
                depth_min, depth_max = self.depth_clamp_ranges[camera_key]
                processed = torch.clamp(processed, min=depth_min, max=depth_max)
            result[camera_key] = processed

        return result
