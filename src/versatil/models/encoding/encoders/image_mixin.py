"""Mixin for image encoders with multi-camera support."""

import abc

import torch

from versatil.data.constants import CameraModality, SampleKey
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.constants import EncoderOutputKeys


def resize_to_target_size(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Resize images preserving aspect ratio and zero-pad to target size.

    Scales down so the largest dimension fits the target, then zero-pads
    the shorter dimension. No-op if images already match the target.

    Args:
        images: Image tensor of shape (B, C, H, W).
        target_height: Target height in pixels.
        target_width: Target width in pixels.

    Returns:
        Resized and padded tensor of shape (B, C, target_height, target_width).
    """
    current_height, current_width = images.shape[2], images.shape[3]
    if current_height == target_height and current_width == target_width:
        return images
    ratio = max(current_height / target_height, current_width / target_width)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = torch.nn.functional.interpolate(
        images,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    pad_height = target_height - resized_height
    pad_width = target_width - resized_width
    if pad_height > 0 or pad_width > 0:
        resized = torch.nn.functional.pad(
            resized, (0, pad_width, 0, pad_height), value=0.0
        )
    return resized


_NON_CAMERA_INPUT_KEYS = frozenset(
    {
        SampleKey.TOKENIZED_OBSERVATIONS.value,
        SampleKey.IS_PAD_OBSERVATION.value,
    }
)


class ImageEncoderMixin(abc.ABC):
    """Shared logic for encoders that process camera images.

    Provides camera key storage, multi-camera detection, vision feature naming,
    and multi-camera encode dispatch. Camera modality checks are performed by
    experiment validation because they require observation-space metadata.
    """

    @property
    @abc.abstractmethod
    def _output_modality(self) -> str:
        """Output modality prefixed to feature names (e.g. ``'rgb'``, ``'depth'``)."""
        raise NotImplementedError

    @staticmethod
    def _resolve_intermediate_layer_index(
        intermediate_layer_index: int | None,
        output_count: int,
    ) -> int:
        """Resolve an intermediate layer index.

        Args:
            intermediate_layer_index: Optional intermediate layer index.
                Negative values index from the end; ``None`` selects the final
                layer.
            output_count: Number of intermediate layers available.

        Returns:
            Non-negative intermediate layer index.
        """
        layer_index = (
            output_count - 1
            if intermediate_layer_index is None
            else intermediate_layer_index
        )
        if layer_index < 0:
            layer_index = output_count + layer_index
        if layer_index < 0 or layer_index >= output_count:
            raise ValueError(
                f"intermediate_layer_index={intermediate_layer_index} is outside "
                f"the valid range for {output_count} intermediate layers."
            )
        return layer_index

    def _setup_camera_keys(self, input_keys: list[str]) -> None:
        """Store declared camera input keys.

        Args:
            input_keys: Camera input keys declared by the encoder.
        """
        self.camera_keys = [
            key for key in input_keys if key not in _NON_CAMERA_INPUT_KEYS
        ]
        self.camera_metadata: dict[str, CameraMetadata] = {}
        self.is_multi_camera = len(self.camera_keys) > 1

    def set_camera_metadata(self, camera_metadata: dict[str, CameraMetadata]) -> None:
        """Store observation-space camera metadata for runtime camera routing.

        Args:
            camera_metadata: Observation-space camera metadata keyed by
                observation key.
        """
        declared_camera_keys = [
            key
            for key in self.input_specification.keys
            if key not in _NON_CAMERA_INPUT_KEYS
        ]
        unknown_keys = [
            key for key in declared_camera_keys if key not in camera_metadata
        ]
        if unknown_keys:
            raise ValueError(
                f"{type(self).__name__} declares camera keys {unknown_keys} "
                "that are not part of the observation-space cameras "
                f"{sorted(camera_metadata)}."
            )
        self.camera_keys = declared_camera_keys
        self.camera_metadata = camera_metadata
        self.is_multi_camera = len(self.camera_keys) > 1

    def _camera_key_for_modality(self, modality: CameraModality) -> str:
        """Return the first configured camera key with the requested modality."""
        for camera_key in self.camera_keys:
            if self.camera_metadata[camera_key].modality == modality:
                return camera_key
        raise RuntimeError(
            f"{type(self).__name__} has no configured {modality.value} camera. "
            "Run experiment validation and EncodingPipeline setup before forward."
        )

    def _get_vision_feature_names(self) -> list[str]:
        """Get output feature names based on camera configuration.

        Returns:
            Single camera: ``['{modality}']``.
            Multi-camera: ``['{modality}:{cam1}', '{modality}:{cam2}', ...]``.
        """
        modality = self._output_modality
        if self.is_multi_camera:
            return [f"{modality}:{key}" for key in self.camera_keys]
        else:
            return [modality]

    @abc.abstractmethod
    def _encode_single_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a single camera's images into features.

        Args:
            images: Image tensor of shape (B, C, H, W).

        Returns:
            Feature tensor.
        """
        raise NotImplementedError

    def _encode_vision(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Dispatch single-image encoding across cameras.

        Args:
            inputs: Dict mapping camera keys to image tensors (B, C, H, W).

        Returns:
            Dict with features keyed by modality (single) or
            ``modality:camera_key`` (multi-camera).
        """
        modality = self._output_modality
        if self.is_multi_camera:
            result = {}
            for camera_key in self.camera_keys:
                features = self._encode_single_image(inputs[camera_key])
                result[f"{modality}:{camera_key}"] = features
            return result
        else:
            features = self._encode_single_image(inputs[self.camera_keys[0]])
            return {modality: features}


class RGBEncoderMixin(ImageEncoderMixin):
    """Mixin for encoders processing RGB camera images."""

    @property
    def _output_modality(self) -> str:
        return EncoderOutputKeys.RGB.value


class DepthEncoderMixin(ImageEncoderMixin):
    """Mixin for encoders processing single-channel depth camera images."""

    @property
    def _output_modality(self) -> str:
        return EncoderOutputKeys.DEPTH.value


class RGBDEncoderMixin(ImageEncoderMixin):
    """Mixin for encoders processing both RGB and depth camera images."""

    @property
    def _output_modality(self) -> str:
        return EncoderOutputKeys.RGBD.value

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Validate that RGBD inputs use camera metadata.

        Args:
            key: Observation key being validated.
            metadata: Metadata from the observation space for this key.

        Returns:
            Error message if incompatible, None if valid.
        """
        if not isinstance(metadata, CameraMetadata):
            return f"Expected CameraMetadata for '{key}', got {type(metadata).__name__}"
        return None
