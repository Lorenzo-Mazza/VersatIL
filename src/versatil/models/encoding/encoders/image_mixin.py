"""Mixin for image encoders with multi-camera support."""

import abc

import torch

from versatil.data.constants import RGB_CAMERAS
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


class ImageEncoderMixin(abc.ABC):
    """Shared logic for encoders that process camera images.

    Provides camera key extraction, multi-camera detection, vision feature
    naming, and multi-camera encode dispatch. Mixed into CNN, ViT,
    ConditionalCNN, and VLM encoders.
    """

    def _setup_camera_keys(self, input_keys: list[str]) -> None:
        """Extract camera keys from input keys.

        Args:
            input_keys: All input keys for this encoder.
        """
        self.camera_keys = [key for key in input_keys if key in RGB_CAMERAS]
        self.is_multi_camera = len(self.camera_keys) > 1

    def _get_vision_feature_names(self) -> list[str]:
        """Get output feature names based on camera configuration.

        Returns:
            Single camera: ``['rgb']``.
            Multi-camera: ``['rgb.left', 'rgb.right', ...]``.
        """
        modality = EncoderOutputKeys.RGB.value
        if self.is_multi_camera:
            return [f"{modality}.{key}" for key in self.camera_keys]
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
            Dict with RGB features keyed by modality (single) or
            ``modality.camera_key`` (multi-camera).
        """
        modality = EncoderOutputKeys.RGB.value
        if self.is_multi_camera:
            result = {}
            for camera_key in self.camera_keys:
                features = self._encode_single_image(inputs[camera_key])
                result[f"{modality}.{camera_key}"] = features
            return result
        else:
            features = self._encode_single_image(inputs[self.camera_keys[0]])
            return {modality: features}
