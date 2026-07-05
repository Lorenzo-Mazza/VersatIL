"""Spatial RGB encoder producing (B, C, H, W) feature maps via timm features_only."""

from versatil.data.constants import CameraModality
from versatil.models.encoding.encoders.image_mixin import RGBEncoderMixin
from versatil.models.encoding.encoders.spatial_backbone import SpatialBackboneEncoder


class SpatialRGBEncoder(RGBEncoderMixin, SpatialBackboneEncoder):
    """RGB encoder for backbones that output spatial feature maps.

    Supports any timm backbone compatible with ``features_only=True``,
    regardless of whether the architecture is convolutional (ResNet,
    EfficientNet, ConvNeXt) or attention-based (Swin, TinyViT).
    Handles both NCHW and NHWC output layouts transparently.
    """

    _input_channels = 3
    _camera_modality = CameraModality.RGB
