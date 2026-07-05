"""Spatial depth encoder producing (B, C, H, W) feature maps from single-channel depth images."""

from versatil.data.constants import CameraModality
from versatil.models.encoding.encoders.image_mixin import DepthEncoderMixin
from versatil.models.encoding.encoders.spatial_backbone import SpatialBackboneEncoder


class SpatialDepthEncoder(DepthEncoderMixin, SpatialBackboneEncoder):
    """Depth encoder for backbones that output spatial feature maps.

    Accepts any timm backbone compatible with ``features_only=True`` and
    ``in_chans=1``. Handles both NCHW and NHWC output layouts.
    """

    _input_channels = 1
    _camera_modality = CameraModality.DEPTH
