"""Taken from DFormerV2 paper: https://arxiv.org/pdf/2504.04701"""
from .depth_decay import DepthAwareDecayMask
from .geometric_attention import GeometricSelfAttention
from .geometric_bias import GeometricAttentionBias
from .spatial_decay import SpatialDecayMask

__all__ = ['GeometricSelfAttention', 'GeometricAttentionBias', 'DepthAwareDecayMask', 'SpatialDecayMask']
