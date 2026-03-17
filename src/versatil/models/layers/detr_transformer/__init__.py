"""
Re-implementation of DETR Transformer modules, using custom Flash Attention implementation.
Original reference implementation: https://github.com/facebookresearch/detr/tree/main/models
"""

from .transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
]
