"""Diffusion Transformer architectures.

This package provides three DiT variants with different conditioning mechanisms:

DiTBlock (Original DiT)
    Encoder-decoder architecture where encoder tokens are processed bidirectionally
    and pooled to a single conditioning vector. The decoder uses AdaLN modulation
    from the sum of pooled features and timestep embedding. Supports encoder caching
    during inference since the pooled vector is static across denoising steps.

CrossAttentionDiT (PixArt Style)
    Decoder-only architecture that conditions via cross-attention to encoder tokens.
    External embeddings are passed directly to decoder layers without internal encoder
    processing. Each decoder layer attends to the full conditioning sequence through
    cross-attention while using AdaLN for timestep modulation.

MMDiT (Stable Diffusion 3 Style)
    Dual-stream architecture where two token sequences are processed jointly through
    shared attention layers. Both streams have independent weights but attend to
    concatenated key-value pairs, enabling bidirectional information flow. No caching
    is possible since both streams are modified at each layer.

Each architecture has corresponding decoder and layer classes:
    - DiTBlock: DiffusionTransformerDecoder, DecoderLayer
    - CrossAttentionDiT: CrossConditioningDecoder, CrossConditioningDecoderLayer
    - MMDiT: MMDiTDecoder, MMDiTLayer (with JointAttention)
"""

from versatil.models.layers.diffusion_transformer.cross_attention_dit_decoder import (
    CrossConditioningDecoder,
)
from versatil.models.layers.diffusion_transformer.cross_attention_dit_layer import (
    CrossConditioningDecoderLayer,
)
from versatil.models.layers.diffusion_transformer.cross_attention_dit import (
    CrossAttentionDiT,
)
from versatil.models.layers.diffusion_transformer.dit_decoder import DiffusionTransformerDecoder
from versatil.models.layers.diffusion_transformer.dit_decoder_layer import DecoderLayer
from versatil.models.layers.diffusion_transformer.final_prediction_layer import FinalPredictionLayer
from versatil.models.layers.diffusion_transformer.joint_attention import JointAttention
from versatil.models.layers.diffusion_transformer.mmdit_decoder import MMDiTDecoder
from versatil.models.layers.diffusion_transformer.mmdit_layer import MMDiTLayer
from versatil.models.layers.diffusion_transformer.mmdit_transformer import MMDiTTransformer
from versatil.models.layers.diffusion_transformer.query_key_norm import QueryKeyNorm
from versatil.models.layers.diffusion_transformer.dit_block_transformer import DiTBlock

__all__ = [
    "CrossAttentionDiT",
    "CrossConditioningDecoder",
    "CrossConditioningDecoderLayer",
    "DecoderLayer",
    "DiffusionTransformerDecoder",
    "DiTBlock",
    "FinalPredictionLayer",
    "JointAttention",
    "MMDiTDecoder",
    "MMDiTLayer",
    "MMDiTTransformer",
    "QueryKeyNorm",
]