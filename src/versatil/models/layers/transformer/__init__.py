"""Transformer layers package, with custom Flash Attention implementation, cache support and modular layer customization."""

from versatil.models.layers.transformer.attention import CachedAttention
from versatil.models.layers.transformer.autoregressive_decoder import GPTDecoder
from versatil.models.layers.transformer.bidirectional_decoder import (
    BidirectionalDecoder,
)
from versatil.models.layers.transformer.conditional_bidirectional_decoder import (
    ConditionalBidirectionalDecoder,
)
from versatil.models.layers.transformer.conditional_decoder_layer import (
    ConditionalTransformerDecoderLayer,
)
from versatil.models.layers.transformer.decoder_layer import TransformerDecoderLayer
from versatil.models.layers.transformer.encoder import TransformerEncoder
from versatil.models.layers.transformer.encoder_layer import TransformerEncoderLayer
from versatil.models.layers.transformer.kv_cache import (
    DecoderKVCache,
    LayerKVCache,
    initialize_decoder_cache,
    update_layer_cache,
)
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.transformer.positional_encoding import (
    apply_rope_positional_encoding,
    create_positional_encoding,
)

__all__ = [
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "BidirectionalDecoder",
    "ConditionalBidirectionalDecoder",
    "ConditionalTransformerDecoderLayer",
    "GPTDecoder",
    "TransformerDecoderLayer",
    "CachedAttention",
    "LayerKVCache",
    "DecoderKVCache",
    "initialize_decoder_cache",
    "update_layer_cache",
    "create_normalization_layer",
    "create_positional_encoding",
    "apply_rope_positional_encoding",
]
