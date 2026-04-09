"""Shared functionality for transformer encoder and decoder models."""

import math

import torch
import torch.nn as nn

from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.rms_norm import RMSNorm
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)
from versatil.models.layers.transformer.positional_encoding import (
    create_positional_encoding,
)

RESIDUAL_STREAM_FLAG = "SQUARE_ROOT_WEIGHT"


class TransformerMixin:
    """Shared methods for transformer encoder and decoder models.

    Subclasses must set these attributes before calling mixin methods:
        number_of_layers, initializer_range, number_of_residual_blocks
    """

    number_of_layers: int
    initializer_range: float
    number_of_residual_blocks: int
    positional_encoding: nn.Module | None

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style weight initialization with residual stream scaling."""
        if hasattr(module, RESIDUAL_STREAM_FLAG):
            std = self.initializer_range / math.sqrt(
                self.number_of_residual_blocks * self.number_of_layers
            )
        else:
            std = self.initializer_range
        if isinstance(module, nn.Linear):
            if hasattr(module, "_is_modulation_layer") and module._is_modulation_layer:
                return
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm, AdaNorm)):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)

    @staticmethod
    def _expand_padding_mask(
        padding_mask: torch.Tensor,
        query_length: int,
    ) -> torch.Tensor:
        """Expand 2D padding mask to 4D attention mask.

        Args:
            padding_mask: (B, key_length) where True means masked/padded.
            query_length: Length of query sequence.

        Returns:
            Attention mask (B, 1, query_length, key_length) where True means masked.
        """
        return padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, query_length, -1)

    def _setup_positional_encoding(
        self,
        positional_encoding_type: str | None,
        embedding_dimension: int,
        maximum_sequence_length: int,
        number_of_heads: int,
    ) -> None:
        """Initialize positional encoding module.

        Args:
            positional_encoding_type: Type of positional encoding (or None).
            embedding_dimension: Model embedding dimension.
            maximum_sequence_length: Maximum sequence length.
            number_of_heads: Number of attention heads.
        """
        self.positional_encoding = None
        if positional_encoding_type is not None:
            self.positional_encoding = create_positional_encoding(
                encoding_type=positional_encoding_type,
                embedding_dimension=embedding_dimension,
                maximum_length=maximum_sequence_length,
                num_heads=number_of_heads,
            )

    def _apply_positional_encoding(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, RotaryPositionalEncoding | None]:
        """Apply additive positional encoding and extract rotary encoding.

        Args:
            hidden_states: Input tensor (B, T, D).

        Returns:
            Tuple of (hidden_states with additive PE applied, rotary PE or None).
        """
        if isinstance(
            self.positional_encoding,
            (SinusoidalPositionalEncoding1D, LearnedPositionalEncoding1D),
        ):
            hidden_states = hidden_states + self.positional_encoding(hidden_states)
        rotary_positional_encoding = (
            self.positional_encoding
            if isinstance(self.positional_encoding, RotaryPositionalEncoding)
            else None
        )
        return hidden_states, rotary_positional_encoding
