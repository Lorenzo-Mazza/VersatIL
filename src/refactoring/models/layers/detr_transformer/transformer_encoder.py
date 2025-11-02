import copy

import torch
import torch.nn as nn

from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.positional_encoding.base import add_positional_encoding


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre/post normalization support."""
    def __init__(
            self,
            embedding_dimension: int,
            number_of_heads: int,
            feedforward_dimension: int = 2048,
            dropout: float = 0.1,
            activation: str = ActivationFunction.RELU.value,
            normalize_before: bool = False,
    ):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension, num_heads=number_of_heads, dropout=dropout, batch_first=False
        )
        self.feedforward_linear1 = nn.Linear(embedding_dimension, feedforward_dimension)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_linear2 = nn.Linear(feedforward_dimension, embedding_dimension)
        self.normalization1 = nn.LayerNorm(embedding_dimension)
        self.normalization2 = nn.LayerNorm(embedding_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = ActivationFunction(activation).to_torch_activation()()


    def forward(
            self,
            source: torch.Tensor,
            source_mask: torch.Tensor | None = None,
            source_key_padding_mask: torch.Tensor | None = None,
            positional_encoding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Pre-normalization:  Normalization -> Operation -> Add
        Post-normalization: Operation -> Add -> Normalization
        """
        residual = source
        source = self.normalization1(source) if self.normalize_before else source
        # DETR-style: Add positional encoding to queries and keys, but not to values
        query = key = add_positional_encoding(source, positional_encoding)
        source = self.self_attention(
            query, key, value=source,
            attn_mask=source_mask,
            key_padding_mask=source_key_padding_mask,
        )[0]
        source = residual + self.dropout1(source)
        source = source if self.normalize_before else self.normalization1(source)
        residual = source
        source = self.normalization2(source) if self.normalize_before else source
        source = self.feedforward_linear2(
            self.feedforward_dropout(
                self.activation(self.feedforward_linear1(source))
            )
        )
        source = residual + self.dropout2(source)
        source = source if self.normalize_before else self.normalization2(source)
        return source


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    def __init__(
            self,
            encoder_layer: TransformerEncoderLayer,
            number_of_layers: int,
            normalization: nn.Module | None = None,
    ):
        """Initialize transformer encoder.

        Args:
            encoder_layer: Single encoder layer to be stacked.
            number_of_layers: Number of encoder layers.
            normalization: Optional final normalization layer.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(number_of_layers)
        ])
        self.number_of_layers = number_of_layers
        self.normalization = normalization


    def forward(
            self,
            source: torch.Tensor,
            mask: torch.Tensor | None = None,
            source_key_padding_mask: torch.Tensor | None = None,
            positional_encoding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through all encoder layers.

        Args:
            source: Input tensor of shape (sequence_length, batch, model_dimension).
            mask: Attention mask of shape (sequence_length, sequence_length).
            source_key_padding_mask: Padding mask of shape (batch, sequence_length).
            positional_encoding: Positional encoding of shape (sequence_length, batch, model_dimension).

        Returns:
            Output tensor of shape (sequence_length, batch, model_dimension).
        """
        output = source
        for layer in self.layers:
            output = layer(
                output,
                source_mask=mask,
                source_key_padding_mask=source_key_padding_mask,
                positional_encoding=positional_encoding,
            )

        if self.normalization is not None:
            output = self.normalization(output)
        return output
