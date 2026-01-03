import torch
import torch.nn as nn

from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.detr_transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from refactoring.models.layers.detr_transformer.transformer_encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


class Transformer(nn.Module):
    """Transformer with encoder-decoder architecture and DETR-style positional encodings."""

    def __init__(
        self,
        embedding_dimension: int = 512,
        number_of_heads: int = 8,
        number_of_encoder_layers: int = 6,
        number_of_decoder_layers: int = 6,
        feedforward_dimension: int = 2048,
        dropout: float = 0.1,
        activation: str = ActivationFunction.RELU.value,
        normalize_before: bool = False,
        return_intermediate_decoder: bool = False,
    ):
        """Initialize transformer.

        Args:
            embedding_dimension: Model embedding dimension.
            number_of_heads: Number of attention heads.
            number_of_encoder_layers: Number of encoder layers.
            number_of_decoder_layers: Number of decoder layers.
            feedforward_dimension: Dimension of feedforward network.
            dropout: Dropout rate.
            activation: Activation function name from ActivationFunction enum.
            normalize_before: If True, use pre-normalization. If False, use post-normalization.
            return_intermediate_decoder: If True, return outputs from all decoder layers.

        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        encoder_layer = TransformerEncoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        encoder_normalization = (
            nn.LayerNorm(embedding_dimension) if normalize_before else None
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            number_of_layers=number_of_encoder_layers,
            normalization=encoder_normalization,
        )

        decoder_layer = TransformerDecoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        decoder_normalization = nn.LayerNorm(embedding_dimension)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            number_of_layers=number_of_decoder_layers,
            normalization=decoder_normalization,
            return_intermediate=return_intermediate_decoder,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        source_key_padding_mask: torch.Tensor | None = None,
        target_key_padding_mask: torch.Tensor | None = None,
        source_positional_encoding: torch.Tensor | None = None,
        target_positional_encoding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through transformer.

        Args:
            source: Input tensor of shape (batch size, source_length, embedding_dimension).
            target: Target tensor of shape (batch size, target_length, embedding_dimension).
            source_mask: Source attention mask of shape (source_length, source_length).
            target_mask: Target attention mask of shape (target_length, target_length).
            source_key_padding_mask: Source padding mask of shape (batch, source_length).
            target_key_padding_mask: Target padding mask of shape (batch, target_length).
            source_positional_encoding: Source PE of shape (batch size, source_length, embedding_dimension).
            target_positional_encoding: Target PE of shape (batch size, target_length, embedding_dimension).

        Returns:
            If return_intermediate is True, a tensor with shape (number_of_layers, batch_size, target_length,
             embedding_dimension). Otherwise, with shape  (1, batch_size, target_length, embedding_dimension).
        """
        memory = self.encoder(
            source=source,
            mask=source_mask,
            source_key_padding_mask=source_key_padding_mask,
            positional_encoding=source_positional_encoding,
        )

        output = self.decoder(
            target=target,
            memory=memory,
            target_mask=target_mask,
            memory_mask=None,
            target_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=source_key_padding_mask,
            memory_positional_encoding=source_positional_encoding,
            query_positional_encoding=target_positional_encoding,
        )
        return output  # type: ignore[no-any-return]
