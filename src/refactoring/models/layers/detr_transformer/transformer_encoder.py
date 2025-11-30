import copy
import math

import torch
import torch.nn as nn
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.detr_transformer.attention import FlashAttention
from refactoring.models.layers.normalization.ada_norm import AdaNorm
from refactoring.models.layers.normalization.rms_norm import RMSNorm


RESIDUAL_STREAM_FLAG = "SQUARE_ROOT_WEIGHT"


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre- and post- normalization support."""
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
        self.self_attention = FlashAttention(embedding_dimension=embedding_dimension,
                                        number_of_heads=number_of_heads, dropout=dropout)
        self.feedforward_linear1 = nn.Linear(embedding_dimension, feedforward_dimension)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_linear2 = nn.Linear(feedforward_dimension, embedding_dimension)
        self.normalization1 = nn.LayerNorm(embedding_dimension)
        self.normalization2 = nn.LayerNorm(embedding_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if activation == ActivationFunction.SWIGLU.value:
            self.activation = ActivationFunction(activation).to_torch_activation()(
                input_dim=embedding_dimension, hidden_dim=feedforward_dimension)
            self.feedforward_network = nn.Sequential(
                self.activation,
                self.feedforward_dropout,
                self.feedforward_linear2,
            )
        else:
            self.activation = ActivationFunction(activation).to_torch_activation()()
            self.feedforward_network = nn.Sequential(
                self.feedforward_linear1,
                self.activation,
                self.feedforward_dropout,
                self.feedforward_linear2,
            )
        self.feedforward_linear2.SQUARE_ROOT_WEIGHT = True  # Flag for initialization (GPT-2 style)


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
        source = self.self_attention(
            query=source,
            key=source,
            value=source,
            query_positional_encoding=positional_encoding,
            attention_mask=source_mask,
            key_padding_mask=source_key_padding_mask,
        )
        source = residual + self.dropout1(source)
        source = source if self.normalize_before else self.normalization1(source)
        residual = source
        source = self.normalization2(source) if self.normalize_before else source
        source = self.feedforward_network(source)
        source = residual + self.dropout2(source)
        return source if self.normalize_before else self.normalization2(source) # (B, T, C)


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    def __init__(
            self,
            encoder_layer: TransformerEncoderLayer,
            number_of_layers: int,
            normalization: nn.Module | None = None,
            initializer_range: float = 0.02,
    ):
        """Initialize transformer encoder.

        Args:
            encoder_layer: Single encoder layer to be stacked.
            number_of_layers: Number of encoder layers.
            normalization: Optional final normalization layer.
            initializer_range: Standard deviation for weight initialization.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(number_of_layers)
        ])
        self.number_of_layers = number_of_layers
        self.initializer_range = initializer_range
        self.normalization = normalization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        # > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        # > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        # > -- GPT-2 :: https://openai.com/blog/better-language-models/
        if hasattr(module, RESIDUAL_STREAM_FLAG):  # Residual stream correction
            num_norm_layers = 2
            std = self.initializer_range / math.sqrt(num_norm_layers * self.number_of_layers)
        else:
            std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm, AdaNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)


    def forward(
            self,
            source: torch.Tensor,
            mask: torch.Tensor | None = None,
            source_key_padding_mask: torch.Tensor | None = None,
            positional_encoding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through all encoder layers.

        Args:
            source: Input tensor of shape (batch size, sequence_length, embedding_dimension).
            mask: Attention mask of shape (sequence_length, sequence_length) where True indicates padding tokens.
            source_key_padding_mask: Padding mask of shape (batch size, sequence_length), where True indicates padding tokens.
            positional_encoding: Positional encoding of shape (batch size, sequence_length, embedding_dimension).

        Returns:
            Output tensor of shape (batch size, sequence_length, embedding_dimension).
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
