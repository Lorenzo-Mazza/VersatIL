"""Feedforward block: norm -> FFN -> gated residual."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.normalization.typedefs import BlockNormalization
from versatil.models.layers.transformer.block.base import TransformerBlock


def build_feedforward(
    embedding_dimension: int,
    feedforward_dimension: int,
    activation: str = ActivationFunction.SWIGLU.value,
    dropout: float = 0.1,
    bias: bool = True,
) -> nn.Sequential:
    """Build a gated or standard feedforward layer.

    Args:
        embedding_dimension: Input and output dimension.
        feedforward_dimension: Hidden dimension.
        activation: Activation function name from ActivationFunction enum.
        dropout: Dropout rate.
        bias: Whether to use bias in linear layers.

    Note:
        Sets `SQUARE_ROOT_WEIGHT flag` to true, for scaling the variance of residual connections.

    Returns:
        Sequential with `SQUARE_ROOT_WEIGHT flag` on the final linear.
    """
    activation_enum = ActivationFunction(activation)
    if activation_enum.is_gated:
        feedforward = nn.Sequential(
            activation_enum.to_torch_activation()(
                input_dim=embedding_dimension,
                hidden_dim=feedforward_dimension,
                bias=bias,
            ),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
        )
    else:
        feedforward = nn.Sequential(
            nn.Linear(embedding_dimension, feedforward_dimension, bias=bias),
            activation_enum.to_torch_activation()(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
        )
    feedforward[-1].SQUARE_ROOT_WEIGHT = True
    return feedforward


class FeedforwardBlock(TransformerBlock):
    """Norm -> feedforward -> gated residual."""

    def __init__(
        self,
        feedforward: nn.Module,
        normalization: BlockNormalization,
        dropout: float = 0.1,
    ):
        super().__init__(normalization=normalization, dropout=dropout)
        self.feedforward = feedforward

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Norm -> feedforward -> gated residual.

        Args:
            hidden_states: Input embeddings (B, S, D).
            conditioning: Conditioning vector for AdaNorm (B, C). Ignored by UnconditionedNorm.

        Returns:
            Output hidden states (B, S, D).
        """
        residual = hidden_states
        hidden_states, gate = self.normalization(
            x=hidden_states, condition=conditioning
        )
        feedforward_output = self.feedforward(hidden_states)
        hidden_states = self.apply_residual(residual, feedforward_output, gate)
        return hidden_states
