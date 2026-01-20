"""DiT decoder layer with modulation-based conditioning.

This layer uses FiLM-style modulation instead of cross-attention to condition
on timestep and encoder outputs.
"""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.transformer.attention import CachedAttention


class DiTDecoderLayer(nn.Module):
    """A decoder block with self-attention and modulation, inspired by DiT."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        activation_name: str = "gelu",
        normalization_type: str = NormalizationType.ADALN.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
    ) -> None:
        """Initialize the DiTDecoderLayer."""
        super().__init__()

        # Self-attention (custom attention uses batch-first inputs)
        self.self_attention = CachedAttention(
            embedding_dimension=embedding_dim,
            number_of_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
        )

        # Feedforward subnetwork
        self.linear1 = nn.Linear(embedding_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, embedding_dim)

        # Normalization layers
        condition_dim = None
        if normalization_type in (
            NormalizationType.ADALN.value,
            NormalizationType.ADARMS.value,
        ):
            condition_dim = embedding_dim
        self.norm1 = create_normalization_layer(
            normalization_type, embedding_dim, condition_dim=condition_dim
        )
        self.norm2 = create_normalization_layer(
            normalization_type, embedding_dim, condition_dim=condition_dim
        )

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation function
        if activation_name == "relu":
            self.activation = nn.ReLU()
        elif activation_name == "gelu":
            self.activation = nn.GELU(approximate="tanh")
        elif activation_name == "glu":
            self.activation = nn.GLU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        # Modulation layers for attention and MLP, conditioned on timestep + condition mean.
        # Mod1: Shift+Scale before attention (after norm)
        self.attention_mod1 = ConditionalModulation(
            condition_dim=embedding_dim,
            feature_dim=embedding_dim,
            use_shift=True,
            activation=ActivationFunction.SILU.value,
            init_strategy="xavier",
        )
        # Mod2: Scale-only after attention (zero-initialized)
        self.attention_mod2 = ConditionalModulation(
            condition_dim=embedding_dim,
            feature_dim=embedding_dim,
            use_shift=False,
            activation=ActivationFunction.SILU.value,
            init_strategy="zero",
        )
        # Mod3: Shift+Scale before MLP (after norm)
        self.mlp_mod1 = ConditionalModulation(
            condition_dim=embedding_dim,
            feature_dim=embedding_dim,
            use_shift=True,
            activation=ActivationFunction.SILU.value,
            init_strategy="xavier",
        )
        # Mod4: Scale-only after MLP (zero-initialized)
        self.mlp_mod2 = ConditionalModulation(
            condition_dim=embedding_dim,
            feature_dim=embedding_dim,
            use_shift=False,
            activation=ActivationFunction.SILU.value,
            init_strategy="zero",
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        timestep_embedding: torch.Tensor,
        condition_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the decoder block."""
        # Combine condition by taking mean over sequence length and adding to timestep.
        # condition_tensor: (S, B, D) -> mean -> (B, D)
        condition_mean = torch.mean(condition_tensor, dim=0)
        combined_condition = condition_mean + timestep_embedding  # (B, D)

        # Attention branch with modulation
        # Input: (S, B, D), Condition: (B, D)
        normalized_input = self._apply_norm(
            self.norm1, input_tensor, combined_condition
        )  # (S, B, D)
        modulated_input = self.attention_mod1(
            normalized_input, combined_condition
        )  # (S, B, D)
        modulated_input_bf = modulated_input.transpose(0, 1)
        attended_bf, _ = self.self_attention(
            modulated_input_bf,
            modulated_input_bf,
            modulated_input_bf,
            attention_mask=None,
        )  # (B, S, D)
        attended_tensor = attended_bf.transpose(0, 1)  # (S, B, D)
        input_tensor = (
            self.attention_mod2(self.dropout1(attended_tensor), combined_condition)
            + input_tensor
        )

        # MLP branch with modulation
        normalized_input = self._apply_norm(
            self.norm2, input_tensor, combined_condition
        )  # (S, B, D)
        modulated_input = self.mlp_mod1(normalized_input, combined_condition)  # (S, B, D)
        feedforward_tensor = self.linear2(
            self.dropout2(self.activation(self.linear1(modulated_input)))
        )  # (S, B, D)
        feedforward_tensor = self.mlp_mod2(
            self.dropout3(feedforward_tensor), combined_condition
        )  # (S, B, D)
        return input_tensor + feedforward_tensor

    def _apply_norm(
        self,
        norm_layer: nn.Module,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(norm_layer, AdaNorm):
            result = norm_layer(x, condition)
            if isinstance(result, tuple):
                return result[0]
            return result
        return norm_layer(x)

    def reset_parameters(self) -> None:
        """Reset parameters to zeros (DiT initialization)."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        # Reset modulation parameters
        for mod in (self.attention_mod1, self.attention_mod2, self.mlp_mod1, self.mlp_mod2):
            mod.init_parameters()

