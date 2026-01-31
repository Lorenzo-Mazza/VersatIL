"""MMDiT (Multimodal Diffusion Transformer) layer implementation.

Implements dual-stream processing with joint attention as described in the
Stable Diffusion 3 paper. Each stream (observations and actions) has independent
weights but shares attention through key-value concatenation.

References:
    Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2403.03206
"""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.diffusion_transformer.joint_attention import JointAttention
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.swiglu import SwiGLU


class MMDiTLayer(nn.Module):
    """Multimodal Diffusion Transformer layer.

    Combines JointAttention with AdaNorm modulation and feedforward networks
    for dual-stream processing. Uses AdaNorm with gating (AdaLN-Zero) for stable training.

    Shape notation:
        B: batch size
        S: observation sequence length
        T: action sequence length
        D: embedding dimension
    """

    def __init__(
        self,
        embedding_dimension: int,
        conditioning_dimension: int,
        number_of_heads: int,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        normalization_epsilon: float = 1e-6,
        use_query_key_norm: bool = True,
        use_gating: bool = True,
        bias: bool = True,
    ):
        """Initialize MMDiT layer.

        Args:
            embedding_dimension: Hidden dimension for both streams.
            conditioning_dimension: Dimension of the conditioning vector (timestep embedding).
            number_of_heads: Number of attention heads.
            feedforward_dimension: FFN hidden dimension. Defaults to 4 * embedding_dimension.
            dropout: Dropout rate for residual connections.
            attention_dropout: Dropout rate for attention weights.
            activation: Activation function for FFN.
            normalization_type: Type of normalization layer.
            normalization_epsilon: Epsilon for normalization layers.
            use_query_key_norm: Whether to apply QK-normalization before attention.
            use_gating: Whether to use gating in AdaNorm (AdaLN-Zero style).
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.use_gating = use_gating
        if feedforward_dimension is None:
            feedforward_dimension = 4 * embedding_dimension
        base_normalization_layer = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.attention_normalization_observation = AdaNorm(
            base_norm=base_normalization_layer,
            condition_dim=conditioning_dimension,
            feature_dim=embedding_dimension,
            use_gate=use_gating,
        )
        self.attention_normalization_action = AdaNorm(
            base_norm=base_normalization_layer,
            condition_dim=conditioning_dimension,
            feature_dim=embedding_dimension,
            use_gate=use_gating,
        )
        self.joint_attention = JointAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            dropout=attention_dropout,
            use_query_key_norm=use_query_key_norm,
            normalization_epsilon=normalization_epsilon,
            bias=bias,
        )
        self.feedforward_normalization_observation = AdaNorm(
            base_norm=base_normalization_layer,
            condition_dim=conditioning_dimension,
            feature_dim=embedding_dimension,
            use_gate=use_gating,
        )
        self.feedforward_normalization_action = AdaNorm(
            base_norm=base_normalization_layer,
            condition_dim=conditioning_dimension,
            feature_dim=embedding_dimension,
            use_gate=use_gating,
        )
        self.attention_dropout_observation = nn.Dropout(dropout)
        self.attention_dropout_action = nn.Dropout(dropout)
        self.feedforward_dropout_observation = nn.Dropout(dropout)
        self.feedforward_dropout_action = nn.Dropout(dropout)
        if activation == ActivationFunction.SWIGLU.value:
            self.feedforward_observation = nn.Sequential(
                SwiGLU(embedding_dimension, feedforward_dimension, bias=bias),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
            self.feedforward_action = nn.Sequential(
                SwiGLU(embedding_dimension, feedforward_dimension, bias=bias),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
        else:
            activation_function = ActivationFunction(activation).to_torch_activation()()
            self.feedforward_observation = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension, bias=bias),
                activation_function,
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
            self.feedforward_action = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension, bias=bias),
                activation_function,
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
        self.feedforward_observation[-1].SQUARE_ROOT_WEIGHT = True
        self.feedforward_action[-1].SQUARE_ROOT_WEIGHT = True

    def forward(
        self,
        hidden_states_observation: torch.Tensor,
        hidden_states_action: torch.Tensor,
        conditioning: torch.Tensor,
        attention_mask_observation: torch.Tensor | None = None,
        attention_mask_action: torch.Tensor | None = None,
        positional_encoding_observation: RotaryPositionalEncoding | None = None,
        positional_encoding_action: RotaryPositionalEncoding | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dual-stream joint attention.

        Args:
            hidden_states_observation: Observation stream (B, S, D).
            hidden_states_action: Action stream (B, T, D).
            conditioning: Conditioning vector from timestep embedding (B, D).
            attention_mask_observation: Padding mask for observations (B, S) where True = masked.
            attention_mask_action: Padding mask for actions (B, T) where True = masked.
            positional_encoding_observation: Optional RoPE for observation stream.
            positional_encoding_action: Optional RoPE for action stream.

        Returns:
            Tuple of (observation_output, action_output) with same shapes as inputs.
        """
        residual_observation = hidden_states_observation
        residual_action = hidden_states_action
        if self.use_gating:
            (
                normed_observation,
                gate_attention_observation,
            ) = self.attention_normalization_observation(
                x=hidden_states_observation, condition=conditioning
            )
            normed_action, gate_attention_action = self.attention_normalization_action(
                x=hidden_states_action, condition=conditioning
            )
        else:
            normed_observation = self.attention_normalization_observation(
                x=hidden_states_observation, condition=conditioning
            )
            normed_action = self.attention_normalization_action(
                x=hidden_states_action, condition=conditioning
            )
            gate_attention_observation = 1.0
            gate_attention_action = 1.0

        attention_output_observation, attention_output_action = self.joint_attention(
            hidden_states_observation=normed_observation,
            hidden_states_action=normed_action,
            attention_mask_observation=attention_mask_observation,
            attention_mask_action=attention_mask_action,
            positional_encoding_observation=positional_encoding_observation,
            positional_encoding_action=positional_encoding_action,
        )
        hidden_states_observation = (
            residual_observation
            + gate_attention_observation
            * self.attention_dropout_observation(attention_output_observation)
        )
        hidden_states_action = (
            residual_action
            + gate_attention_action
            * self.attention_dropout_action(attention_output_action)
        )
        residual_observation = hidden_states_observation
        residual_action = hidden_states_action
        if self.use_gating:
            (
                normed_observation,
                gate_feedforward_observation,
            ) = self.feedforward_normalization_observation(
                x=hidden_states_observation, condition=conditioning
            )
            (
                normed_action,
                gate_feedforward_action,
            ) = self.feedforward_normalization_action(
                x=hidden_states_action, condition=conditioning
            )
        else:
            normed_observation = self.feedforward_normalization_observation(
                x=hidden_states_observation, condition=conditioning
            )
            normed_action = self.feedforward_normalization_action(
                x=hidden_states_action, condition=conditioning
            )
            gate_feedforward_observation = 1.0
            gate_feedforward_action = 1.0

        feedforward_output_observation = self.feedforward_observation(
            normed_observation
        )
        feedforward_output_action = self.feedforward_action(normed_action)
        hidden_states_observation = (
            residual_observation
            + gate_feedforward_observation
            * self.feedforward_dropout_observation(feedforward_output_observation)
        )
        hidden_states_action = (
            residual_action
            + gate_feedforward_action
            * self.feedforward_dropout_action(feedforward_output_action)
        )
        return hidden_states_observation, hidden_states_action
