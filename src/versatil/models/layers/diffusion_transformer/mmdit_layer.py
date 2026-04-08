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


class MMDiTLayer(nn.Module):
    """Multimodal Diffusion Transformer layer with dual-stream joint attention.

    Combines ``JointAttention`` with optional AdaNorm conditioning and
    per-stream feedforward networks. Supports asymmetric stream dimensions,
    GQA, and precomputed primary Q/K/V for VLA decoders.

    Shape notation:
        B: batch size
        S: primary (observation) sequence length
        T: secondary (action) sequence length
        D_p: primary embedding dimension
        D_s: secondary embedding dimension
    """

    def __init__(
        self,
        embedding_dimension: int,
        conditioning_dimension: int,
        number_of_heads: int,
        secondary_embedding_dimension: int | None = None,
        number_of_key_value_heads: int | None = None,
        head_dimension: int | None = None,
        feedforward_dimension: int | None = None,
        secondary_feedforward_dimension: int | None = None,
        precomputed_primary_stream: bool = False,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        normalization_epsilon: float = 1e-6,
        use_query_key_norm: bool = True,
        use_conditioning: bool = True,
        use_gating: bool = True,
        bias: bool = True,
    ):
        """Initialize MMDiT layer.

        Args:
            embedding_dimension: Hidden dimension for the observation (primary) stream.
            conditioning_dimension: Dimension of the conditioning vector.
            number_of_heads: Number of query attention heads for both streams.
            secondary_embedding_dimension: Hidden dimension for the action (secondary) stream.
                Defaults to ``embedding_dimension`` for symmetric streams.
            number_of_key_value_heads: Number of key/value heads for GQA.
            head_dimension: Per-head dimension override.
            feedforward_dimension: Observation FFN hidden dimension.
                Defaults to 4 * embedding_dimension.
            secondary_feedforward_dimension: Action FFN hidden dimension.
                Defaults to ``feedforward_dimension``.
            precomputed_primary_stream: When ``True``, the primary stream's Q/K/V
                are provided via ``precomputed_observation`` at forward time. Skips
                creating primary norms and FFN.
            dropout: Dropout rate for residual connections.
            attention_dropout: Dropout rate for attention weights.
            activation: Activation function for FFN.
            normalization_type: Type of normalization layer.
            normalization_epsilon: Epsilon for normalization layers.
            use_query_key_norm: Whether to apply QK-normalization before attention.
            use_conditioning: Whether to use AdaNorm (conditioned normalization).
                When ``False``, uses plain normalization and ignores the conditioning
                input. Avoids wasteful zero-conditioning.
            use_gating: Whether to use gating in AdaNorm (AdaLN-Zero style).
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        secondary_embedding_dimension = (
            secondary_embedding_dimension or embedding_dimension
        )
        if feedforward_dimension is None:
            feedforward_dimension = 4 * embedding_dimension
        secondary_feedforward_dimension = (
            secondary_feedforward_dimension or feedforward_dimension
        )
        self.embedding_dimension = embedding_dimension
        self.secondary_embedding_dimension = secondary_embedding_dimension
        self.precomputed_primary_stream = precomputed_primary_stream
        self.use_conditioning = use_conditioning
        self.use_gating = use_gating and use_conditioning
        if not precomputed_primary_stream:
            if use_conditioning:
                self.attention_normalization_observation = AdaNorm(
                    base_norm=create_normalization_layer(
                        normalization_type=normalization_type,
                        dimension=embedding_dimension,
                        epsilon=normalization_epsilon,
                    ),
                    condition_dim=conditioning_dimension,
                    feature_dim=embedding_dimension,
                    use_gate=self.use_gating,
                )
                self.feedforward_normalization_observation = AdaNorm(
                    base_norm=create_normalization_layer(
                        normalization_type=normalization_type,
                        dimension=embedding_dimension,
                        epsilon=normalization_epsilon,
                    ),
                    condition_dim=conditioning_dimension,
                    feature_dim=embedding_dimension,
                    use_gate=self.use_gating,
                )
            else:
                self.attention_normalization_observation = create_normalization_layer(
                    normalization_type=normalization_type,
                    dimension=embedding_dimension,
                    epsilon=normalization_epsilon,
                )
                self.feedforward_normalization_observation = create_normalization_layer(
                    normalization_type=normalization_type,
                    dimension=embedding_dimension,
                    epsilon=normalization_epsilon,
                )
            self.feedforward_observation = self._build_feedforward(
                embedding_dimension,
                feedforward_dimension,
                activation,
                dropout,
                bias,
            )
            self.feedforward_observation[-1].SQUARE_ROOT_WEIGHT = True
            self.attention_dropout_observation = nn.Dropout(dropout)
            self.feedforward_dropout_observation = nn.Dropout(dropout)
        if use_conditioning:
            self.attention_normalization_action = AdaNorm(
                base_norm=create_normalization_layer(
                    normalization_type=normalization_type,
                    dimension=secondary_embedding_dimension,
                    epsilon=normalization_epsilon,
                ),
                condition_dim=conditioning_dimension,
                feature_dim=secondary_embedding_dimension,
                use_gate=self.use_gating,
            )
        else:
            self.attention_normalization_action = create_normalization_layer(
                normalization_type=normalization_type,
                dimension=secondary_embedding_dimension,
                epsilon=normalization_epsilon,
            )
        self.joint_attention = JointAttention(
            primary_embedding_dimension=embedding_dimension,
            secondary_embedding_dimension=secondary_embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            head_dimension=head_dimension,
            precomputed_primary_stream=precomputed_primary_stream,
            dropout=attention_dropout,
            use_query_key_norm=use_query_key_norm,
            normalization_epsilon=normalization_epsilon,
            bias=bias,
        )
        if use_conditioning:
            self.feedforward_normalization_action = AdaNorm(
                base_norm=create_normalization_layer(
                    normalization_type=normalization_type,
                    dimension=secondary_embedding_dimension,
                    epsilon=normalization_epsilon,
                ),
                condition_dim=conditioning_dimension,
                feature_dim=secondary_embedding_dimension,
                use_gate=self.use_gating,
            )
        else:
            self.feedforward_normalization_action = create_normalization_layer(
                normalization_type=normalization_type,
                dimension=secondary_embedding_dimension,
                epsilon=normalization_epsilon,
            )
        self.attention_dropout_action = nn.Dropout(dropout)
        self.feedforward_dropout_action = nn.Dropout(dropout)
        self.feedforward_action = self._build_feedforward(
            secondary_embedding_dimension,
            secondary_feedforward_dimension,
            activation,
            dropout,
            bias,
        )
        self.feedforward_action[-1].SQUARE_ROOT_WEIGHT = True

    @staticmethod
    def _build_feedforward(
        embedding_dimension: int,
        feedforward_dimension: int,
        activation: str,
        dropout: float,
        bias: bool,
    ) -> nn.Sequential:
        activation_enum = ActivationFunction(activation)
        if activation_enum.is_gated:
            gated_unit = activation_enum.to_torch_activation()(
                embedding_dimension, feedforward_dimension, bias=bias
            )
            return nn.Sequential(
                gated_unit,
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
        return nn.Sequential(
            nn.Linear(embedding_dimension, feedforward_dimension, bias=bias),
            activation_enum.to_torch_activation()(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
        )

    def forward(
        self,
        hidden_states_observation: torch.Tensor,
        hidden_states_action: torch.Tensor,
        conditioning: torch.Tensor | None,
        attention_mask_observation: torch.Tensor | None = None,
        attention_mask_action: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        precomputed_observation: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        positional_encoding_observation: RotaryPositionalEncoding | None = None,
        positional_encoding_action: RotaryPositionalEncoding | None = None,
        precomputed_action_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dual-stream joint attention.

        Args:
            hidden_states_observation: Observation stream (B, S, D_p).
            hidden_states_action: Action stream (B, T, D_s).
            conditioning: Conditioning vector from timestep embedding (B, C).
            attention_mask_observation: Per-stream padding mask (B, S), True = masked.
            attention_mask_action: Per-stream padding mask (B, T), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T). When provided,
                used directly instead of building from per-stream masks.
            precomputed_observation: Pre-projected primary (Q, K, V) tuple.
                Skips observation normalization and projection when provided.
            positional_encoding_observation: Optional RoPE for observation stream.
            positional_encoding_action: Optional RoPE for action stream.
            precomputed_action_rope: Pre-computed (cos, sin) for action positions.

        Returns:
            Tuple of (observation_output, action_output).
        """
        residual_observation = hidden_states_observation
        residual_action = hidden_states_action
        if precomputed_observation is not None:
            normed_observation = hidden_states_observation
            gate_attention_observation = 1.0
        else:
            normed_observation, gate_attention_observation = self._apply_norm(
                self.attention_normalization_observation,
                hidden_states_observation,
                conditioning,
            )
        normed_action, gate_attention_action = self._apply_norm(
            self.attention_normalization_action,
            hidden_states_action,
            conditioning,
        )
        attention_output_observation, attention_output_action = self.joint_attention(
            hidden_states_observation=normed_observation,
            hidden_states_action=normed_action,
            attention_mask_observation=attention_mask_observation,
            attention_mask_action=attention_mask_action,
            joint_attention_mask=joint_attention_mask,
            precomputed_observation=precomputed_observation,
            positional_encoding_observation=positional_encoding_observation,
            positional_encoding_action=positional_encoding_action,
            precomputed_action_rope=precomputed_action_rope,
        )
        if not self.precomputed_primary_stream:
            hidden_states_observation = (
                residual_observation
                + gate_attention_observation
                * self.attention_dropout_observation(attention_output_observation)
            )
        else:
            hidden_states_observation = attention_output_observation
        hidden_states_action = (
            residual_action
            + gate_attention_action
            * self.attention_dropout_action(attention_output_action)
        )
        if not self.precomputed_primary_stream:
            residual_observation = hidden_states_observation
            normed_observation, gate_feedforward_observation = self._apply_norm(
                self.feedforward_normalization_observation,
                hidden_states_observation,
                conditioning,
            )
            hidden_states_observation = (
                residual_observation
                + gate_feedforward_observation
                * self.feedforward_dropout_observation(
                    self.feedforward_observation(normed_observation)
                )
            )
        residual_action = hidden_states_action
        normed_action, gate_feedforward_action = self._apply_norm(
            self.feedforward_normalization_action,
            hidden_states_action,
            conditioning,
        )
        hidden_states_action = (
            residual_action
            + gate_feedforward_action
            * self.feedforward_dropout_action(self.feedforward_action(normed_action))
        )
        return hidden_states_observation, hidden_states_action

    def _apply_norm(
        self,
        norm: nn.Module,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply normalization, handling both AdaNorm and plain norm."""
        if self.use_conditioning:
            return norm(x=hidden_states, condition=conditioning)
        else:
            return norm(hidden_states), torch.ones(
                1, dtype=hidden_states.dtype, device=hidden_states.device
            )
