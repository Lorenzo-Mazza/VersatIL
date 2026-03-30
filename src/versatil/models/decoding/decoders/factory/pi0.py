"""Pi0/Pi0.5 interleaved VLM-expert decoder with joint attention."""

import enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_masking import make_attention_mask
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.layers.diffusion_transformer.mmdit_layer import MMDiTLayer
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.base import PositionSource
from versatil.models.layers.positional_encoding.sinusoidal import (
    PeriodInterpolationPositionalEncoding1D,
)


class Pi0TimeConditioning(enum.StrEnum):
    """Timestep conditioning strategy for Pi0 variants."""

    CONCAT_MLP = "concat_mlp"
    ADANORM = "adanorm"


class Pi0Decoder(ActionDecoder):
    """Pi0/Pi0.5 decoder with pretrained VLM backbone and learned action expert.

    All VLM layers are paired 1:1 with expert layers via joint attention
    (``MMDiTLayer`` with ``precomputed_primary_stream=True``). VLM layers
    are referenced directly from the encoder — not duplicated.
    """

    def __init__(
        self,
        input_keys: list[str],
        action_space: ActionSpace,
        action_heads: dict[str, ActionHead],
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        expert_hidden_size: int,
        expert_intermediate_size: int,
        expert_number_of_heads: int,
        expert_number_of_key_value_heads: int,
        expert_number_of_layers: int,
        expert_head_dimension: int,
        time_conditioning: str = Pi0TimeConditioning.CONCAT_MLP.value,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        use_state_token: bool = True,
        state_dimension: int = 0,
        proprioceptive_feature_key: str = "state",
        normalization_type: str = NormalizationType.RMS_NORM.value,
        dropout: float = 0.1,
    ):
        """Initialize the Pi0 decoder.

        Args:
            input_keys: Feature keys from the encoding pipeline.
            action_space: Action space configuration.
            action_heads: Action prediction heads.
            observation_space: Observation space configuration.
            observation_horizon: Number of observation timesteps.
            prediction_horizon: Number of action steps to predict.
            device: Device string.
            expert_hidden_size: Hidden dimension of the action expert.
            expert_intermediate_size: Expert feedforward hidden dimension.
            expert_number_of_heads: Number of query heads in the expert.
            expert_number_of_key_value_heads: Number of key/value heads (GQA).
            expert_number_of_layers: Number of expert transformer layers.
            expert_head_dimension: Per-head dimension of the expert.
            time_conditioning: ``"concat_mlp"`` (Pi0) or ``"adanorm"`` (Pi0.5).
            min_period: Minimum period for sinusoidal timestep embedding.
            max_period: Maximum period for sinusoidal timestep embedding.
            use_state_token: Whether to prepend proprioceptive state as suffix token.
            state_dimension: Dimension of proprioceptive state input.
            proprioceptive_feature_key: Key for proprioceptive state in features dict.
            normalization_type: Normalization layer type.
            dropout: Dropout rate.
        """
        decoder_input = DecoderInput(
            keys=input_keys,
            requires_actions=True,
            requires_vlm_backbone=True,
        )
        super().__init__(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
        )
        self.time_conditioning = time_conditioning
        self.expert_hidden_size = expert_hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.expert_number_of_heads = expert_number_of_heads
        self.expert_number_of_key_value_heads = expert_number_of_key_value_heads
        self.expert_number_of_layers = expert_number_of_layers
        self.expert_head_dimension = expert_head_dimension
        self.use_state_token = use_state_token
        self.state_dimension = state_dimension
        self.proprioceptive_feature_key = proprioceptive_feature_key
        self.normalization_type = normalization_type
        self._dropout = dropout
        self.action_input_projection = nn.Linear(self.action_dim, expert_hidden_size)
        self.action_output_projection = nn.Linear(expert_hidden_size, self.action_dim)
        self.timestep_embedding = PeriodInterpolationPositionalEncoding1D(
            embedding_dimension=expert_hidden_size,
            min_period=min_period,
            max_period=max_period,
            position_source=PositionSource.SCALAR.value,
        )
        match time_conditioning:
            case Pi0TimeConditioning.CONCAT_MLP.value:
                self.action_time_fusion_input = nn.Linear(
                    expert_hidden_size * 2, expert_hidden_size
                )
                self.action_time_fusion_output = nn.Linear(
                    expert_hidden_size, expert_hidden_size
                )
                self.state_projection = (
                    nn.Linear(state_dimension, expert_hidden_size)
                    if use_state_token and state_dimension > 0
                    else None
                )
            case Pi0TimeConditioning.ADANORM.value:
                self.time_mlp_input = nn.Linear(expert_hidden_size, expert_hidden_size)
                self.time_mlp_output = nn.Linear(expert_hidden_size, expert_hidden_size)
                self.state_projection = None
            case _:
                raise ValueError(
                    f"Unknown time_conditioning: {time_conditioning}. Use {[m.value for m in Pi0TimeConditioning]}"
                )
        self.vlm_layers: nn.ModuleList | None = None
        self.vlm_rotary_embedding: nn.Module | None = None
        self.vlm_hidden_dimension: int | None = None
        self.expert_layers: nn.ModuleList | None = None
        self.expert_final_normalization: nn.Module | None = None
        self._encoder_cache_enabled = False
        self.to(self.device)

    def set_backbone(
        self,
        vlm_layers: nn.ModuleList,
        rotary_emb: nn.Module,
        vlm_hidden_dim: int,
        vlm_text_config: PretrainedConfig,
    ) -> None:
        """Reference pretrained VLM layers and create expert layers.

        Args:
            vlm_layers: VLM transformer layers (referenced directly, not copied).
            rotary_emb: VLM rotary positional encoding module.
            vlm_hidden_dim: VLM hidden dimension.
            vlm_text_config: VLM text model config.
        """
        if len(vlm_layers) != self.expert_number_of_layers:
            raise ValueError(
                f"Pi0 requires equal VLM ({len(vlm_layers)}) and expert ({self.expert_number_of_layers}) layer counts."
            )
        self.vlm_layers = vlm_layers
        self.vlm_rotary_embedding = rotary_emb
        self.vlm_hidden_dimension = vlm_hidden_dim
        use_conditioning = self.time_conditioning == Pi0TimeConditioning.ADANORM.value
        self.expert_layers = nn.ModuleList(
            [
                MMDiTLayer(
                    embedding_dimension=vlm_hidden_dim,
                    conditioning_dimension=self.expert_hidden_size,
                    number_of_heads=self.expert_number_of_heads,
                    secondary_embedding_dimension=self.expert_hidden_size,
                    number_of_key_value_heads=self.expert_number_of_key_value_heads,
                    head_dimension=self.expert_head_dimension,
                    secondary_feedforward_dimension=self.expert_intermediate_size,
                    precomputed_primary_stream=True,
                    normalization_type=self.normalization_type,
                    use_gating=use_conditioning,
                    dropout=self._dropout,
                    bias=False,
                )
                for _ in range(self.expert_number_of_layers)
            ]
        )
        self.expert_final_normalization = create_normalization_layer(
            normalization_type=self.normalization_type,
            dimension=self.expert_hidden_size,
        )

    def enable_encoder_cache(self) -> None:
        """Enable prefix caching for multi-step denoising inference."""
        self._encoder_cache_enabled = True

    def disable_encoder_cache(self) -> None:
        """Disable prefix caching and clear stored states."""
        self._encoder_cache_enabled = False

    def _embed_suffix(
        self,
        actions: dict[str, torch.Tensor],
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Project actions and apply time conditioning.

        Returns:
            (suffix_embeddings, adaptive_norm_conditioning_or_none)
        """
        action_tensors = [actions[key] for key in sorted(self.action_heads.keys())]
        expert_hidden = self.action_input_projection(torch.cat(action_tensors, dim=-1))
        time_embedding = self.timestep_embedding(timestep)  # (B, D)
        match self.time_conditioning:
            case Pi0TimeConditioning.CONCAT_MLP.value:
                time_expanded = time_embedding.unsqueeze(1).expand_as(expert_hidden)
                fused = torch.cat([expert_hidden, time_expanded], dim=-1)  # (B, H, 2D)
                return self.action_time_fusion_output(
                    F.silu(self.action_time_fusion_input(fused))
                ), None
            case Pi0TimeConditioning.ADANORM.value:
                conditioning = F.silu(
                    self.time_mlp_output(F.silu(self.time_mlp_input(time_embedding)))
                )
                return expert_hidden, conditioning

    @staticmethod
    def _extract_vlm_query_key_value(
        vlm_layer: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract Q/K/V from a pretrained VLM layer, reshaped for multi-head attention.

        Args:
            vlm_layer: Pretrained VLM transformer layer.
            hidden_states: VLM hidden states (B, P, D_vlm).

        Returns:
            (query, key, value) each (B, heads, P, head_dimension).
        """
        normalized = vlm_layer.input_layernorm(hidden_states)
        attention = vlm_layer.self_attn
        batch_size, sequence_length, _ = normalized.shape
        head_dimension = attention.head_dim
        number_of_heads = attention.config.num_attention_heads
        number_of_key_value_heads = attention.config.num_key_value_heads
        query = (
            attention.q_proj(normalized)
            .view(
                batch_size,
                sequence_length,
                number_of_heads,
                head_dimension,
            )
            .transpose(1, 2)
        )  # (B, H, P, D_head)
        key = (
            attention.k_proj(normalized)
            .view(
                batch_size,
                sequence_length,
                number_of_key_value_heads,
                head_dimension,
            )
            .transpose(1, 2)
        )  # (B, KV_H, P, D_head)
        value = (
            attention.v_proj(normalized)
            .view(
                batch_size,
                sequence_length,
                number_of_key_value_heads,
                head_dimension,
            )
            .transpose(1, 2)
        )  # (B, KV_H, P, D_head)
        return query, key, value

    @staticmethod
    def _apply_vlm_post_attention(
        vlm_layer: nn.Module,
        vlm_residual: torch.Tensor,
        vlm_attention_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply VLM layer's O-projection, residual, and feedforward.

        Args:
            vlm_layer: Pretrained VLM transformer layer.
            vlm_residual: VLM hidden states before attention (B, P, D_vlm).
            vlm_attention_output: Raw attention output (B, P, H * D_head).

        Returns:
            Updated VLM hidden states (B, P, D_vlm).
        """
        vlm_attention_output = vlm_layer.self_attn.o_proj(vlm_attention_output)
        hidden_states = vlm_residual + vlm_attention_output
        if hasattr(vlm_layer, "post_attention_layernorm"):
            hidden_states = vlm_residual + vlm_layer.post_attention_layernorm(
                vlm_attention_output
            )
        residual = hidden_states
        if hasattr(vlm_layer, "pre_feedforward_layernorm"):
            hidden_states = vlm_layer.pre_feedforward_layernorm(hidden_states)
        else:
            hidden_states = vlm_layer.post_attention_layernorm(hidden_states)
        hidden_states = vlm_layer.mlp(hidden_states)
        if hasattr(vlm_layer, "post_feedforward_layernorm"):
            hidden_states = vlm_layer.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through interleaved VLM + expert layers.

        Args:
            features: Encoded observation features from the encoding pipeline.
            actions: Noisy action tensors keyed by action name.

        Returns:
            Predicted action tensors keyed by action name.
        """
        if self.expert_layers is None:
            raise RuntimeError("set_backbone() must be called before forward().")
        prefix_embeddings = features[self.decoder_input.keys[0]]
        timestep = features.get(DecoderOutputKey.TIMESTEP.value)
        if timestep is None:
            timestep = torch.zeros(
                prefix_embeddings.shape[0], device=prefix_embeddings.device
            )
        expert_hidden, adaptive_norm_conditioning = self._embed_suffix(
            actions, timestep
        )
        if (
            self.state_projection is not None
            and self.time_conditioning == Pi0TimeConditioning.CONCAT_MLP.value
        ):
            state = features.get(self.proprioceptive_feature_key)
            if state is not None:
                expert_hidden = torch.cat(
                    [self.state_projection(state).unsqueeze(1), expert_hidden], dim=1
                )
        attention_mask, _ = make_attention_mask(
            action_tokens=expert_hidden,
            feature_tokens=prefix_embeddings,
        )
        if adaptive_norm_conditioning is None:
            adaptive_norm_conditioning = torch.zeros(
                expert_hidden.shape[0],
                self.expert_hidden_size,
                device=expert_hidden.device,
            )
        vlm_hidden = prefix_embeddings
        for layer_index in range(self.expert_number_of_layers):
            vlm_layer = self.vlm_layers[layer_index]
            with torch.no_grad():
                vlm_query, vlm_key, vlm_value = self._extract_vlm_query_key_value(
                    vlm_layer, vlm_hidden
                )
            vlm_attention_output, expert_hidden = self.expert_layers[layer_index](
                hidden_states_observation=vlm_hidden,
                hidden_states_action=expert_hidden,
                conditioning=adaptive_norm_conditioning,
                joint_attention_mask=attention_mask,
                precomputed_observation=(vlm_query, vlm_key, vlm_value),
            )
            with torch.no_grad():
                vlm_hidden = self._apply_vlm_post_attention(
                    vlm_layer, vlm_hidden, vlm_attention_output
                )
        expert_hidden = self.expert_final_normalization(expert_hidden)
        action_output = self.action_output_projection(
            expert_hidden[:, -self.prediction_horizon :, :]
        )
        predictions: dict[str, torch.Tensor] = {}
        offset = 0
        for key in sorted(self.action_heads.keys()):
            dimension = self.action_heads[key].output_dim
            predictions[key] = action_output[:, :, offset : offset + dimension]
            offset += dimension
        return predictions
