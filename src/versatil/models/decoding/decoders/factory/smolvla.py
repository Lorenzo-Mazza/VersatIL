"""SmolVLA decoder with cross-attention and periodic self-attention."""

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
from versatil.models.decoding.decoders.vla_interleaved import (
    VLACrossAttentionLayer,
    VLAJointAttentionLayer,
)
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm import (
    GenerativeVLMEncoder,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.feature_projection import FeatureProjection
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.base import PositionSource
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.positional_encoding.sinusoidal import (
    PeriodInterpolationPositionalEncoding1D,
)


class SmolVLALayerType(enum.StrEnum):
    """Layer routing types in the SmolVLA decoder forward loop."""

    VLM_ONLY = "vlm_only"
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"


class SmolVLADecoder(ActionDecoder):
    """SmolVLA decoder with interleaved VLM and expert processing.

    Alternates between joint self-attention (expert attends alongside
    VLM tokens) and cross-attention (expert attends to VLM key/values)
    layers.
    Modules are created lazily in ``set_backbone`` from the VLM text config.
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
        expert_width_multiplier: float = 0.75,
        num_expert_layers: int = -1,
        num_vlm_layers: int = 16,
        self_attention_every_n_layers: int = 2,
        proprioceptive_feature_key: str | None = None,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        freeze_vlm: bool = True,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        activation: str = ActivationFunction.SWIGLU.value,
        dropout: float = 0.1,
    ):
        """Initialize the SmolVLA decoder.

        Args:
            input_keys: Feature keys from the encoding pipeline.
            action_space: Action space configuration.
            action_heads: Action prediction heads.
            observation_space: Observation space configuration.
            observation_horizon: Number of observation timesteps.
            prediction_horizon: Number of action steps to predict.
            device: Device string.
            expert_width_multiplier: Expert hidden size as fraction of VLM hidden size.
            num_expert_layers: Number of expert layers. ``-1`` uses the same count as VLM.
            num_vlm_layers: Number of VLM layers to use (truncates if fewer than available).
            self_attention_every_n_layers: Period for joint self-attention layers.
                ``0`` disables joint self-attention (all cross-attention).
            proprioceptive_feature_key: Feature key for proprioceptive state from the
                encoding pipeline. When set, the feature is prepended to the VLM
                prefix before interleaved processing. None disables state prepend.
            min_period: Minimum period for sinusoidal timestep embedding.
            max_period: Maximum period for sinusoidal timestep embedding.
            freeze_vlm: Whether to freeze VLM layer parameters (disable gradients).
            normalization_type: Normalization layer type.
            activation: Activation function for expert feedforward layers.
            dropout: Dropout rate.
        """
        decoder_input = DecoderInput(
            keys=input_keys, requires_actions=True, requires_vlm_backbone=True
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
        self.expert_width_multiplier = expert_width_multiplier
        self.num_expert_layers = num_expert_layers
        self.num_vlm_layers = num_vlm_layers
        self.self_attention_every_n_layers = self_attention_every_n_layers
        self.proprioceptive_feature_key = proprioceptive_feature_key
        self.min_period = min_period
        self.max_period = max_period
        self.normalization_type = normalization_type
        self.activation = activation
        self._dropout = dropout
        self.vlm_layers: nn.ModuleList | None = None
        self.expert_layers: nn.ModuleList | None = None
        self._layer_types: list[str] | None = None
        self._expert_to_vlm_index: dict[int, int] | None = None
        self.freeze_vlm = freeze_vlm
        self.action_input_projection: nn.Linear | None = None
        self.action_output_projection: nn.Linear | None = None
        self.action_time_fusion_input: nn.Linear | None = None
        self.action_time_fusion_output: nn.Linear | None = None
        self.timestep_embedding: PeriodInterpolationPositionalEncoding1D | None = None
        self.expert_final_norm: nn.Module | None = None
        self.vlm_hidden_dimension: int | None = None
        self._encoder_cache_enabled = False
        self._prefix_cache: dict[int, dict[str, torch.Tensor]] | None = None

    @staticmethod
    def _get_intermediate_size(
        hidden_dimension: int, feedforward_multiplier: int = 4, multiple_of: int = 256
    ) -> int:
        """Compute feedforward intermediate size rounded to a multiple."""
        intermediate = feedforward_multiplier * int(2 * hidden_dimension / 3)
        return multiple_of * ((intermediate + multiple_of - 1) // multiple_of)

    def set_backbone(
        self,
        vlm_layers: nn.ModuleList,
        rotary_emb: nn.Module,
        vlm_hidden_dimension: int,
        vlm_text_config: PretrainedConfig,
    ) -> None:
        """Create layers and projections from VLM config."""
        self.vlm_hidden_dimension = vlm_hidden_dimension
        self.vlm_rotary_emb = rotary_emb
        actual_vlm_count = (
            len(vlm_layers)
            if self.num_vlm_layers <= 0
            else min(self.num_vlm_layers, len(vlm_layers))
        )
        if self.freeze_vlm:
            for parameter in vlm_layers.parameters():
                parameter.requires_grad = False
        expert_hidden_size = int(vlm_hidden_dimension * self.expert_width_multiplier)
        expert_intermediate_size = self._get_intermediate_size(expert_hidden_size)
        vlm_head_dimension = getattr(
            vlm_text_config,
            "head_dim",
            vlm_hidden_dimension // vlm_text_config.num_attention_heads,
        )
        vlm_num_heads = vlm_text_config.num_attention_heads
        vlm_num_key_value_heads = getattr(
            vlm_text_config, "num_key_value_heads", vlm_num_heads
        )
        vlm_key_value_dimension = vlm_num_key_value_heads * vlm_head_dimension
        expert_num_heads = vlm_num_heads
        expert_num_key_value_heads = vlm_num_key_value_heads
        expert_head_dimension = vlm_head_dimension
        actual_expert_count = (
            self.num_expert_layers if self.num_expert_layers > 0 else actual_vlm_count
        )
        self.action_input_projection = nn.Linear(self.action_dim, expert_hidden_size)
        self.action_output_projection = nn.Linear(expert_hidden_size, self.action_dim)
        self.action_time_fusion_input = nn.Linear(
            expert_hidden_size * 2, expert_hidden_size
        )
        self.action_time_fusion_output = nn.Linear(
            expert_hidden_size, expert_hidden_size
        )
        if self.proprioceptive_feature_key is not None:
            self.proprioceptive_projection = FeatureProjection(
                embedding_dim=vlm_hidden_dimension
            )
        self.timestep_embedding = PeriodInterpolationPositionalEncoding1D(
            embedding_dimension=expert_hidden_size,
            min_period=self.min_period,
            max_period=self.max_period,
            position_source=PositionSource.SCALAR.value,
        )
        self.expert_final_norm = create_normalization_layer(
            normalization_type=self.normalization_type, dimension=expert_hidden_size
        )
        stride = (
            1
            if actual_vlm_count == actual_expert_count
            else actual_vlm_count // actual_expert_count
        )
        layer_has_expert = [False] * actual_vlm_count
        expert_idx = 0
        for vlm_idx in range(actual_vlm_count):
            if expert_idx < actual_expert_count and vlm_idx % stride == 0:
                layer_has_expert[vlm_idx] = True
                expert_idx += 1
        self.vlm_layers = vlm_layers[:actual_vlm_count]
        self.expert_layers = nn.ModuleList()
        self._layer_types = []
        self._expert_to_vlm_index: dict[int, int] = {}
        expert_counter = 0
        for vlm_idx in range(actual_vlm_count):
            if not layer_has_expert[vlm_idx]:
                self._layer_types.append(SmolVLALayerType.VLM_ONLY.value)
            elif (
                self.self_attention_every_n_layers > 0
                and expert_counter % self.self_attention_every_n_layers == 0
            ):
                self._expert_to_vlm_index[len(self.expert_layers)] = vlm_idx
                self.expert_layers.append(
                    VLAJointAttentionLayer(
                        vlm_embedding_dimension=vlm_hidden_dimension,
                        expert_embedding_dimension=expert_hidden_size,
                        number_of_heads=expert_num_heads,
                        number_of_key_value_heads=expert_num_key_value_heads,
                        head_dimension=expert_head_dimension,
                        expert_feedforward_dimension=expert_intermediate_size,
                        normalization_type=self.normalization_type,
                        activation=self.activation,
                        dropout=self._dropout,
                    )
                )
                self._layer_types.append(SmolVLALayerType.SELF_ATTENTION.value)
                expert_counter += 1
            else:
                self._expert_to_vlm_index[len(self.expert_layers)] = vlm_idx
                self.expert_layers.append(
                    VLACrossAttentionLayer(
                        expert_embedding_dimension=expert_hidden_size,
                        vlm_key_value_dimension=vlm_key_value_dimension,
                        expert_number_of_heads=expert_num_heads,
                        expert_number_of_key_value_heads=expert_num_key_value_heads,
                        expert_head_dimension=expert_head_dimension,
                        expert_feedforward_dimension=expert_intermediate_size,
                        normalization_type=self.normalization_type,
                        activation=self.activation,
                        dropout=self._dropout,
                    )
                )
                self._layer_types.append(SmolVLALayerType.CROSS_ATTENTION.value)
                expert_counter += 1
        self.to(self.device)

    def enable_encoder_cache(self) -> None:
        """Enable prefix caching for multi-step denoising inference."""
        self._encoder_cache_enabled = True
        self._prefix_cache: dict[int, dict[str, torch.Tensor]] | None = None

    def disable_encoder_cache(self) -> None:
        """Disable prefix caching and clear stored states."""
        self._encoder_cache_enabled = False
        self._prefix_cache = None

    def _compute_rope(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (cos, sin) RoPE components for given positions.

        Args:
            hidden_states: Tensor whose dtype/device to match.
            position_ids: Position indices (B, S).

        Returns:
            (cos, sin) broadcastable to (B, 1, S, head_dim).
        """
        cos, sin = self.vlm_rotary_emb(hidden_states, position_ids)
        # (B, S, head_dim) → (B, 1, S, head_dim) for head broadcast
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def _compute_vlm_position_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute position embeddings in HF format for VLM layer calls.

        Args:
            hidden_states: Tensor whose dtype/device to match.
            position_ids: Position indices (B, S).

        Returns:
            (cos, sin) each (B, S, head_dim) — raw format for HF layers.
        """
        return self.vlm_rotary_emb(hidden_states, position_ids)

    def _extract_key_value_with_rope(
        self,
        vlm_layer: nn.Module,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract K/V from a VLM layer with RoPE applied to keys.

        Args:
            vlm_layer: Pretrained VLM transformer layer.
            hidden_states: VLM hidden states (B, P, D_vlm).
            position_ids: Position indices (B, total_length).

        Returns:
            (keys, values) each (B, P, key_value_dimension) with RoPE on keys.
        """
        normalized = vlm_layer.input_layernorm(hidden_states)
        attention = vlm_layer.self_attn
        batch_size, sequence_length, _ = normalized.shape
        head_dimension = attention.head_dim
        number_of_key_value_heads = attention.config.num_key_value_heads

        keys_flat = attention.k_proj(normalized)
        values_flat = attention.v_proj(normalized)

        keys_headed = keys_flat.view(
            batch_size, sequence_length, number_of_key_value_heads, head_dimension
        ).transpose(1, 2)
        cos, sin = self.vlm_rotary_emb(keys_headed, position_ids[:, :sequence_length])
        # (B, S, head_dim) → (B, 1, S, head_dim) for head broadcast
        keys_headed = RotaryPositionalEncoding.apply_rotation_half(
            keys_headed, sin.unsqueeze(1), cos.unsqueeze(1)
        )
        keys_with_rope = (
            keys_headed.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, -1)
        )
        return keys_with_rope, values_flat

    def _embed_suffix(
        self, actions: dict[str, torch.Tensor], timestep: torch.Tensor
    ) -> torch.Tensor:
        """Project actions and fuse with timestep via concat-MLP conditioning."""
        action_tensors = [actions[key] for key in sorted(self.action_heads.keys())]
        action_embedding = self.action_input_projection(
            torch.cat(action_tensors, dim=-1)
        )
        time_embedding = (
            self.timestep_embedding(timestep).unsqueeze(1).expand_as(action_embedding)
        )
        fused = torch.cat([action_embedding, time_embedding], dim=-1)  # (B, H, 2D)
        return self.action_time_fusion_output(
            F.silu(self.action_time_fusion_input(fused))
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run VLM prefix layers and expert suffix layers with cross-attention.

        Args:
            features: Encoded observation features from the encoding pipeline.
            actions: Noisy action tensors keyed by action name.

        Returns:
            Predicted action tensors keyed by action name.
        """
        if self.expert_layers is None:
            raise RuntimeError("set_backbone() must be called before forward().")
        if actions is None:
            raise ValueError(
                "SmolVLADecoder requires actions during forward (noisy actions for denoising)."
            )
        feature_key = self.decoder_input.keys[0]
        padding_mask_key = f"{feature_key}_{EncoderOutputKeys.PADDING_MASK.value}"
        prefix_embeddings = features[feature_key]
        prefix_padding_mask = features.get(padding_mask_key)
        if DecoderOutputKey.TIMESTEP.value not in features:
            raise ValueError(
                f"Missing '{DecoderOutputKey.TIMESTEP.value}' in features dict. "
                "The algorithm should inject timesteps into features."
            )
        timestep = features[DecoderOutputKey.TIMESTEP.value]
        proprio = (
            features.get(self.proprioceptive_feature_key)
            if self.proprioceptive_feature_key is not None
            else None
        )
        causal_prefix_suffix_length = 0
        if proprio is not None:
            projected = self.proprioceptive_projection(
                {self.proprioceptive_feature_key: proprio}
            )
            proprio_token = projected[self.proprioceptive_feature_key]
            if proprio_token.ndim == 2:
                proprio_token = proprio_token.unsqueeze(1)  # (B, D) → (B, 1, D)
            prefix_embeddings = torch.cat([prefix_embeddings, proprio_token], dim=1)
            if prefix_padding_mask is not None:
                proprio_valid = torch.zeros(
                    prefix_padding_mask.shape[0],
                    1,
                    dtype=torch.bool,
                    device=prefix_padding_mask.device,
                )
                prefix_padding_mask = torch.cat(
                    [prefix_padding_mask, proprio_valid], dim=1
                )
            causal_prefix_suffix_length = 1
        expert_hidden = self._embed_suffix(actions, timestep)
        attention_mask, key_padding_mask = make_attention_mask(
            action_tokens=expert_hidden,
            feature_tokens=prefix_embeddings,
            feature_token_mask=prefix_padding_mask,
            causal_actions=True,
            causal_prefix_suffix_length=causal_prefix_suffix_length,
        )
        # Non-padded tokens get incrementing positions; padded tokens stay at 0
        pad_mask = ~key_padding_mask.bool()
        position_ids = (pad_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        prefix_length = prefix_embeddings.shape[1]
        expert_position_ids = position_ids[:, prefix_length:]
        expert_action_rope = self._compute_rope(expert_hidden, expert_position_ids)
        use_cached_prefix = (
            self._encoder_cache_enabled and self._prefix_cache is not None
        )
        if use_cached_prefix:
            expert_hidden = self._run_expert_with_cache(
                expert_hidden=expert_hidden,
                vlm_cache=self._prefix_cache,
                attention_mask=attention_mask,
                expert_action_rope=expert_action_rope,
            )
        elif self._encoder_cache_enabled:
            vlm_attention_mask = None
            if prefix_padding_mask is not None and prefix_padding_mask.any():
                vlm_attention_mask = (
                    (~prefix_padding_mask).unsqueeze(1).unsqueeze(1)
                )  # (B, P) → (B, 1, 1, P)
            vlm_cache = self._fill_prefix_cache(
                prefix_embeddings, position_ids, vlm_attention_mask
            )
            self._prefix_cache = vlm_cache
            expert_hidden = self._run_expert_with_cache(
                expert_hidden=expert_hidden,
                vlm_cache=vlm_cache,
                attention_mask=attention_mask,
                expert_action_rope=expert_action_rope,
            )
        else:
            vlm_train_mask = None
            if prefix_padding_mask is not None and prefix_padding_mask.any():
                vlm_train_mask = (
                    (~prefix_padding_mask).unsqueeze(1).unsqueeze(1)
                )  # (B, P) → (B, 1, 1, P)
            expert_hidden = self._run_training_forward(
                prefix_embeddings=prefix_embeddings,
                expert_hidden=expert_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                expert_action_rope=expert_action_rope,
                vlm_prefix_attention_mask=vlm_train_mask,
            )
        expert_hidden = self.expert_final_norm(expert_hidden)
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

    def _run_training_forward(
        self,
        prefix_embeddings: torch.Tensor,
        expert_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        expert_action_rope: tuple[torch.Tensor, torch.Tensor],
        vlm_prefix_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Interleaved VLM + expert forward for training.

        VLM sees expert tokens in joint attention layers, matching the
        reference where both streams are processed simultaneously.

        Args:
            prefix_embeddings: VLM prefix token embeddings (B, P, D).
            expert_hidden: Expert action token embeddings (B, A, D_expert).
            attention_mask: Full attention mask (B, 1, P+A, P+A).
            position_ids: Position IDs (B, P+A).
            expert_action_rope: Pre-computed (cos, sin) for expert RoPE.
            vlm_prefix_attention_mask: Optional HF-style mask for VLM layers
                (B, P) where 1=attend, 0=masked.
        """

        vlm_hidden = prefix_embeddings
        vlm_position_embeddings = self._compute_vlm_position_embeddings(
            prefix_embeddings, position_ids[:, : prefix_embeddings.shape[1]]
        )
        vlm_layer_index = 0
        expert_layer_index = 0
        for layer_type in self._layer_types:
            vlm_layer = self.vlm_layers[vlm_layer_index]
            match layer_type:
                case SmolVLALayerType.VLM_ONLY.value:
                    with torch.no_grad():
                        vlm_output = vlm_layer(
                            vlm_hidden,
                            attention_mask=vlm_prefix_attention_mask,
                            position_embeddings=vlm_position_embeddings,
                        )
                        vlm_hidden = (
                            vlm_output[0]
                            if isinstance(vlm_output, tuple)
                            else vlm_output
                        )
                    vlm_layer_index += 1
                case SmolVLALayerType.SELF_ATTENTION.value:
                    with torch.no_grad():
                        vlm_query, vlm_key, vlm_value = (
                            GenerativeVLMEncoder.extract_query_key_value(
                                vlm_layer,
                                vlm_hidden,
                                self.vlm_rotary_emb,
                                position_ids,
                            )
                        )
                    vlm_attention_output, expert_hidden = self.expert_layers[
                        expert_layer_index
                    ](
                        precomputed_primary=(vlm_query, vlm_key, vlm_value),
                        hidden_states_secondary=expert_hidden,
                        joint_attention_mask=attention_mask,
                        precomputed_secondary_rope=expert_action_rope,
                    )
                    with torch.no_grad():
                        vlm_hidden = GenerativeVLMEncoder.apply_residual_feedforward(
                            vlm_layer,
                            vlm_hidden,
                            vlm_attention_output,
                        )
                    vlm_layer_index += 1
                    expert_layer_index += 1
                case SmolVLALayerType.CROSS_ATTENTION.value:
                    with torch.no_grad():
                        vlm_keys, vlm_values = self._extract_key_value_with_rope(
                            vlm_layer, vlm_hidden, position_ids
                        )
                        vlm_output = vlm_layer(
                            vlm_hidden,
                            attention_mask=vlm_prefix_attention_mask,
                            position_embeddings=vlm_position_embeddings,
                        )
                        vlm_hidden = (
                            vlm_output[0]
                            if isinstance(vlm_output, tuple)
                            else vlm_output
                        )
                    expert_hidden = self.expert_layers[expert_layer_index](
                        expert_hidden_states=expert_hidden,
                        vlm_key_states=vlm_keys,
                        vlm_value_states=vlm_values,
                        precomputed_query_rope=expert_action_rope,
                    )
                    vlm_layer_index += 1
                    expert_layer_index += 1
        return expert_hidden

    def _fill_prefix_cache(
        self,
        prefix_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor | None = None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Run VLM layers as plain self-attention and cache K/V for inference.

        During cached inference, the VLM doesn't see expert tokens (matching
        the reference fill_kv_cache=True path).

        Args:
            prefix_embeddings: Prefix token embeddings (B, P, D).
            position_ids: Full position IDs (B, P + A). Only the prefix
                portion [:, :P] is used.
            prefix_attention_mask: Optional (B, P) mask where 1 means attend
                and 0 means ignore. Passed to each VLM layer so padded tokens
                do not participate in self-attention.
        """
        vlm_cache: dict[int, dict[str, torch.Tensor]] = {}
        vlm_hidden = prefix_embeddings
        prefix_position_ids = position_ids[:, : prefix_embeddings.shape[1]]
        vlm_position_embeddings = self._compute_vlm_position_embeddings(
            prefix_embeddings, prefix_position_ids
        )
        with torch.no_grad():
            for vlm_layer_index, layer_type in enumerate(self._layer_types):
                vlm_layer = self.vlm_layers[vlm_layer_index]
                match layer_type:
                    case SmolVLALayerType.SELF_ATTENTION.value:
                        vlm_query, vlm_key, vlm_value = (
                            GenerativeVLMEncoder.extract_query_key_value(
                                vlm_layer,
                                vlm_hidden,
                                self.vlm_rotary_emb,
                                prefix_position_ids,
                            )
                        )
                        vlm_cache[vlm_layer_index] = {
                            "query": vlm_query,
                            "key": vlm_key,
                            "value": vlm_value,
                            "hidden": vlm_hidden,
                        }
                    case SmolVLALayerType.CROSS_ATTENTION.value:
                        vlm_keys, vlm_values = self._extract_key_value_with_rope(
                            vlm_layer, vlm_hidden, prefix_position_ids
                        )
                        vlm_cache[vlm_layer_index] = {
                            "key": vlm_keys,
                            "value": vlm_values,
                        }
                vlm_output = vlm_layer(
                    vlm_hidden,
                    attention_mask=prefix_attention_mask,
                    position_embeddings=vlm_position_embeddings,
                )
                vlm_hidden = (
                    vlm_output[0] if isinstance(vlm_output, tuple) else vlm_output
                )
        return vlm_cache

    def _run_expert_with_cache(
        self,
        expert_hidden: torch.Tensor,
        vlm_cache: dict[int, dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        expert_action_rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run expert layers using cached VLM states (inference only)."""
        vlm_layer_index = 0
        expert_layer_index = 0
        for layer_type in self._layer_types:
            match layer_type:
                case SmolVLALayerType.VLM_ONLY.value:
                    vlm_layer_index += 1
                case SmolVLALayerType.SELF_ATTENTION.value:
                    cached = vlm_cache[vlm_layer_index]
                    _, expert_hidden = self.expert_layers[expert_layer_index](
                        precomputed_primary=(
                            cached["query"],
                            cached["key"],
                            cached["value"],
                        ),
                        hidden_states_secondary=expert_hidden,
                        joint_attention_mask=attention_mask,
                        precomputed_secondary_rope=expert_action_rope,
                    )
                    vlm_layer_index += 1
                    expert_layer_index += 1
                case SmolVLALayerType.CROSS_ATTENTION.value:
                    cached = vlm_cache[vlm_layer_index]
                    expert_hidden = self.expert_layers[expert_layer_index](
                        expert_hidden_states=expert_hidden,
                        vlm_key_states=cached["key"],
                        vlm_value_states=cached["value"],
                        precomputed_query_rope=expert_action_rope,
                    )
                    vlm_layer_index += 1
                    expert_layer_index += 1
        return expert_hidden
