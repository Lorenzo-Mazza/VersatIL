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
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.encoders.cross_modal.vision_language.generative_vlm import (
    GenerativeVLMEncoder,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.feature_projection import FeatureProjection
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.base import PositionSource
from versatil.models.layers.positional_encoding.sinusoidal import (
    PeriodInterpolationPositionalEncoding1D,
)
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningCache,
    ConditioningLayerCache,
)
from versatil.models.layers.transformer.layer.precomputed_dual_stream_layer import (
    PrecomputedDualStreamLayer,
)
from versatil.models.layers.transformer.layer.precomputed_kv_layer import (
    PrecomputedKVCrossAttentionLayer,
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
    Modules are created lazily in ``set_backbone`` from the VLM config.
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
        self.vlm_rotary_embedding: nn.Module | None = None
        self.freeze_vlm = freeze_vlm
        self.action_input_projection: nn.Linear | None = None
        self.action_output_projection: nn.Linear | None = None
        self.action_time_fusion_input: nn.Linear | None = None
        self.action_time_fusion_output: nn.Linear | None = None
        self.timestep_embedding: PeriodInterpolationPositionalEncoding1D | None = None
        self.expert_final_norm: nn.Module | None = None
        self.vlm_hidden_dimension: int | None = None
        self.proprioceptive_projection: FeatureProjection | None = None
        self._encoder_cache_enabled = False
        self._prefix_cache: ConditioningCache | None = None

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
        self.vlm_rotary_embedding = rotary_emb
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
                    PrecomputedDualStreamLayer(
                        primary_embedding_dimension=expert_hidden_size,
                        secondary_embedding_dimension=vlm_hidden_dimension,
                        number_of_heads=expert_num_heads,
                        number_of_key_value_heads=expert_num_key_value_heads,
                        head_dimension=expert_head_dimension,
                        primary_feedforward_dimension=expert_intermediate_size,
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
                    PrecomputedKVCrossAttentionLayer(
                        embedding_dimension=expert_hidden_size,
                        conditioning_key_value_dimension=vlm_key_value_dimension,
                        number_of_heads=expert_num_heads,
                        number_of_key_value_heads=expert_num_key_value_heads,
                        head_dimension=expert_head_dimension,
                        feedforward_dimension=expert_intermediate_size,
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
        self._prefix_cache: ConditioningCache | None = None

    def disable_encoder_cache(self) -> None:
        """Disable prefix caching and clear stored states."""
        self._encoder_cache_enabled = False
        self._prefix_cache = None

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
        if (
            self.expert_layers is None
            or self.expert_final_norm is None
            or self.action_output_projection is None
        ):
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
        prefix_length = prefix_embeddings.shape[1]
        prefix_attention_mask = attention_mask[:, :, :prefix_length, :prefix_length]
        vlm_prefix_attention_mask = GenerativeVLMEncoder.build_additive_attention_mask(
            attention_mask=prefix_attention_mask,
            dtype=prefix_embeddings.dtype,
        )
        # Reorder attention mask from [prefix(P), action(A)] to [expert(A), VLM(P)]
        # so _joint_sdpa's primary/secondary slicing gets the correct blocks.
        # key_padding_mask and position_ids stay in [prefix, action] order for RoPE.
        action_length = expert_hidden.shape[1]
        perm = torch.cat(
            [
                torch.arange(
                    prefix_length,
                    prefix_length + action_length,
                    device=expert_hidden.device,
                ),
                torch.arange(prefix_length, device=expert_hidden.device),
            ]
        )
        attention_mask = attention_mask[:, :, perm, :][:, :, :, perm]
        # Expert→VLM cross-attention mask: expert query rows, VLM key columns
        # A = action_length (expert tokens), P = prefix_length (VLM tokens)
        cross_attention_mask = attention_mask[
            :, :, :action_length, action_length:
        ]  # (B, 1, A, P)
        # Non-padded tokens get incrementing positions; padded tokens stay at 0
        pad_mask = ~key_padding_mask.bool()
        position_ids = (pad_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        expert_position_ids = position_ids[:, prefix_length:]
        expert_action_rope = GenerativeVLMEncoder.compute_rope(
            rotary_embedding=self.vlm_rotary_embedding,
            hidden_states=expert_hidden,
            position_ids=expert_position_ids,
        )
        # Cross-attention rotates expert queries in a frame independent of the
        # prefix length: positions are shifted to start from 0 so the relative
        # distance to the VLM keys (also at [0, P)) covers the informative range
        # of RoPE. Matches the reference SmolVLA implementation.
        expert_cross_attn_position_ids = (
            expert_position_ids - expert_position_ids.min(dim=1, keepdim=True).values
        )
        expert_cross_attn_rope = GenerativeVLMEncoder.compute_rope(
            rotary_embedding=self.vlm_rotary_embedding,
            hidden_states=expert_hidden,
            position_ids=expert_cross_attn_position_ids,
        )
        use_cached_prefix = (
            self._encoder_cache_enabled and self._prefix_cache is not None
        )
        if use_cached_prefix:
            expert_hidden = self._run_expert_with_cache(
                expert_hidden=expert_hidden,
                vlm_cache=self._prefix_cache,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                expert_action_rope=expert_action_rope,
                expert_cross_attn_rope=expert_cross_attn_rope,
            )
        elif self._encoder_cache_enabled:
            vlm_cache = self._fill_prefix_cache(
                prefix_embeddings=prefix_embeddings,
                position_ids=position_ids,
                prefix_attention_mask=vlm_prefix_attention_mask,
            )
            self._prefix_cache = vlm_cache
            expert_hidden = self._run_expert_with_cache(
                expert_hidden=expert_hidden,
                vlm_cache=vlm_cache,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                expert_action_rope=expert_action_rope,
                expert_cross_attn_rope=expert_cross_attn_rope,
            )
        else:
            expert_hidden = self._run_training_forward(
                prefix_embeddings=prefix_embeddings,
                expert_hidden=expert_hidden,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                position_ids=position_ids,
                expert_action_rope=expert_action_rope,
                expert_cross_attn_rope=expert_cross_attn_rope,
                vlm_prefix_attention_mask=vlm_prefix_attention_mask,
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
        cross_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        expert_action_rope: tuple[torch.Tensor, torch.Tensor],
        expert_cross_attn_rope: tuple[torch.Tensor, torch.Tensor],
        vlm_prefix_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Interleaved VLM + expert forward for training.

        VLM sees expert tokens in joint attention layers, matching the
        reference where both streams are processed simultaneously.

        Args:
            prefix_embeddings: VLM prefix token embeddings (B, P, D).
            expert_hidden: Expert action token embeddings (B, A, D_expert).
            attention_mask: Joint mask (B, 1, A+P, A+P) for dual-stream layers.
            cross_attention_mask: Expert→VLM mask (B, 1, A, P) for cross-attention layers.
            position_ids: Position IDs (B, P+A).
            expert_action_rope: Pre-computed (cos, sin) for expert RoPE in the
                joint position frame (positions [P, P+A)).
            expert_cross_attn_rope: Pre-computed (cos, sin) for expert RoPE in
                the shifted cross-attn frame (positions [0, A)).
            vlm_prefix_attention_mask: Optional additive mask for VLM prefix
                self-attention, shaped ``(B, 1, P, P)``.
        """

        vlm_hidden = prefix_embeddings
        vlm_position_embeddings = self.vlm_rotary_embedding(
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
                                self.vlm_rotary_embedding,
                                position_ids,
                            )
                        )
                    expert_hidden, vlm_attention_output = self.expert_layers[
                        expert_layer_index
                    ].forward_with_secondary(
                        hidden_states_primary=expert_hidden,
                        conditioning_cache=ConditioningLayerCache(
                            queries=vlm_query, keys=vlm_key, values=vlm_value
                        ),
                        joint_attention_mask=attention_mask,
                        precomputed_primary_rope=expert_action_rope,
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
                        vlm_keys, vlm_values = (
                            GenerativeVLMEncoder.extract_key_value_with_rope(
                                vlm_layer=vlm_layer,
                                hidden_states=vlm_hidden,
                                rotary_embedding=self.vlm_rotary_embedding,
                                position_ids=position_ids,
                            )
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
                        hidden_states=expert_hidden,
                        conditioning_cache=ConditioningLayerCache(
                            keys=vlm_keys, values=vlm_values
                        ),
                        attention_mask=cross_attention_mask,
                        precomputed_rope=expert_cross_attn_rope,
                    )
                    vlm_layer_index += 1
                    expert_layer_index += 1
        return expert_hidden

    def _fill_prefix_cache(
        self,
        prefix_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor | None = None,
    ) -> ConditioningCache:
        """Run VLM layers as plain self-attention and cache K/V for inference.

        During cached inference, the VLM doesn't see expert tokens (matching
        the reference fill_kv_cache=True path).

        Args:
            prefix_embeddings: Prefix token embeddings (B, P, D).
            position_ids: Full position IDs (B, P + A). Only the prefix
                portion [:, :P] is used.
            prefix_attention_mask: Optional additive mask for VLM prefix
                self-attention, shaped ``(B, 1, P, P)``.

        Returns:
            ConditioningCache with one entry per expert layer (VLM_ONLY layers
            are skipped since they have no expert counterpart).
        """
        layer_caches: list[ConditioningLayerCache] = []
        vlm_hidden = prefix_embeddings
        prefix_position_ids = position_ids[:, : prefix_embeddings.shape[1]]
        vlm_position_embeddings = self.vlm_rotary_embedding(
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
                                self.vlm_rotary_embedding,
                                prefix_position_ids,
                            )
                        )
                        layer_caches.append(
                            ConditioningLayerCache(
                                queries=vlm_query, keys=vlm_key, values=vlm_value
                            )
                        )
                    case SmolVLALayerType.CROSS_ATTENTION.value:
                        vlm_keys, vlm_values = (
                            GenerativeVLMEncoder.extract_key_value_with_rope(
                                vlm_layer=vlm_layer,
                                hidden_states=vlm_hidden,
                                rotary_embedding=self.vlm_rotary_embedding,
                                position_ids=prefix_position_ids,
                            )
                        )
                        layer_caches.append(
                            ConditioningLayerCache(keys=vlm_keys, values=vlm_values)
                        )
                vlm_output = vlm_layer(
                    vlm_hidden,
                    attention_mask=prefix_attention_mask,
                    position_embeddings=vlm_position_embeddings,
                )
                vlm_hidden = (
                    vlm_output[0] if isinstance(vlm_output, tuple) else vlm_output
                )
        return ConditioningCache(layers=layer_caches)

    def _run_expert_with_cache(
        self,
        expert_hidden: torch.Tensor,
        vlm_cache: ConditioningCache,
        attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        expert_action_rope: tuple[torch.Tensor, torch.Tensor],
        expert_cross_attn_rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run expert layers using cached VLM states (inference only).

        Args:
            expert_hidden: Expert action tokens (B, A, D_expert).
            vlm_cache: Precomputed VLM K/V/Q per layer.
            attention_mask: Joint mask (B, 1, A+P, A+P) for dual-stream layers.
            cross_attention_mask: Expert→VLM mask (B, 1, A, P) for cross-attention layers.
            expert_action_rope: Precomputed (cos, sin) for expert positions in
                the joint frame (used by joint self-attention layers).
            expert_cross_attn_rope: Precomputed (cos, sin) for expert positions
                shifted to start from 0 (used by cross-attention layers).
        """
        for expert_layer_index, expert_layer in enumerate(self.expert_layers):
            is_dual_stream = isinstance(expert_layer, PrecomputedDualStreamLayer)
            expert_hidden = expert_layer(
                hidden_states=expert_hidden,
                conditioning_cache=vlm_cache.layers[expert_layer_index],
                attention_mask=attention_mask
                if is_dual_stream
                else cross_attention_mask,
                precomputed_rope=expert_action_rope
                if is_dual_stream
                else expert_cross_attn_rope,
            )
        return expert_hidden
