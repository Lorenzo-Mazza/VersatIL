"""Base decoder for interleaved (vision-language-model(VLM), action-expert) architectures, like pi0 and SmolVLA."""

import abc
import enum
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_masking import make_attention_mask
from versatil.models.decoding.constants import (
    ActionHeadLayout,
    DecoderOutputKey,
    TimeConditioning,
)
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.decoding.decoders.vlm import VLMBackboneDecoderMixin
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)
from versatil.models.layers.feature_projection import FeatureProjection
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.base import PositionSource
from versatil.models.layers.positional_encoding.sinusoidal import (
    PeriodInterpolationPositionalEncoding1D,
)
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningCache,
    ConditioningLayerCache,
)


class InterleavedLayerType(enum.StrEnum):
    """Layer routing types for interleaved VLM/action-expert decoders."""

    VLM_ONLY = "vlm_only"
    JOINT_SELF_ATTENTION = "joint_self_attention"
    CROSS_ATTENTION = "cross_attention"


@dataclass(frozen=True)
class InterleavedVLMAttentionState:
    """Masks, positions, and RoPE tensors shared by interleaved VLA decoders.

    Attributes:
        attention_mask: Expert-first attention mask with shape
            ``(batch_size, 1, action_token_count + prefix_token_count,
            action_token_count + prefix_token_count)``. ``True`` marks masked
            positions.
        key_padding_mask: Key padding mask in original prefix-first order with
            shape ``(batch_size, prefix_token_count + action_token_count)``.
        vlm_prefix_attention_mask: Additive (huggingface convention) VLM self-attention mask with shape
            ``(batch_size, 1, prefix_token_count, prefix_token_count)`` or
            ``None`` when no prefix positions are masked.
        position_ids: Prefix-first position ids with shape
            ``(batch_size, prefix_token_count + action_token_count)``.
        expert_position_ids: Action expert position ids with shape
            ``(batch_size, action_token_count)``.
        expert_action_rope: Rotary embeddings for action expert tokens.
        prefix_token_count: Number of prefix tokens.
        action_token_count: Number of action expert tokens.
    """

    attention_mask: torch.Tensor
    key_padding_mask: torch.Tensor
    vlm_prefix_attention_mask: torch.Tensor | None
    position_ids: torch.Tensor
    expert_position_ids: torch.Tensor
    expert_action_rope: tuple[torch.Tensor, torch.Tensor]
    prefix_token_count: int
    action_token_count: int


class BaseInterleavedVLMDecoder(VLMBackboneDecoderMixin, ActionDecoder, abc.ABC):
    """Base class for VLA decoders with interleaved VLM and action expert layers."""

    action_head_layout: ActionHeadLayout = ActionHeadLayout.JOINT

    def __init__(
        self,
        input_keys: list[str],
        action_space: ActionSpace,
        action_heads: dict[str, ActionHead],
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        vlm_backbone: GenerativeVLM,
        requires_actions: bool = True,
    ) -> None:
        """Initialize decoder input wiring for a configured VLM backbone.

        Args:
            input_keys: Encoded feature keys consumed by the action expert,
                excluding the raw observation keys consumed by ``vlm_backbone``.
            action_space: Task action-space metadata.
            action_heads: Exactly one joint action head mapping expert tokens
                to the full continuous action vector.
            observation_space: Task observation-space metadata.
            observation_horizon: Number of observation timesteps in each sample.
            prediction_horizon: Number of action timesteps predicted per sample.
            device: Device used by decoder modules.
            vlm_backbone: VLM backbone that consumes normalized/tokenized
                observations and emits prefix embeddings with shape
                ``(B, P, D_vlm)``.
            requires_actions: Whether the decoder forward pass requires
                ground-truth actions.
        """
        if vlm_backbone is None:
            raise ValueError(f"{type(self).__name__} requires a vlm_backbone.")
        decoder_keys = self._vlm_decoder_input_keys(
            input_keys=input_keys,
            vlm_backbone=vlm_backbone,
        )
        decoder_input = DecoderInput(
            keys=decoder_keys,
            requires_actions=requires_actions,
            needs_raw_observations=True,
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
        self.vlm_backbone: GenerativeVLM = vlm_backbone
        self._encoder_cache_enabled = False
        self._prefix_cache: ConditioningCache | None = None

    def enable_encoder_cache(self) -> None:
        """Enable reusable VLM prefix caching for inference."""
        self._encoder_cache_enabled = True
        self._prefix_cache = None

    def disable_encoder_cache(self) -> None:
        """Disable reusable VLM prefix caching and clear stored cache."""
        self._encoder_cache_enabled = False
        self._prefix_cache = None

    def set_vlm_backbone(self, vlm_backbone: GenerativeVLM) -> None:
        """Attach a VLM backbone and initialize architecture-specific layers.

        Args:
            vlm_backbone: VLM backbone exposing transformer layers, RoPE, text
                config, hidden size, and a prefix builder returning embeddings
                with shape ``(B, P, D_vlm)``.
        """
        self.vlm_backbone = vlm_backbone
        self.build_action_expert(
            vlm_layers=vlm_backbone.layers,
            rotary_emb=vlm_backbone.rotary_embedding,
            vlm_hidden_dimension=vlm_backbone.hidden_dim,
            vlm_text_config=vlm_backbone.text_config,
        )

    @abc.abstractmethod
    def build_action_expert(
        self,
        vlm_layers: torch.nn.ModuleList,
        rotary_emb: torch.nn.Module,
        vlm_hidden_dimension: int,
        vlm_text_config: PretrainedConfig,
    ) -> None:
        """Build decoder-specific action expert layers from VLM internals.

        Note:
            This is called by ``BaseInterleavedVLMDecoder.set_vlm_backbone()``
            after the full VLM backbone is attached.

        Args:
            vlm_layers: Transformer layers copied or referenced by the decoder.
            rotary_emb: Rotary embedding module used by the VLM layers.
            vlm_hidden_dimension: Hidden dimension of VLM prefix tokens.
            vlm_text_config: HuggingFace text config for the VLM language tower.
        """
        raise NotImplementedError

    def _set_action_suffix_modules(
        self,
        expert_hidden_dimension: int,
        time_conditioning: str,
        min_period: float,
        max_period: float,
        normalization_type: str,
    ) -> None:
        """Create shared action suffix, timestep, and output modules.

        Args:
            expert_hidden_dimension: Expert stream hidden dimension.
            time_conditioning: Timestep-conditioning mode from
                ``TimeConditioning``.
            min_period: Minimum period for sinusoidal timestep embedding.
            max_period: Maximum period for sinusoidal timestep embedding.
            normalization_type: Final expert normalization type.

        Side Effects:
            Sets ``action_input_projection``, ``timestep_embedding``,
            ``time_conditioning_input``, ``time_conditioning_output``, and
            ``expert_final_normalization``.
        """
        self.action_input_projection = nn.Linear(
            self.action_dim,
            expert_hidden_dimension,
        )
        self._validate_joint_action_head_input_dimension(
            expected_input_dimension=expert_hidden_dimension
        )
        self.timestep_embedding = PeriodInterpolationPositionalEncoding1D(
            embedding_dimension=expert_hidden_dimension,
            min_period=min_period,
            max_period=max_period,
            position_source=PositionSource.SCALAR.value,
        )
        match time_conditioning:
            case TimeConditioning.CONCAT_MLP.value:
                self.time_conditioning_input = nn.Linear(
                    expert_hidden_dimension * 2,
                    expert_hidden_dimension,
                )
            case TimeConditioning.ADANORM.value:
                self.time_conditioning_input = nn.Linear(
                    expert_hidden_dimension,
                    expert_hidden_dimension,
                )
            case _:
                raise ValueError(
                    f"Unknown time_conditioning: {time_conditioning}. "
                    f"Use {[mode.value for mode in TimeConditioning]}"
                )
        self.time_conditioning_output = nn.Linear(
            expert_hidden_dimension,
            expert_hidden_dimension,
        )
        self.expert_final_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=expert_hidden_dimension,
        )

    def _validate_joint_action_head_input_dimension(
        self,
        expected_input_dimension: int,
    ) -> None:
        """Validate that the joint action head consumes expert hidden states."""
        joint_action_head = self._single_action_head()
        if joint_action_head.input_dim == expected_input_dimension:
            return
        raise ValueError(
            f"{type(self).__name__} joint action head input_dim must equal "
            f"expert hidden dimension {expected_input_dimension}, got "
            f"{joint_action_head.input_dim}."
        )

    @staticmethod
    def _get_vlm_head_dimension(
        vlm_text_config: PretrainedConfig,
        vlm_hidden_dimension: int,
    ) -> int:
        """Return the VLM attention head dimension.

        Args:
            vlm_text_config: HuggingFace text config for the VLM language tower.
            vlm_hidden_dimension: VLM hidden dimension.

        Returns:
            Attention head dimension.
        """
        return int(
            vlm_text_config.head_dim
            if hasattr(vlm_text_config, "head_dim")
            else vlm_hidden_dimension // vlm_text_config.num_attention_heads
        )

    @staticmethod
    def _get_vlm_num_key_value_heads(vlm_text_config: PretrainedConfig) -> int:
        """Return the VLM key/value head count.

        Args:
            vlm_text_config: HuggingFace text config for the VLM language tower.

        Returns:
            Number of key/value heads, falling back to attention heads for MHA
            configs without grouped-query attention.
        """
        return int(
            vlm_text_config.num_key_value_heads
            if hasattr(vlm_text_config, "num_key_value_heads")
            else vlm_text_config.num_attention_heads
        )

    def _build_prefix(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return VLM prefix embeddings from the configured backbone.

        Args:
            features: Normalized/tokenized observation tensors keyed by the VLM
                input specification. Image tensors keep their configured camera
                shape, normally ``(B, T, C, H, W)`` or ``(B, C, H, W)``.
                Tokenized text has shape ``(B, T, S)`` or ``(B, S)`` depending
                on the dataloader path.

        Returns:
            Prefix embeddings with shape ``(B, P, D_vlm)`` and optional padding
            mask with shape ``(B, P)`` where ``True`` marks padding.
        """
        return self._build_vlm_prefix(features=features)

    @staticmethod
    def _append_valid_prefix_tokens(
        prefix_embeddings: torch.Tensor,
        prefix_padding_mask: torch.Tensor | None,
        prefix_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Append valid tokens to a VLM prefix.

        Args:
            prefix_embeddings: Prefix embeddings with shape
                ``(batch_size, prefix_token_count, vlm_hidden_dimension)``.
            prefix_padding_mask: Optional prefix padding mask with shape
                ``(batch_size, prefix_token_count)`` where ``True`` marks
                padding.
            prefix_tokens: Tokens to append with shape
                ``(batch_size, appended_token_count, vlm_hidden_dimension)`` or
                ``(batch_size, vlm_hidden_dimension)``.

        Returns:
            Updated prefix embeddings, updated padding mask, and number of
            appended tokens.
        """
        if prefix_tokens.ndim == 2:
            prefix_tokens = prefix_tokens.unsqueeze(1)  # (B, D) -> (B, 1, D)
        updated_prefix_embeddings = torch.cat(
            [prefix_embeddings, prefix_tokens],
            dim=1,
        )
        appended_token_count = prefix_tokens.shape[1]
        if prefix_padding_mask is None:
            return updated_prefix_embeddings, None, appended_token_count

        appended_padding_mask = torch.zeros(
            prefix_padding_mask.shape[0],
            appended_token_count,
            dtype=torch.bool,
            device=prefix_padding_mask.device,
        )
        updated_prefix_padding_mask = torch.cat(
            [prefix_padding_mask, appended_padding_mask],
            dim=1,
        )
        return (
            updated_prefix_embeddings,
            updated_prefix_padding_mask,
            appended_token_count,
        )

    def _append_projected_prefix_feature(
        self,
        prefix_embeddings: torch.Tensor,
        prefix_padding_mask: torch.Tensor | None,
        features: dict[str, torch.Tensor],
        feature_key: str,
        projection: FeatureProjection,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Project one encoded feature and append it to the VLM prefix.

        Args:
            prefix_embeddings: Prefix embeddings with shape
                ``(batch_size, prefix_token_count, vlm_hidden_dimension)``.
            prefix_padding_mask: Optional prefix padding mask with shape
                ``(batch_size, prefix_token_count)`` where ``True`` marks
                padding.
            features: Feature dictionary containing ``feature_key``.
            feature_key: Encoded feature name to project.
            projection: Projection module that maps ``feature_key`` to the VLM
                hidden dimension.

        Returns:
            Updated prefix embeddings, updated padding mask, and number of
            appended tokens.
        """
        if feature_key not in features:
            raise ValueError(
                f"Missing '{feature_key}' in features for projected VLM prefix token."
            )
        projected_features = projection(features={feature_key: features[feature_key]})
        return self._append_valid_prefix_tokens(
            prefix_embeddings=prefix_embeddings,
            prefix_padding_mask=prefix_padding_mask,
            prefix_tokens=projected_features[feature_key],
        )

    def _append_optional_projected_prefix_feature(
        self,
        prefix_embeddings: torch.Tensor,
        prefix_padding_mask: torch.Tensor | None,
        features: dict[str, torch.Tensor],
        feature_key: str | None,
        projection: FeatureProjection | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Append an encoded feature to the VLM prefix when configured.

        Args:
            prefix_embeddings: Prefix embeddings with shape
                ``(batch_size, prefix_token_count, vlm_hidden_dimension)``.
            prefix_padding_mask: Optional prefix padding mask with shape
                ``(batch_size, prefix_token_count)`` where ``True`` marks
                padding.
            features: Feature dictionary containing ``feature_key`` when the
                optional projection is configured.
            feature_key: Optional encoded feature name to append.
            projection: Optional projection module for ``feature_key``.

        Returns:
            Updated prefix embeddings, updated padding mask, and number of
            appended tokens. The token count is zero when no feature is
            configured.
        """
        if feature_key is None or projection is None:
            return prefix_embeddings, prefix_padding_mask, 0
        return self._append_projected_prefix_feature(
            prefix_embeddings=prefix_embeddings,
            prefix_padding_mask=prefix_padding_mask,
            features=features,
            feature_key=feature_key,
            projection=projection,
        )

    @staticmethod
    def _build_interleaved_attention_state(
        prefix_embeddings: torch.Tensor,
        prefix_padding_mask: torch.Tensor | None,
        expert_tokens: torch.Tensor,
        rotary_embedding: torch.nn.Module,
        causal_actions: bool,
        causal_prefix_suffix_length: int = 0,
    ) -> InterleavedVLMAttentionState:
        """Build shared masks, position ids, and action RoPE for expert layers.

        Args:
            prefix_embeddings: VLM prefix embeddings with shape
                ``(batch_size, prefix_token_count, vlm_hidden_dimension)``.
            prefix_padding_mask: Optional prefix padding mask with shape
                ``(batch_size, prefix_token_count)`` where ``True`` marks
                padding.
            expert_tokens: Action expert tokens with shape
                ``(batch_size, action_token_count, expert_hidden_dimension)``.
            rotary_embedding: Rotary embedding module from the VLM backbone.
            causal_actions: Whether action expert tokens use causal attention.
            causal_prefix_suffix_length: Number of trailing prefix tokens that
                are masked causally relative to the earlier prefix.

        Returns:
            Shared interleaved attention state. The returned ``attention_mask``
            is reordered from prefix-first order to expert-first order for
            dual-stream expert layers.
        """
        attention_mask, key_padding_mask = make_attention_mask(
            action_tokens=expert_tokens,
            feature_tokens=prefix_embeddings,
            feature_token_mask=prefix_padding_mask,
            causal_actions=causal_actions,
            causal_prefix_suffix_length=causal_prefix_suffix_length,
        )
        prefix_token_count = prefix_embeddings.shape[1]
        action_token_count = expert_tokens.shape[1]
        prefix_attention_mask = attention_mask[
            :, :, :prefix_token_count, :prefix_token_count
        ]
        vlm_prefix_attention_mask = GenerativeVLM.build_additive_attention_mask(
            attention_mask=prefix_attention_mask,
            dtype=prefix_embeddings.dtype,
        )
        permutation = torch.cat(
            [
                torch.arange(
                    prefix_token_count,
                    prefix_token_count + action_token_count,
                    device=expert_tokens.device,
                ),
                torch.arange(prefix_token_count, device=expert_tokens.device),
            ]
        )
        expert_first_attention_mask = attention_mask[:, :, permutation, :][
            :, :, :, permutation
        ]
        valid_token_mask = ~key_padding_mask.bool()
        position_ids = (valid_token_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        expert_position_ids = position_ids[:, prefix_token_count:]
        expert_action_rope = GenerativeVLM.compute_rope(
            rotary_embedding=rotary_embedding,
            hidden_states=expert_tokens,
            position_ids=expert_position_ids,
        )
        return InterleavedVLMAttentionState(
            attention_mask=expert_first_attention_mask,
            key_padding_mask=key_padding_mask,
            vlm_prefix_attention_mask=vlm_prefix_attention_mask,
            position_ids=position_ids,
            expert_position_ids=expert_position_ids,
            expert_action_rope=expert_action_rope,
            prefix_token_count=prefix_token_count,
            action_token_count=action_token_count,
        )

    @staticmethod
    def _slice_expert_to_vlm_attention_mask(
        attention_mask: torch.Tensor,
        action_token_count: int,
    ) -> torch.Tensor:
        """Slice the expert-query to VLM-key block from an expert-first mask.

        Args:
            attention_mask: Expert-first attention mask with shape
                ``(batch_size, 1, action_token_count + prefix_token_count,
                action_token_count + prefix_token_count)``.
            action_token_count: Number of action expert tokens.

        Returns:
            Cross-attention mask with shape
            ``(batch_size, 1, action_token_count, prefix_token_count)``.
        """
        return attention_mask[:, :, :action_token_count, action_token_count:]

    @staticmethod
    def _zero_based_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
        """Shift per-sample position ids so each row starts at zero.

        Args:
            position_ids: Position ids with shape ``(batch_size, token_count)``.

        Returns:
            Shifted position ids with shape ``(batch_size, token_count)``.
        """
        return position_ids - position_ids.min(dim=1, keepdim=True).values

    def _require_forward_actions(
        self,
        actions: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        """Return denoising action inputs or raise a decoder-specific error.

        Args:
            actions: Noisy action tensors keyed by action name, or ``None``.

        Returns:
            Action tensors keyed by action name.
        """
        if actions is None:
            raise ValueError(
                f"{type(self).__name__} requires actions during forward "
                "(noisy actions for denoising)."
            )
        return actions

    def _get_forward_timestep(
        self,
        features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Read the timestep injected by the denoising or flow algorithm.

        Args:
            features: Feature dictionary expected to contain
                ``DecoderOutputKey.TIMESTEP.value`` with shape ``(batch_size,)``.

        Returns:
            Timestep tensor with shape ``(batch_size,)``.
        """
        if DecoderOutputKey.TIMESTEP.value not in features:
            raise ValueError(
                f"Missing '{DecoderOutputKey.TIMESTEP.value}' in features dict. "
                "The algorithm should inject timesteps into features."
            )
        return features[DecoderOutputKey.TIMESTEP.value]

    def _project_expert_actions(
        self,
        expert_hidden: torch.Tensor,
        expert_final_normalization: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Project expert tokens and split them by configured action head.

        Args:
            expert_hidden: Action expert tokens with shape
                ``(batch_size, action_token_count, expert_hidden_dimension)``.
            expert_final_normalization: Final expert normalization layer.

        Returns:
            Predicted action tensors keyed by action name. Each value has shape
            ``(batch_size, prediction_horizon, action_dimension)``.
        """
        expert_hidden = expert_final_normalization(expert_hidden)
        action_output = self._single_action_head()(
            expert_hidden[:, -self.prediction_horizon :, :]
        )
        return self.action_space.split_action_tensor(
            action_tensor=action_output,
            owner_name=type(self).__name__,
        )

    def _embed_timestep_conditioned_action_suffix(
        self,
        actions: dict[str, torch.Tensor],
        timestep: torch.Tensor,
        action_input_projection: torch.nn.Module,
        timestep_embedding: torch.nn.Module,
        time_conditioning: str,
        conditioning_input_layer: torch.nn.Module,
        conditioning_output_layer: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Embed action tokens and apply timestep conditioning.

        Args:
            actions: Action tensors keyed by action name. Each tensor has shape
                ``(batch_size, action_token_count, action_dimension)``.
            timestep: Denoising or flow timestep tensor with shape
                ``(batch_size,)``.
            action_input_projection: Module that maps concatenated action
                tensors to expert tokens with shape
                ``(batch_size, action_token_count, expert_hidden_dimension)``.
            timestep_embedding: Module that maps ``timestep`` to shape
                ``(batch_size, expert_hidden_dimension)``.
            time_conditioning: Time-conditioning mode from
                ``TimeConditioning``.
            conditioning_input_layer: First layer of the two-layer conditioning
                MLP. In concat-MLP mode it consumes concatenated action and
                time embeddings. In AdaNorm mode it consumes the time
                embedding.
            conditioning_output_layer: Second layer of the two-layer
                conditioning MLP.

        Returns:
            Expert action tokens with shape
            ``(batch_size, action_token_count, expert_hidden_dimension)`` and
            optional AdaNorm conditioning with shape
            ``(batch_size, expert_hidden_dimension)``.
        """
        concatenated_actions = self.action_space.concatenate_action_tensors(
            actions=actions,
            prediction_horizon=self.prediction_horizon,
            owner_name=type(self).__name__,
        )
        action_embedding = action_input_projection(concatenated_actions)
        time_embedding = timestep_embedding(timestep)
        match time_conditioning:
            case TimeConditioning.CONCAT_MLP.value:
                time_expanded = time_embedding.unsqueeze(1).expand_as(action_embedding)
                fused_embedding = torch.cat(
                    [action_embedding, time_expanded],
                    dim=-1,
                )
                suffix_embedding = conditioning_output_layer(
                    F.silu(conditioning_input_layer(fused_embedding))
                )
                return suffix_embedding, None
            case TimeConditioning.ADANORM.value:
                adaptive_norm_conditioning = F.silu(
                    conditioning_output_layer(
                        F.silu(conditioning_input_layer(time_embedding))
                    )
                )
                return action_embedding, adaptive_norm_conditioning
            case _:
                raise ValueError(
                    f"Unknown time_conditioning: {time_conditioning}. "
                    f"Use {[mode.value for mode in TimeConditioning]}"
                )

    def _run_interleaved_layers(
        self,
        prefix_embeddings: torch.Tensor,
        expert_hidden: torch.Tensor,
        attention_state: InterleavedVLMAttentionState,
        adaptive_norm_conditioning: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        expert_cross_attention_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run training or cached inference through routed interleaved layers.

        Args:
            prefix_embeddings: VLM prefix token embeddings with shape
                ``(batch_size, prefix_token_count, vlm_hidden_dimension)``.
            expert_hidden: Action expert tokens with shape
                ``(batch_size, action_token_count, expert_hidden_dimension)``.
            attention_state: Shared masks, position ids, and joint-action RoPE
                for the current prefix/action sequence.
            adaptive_norm_conditioning: Optional AdaNorm conditioning with
                shape ``(batch_size, expert_hidden_dimension)``.
            cross_attention_mask: Optional expert-query to VLM-key mask with
                shape ``(batch_size, 1, action_token_count, prefix_token_count)``.
            expert_cross_attention_rope: Optional rotary embeddings for cross-
                attention action queries.

        Returns:
            Updated action expert tokens with shape
            ``(batch_size, action_token_count, expert_hidden_dimension)``.
        """
        if self._encoder_cache_enabled and self._prefix_cache is not None:
            return self._run_expert_with_cache(
                expert_hidden=expert_hidden,
                vlm_cache=self._prefix_cache,
                attention_mask=attention_state.attention_mask,
                expert_action_rope=attention_state.expert_action_rope,
                adaptive_norm_conditioning=adaptive_norm_conditioning,
                cross_attention_mask=cross_attention_mask,
                expert_cross_attention_rope=expert_cross_attention_rope,
            )
        if self._encoder_cache_enabled:
            vlm_cache = self._fill_prefix_cache(
                prefix_embeddings=prefix_embeddings,
                position_ids=attention_state.position_ids,
                prefix_attention_mask=attention_state.vlm_prefix_attention_mask,
            )
            self._prefix_cache = vlm_cache
            return self._run_expert_with_cache(
                expert_hidden=expert_hidden,
                vlm_cache=vlm_cache,
                attention_mask=attention_state.attention_mask,
                expert_action_rope=attention_state.expert_action_rope,
                adaptive_norm_conditioning=adaptive_norm_conditioning,
                cross_attention_mask=cross_attention_mask,
                expert_cross_attention_rope=expert_cross_attention_rope,
            )
        return self._run_training_forward(
            prefix_embeddings=prefix_embeddings,
            expert_hidden=expert_hidden,
            attention_mask=attention_state.attention_mask,
            position_ids=attention_state.position_ids,
            expert_action_rope=attention_state.expert_action_rope,
            adaptive_norm_conditioning=adaptive_norm_conditioning,
            cross_attention_mask=cross_attention_mask,
            expert_cross_attention_rope=expert_cross_attention_rope,
            vlm_prefix_attention_mask=attention_state.vlm_prefix_attention_mask,
        )

    def _run_training_forward(
        self,
        prefix_embeddings: torch.Tensor,
        expert_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        expert_action_rope: tuple[torch.Tensor, torch.Tensor],
        adaptive_norm_conditioning: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        expert_cross_attention_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        vlm_prefix_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run routed VLM and action-expert layers during training.

        Args:
            prefix_embeddings: VLM prefix token embeddings with shape
                ``(batch_size, prefix_token_count, vlm_hidden_dimension)``.
            expert_hidden: Action expert tokens with shape
                ``(batch_size, action_token_count, expert_hidden_dimension)``.
            attention_mask: Expert-first joint attention mask with shape
                ``(batch_size, 1, action_token_count + prefix_token_count,
                action_token_count + prefix_token_count)``.
            position_ids: Prefix-first position ids with shape
                ``(batch_size, prefix_token_count + action_token_count)``.
            expert_action_rope: Rotary embeddings for joint self-attention
                action tokens.
            adaptive_norm_conditioning: Optional AdaNorm conditioning with
                shape ``(batch_size, expert_hidden_dimension)``.
            cross_attention_mask: Optional expert-query to VLM-key mask with
                shape ``(batch_size, 1, action_token_count, prefix_token_count)``.
            expert_cross_attention_rope: Optional rotary embeddings for cross-
                attention action queries.
            vlm_prefix_attention_mask: Optional additive VLM prefix
                self-attention mask with shape
                ``(batch_size, 1, prefix_token_count, prefix_token_count)``.

        Returns:
            Updated action expert tokens with shape
            ``(batch_size, action_token_count, expert_hidden_dimension)``.
        """
        vlm_hidden = prefix_embeddings
        vlm_position_embeddings = self.vlm_rotary_embedding(
            prefix_embeddings, position_ids[:, : prefix_embeddings.shape[1]]
        )
        # Detach the VLM stream only when no VLM layer parameter trains
        # (frozen backbone without LoRA). A trainable stream also lets
        # gradients reach the vision tower and projector through the prefix.
        vlm_gradients_enabled = torch.is_grad_enabled() and any(
            parameter.requires_grad for parameter in self.vlm_layers.parameters()
        )
        vlm_layer_index = 0
        expert_layer_index = 0
        for layer_type in self._layer_types:
            vlm_layer = self.vlm_layers[vlm_layer_index]
            match layer_type:
                case InterleavedLayerType.VLM_ONLY.value:
                    with torch.set_grad_enabled(vlm_gradients_enabled):
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
                case InterleavedLayerType.JOINT_SELF_ATTENTION.value:
                    with torch.set_grad_enabled(vlm_gradients_enabled):
                        vlm_query, vlm_key, vlm_value = (
                            GenerativeVLM.extract_query_key_value(
                                vlm_layer=vlm_layer,
                                hidden_states=vlm_hidden,
                                rotary_embedding=self.vlm_rotary_embedding,
                                position_ids=position_ids,
                            )
                        )
                    expert_hidden, vlm_attention_output = self.expert_layers[
                        expert_layer_index
                    ].forward_with_secondary(
                        hidden_states_primary=expert_hidden,
                        conditioning_cache=ConditioningLayerCache(
                            queries=vlm_query, keys=vlm_key, values=vlm_value
                        ),
                        conditioning=adaptive_norm_conditioning,
                        joint_attention_mask=attention_mask,
                        precomputed_primary_rope=expert_action_rope,
                    )
                    with torch.set_grad_enabled(vlm_gradients_enabled):
                        vlm_hidden = GenerativeVLM.apply_residual_feedforward(
                            vlm_layer=vlm_layer,
                            vlm_residual=vlm_hidden,
                            vlm_attention_output=vlm_attention_output,
                        )
                    vlm_layer_index += 1
                    expert_layer_index += 1
                case InterleavedLayerType.CROSS_ATTENTION.value:
                    if (
                        cross_attention_mask is None
                        or expert_cross_attention_rope is None
                    ):
                        raise RuntimeError(
                            "Cross-attention interleaved layers require "
                            "cross_attention_mask and expert_cross_attention_rope."
                        )
                    with torch.set_grad_enabled(vlm_gradients_enabled):
                        vlm_keys, vlm_values = (
                            GenerativeVLM.extract_key_value_with_rope(
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
                        precomputed_rope=expert_cross_attention_rope,
                    )
                    vlm_layer_index += 1
                    expert_layer_index += 1
                case _:
                    raise ValueError(f"Unknown interleaved layer type: {layer_type}.")
        return expert_hidden

    def _fill_prefix_cache(
        self,
        prefix_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor | None = None,
    ) -> ConditioningCache:
        """Run VLM prefix layers and cache states for expert inference.

        Args:
            prefix_embeddings: VLM prefix token embeddings with shape
                ``(batch_size, prefix_token_count, vlm_hidden_dimension)``.
            position_ids: Prefix-first position ids with shape
                ``(batch_size, prefix_token_count + action_token_count)``.
            prefix_attention_mask: Optional additive VLM prefix self-attention
                mask with shape
                ``(batch_size, 1, prefix_token_count, prefix_token_count)``.

        Returns:
            Cache with one entry per expert layer. VLM-only layers update the
            prefix hidden state but do not add cache entries.
        """
        if self.vlm_rotary_embedding is None:
            raise RuntimeError(
                "VLM rotary embedding not set. build_action_expert() must be called."
            )
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
                    case InterleavedLayerType.VLM_ONLY.value:
                        pass
                    case InterleavedLayerType.JOINT_SELF_ATTENTION.value:
                        vlm_query, vlm_key, vlm_value = (
                            GenerativeVLM.extract_query_key_value(
                                vlm_layer=vlm_layer,
                                hidden_states=vlm_hidden,
                                rotary_embedding=self.vlm_rotary_embedding,
                                position_ids=prefix_position_ids,
                            )
                        )
                        layer_caches.append(
                            ConditioningLayerCache(
                                queries=vlm_query, keys=vlm_key, values=vlm_value
                            )
                        )
                    case InterleavedLayerType.CROSS_ATTENTION.value:
                        vlm_keys, vlm_values = (
                            GenerativeVLM.extract_key_value_with_rope(
                                vlm_layer=vlm_layer,
                                hidden_states=vlm_hidden,
                                rotary_embedding=self.vlm_rotary_embedding,
                                position_ids=prefix_position_ids,
                            )
                        )
                        layer_caches.append(
                            ConditioningLayerCache(keys=vlm_keys, values=vlm_values)
                        )
                    case _:
                        raise ValueError(
                            f"Unknown interleaved layer type: {layer_type}."
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
        expert_action_rope: tuple[torch.Tensor, torch.Tensor],
        adaptive_norm_conditioning: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        expert_cross_attention_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run action expert layers using cached VLM states.

        Args:
            expert_hidden: Action expert tokens with shape
                ``(batch_size, action_token_count, expert_hidden_dimension)``.
            vlm_cache: Cached VLM query/key/value states, one entry per expert
                layer.
            attention_mask: Expert-first joint attention mask with shape
                ``(batch_size, 1, action_token_count + prefix_token_count,
                action_token_count + prefix_token_count)``.
            expert_action_rope: Rotary embeddings for joint self-attention
                action tokens.
            adaptive_norm_conditioning: Optional AdaNorm conditioning with
                shape ``(batch_size, expert_hidden_dimension)``.
            cross_attention_mask: Optional expert-query to VLM-key mask with
                shape ``(batch_size, 1, action_token_count, prefix_token_count)``.
            expert_cross_attention_rope: Optional rotary embeddings for cross-
                attention action queries.

        Returns:
            Updated action expert tokens with shape
            ``(batch_size, action_token_count, expert_hidden_dimension)``.
        """
        expert_layer_index = 0
        for layer_type in self._layer_types:
            if layer_type == InterleavedLayerType.VLM_ONLY.value:
                continue
            expert_layer = self.expert_layers[expert_layer_index]
            match layer_type:
                case InterleavedLayerType.JOINT_SELF_ATTENTION.value:
                    expert_hidden = expert_layer(
                        hidden_states=expert_hidden,
                        conditioning_cache=vlm_cache.layers[expert_layer_index],
                        conditioning=adaptive_norm_conditioning,
                        attention_mask=attention_mask,
                        precomputed_rope=expert_action_rope,
                    )
                case InterleavedLayerType.CROSS_ATTENTION.value:
                    if (
                        cross_attention_mask is None
                        or expert_cross_attention_rope is None
                    ):
                        raise RuntimeError(
                            "Cross-attention interleaved layers require "
                            "cross_attention_mask and expert_cross_attention_rope."
                        )
                    expert_hidden = expert_layer(
                        hidden_states=expert_hidden,
                        conditioning_cache=vlm_cache.layers[expert_layer_index],
                        attention_mask=cross_attention_mask,
                        precomputed_rope=expert_cross_attention_rope,
                    )
                case _:
                    raise ValueError(f"Unknown interleaved layer type: {layer_type}.")
            expert_layer_index += 1
        return expert_hidden
