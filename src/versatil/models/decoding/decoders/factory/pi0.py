"""Pi0/Pi0.5 interleaved VLM-expert decoder with joint attention.

References:
    Pi0: https://github.com/Physical-Intelligence/openpi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_masking import make_attention_mask
from versatil.models.decoding.constants import DecoderOutputKey, TimeConditioning
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
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


class Pi0Decoder(ActionDecoder):
    """Pi0/Pi0.5 decoder with pretrained VLM backbone and learned action expert.

    Each VLM layer is paired 1:1 with an expert layer via joint
    self-attention. Pi0 fuses timestep into action tokens via MLP
    before the expert layers. Pi0.5 modulates each expert layer
    via adaptive normalization.
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
        expert_hidden_size: int,
        expert_intermediate_size: int,
        expert_number_of_heads: int,
        expert_number_of_key_value_heads: int,
        expert_number_of_layers: int,
        expert_head_dimension: int,
        time_conditioning: str = TimeConditioning.CONCAT_MLP.value,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        proprioceptive_feature_key: str | None = None,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        activation: str = ActivationFunction.GEGLU.value,
        dropout: float = 0.0,
    ):
        """Initialize Pi0 decoder.

        Args:
            input_keys: Feature keys from the encoding pipeline.
            action_space: Action space configuration.
            action_heads: Action prediction heads.
            observation_space: Observation space configuration.
            observation_horizon: Number of observation timesteps.
            prediction_horizon: Number of action steps to predict.
            device: Device string.
            expert_hidden_size: Expert network hidden dimension.
            expert_intermediate_size: Expert feedforward intermediate dimension.
            expert_number_of_heads: Number of attention heads in expert layers.
            expert_number_of_key_value_heads: Number of K/V heads in expert layers.
            expert_number_of_layers: Number of expert layers (must match VLM layers).
            expert_head_dimension: Per-head dimension in expert layers.
            time_conditioning: Timestep conditioning mode (use TimeConditioning enum values).
            min_period: Minimum period for sinusoidal timestep embedding.
            max_period: Maximum period for sinusoidal timestep embedding.
            proprioceptive_feature_key: Feature key for proprioceptive state.
                When set, the feature is prepended to the VLM prefix.
            normalization_type: Normalization layer type.
            activation: Activation function for expert feedforward layers.
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
        self.proprioceptive_feature_key = proprioceptive_feature_key
        self.normalization_type = normalization_type
        self.activation = activation
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
            case TimeConditioning.CONCAT_MLP.value:
                self.action_time_fusion_input = nn.Linear(
                    expert_hidden_size * 2, expert_hidden_size
                )
                self.action_time_fusion_output = nn.Linear(
                    expert_hidden_size, expert_hidden_size
                )
                # Deferred to set_backbone — needs vlm_hidden_dimension
                self.proprioceptive_projection: FeatureProjection | None = None
            case TimeConditioning.ADANORM.value:
                self.time_mlp_input = nn.Linear(expert_hidden_size, expert_hidden_size)
                self.time_mlp_output = nn.Linear(expert_hidden_size, expert_hidden_size)
                self.proprioceptive_projection = None
            case _:
                raise ValueError(
                    f"Unknown time_conditioning: {time_conditioning}. "
                    f"Use {[m.value for m in TimeConditioning]}"
                )
        self.vlm_layers: nn.ModuleList | None = None
        self.vlm_rotary_embedding: nn.Module | None = None
        self.vlm_hidden_dimension: int | None = None
        self.expert_layers: nn.ModuleList | None = None
        self.expert_final_normalization: nn.Module | None = None
        self._encoder_cache_enabled = False
        self._prefix_cache: ConditioningCache | None = None
        self.to(self.device)

    def set_backbone(
        self,
        vlm_layers: nn.ModuleList,
        rotary_emb: nn.Module,
        vlm_hidden_dimension: int,
        vlm_text_config: PretrainedConfig,
    ) -> None:
        """Reference pretrained VLM layers and create expert layers.

        Args:
            vlm_layers: VLM transformer layers (referenced directly, not copied).
            rotary_emb: VLM rotary positional encoding module.
            vlm_hidden_dimension: VLM hidden dimension.
            vlm_text_config: VLM text model config.

        Raises:
            ValueError: If VLM and expert layer counts don't match.
        """
        if len(vlm_layers) != self.expert_number_of_layers:
            raise ValueError(
                f"Pi0 requires equal VLM ({len(vlm_layers)}) and expert "
                f"({self.expert_number_of_layers}) layer counts."
            )
        self.vlm_layers = vlm_layers
        self.vlm_rotary_embedding = rotary_emb
        self.vlm_hidden_dimension = vlm_hidden_dimension
        use_conditioning = self.time_conditioning == TimeConditioning.ADANORM.value
        self.expert_layers = nn.ModuleList(
            [
                PrecomputedDualStreamLayer(
                    primary_embedding_dimension=self.expert_hidden_size,
                    secondary_embedding_dimension=vlm_hidden_dimension,
                    number_of_heads=self.expert_number_of_heads,
                    number_of_key_value_heads=self.expert_number_of_key_value_heads,
                    head_dimension=self.expert_head_dimension,
                    primary_feedforward_dimension=self.expert_intermediate_size,
                    normalization_type=self.normalization_type,
                    conditioning_dimension=self.expert_hidden_size
                    if use_conditioning
                    else None,
                    use_gating=use_conditioning,
                    dropout=self._dropout,
                    activation=self.activation,
                )
                for _ in range(self.expert_number_of_layers)
            ]
        )
        self.expert_final_normalization = create_normalization_layer(
            normalization_type=self.normalization_type,
            dimension=self.expert_hidden_size,
        )
        if (
            self.proprioceptive_feature_key is not None
            and self.time_conditioning == TimeConditioning.CONCAT_MLP.value
        ):
            self.proprioceptive_projection = FeatureProjection(
                embedding_dim=vlm_hidden_dimension
            )

    def enable_encoder_cache(self) -> None:
        """Enable prefix caching for multi-step denoising inference."""
        self._encoder_cache_enabled = True
        self._prefix_cache = None

    def disable_encoder_cache(self) -> None:
        """Disable prefix caching and clear stored states."""
        self._encoder_cache_enabled = False
        self._prefix_cache = None

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
        time_embedding = self.timestep_embedding(timestep)
        match self.time_conditioning:
            case TimeConditioning.CONCAT_MLP.value:
                time_expanded = time_embedding.unsqueeze(1).expand_as(expert_hidden)
                fused = torch.cat([expert_hidden, time_expanded], dim=-1)
                return self.action_time_fusion_output(
                    F.silu(self.action_time_fusion_input(fused))
                ), None
            case TimeConditioning.ADANORM.value:
                conditioning = F.silu(
                    self.time_mlp_output(F.silu(self.time_mlp_input(time_embedding)))
                )
                return expert_hidden, conditioning

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
        if self.expert_layers is None or self.expert_final_normalization is None:
            raise RuntimeError("set_backbone() must be called before forward().")
        if actions is None:
            raise ValueError(
                "Pi0Decoder requires actions during forward (noisy actions for denoising)."
            )
        prefix_embeddings = features[self.decoder_input.keys[0]]
        if DecoderOutputKey.TIMESTEP.value not in features:
            raise ValueError(
                f"Missing '{DecoderOutputKey.TIMESTEP.value}' in features dict. "
                "The algorithm should inject timesteps into features."
            )
        timestep = features[DecoderOutputKey.TIMESTEP.value]
        expert_hidden, adaptive_norm_conditioning = self._embed_suffix(
            actions=actions, timestep=timestep
        )
        causal_prefix_suffix_length = 0
        if (
            self.proprioceptive_projection is not None
            and self.time_conditioning == TimeConditioning.CONCAT_MLP.value
        ):
            proprio = (
                features.get(self.proprioceptive_feature_key)
                if self.proprioceptive_feature_key is not None
                else None
            )
            if proprio is not None:
                projected = self.proprioceptive_projection(
                    {self.proprioceptive_feature_key: proprio}
                )
                proprio_token = projected[self.proprioceptive_feature_key]
                if proprio_token.ndim == 2:
                    proprio_token = proprio_token.unsqueeze(1)  # (B, D) → (B, 1, D)
                prefix_embeddings = torch.cat([prefix_embeddings, proprio_token], dim=1)
                causal_prefix_suffix_length = 1
        attention_mask, key_padding_mask = make_attention_mask(
            action_tokens=expert_hidden,
            feature_tokens=prefix_embeddings,
            causal_actions=False,
            causal_prefix_suffix_length=causal_prefix_suffix_length,
        )
        pad_mask = ~key_padding_mask.bool()
        position_ids = (pad_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        prefix_length = prefix_embeddings.shape[1]
        expert_position_ids = position_ids[:, prefix_length:]
        expert_action_rope = GenerativeVLMEncoder.compute_rope(
            rotary_embedding=self.vlm_rotary_embedding,
            hidden_states=expert_hidden,
            position_ids=expert_position_ids,
        )

        use_cached_prefix = (
            self._encoder_cache_enabled and self._prefix_cache is not None
        )
        if use_cached_prefix:
            expert_hidden = self._run_expert_with_cache(
                expert_hidden=expert_hidden,
                vlm_cache=self._prefix_cache,
                attention_mask=attention_mask,
                expert_action_rope=expert_action_rope,
                adaptive_norm_conditioning=adaptive_norm_conditioning,
            )
        elif self._encoder_cache_enabled:
            vlm_cache = self._fill_prefix_cache(
                prefix_embeddings=prefix_embeddings,
                position_ids=position_ids,
            )
            self._prefix_cache = vlm_cache
            expert_hidden = self._run_expert_with_cache(
                expert_hidden=expert_hidden,
                vlm_cache=vlm_cache,
                attention_mask=attention_mask,
                expert_action_rope=expert_action_rope,
                adaptive_norm_conditioning=adaptive_norm_conditioning,
            )
        else:
            expert_hidden = self._run_training_forward(
                prefix_embeddings=prefix_embeddings,
                expert_hidden=expert_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                expert_action_rope=expert_action_rope,
                adaptive_norm_conditioning=adaptive_norm_conditioning,
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

    def _run_training_forward(
        self,
        prefix_embeddings: torch.Tensor,
        expert_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        expert_action_rope: tuple[torch.Tensor, torch.Tensor],
        adaptive_norm_conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
        """Interleaved VLM + expert forward for training."""
        vlm_hidden = prefix_embeddings
        for layer_index in range(self.expert_number_of_layers):
            vlm_layer = self.vlm_layers[layer_index]
            with torch.no_grad():
                vlm_query, vlm_key, vlm_value = (
                    GenerativeVLMEncoder.extract_query_key_value(
                        vlm_layer=vlm_layer,
                        hidden_states=vlm_hidden,
                        rotary_embedding=self.vlm_rotary_embedding,
                        position_ids=position_ids,
                    )
                )
            expert_hidden, vlm_attention_output = self.expert_layers[
                layer_index
            ].forward_with_secondary(
                hidden_states_primary=expert_hidden,
                conditioning_cache=ConditioningLayerCache(
                    queries=vlm_query, keys=vlm_key, values=vlm_value
                ),
                conditioning=adaptive_norm_conditioning,
                joint_attention_mask=attention_mask,
                precomputed_primary_rope=expert_action_rope,
            )
            with torch.no_grad():
                vlm_hidden = GenerativeVLMEncoder.apply_residual_feedforward(
                    vlm_layer=vlm_layer,
                    vlm_residual=vlm_hidden,
                    vlm_attention_output=vlm_attention_output,
                )
        return expert_hidden

    def _fill_prefix_cache(
        self,
        prefix_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> ConditioningCache:
        """Run VLM layers and cache Q/K/V for inference."""
        if self.vlm_rotary_embedding is None:
            raise RuntimeError(
                "VLM rotary embedding not set. set_backbone() must be called."
            )
        layer_caches: list[ConditioningLayerCache] = []
        vlm_hidden = prefix_embeddings
        prefix_position_ids = position_ids[:, : prefix_embeddings.shape[1]]
        vlm_position_embeddings = self.vlm_rotary_embedding(
            vlm_hidden, prefix_position_ids
        )
        with torch.no_grad():
            for layer_index in range(self.expert_number_of_layers):
                vlm_layer = self.vlm_layers[layer_index]
                vlm_query, vlm_key, vlm_value = (
                    GenerativeVLMEncoder.extract_query_key_value(
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
                vlm_output = vlm_layer(
                    vlm_hidden,
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
        adaptive_norm_conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run expert layers using cached VLM Q/K/V (inference only)."""
        for layer_index in range(self.expert_number_of_layers):
            expert_hidden = self.expert_layers[layer_index](
                hidden_states=expert_hidden,
                conditioning_cache=vlm_cache.layers[layer_index],
                conditioning=adaptive_norm_conditioning,
                attention_mask=attention_mask,
                precomputed_rope=expert_action_rope,
            )
        return expert_hidden
