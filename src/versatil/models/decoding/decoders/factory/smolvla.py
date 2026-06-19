"""SmolVLA decoder with interleaved cross-attention and self-attention."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.constants import TimeConditioning
from versatil.models.decoding.decoders.interleaved_vlm import (
    BaseInterleavedVLMDecoder,
    InterleavedLayerType,
)
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.feature_projection import FeatureProjection
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.layer.precomputed_dual_stream_layer import (
    PrecomputedDualStreamLayer,
)
from versatil.models.layers.transformer.layer.precomputed_kv_layer import (
    PrecomputedKVCrossAttentionLayer,
)


class SmolVLADecoder(BaseInterleavedVLMDecoder):
    """SmolVLA decoder with interleaved VLM and expert processing.

    Alternates between joint self-attention (expert attends alongside
    VLM tokens) and cross-attention (expert attends to VLM key/values)
    layers.
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
        vlm_backbone: GenerativeVLM,
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
    ) -> None:
        """Initialize the SmolVLA decoder.

        Args:
            input_keys: Feature keys from the encoding pipeline.
            action_space: Action space configuration.
            action_heads: Exactly one joint action prediction head.
            observation_space: Observation space configuration.
            observation_horizon: Number of observation timesteps.
            prediction_horizon: Number of action steps to predict.
            device: Device string.
            vlm_backbone: Generative VLM backbone that builds the raw
                observation prefix with shape ``(B, P, D_vlm)``.
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
        super().__init__(
            input_keys=input_keys,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            vlm_backbone=vlm_backbone,
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
        self.time_conditioning_input: nn.Linear
        self.time_conditioning_output: nn.Linear
        self.timestep_embedding: nn.Module | None = None
        self.expert_final_normalization: nn.Module | None = None
        self.vlm_hidden_dimension: int | None = None
        self.proprioceptive_projection: FeatureProjection | None = None
        self.set_vlm_backbone(vlm_backbone=self.vlm_backbone)

    @staticmethod
    def _get_intermediate_size(
        hidden_dimension: int, feedforward_multiplier: int = 4, multiple_of: int = 256
    ) -> int:
        """Compute feedforward intermediate size rounded to a multiple."""
        intermediate = feedforward_multiplier * int(2 * hidden_dimension / 3)
        return multiple_of * ((intermediate + multiple_of - 1) // multiple_of)

    def build_action_expert(
        self,
        vlm_layers: nn.ModuleList,
        rotary_emb: nn.Module,
        vlm_hidden_dimension: int,
        vlm_text_config: PretrainedConfig,
    ) -> None:
        """Create SmolVLA expert layers and projections from VLM internals.

        Note:
            This is called by ``BaseInterleavedVLMDecoder.set_vlm_backbone()``
            after the full VLM backbone is attached.

        Args:
            vlm_layers: VLM transformer layers used by the interleaved decoder.
            rotary_emb: VLM rotary positional encoding module.
            vlm_hidden_dimension: VLM hidden dimension.
            vlm_text_config: HuggingFace text config for the VLM language tower.
        """
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
        vlm_head_dimension = self._get_vlm_head_dimension(
            vlm_text_config=vlm_text_config,
            vlm_hidden_dimension=vlm_hidden_dimension,
        )
        vlm_num_heads = vlm_text_config.num_attention_heads
        vlm_num_key_value_heads = self._get_vlm_num_key_value_heads(
            vlm_text_config=vlm_text_config
        )
        vlm_key_value_dimension = vlm_num_key_value_heads * vlm_head_dimension
        expert_num_heads = vlm_num_heads
        expert_num_key_value_heads = vlm_num_key_value_heads
        expert_head_dimension = vlm_head_dimension
        actual_expert_count = (
            self.num_expert_layers if self.num_expert_layers > 0 else actual_vlm_count
        )
        self._set_action_suffix_modules(
            expert_hidden_dimension=expert_hidden_size,
            time_conditioning=TimeConditioning.CONCAT_MLP.value,
            min_period=self.min_period,
            max_period=self.max_period,
            normalization_type=self.normalization_type,
        )
        if self.proprioceptive_feature_key is not None:
            self.proprioceptive_projection = FeatureProjection(
                embedding_dim=vlm_hidden_dimension
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
        for vlm_idx in range(actual_vlm_count):
            if not layer_has_expert[vlm_idx]:
                self._layer_types.append(InterleavedLayerType.VLM_ONLY.value)
            elif (
                self.self_attention_every_n_layers > 0
                and vlm_idx % self.self_attention_every_n_layers == 0
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
                self._layer_types.append(
                    InterleavedLayerType.JOINT_SELF_ATTENTION.value
                )
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
                self._layer_types.append(InterleavedLayerType.CROSS_ATTENTION.value)
        self.to(self.device)

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
        if self.expert_layers is None or self.expert_final_normalization is None:
            raise RuntimeError("build_action_expert() must be called before forward().")
        actions = self._require_forward_actions(actions=actions)
        prefix_embeddings, prefix_padding_mask = self._build_prefix(features=features)
        timestep = self._get_forward_timestep(features=features)
        prefix_embeddings, prefix_padding_mask, causal_prefix_suffix_length = (
            self._append_optional_projected_prefix_feature(
                prefix_embeddings=prefix_embeddings,
                prefix_padding_mask=prefix_padding_mask,
                features=features,
                feature_key=self.proprioceptive_feature_key,
                projection=self.proprioceptive_projection,
            )
        )
        expert_hidden, _ = self._embed_timestep_conditioned_action_suffix(
            actions=actions,
            timestep=timestep,
            action_input_projection=self.action_input_projection,
            timestep_embedding=self.timestep_embedding,
            time_conditioning=TimeConditioning.CONCAT_MLP.value,
            conditioning_input_layer=self.time_conditioning_input,
            conditioning_output_layer=self.time_conditioning_output,
        )
        attention_state = self._build_interleaved_attention_state(
            expert_tokens=expert_hidden,
            prefix_embeddings=prefix_embeddings,
            prefix_padding_mask=prefix_padding_mask,
            rotary_embedding=self.vlm_rotary_embedding,
            causal_actions=True,
            causal_prefix_suffix_length=causal_prefix_suffix_length,
        )
        cross_attention_mask = self._slice_expert_to_vlm_attention_mask(
            attention_mask=attention_state.attention_mask,
            action_token_count=attention_state.action_token_count,
        )
        # Cross-attention rotates expert queries in a frame independent of the
        # prefix length: positions are shifted to start from 0 so the relative
        # distance to the VLM keys (also at [0, P)) covers the informative range
        # of RoPE. Matches the reference SmolVLA implementation.
        expert_cross_attention_position_ids = self._zero_based_position_ids(
            attention_state.expert_position_ids
        )
        expert_cross_attention_rope = GenerativeVLM.compute_rope(
            rotary_embedding=self.vlm_rotary_embedding,
            hidden_states=expert_hidden,
            position_ids=expert_cross_attention_position_ids,
        )
        expert_hidden = self._run_interleaved_layers(
            prefix_embeddings=prefix_embeddings,
            expert_hidden=expert_hidden,
            attention_state=attention_state,
            cross_attention_mask=cross_attention_mask,
            expert_cross_attention_rope=expert_cross_attention_rope,
        )
        return self._project_expert_actions(
            expert_hidden=expert_hidden,
            expert_final_normalization=self.expert_final_normalization,
        )
