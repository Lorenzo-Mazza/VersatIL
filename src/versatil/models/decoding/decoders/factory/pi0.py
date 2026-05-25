"""Pi0/Pi0.5 interleaved VLM-expert decoder with joint attention.

References:
    Pi0: https://github.com/Physical-Intelligence/openpi
"""

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


class Pi0Decoder(BaseInterleavedVLMDecoder):
    """Pi0/Pi0.5 decoder with pretrained VLM backbone and learned action expert.

    Each VLM layer is paired 1:1 with an expert layer via joint
    self-attention. Pi0 variant fuses timestep into action tokens via MLP
    before the expert layers. Pi0.5 variant modulates each expert layer
    via adaptive normalization.
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
    ) -> None:
        """Initialize Pi0 decoder.

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
        self.action_input_projection: nn.Linear
        self.timestep_embedding: nn.Module
        self.time_conditioning_input: nn.Linear
        self.time_conditioning_output: nn.Linear
        self.expert_final_normalization: nn.Module
        self.proprioceptive_projection: FeatureProjection | None = None
        self._set_action_suffix_modules(
            expert_hidden_dimension=expert_hidden_size,
            time_conditioning=time_conditioning,
            min_period=min_period,
            max_period=max_period,
            normalization_type=normalization_type,
        )
        self.vlm_layers: nn.ModuleList | None = None
        self.vlm_rotary_embedding: nn.Module | None = None
        self.vlm_hidden_dimension: int | None = None
        self.expert_layers: nn.ModuleList | None = None
        self.set_vlm_backbone(vlm_backbone=self.vlm_backbone)
        self.to(self.device)

    def build_action_expert(
        self,
        vlm_layers: nn.ModuleList,
        rotary_emb: nn.Module,
        vlm_hidden_dimension: int,
        vlm_text_config: PretrainedConfig,
    ) -> None:
        """Reference pretrained VLM layers and create expert layers.

        Note:
            This is called by ``BaseInterleavedVLMDecoder.set_vlm_backbone()``
            after the full VLM backbone is attached.

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
        self._layer_types = [
            InterleavedLayerType.JOINT_SELF_ATTENTION.value
            for _ in range(self.expert_number_of_layers)
        ]
        if (
            self.proprioceptive_feature_key is not None
            and self.time_conditioning == TimeConditioning.CONCAT_MLP.value
        ):
            self.proprioceptive_projection = FeatureProjection(
                embedding_dim=vlm_hidden_dimension
            )
        self.to(self.device)

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
            raise RuntimeError("build_action_expert() must be called before forward().")
        actions = self._require_forward_actions(actions=actions)
        prefix_embeddings, prefix_padding_mask = self._build_prefix(features=features)
        timestep = self._get_forward_timestep(features=features)
        expert_hidden, adaptive_norm_conditioning = (
            self._embed_timestep_conditioned_action_suffix(
                actions=actions,
                timestep=timestep,
                action_input_projection=self.action_input_projection,
                timestep_embedding=self.timestep_embedding,
                time_conditioning=self.time_conditioning,
                conditioning_input_layer=self.time_conditioning_input,
                conditioning_output_layer=self.time_conditioning_output,
            )
        )
        prefix_embeddings, prefix_padding_mask, causal_prefix_suffix_length = (
            self._append_optional_projected_prefix_feature(
                prefix_embeddings=prefix_embeddings,
                prefix_padding_mask=prefix_padding_mask,
                features=features,
                feature_key=self.proprioceptive_feature_key,
                projection=self.proprioceptive_projection,
            )
        )
        attention_state = self._build_interleaved_attention_state(
            expert_tokens=expert_hidden,
            prefix_embeddings=prefix_embeddings,
            prefix_padding_mask=prefix_padding_mask,
            rotary_embedding=self.vlm_rotary_embedding,
            causal_actions=False,
            causal_prefix_suffix_length=causal_prefix_suffix_length,
        )
        expert_hidden = self._run_interleaved_layers(
            prefix_embeddings=prefix_embeddings,
            expert_hidden=expert_hidden,
            attention_state=attention_state,
            adaptive_norm_conditioning=adaptive_norm_conditioning,
        )
        return self._project_expert_actions(
            expert_hidden=expert_hidden,
            expert_final_normalization=self.expert_final_normalization,
        )
