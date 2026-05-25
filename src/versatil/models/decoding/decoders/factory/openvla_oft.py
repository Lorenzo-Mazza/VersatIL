"""OpenVLA-OFT-style continuous action-chunk decoder backed by a VLM."""

import torch
import torch.nn as nn

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_heads.base import BaseActionHead
from versatil.models.decoding.constants import ActionHeadLayout, DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.decoding.decoders.llm_prefix_suffix_attention import (
    LLMPrefixSuffixAttentionMixin,
)
from versatil.models.decoding.decoders.timestep_conditioning import (
    extract_timestep_conditioning,
    filter_timestep_feature,
    validate_action_tensors_against_dimensions,
)
from versatil.models.decoding.decoders.vlm import VLMBackboneDecoderMixin
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    PeriodInterpolationPositionalEncoding1D,
)

JOINT_ACTION_HEAD_KEY = "joint_action"


class OpenVLAOFTDecoder(
    LLMPrefixSuffixAttentionMixin,
    VLMBackboneDecoderMixin,
    ActionDecoder,
):
    """Decode continuous action chunks from VLM prefix plus action slots."""

    action_head_layout: ActionHeadLayout = ActionHeadLayout.JOINT

    def __init__(
        self,
        action_heads: dict[str, ActionHead],
        input_keys: list[str],
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        vlm_backbone: GenerativeVLM,
        slots_per_action_dimension: bool = True,
        causal_action_slots: bool = True,
        min_period: float = 4e-3,
        max_period: float = 4.0,
    ) -> None:
        """Initialize a VLM-backed continuous action-chunk decoder.

        Args:
            action_heads: Exactly one joint action head that maps per-timestep
                decoder embeddings to the full continuous action vector. With
                ``slots_per_action_dimension=True``, its input dimension must be
                ``action_dim * language_hidden_dimension``. Otherwise, its
                input dimension must be ``language_hidden_dimension``.
            input_keys: Must be empty. Raw observation keys are declared by
                ``vlm_backbone.input_specification``.
            action_space: Task action-space metadata.
            observation_space: Task observation-space metadata.
            observation_horizon: Number of observation timesteps in each sample.
            prediction_horizon: Number of future action timesteps to predict.
            device: Device used by decoder modules and generated tensors.
            vlm_backbone: Generative VLM that builds image-language prefix
                embeddings and exposes the language tower.
            slots_per_action_dimension: When ``True``, each action scalar owns
                one VLM hidden-state slot before the joint action projection.
                When ``False``, each timestep owns one slot.
            causal_action_slots: Whether action slots use causal self-attention.
            min_period: Minimum period for sinusoidal timestep embeddings used
                by denoising algorithms.
            max_period: Maximum period for sinusoidal timestep embeddings used
                by denoising algorithms.
        """
        self._validate_no_extra_input_keys(
            decoder_name=type(self).__name__,
            input_keys=input_keys,
        )
        self.slots_per_action_dimension = slots_per_action_dimension
        self.causal_action_slots = causal_action_slots
        self.min_period = min_period
        self.max_period = max_period
        self.causal_prefix = False
        self.language_hidden_dimension = int(vlm_backbone.hidden_dim)
        action_projection_input_dimension = (
            self._resolve_action_projection_input_dimension(
                action_space=action_space,
                language_hidden_dimension=self.language_hidden_dimension,
                slots_per_action_dimension=slots_per_action_dimension,
            )
        )

        ActionDecoder.__init__(
            self,
            decoder_input=DecoderInput(
                keys=self._vlm_decoder_input_keys(
                    input_keys=input_keys,
                    vlm_backbone=vlm_backbone,
                ),
                requires_actions=False,
                needs_raw_observations=True,
            ),
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )
        self.vlm_backbone = vlm_backbone
        self.action_projection_input_dimension = action_projection_input_dimension
        self._validate_action_head_input_dimensions(
            expected_input_dimension=action_projection_input_dimension
        )
        self._build_action_slot_components()
        self._validate_context_capacity(
            vlm_backbone=vlm_backbone,
            includes_denoising_timestep=False,
        )
        self.to(self.device)

    @staticmethod
    def _resolve_action_projection_input_dimension(
        action_space: ActionSpace,
        language_hidden_dimension: int,
        slots_per_action_dimension: bool,
    ) -> int:
        """Return the hidden width consumed by the joint action projection."""
        if slots_per_action_dimension:
            return action_space.get_total_action_dim() * language_hidden_dimension
        return language_hidden_dimension

    def _joint_action_head(self) -> BaseActionHead:
        """Return the configured OFT joint action head."""
        return self._single_action_head()

    def _predicted_action_keys(self) -> list[str]:
        """Return action-space keys predicted by the joint action head."""
        return self.action_space.predicted_action_keys

    def _predicted_action_dimensions(self) -> dict[str, int]:
        """Return per-component prediction dimensions from the action space."""
        return self.action_space.predicted_action_dimensions

    def _validate_action_head_input_dimensions(
        self,
        expected_input_dimension: int,
    ) -> None:
        """Validate that action heads consume the decoder action embeddings."""
        mismatched_input_dimensions = {
            action_key: action_head.input_dim
            for action_key, action_head in self.action_heads.items()
            if action_head.input_dim != expected_input_dimension
        }
        if not mismatched_input_dimensions:
            return

        if self.slots_per_action_dimension:
            expected_rule = (
                "slots_per_action_dimension=True uses one slot per action scalar, "
                "so the joint action head input_dim must equal action_dim * "
                "language_hidden_dimension "
                f"({self.action_dim} * {self.language_hidden_dimension} = "
                f"{expected_input_dimension})."
            )
        else:
            expected_rule = (
                "slots_per_action_dimension=False uses one slot per timestep, "
                "so the joint action head input_dim must equal "
                f"language_hidden_dimension ({expected_input_dimension})."
            )
        raise ValueError(
            "OpenVLAOFTDecoder action head input_dim mismatch. "
            f"{expected_rule} Got {mismatched_input_dimensions}."
        )

    def _build_action_slot_components(self) -> None:
        """Create learned action slots and denoising/timestep projections."""
        self.action_slots_per_step = (
            self.action_dim if self.slots_per_action_dimension else 1
        )
        self.action_slot_count = self.prediction_horizon * self.action_slots_per_step
        self.action_slot_embeddings = nn.Embedding(
            self.action_slot_count,
            self.language_hidden_dimension,
        )
        noisy_action_input_dimension = (
            1 if self.slots_per_action_dimension else self.action_dim
        )
        self.noisy_action_projection = nn.Linear(
            noisy_action_input_dimension,
            self.language_hidden_dimension,
        )
        self.timestep_embedding = PeriodInterpolationPositionalEncoding1D(
            embedding_dimension=self.language_hidden_dimension,
            min_period=self.min_period,
            max_period=self.max_period,
        )

    def _validate_context_capacity(
        self,
        vlm_backbone: GenerativeVLM,
        includes_denoising_timestep: bool,
    ) -> None:
        """Validate that the VLM context can hold prefix and action slots."""
        text_config = vlm_backbone.get_text_config()
        max_position_embeddings = text_config.max_position_embeddings
        timestep_token_count = 1 if includes_denoising_timestep else 0
        required_sequence_length = (
            vlm_backbone.total_image_tokens
            + vlm_backbone.max_text_length
            + timestep_token_count
            + self.action_slot_count
        )
        if required_sequence_length > max_position_embeddings:
            raise ValueError(
                "OpenVLAOFTDecoder sequence length exceeds the VLM language "
                "context. Required total_image_tokens + max_text_length + "
                "denoising_timestep_tokens + action_slots = "
                f"{vlm_backbone.total_image_tokens} + "
                f"{vlm_backbone.max_text_length} + {timestep_token_count} + "
                f"{self.action_slot_count} = {required_sequence_length}, "
                f"but max_position_embeddings={max_position_embeddings}."
            )

    def _append_timestep_feature_token(
        self,
        feature_tokens: torch.Tensor,
        feature_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one denoising timestep token to the prefix."""
        timestep_token = self.timestep_embedding(
            timestep.to(device=feature_tokens.device)
        ).to(
            device=feature_tokens.device,
            dtype=feature_tokens.dtype,
        )  # (B, D)
        timestep_token = timestep_token.unsqueeze(1)  # (B, 1, D)
        timestep_mask = torch.zeros(
            feature_tokens.shape[0],
            1,
            dtype=torch.bool,
            device=feature_tokens.device,
        )  # (B, 1)
        return (
            torch.cat([feature_tokens, timestep_token], dim=1),  # (B, P+1, D)
            torch.cat([feature_mask, timestep_mask], dim=1),  # (B, P+1)
        )

    def _build_prefix(
        self,
        features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the VLM image-language conditioning prefix."""
        return self._build_vlm_prefix(features=features)

    def _build_denoising_prefix(
        self,
        features: dict[str, torch.Tensor],
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build a VLM prefix with an appended denoising timestep token."""
        observation_features = filter_timestep_feature(features=features)
        prefix_tokens, prefix_mask = self._build_prefix(features=observation_features)
        if prefix_mask is None:
            prefix_mask = torch.zeros(
                prefix_tokens.shape[:2],
                dtype=torch.bool,
                device=prefix_tokens.device,
            )  # (B, P)
        prefix_tokens, prefix_mask = self._append_timestep_feature_token(
            feature_tokens=prefix_tokens,
            feature_mask=prefix_mask,
            timestep=timestep,
        )
        return prefix_tokens, self._all_false_or_none(prefix_mask)

    def _build_action_slots(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand learned action slots for each sample in the batch."""
        action_slot_indices = torch.arange(self.action_slot_count, device=device)
        slot_embeddings = self.action_slot_embeddings(
            action_slot_indices
        )  # (action_slot_count, D)
        return (
            slot_embeddings.to(dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        )  # (B, action_slot_count, D)

    def _build_denoising_action_slots(
        self,
        actions: dict[str, torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Project noisy/interpolated actions into parallel action slots."""
        action_tensors = [
            actions[action_key].to(device=device, dtype=dtype)
            for action_key in self._predicted_action_keys()
        ]
        noisy_actions = torch.cat(action_tensors, dim=-1)  # (B, H, action_dim)
        if self.slots_per_action_dimension:
            noisy_actions = noisy_actions.reshape(
                noisy_actions.shape[0],
                self.action_slot_count,
                1,
            )  # (B, H*action_dim, 1)
        return self.noisy_action_projection(noisy_actions)  # (B, action_slot_count, D)

    def _run_language_model(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the VLM language tower and return final hidden states."""
        output = self.vlm_backbone.forward_language_model(
            inputs_embeds=tokens,
            attention_mask=attention_mask,
            use_cache=False,
        )
        if output.hidden_states is None:
            raise ValueError(
                f"{type(self).__name__} requires VLM language-model hidden states."
            )
        return output.hidden_states[-1]

    def _reshape_action_slot_states(
        self,
        slot_states: torch.Tensor,
    ) -> torch.Tensor:
        """Convert raw action-slot states to per-step projection inputs."""
        if self.slots_per_action_dimension:
            slot_states = slot_states.reshape(
                slot_states.shape[0],
                self.prediction_horizon,
                self.action_slots_per_step,
                self.language_hidden_dimension,
            )  # (B, H, action_dim, D)
            return slot_states.flatten(start_dim=2)  # (B, H, action_dim*D)
        return slot_states.reshape(
            slot_states.shape[0],
            self.prediction_horizon,
            self.language_hidden_dimension,
        )  # (B, H, D)

    def _project_action_output(
        self,
        action_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Project joint actions and split them into configured components."""
        action_output = self._joint_action_head()(
            action_embeddings
        )  # (B, H, action_dim)
        return self.action_space.split_action_tensor(
            action_tensor=action_output,
            owner_name=type(self).__name__,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict a continuous action chunk from a VLM prefix."""
        if DecoderOutputKey.TIMESTEP.value in features:
            self._validate_context_capacity(
                vlm_backbone=self.vlm_backbone,
                includes_denoising_timestep=True,
            )
            if actions is None:
                raise ValueError(
                    "OpenVLAOFTDecoder with denoising algorithm requires "
                    "ground truth actions during training."
                )
            batch_size, action_device = validate_action_tensors_against_dimensions(
                actions=actions,
                action_dimensions=self._predicted_action_dimensions(),
                prediction_horizon=self.prediction_horizon,
                decoder_name=type(self).__name__,
            )
            timestep = extract_timestep_conditioning(
                features=features,
                batch_size=batch_size,
                action_device=action_device,
            )
            prefix_tokens, prefix_mask = self._build_denoising_prefix(
                features=features,
                timestep=timestep,
            )
            action_slots = self._build_denoising_action_slots(
                actions=actions,
                dtype=prefix_tokens.dtype,
                device=prefix_tokens.device,
            )  # (B, action_slot_count, D)
        else:
            prefix_tokens, prefix_mask = self._build_prefix(features=features)
            action_slots = self._build_action_slots(
                batch_size=prefix_tokens.shape[0],
                dtype=prefix_tokens.dtype,
                device=prefix_tokens.device,
            )  # (B, action_slot_count, D)
        full_token_sequence, attention_mask = self._build_prefix_suffix_inputs(
            prefix_tokens=prefix_tokens,
            suffix_tokens=action_slots,
            prefix_mask=prefix_mask,
            causal_suffix=self.causal_action_slots,
        )
        sequence_output = self._run_language_model(
            tokens=full_token_sequence,
            attention_mask=attention_mask,
        )  # (B, P+action_slot_count, D)
        action_slot_states = sequence_output[
            :, -self.action_slot_count :, :
        ]  # (B, action_slot_count, D)
        action_embeddings = self._reshape_action_slot_states(action_slot_states)
        return self._project_action_output(action_embeddings)
