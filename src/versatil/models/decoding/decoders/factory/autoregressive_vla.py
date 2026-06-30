"""Autoregressive VLA decoder for discrete action-token generation.

This module is the shared implementation behind the OpenVLA and pi0-FAST
Hydra presets. It runs a generative VLM on raw image/text observations to build
the conditioning prefix, then trains or samples discrete action tokens
autoregressively through the VLM language model.
"""

import torch

from versatil.data.constants import ActionTokenIdMappingType, SampleKey
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import ActionTokenizer, Tokenizer
from versatil.data.tokenization.action_discretizer import (
    BinnedActionDiscretizer,
    UniformBinnedActionDiscretizer,
)
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.constants import ActionHeadLayout, DecoderOutputKey
from versatil.models.decoding.decoders.autoregressive_mixin import (
    AutoregressiveDecoderMixin,
    CachedAutoregressiveGenerationState,
    PastKeyValues,
)
from versatil.models.decoding.decoders.base import DecoderInput
from versatil.models.decoding.decoders.discrete import DiscreteDecoder
from versatil.models.decoding.decoders.llm_prefix_suffix_attention import (
    LLMPrefixSuffixAttentionMixin,
)
from versatil.models.decoding.decoders.vlm import VLMBackboneDecoderMixin
from versatil.models.decoding.generative_language_models.base import (
    CausalLanguageModelOutput,
)
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)


class AutoregressiveVLADecoder(
    AutoregressiveDecoderMixin,
    LLMPrefixSuffixAttentionMixin,
    VLMBackboneDecoderMixin,
    DiscreteDecoder,
):
    """Predict autoregressive action tokens from a VLM observation prefix."""

    action_head_layout: ActionHeadLayout = ActionHeadLayout.NONE

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
        max_seq_len: int = 512,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        deterministic: bool = True,
        causal_prefix: bool = False,
    ) -> None:
        """Initialize a VLM-backed causal action-token decoder.

        Args:
            action_heads: Must be empty. This decoder predicts action tokens
                with the VLM language vocabulary head.
            input_keys: Must be empty. Raw observation keys are declared by
                ``vlm_backbone.input_specification``.
            action_space: Task action-space metadata.
            observation_space: Task observation-space metadata.
            observation_horizon: Number of observation timesteps in each sample.
            prediction_horizon: Number of future action timesteps represented
                by generated action tokens.
            device: Device used by decoder modules and generated tensors.
            vlm_backbone: Generative VLM that builds image-language prefix
                embeddings and exposes the causal language model vocabulary.
            max_seq_len: Maximum prefix plus generated action-token length.
            temperature: Softmax temperature for stochastic inference.
            learnable_temperature: Whether ``temperature`` is optimized as a
                model parameter.
            deterministic: Whether inference uses greedy token selection.
            causal_prefix: Whether to use a standard causal padding mask (OpenVLA) for
                the whole sequence instead of bidirectional prefix attention (Pi0-FAST).
        """
        if action_heads:
            raise ValueError(
                "AutoregressiveVLADecoder predicts action tokens with the VLM language "
                "vocabulary head, so action_heads must be empty."
            )
        self._validate_no_extra_input_keys(
            decoder_name=type(self).__name__,
            input_keys=input_keys,
        )
        self.max_seq_len = max_seq_len
        self.causal_prefix = causal_prefix
        self.eos_token_id: int | None = None
        self.valid_generation_token_ids: torch.Tensor | None = None

        DiscreteDecoder.__init__(
            self,
            decoder_input=DecoderInput(
                keys=self._vlm_decoder_input_keys(
                    input_keys=[],
                    vlm_backbone=vlm_backbone,
                ),
                requires_actions=True,
                needs_raw_observations=True,
            ),
            action_space=action_space,
            action_heads={},
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            deterministic=deterministic,
        )
        self.vlm_backbone = vlm_backbone
        self.language_hidden_dimension = int(vlm_backbone.hidden_dim)
        self.to(self.device)

    def set_tokenizer(self, tokenizer: Tokenizer | None = None) -> None:
        """Set a language-vocabulary action tokenizer for autoregressive decoding.

        Args:
            tokenizer: Tokenizer with action IDs mapped into the VLM language
                vocabulary.
        """
        if tokenizer is None or tokenizer.action_tokenizer is None:
            raise ValueError(
                "AutoregressiveVLADecoder requires an action tokenizer with "
                "language-vocabulary token IDs."
            )
        action_tokenizer = tokenizer.action_tokenizer
        mapping_type = action_tokenizer.token_id_mapping.state_dict()["type"]
        if mapping_type != ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value:
            raise ValueError(
                "AutoregressiveVLADecoder requires action_tokenizer.token_id_mapping.type="
                f"{ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value}."
            )
        tokenizer_vocab_size = int(action_tokenizer.vocab_size)
        eos_token_id = int(action_tokenizer.eos_token_id)
        if eos_token_id < 0 or eos_token_id >= tokenizer_vocab_size:
            raise ValueError(
                "AutoregressiveVLADecoder received an action tokenizer with eos_token_id "
                "outside the model vocabulary: "
                f"eos_token_id={eos_token_id}, vocab_size={tokenizer_vocab_size}."
            )
        language_vocab_size = self._resize_vlm_to_action_vocabulary(
            tokenizer_vocab_size=tokenizer_vocab_size
        )
        self.tokenizer = action_tokenizer
        self.vocab_size = language_vocab_size
        self.eos_token_id = eos_token_id
        self.valid_generation_token_ids = self._build_valid_generation_token_ids(
            action_tokenizer=action_tokenizer,
            tokenizer_vocab_size=tokenizer_vocab_size,
            eos_token_id=eos_token_id,
        )

    def _resize_vlm_to_action_vocabulary(self, tokenizer_vocab_size: int) -> int:
        """Ensure the VLM language vocabulary covers all action-token IDs."""
        language_vocab_size = self.vlm_backbone.get_vocab_size()
        if language_vocab_size is None:
            raise ValueError(
                "AutoregressiveVLADecoder vlm_backbone must expose get_vocab_size()."
            )
        if language_vocab_size < tokenizer_vocab_size:
            self.vlm_backbone.resize_token_embeddings(tokenizer_vocab_size)
            language_vocab_size = self.vlm_backbone.get_vocab_size()
        if language_vocab_size is None:
            raise ValueError(
                "AutoregressiveVLADecoder vlm_backbone must expose get_vocab_size() "
                "after resizing token embeddings."
            )
        if language_vocab_size < tokenizer_vocab_size:
            raise ValueError(
                "AutoregressiveVLADecoder VLM language vocabulary must cover the "
                "action tokenizer vocabulary after resizing, got "
                f"tokenizer_vocab_size={tokenizer_vocab_size} and "
                f"vlm_backbone_vocab_size={language_vocab_size}."
            )
        return int(language_vocab_size)

    def _build_valid_generation_token_ids(
        self,
        action_tokenizer: ActionTokenizer,
        tokenizer_vocab_size: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        """Return action-token IDs that inference is allowed to sample."""
        token_count = action_tokenizer.action_discretizer.token_count
        local_token_ids = list(range(int(token_count)))
        action_token_ids = action_tokenizer.token_id_mapping.encode(local_token_ids)
        valid_token_ids = torch.as_tensor(
            action_token_ids,
            dtype=torch.long,
            device=self.device,
        )
        eos_token = torch.tensor([eos_token_id], dtype=torch.long, device=self.device)
        valid_token_ids = torch.cat([valid_token_ids, eos_token], dim=0)
        if (
            valid_token_ids.min().item() < 0
            or valid_token_ids.max().item() >= tokenizer_vocab_size
        ):
            raise ValueError(
                "AutoregressiveVLADecoder valid action-token IDs must lie inside "
                f"the action tokenizer vocabulary [0, {tokenizer_vocab_size})."
            )
        return valid_token_ids

    def _get_action_payload_token_count(self) -> int | None:
        """Return the required action-token count before any optional EOS."""
        if self.tokenizer is None:
            return None
        if not isinstance(
            self.tokenizer.action_discretizer,
            (BinnedActionDiscretizer, UniformBinnedActionDiscretizer),
        ):
            return None
        time_horizon = self.tokenizer.action_discretizer.time_horizon
        action_dim = self.tokenizer.action_discretizer.action_dim
        if time_horizon is None or action_dim is None:
            raise ValueError(
                "AutoregressiveVLADecoder fixed-length generation requires the "
                "action discretizer to know time_horizon and action_dim."
            )
        return int(time_horizon * action_dim)

    def _get_max_generation_steps(self, available_context_steps: int) -> int:
        """Return the action-token generation cap within context capacity."""
        action_payload_token_count = self._get_action_payload_token_count()
        if action_payload_token_count is None:
            return super()._get_max_generation_steps(
                available_context_steps=available_context_steps
            )
        if available_context_steps < action_payload_token_count:
            raise ValueError(
                f"{type(self).__name__} needs {action_payload_token_count} context "
                "slots for fixed-length action-token generation, but only "
                f"{available_context_steps} are available."
            )
        return action_payload_token_count

    def _validate_action_tokenizer_is_set(self) -> None:
        """Ensure action-token metadata was initialized."""
        if (
            self.tokenizer is None
            or self.vocab_size is None
            or self.eos_token_id is None
            or self.valid_generation_token_ids is None
        ):
            raise ValueError(
                "AutoregressiveVLADecoder requires set_tokenizer() to be called before forward."
            )

    def get_auxiliary_output_keys(self) -> set[str]:
        """Return token outputs produced without action heads."""
        return super().get_auxiliary_output_keys() | {
            DecoderOutputKey.ACTION_LOGITS.value,
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value,
        }

    def _get_target_token_ids(
        self,
        actions: dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Read teacher-forcing target token IDs from the action dictionary."""
        action_key = SampleKey.TOKENIZED_ACTIONS.value
        if action_key not in actions:
            raise ValueError(
                f"AutoregressiveVLADecoder training requires '{action_key}' in actions."
            )
        target_token_ids = actions[action_key]
        if target_token_ids.ndim != 2:
            raise ValueError(
                f"'{action_key}' must have shape (B, token_length), "
                f"got {target_token_ids.shape}."
            )
        if target_token_ids.shape[0] != batch_size:
            raise ValueError(
                f"'{action_key}' batch size must match feature batch size "
                f"{batch_size}, got {target_token_ids.shape[0]}."
            )
        return target_token_ids.to(device=self.device, dtype=torch.long)

    def _sample_next_action_token(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the valid action-token subset of the VLM vocabulary."""
        if self.valid_generation_token_ids is None:
            raise ValueError(
                "AutoregressiveVLADecoder valid action-token IDs are not initialized."
            )
        valid_token_ids = self.valid_generation_token_ids.to(
            device=logits.device
        )  # (valid_action_token_count)
        valid_logits = logits.index_select(
            dim=-1,
            index=valid_token_ids,
        )  # (B, 1, valid_action_token_count)
        if self.deterministic:
            selected_indices = torch.argmax(valid_logits, dim=-1)  # (B, 1)
        else:
            scaled_logits = valid_logits / self.temperature.clamp(
                min=0.01
            )  # (B, 1, valid_action_token_count)
            probabilities = torch.softmax(
                scaled_logits,
                dim=-1,
            )  # (B, 1, valid_action_token_count)
            selected_indices = torch.multinomial(
                probabilities.squeeze(1),  # (B, valid_action_token_count)
                num_samples=1,
            )  # (B, 1)
        return valid_token_ids[selected_indices]  # (B, 1)

    def _build_projected_prefix(
        self,
        features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the VLM image-language conditioning prefix."""
        return self._build_vlm_prefix(features=features)

    def _run_language_model_logits(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the VLM causal language model and return vocabulary logits."""
        output = self.vlm_backbone.forward_language_model(
            inputs_embeds=tokens,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return self._extract_language_model_logits(output=output)

    @staticmethod
    def _extract_language_model_logits(
        output: CausalLanguageModelOutput,
    ) -> torch.Tensor:
        """Return vocabulary logits from the VLM causal language-model output."""
        return output.logits

    def _build_cached_generation_attention_mask(
        self,
        prefix_token_mask: torch.Tensor | None,
        prefix_tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build the 2D cache attention mask used after prefix prefill."""
        if prefix_token_mask is None or not prefix_token_mask.any():
            return None
        return (~prefix_token_mask).to(device=prefix_tokens.device, dtype=torch.long)

    @staticmethod
    def _get_prefix_valid_mask(
        prefix_tokens: torch.Tensor,
        prefix_token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the valid-token mask for a VLA prefix.

        Args:
            prefix_tokens: Prefix embeddings with shape ``(B, P, D)``.
            prefix_token_mask: Optional prefix padding mask with shape ``(B, P)``,
                where ``True`` marks padding.

        Returns:
            Boolean mask with shape ``(B, P)``, where ``True`` marks real
            prefix tokens.
        """
        if prefix_token_mask is None:
            return torch.ones(
                prefix_tokens.shape[:2],
                dtype=torch.bool,
                device=prefix_tokens.device,
            )
        expected_shape = prefix_tokens.shape[:2]
        if prefix_token_mask.shape != expected_shape:
            raise ValueError(
                f"prefix_token_mask must have shape {expected_shape}, "
                f"got {prefix_token_mask.shape}."
            )
        return ~prefix_token_mask.to(device=prefix_tokens.device, dtype=torch.bool)

    def _remove_prefix_padding(
        self,
        prefix_tokens: torch.Tensor,
        prefix_token_mask: torch.Tensor | None,
        right_align: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove padding slots from each VLA prefix row.

        Args:
            prefix_tokens: Prefix embeddings with shape ``(B, P, D)``.
            prefix_token_mask: Optional prefix padding mask with shape ``(B, P)``,
                where ``True`` marks padding.
            right_align: If ``True``, place each valid prefix at the right edge.
                If ``False``, place each valid prefix at the left edge.

        Returns:
            Prefix embeddings with shape ``(B, P_valid_max, D)``, output
            padding mask with shape ``(B, P_valid_max)``, and per-sample prefix
            lengths with shape ``(B,)``. ``P_valid_max`` is the longest
            non-padding prefix length in the batch. With ``right_align=False``,
            each row is ``[valid prefix tokens, PAD, ..., PAD]``. With
            ``right_align=True``, each row is
            ``[PAD, ..., PAD, valid prefix tokens]``.
        """
        valid_token_mask = self._get_prefix_valid_mask(
            prefix_tokens=prefix_tokens,
            prefix_token_mask=prefix_token_mask,
        )
        prefix_lengths = valid_token_mask.sum(dim=1)
        if (prefix_lengths == 0).any():
            raise ValueError("AutoregressiveVLADecoder received an empty VLA prefix.")
        max_valid_prefix_length = int(prefix_lengths.max().item())
        prefix_tokens_without_padding = prefix_tokens.new_zeros(
            prefix_tokens.shape[0],
            max_valid_prefix_length,
            prefix_tokens.shape[2],
        )
        output_padding_mask = torch.ones(
            prefix_tokens.shape[0],
            max_valid_prefix_length,
            dtype=torch.bool,
            device=prefix_tokens.device,
        )
        for batch_index in range(prefix_tokens.shape[0]):
            sample_length = int(prefix_lengths[batch_index].item())
            sample_tokens = prefix_tokens[batch_index, valid_token_mask[batch_index]]
            start_index = max_valid_prefix_length - sample_length if right_align else 0
            end_index = start_index + sample_length
            # (sample_length, D) -> row slice inside (P_valid_max, D).
            prefix_tokens_without_padding[batch_index, start_index:end_index] = (
                sample_tokens
            )
            output_padding_mask[batch_index, start_index:end_index] = False
        return prefix_tokens_without_padding, output_padding_mask, prefix_lengths

    @staticmethod
    def _build_position_ids_from_padding_mask(
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build zero-based position IDs over a padded token sequence.

        Args:
            padding_mask: Padding mask with shape ``(B, S)``, where ``True``
                marks padding.

        Returns:
            Position IDs with shape ``(B, S)``. Padded positions are set to 0;
            valid positions count up from 0 within each row.
        """
        valid_token_mask = ~padding_mask
        return (valid_token_mask.long().cumsum(dim=1) - 1).clamp(min=0)

    def _build_training_inputs_without_prefix_padding(
        self,
        prefix_tokens: torch.Tensor,
        prefix_token_mask: torch.Tensor | None,
        action_token_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Build teacher-forcing inputs after removing prefix padding slots.

        Args:
            prefix_tokens: Prefix embeddings with shape ``(B, P, D)``.
            prefix_token_mask: Optional prefix padding mask with shape ``(B, P)``,
                where ``True`` marks padding.
            action_token_embeddings: Target action-token embeddings with shape
                ``(B, A, D)``.

        Returns:
            Full embeddings with shape ``(B, P_valid_max + A, D)``, attention
            mask for the language model, and valid prefix lengths with shape
            ``(B,)``. For each row, action tokens start at ``prefix_lengths[b]``.
            Each row is laid out as
            ``[valid prefix tokens, action tokens, PAD, ..., PAD]``.
        """
        prefix_tokens_without_padding, _, prefix_lengths = self._remove_prefix_padding(
            prefix_tokens=prefix_tokens,
            prefix_token_mask=prefix_token_mask,
            right_align=False,
        )
        action_token_length = action_token_embeddings.shape[1]
        sequence_length = int(prefix_lengths.max().item()) + action_token_length
        full_token_sequence = prefix_tokens.new_zeros(
            prefix_tokens.shape[0],
            sequence_length,
            prefix_tokens.shape[2],
        )
        full_padding_mask = torch.ones(
            prefix_tokens.shape[0],
            sequence_length,
            dtype=torch.bool,
            device=prefix_tokens.device,
        )
        for batch_index in range(prefix_tokens.shape[0]):
            prefix_length = int(prefix_lengths[batch_index].item())
            sample_length = prefix_length + action_token_length
            # Row layout: [valid prefix tokens, action tokens, optional padding].
            full_token_sequence[batch_index, :prefix_length] = (
                prefix_tokens_without_padding[batch_index, :prefix_length]
            )
            full_token_sequence[batch_index, prefix_length:sample_length] = (
                action_token_embeddings[batch_index]
            )
            full_padding_mask[batch_index, :sample_length] = False

        attention_mask = self._build_training_attention_mask_without_prefix_padding(
            full_padding_mask=full_padding_mask,
            prefix_lengths=prefix_lengths,
            action_token_length=action_token_length,
            sequence_length=sequence_length,
        )
        return full_token_sequence, attention_mask, prefix_lengths

    def _build_training_attention_mask_without_prefix_padding(
        self,
        full_padding_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
        action_token_length: int,
        sequence_length: int,
    ) -> torch.Tensor | None:
        """Build the LM attention mask for variable-length prefix/action rows.

        Args:
            full_padding_mask: Full sequence padding mask with shape ``(B, S)``,
                where ``S = P_valid_max + A`` and ``True`` marks padding.
            prefix_lengths: Valid prefix lengths with shape ``(B,)``.
            action_token_length: Number of action tokens ``A``.
            sequence_length: Full sequence length ``S = P_valid_max + A``.

        Returns:
            ``None`` when no mask is needed, a 2D causal-model mask with shape
            ``(B, S)`` for OpenVLA, or a 4D attention mask with shape
            ``(B, 1, S, S)`` for bidirectional-prefix models.
        """
        if self.causal_prefix:
            if not full_padding_mask.any():
                return None
            return (~full_padding_mask).to(dtype=torch.long)
        attention_mask = self._build_variable_prefix_attention_mask(
            prefix_lengths=prefix_lengths,
            action_token_length=action_token_length,
            sequence_length=sequence_length,
            device=full_padding_mask.device,
        )
        if attention_mask.all():
            return None
        return attention_mask

    @staticmethod
    def _build_variable_prefix_attention_mask(
        prefix_lengths: torch.Tensor,
        action_token_length: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a bidirectional-prefix/causal-action mask.

        Args:
            prefix_lengths: Valid prefix lengths with shape ``(B,)``.
            action_token_length: Number of action tokens ``A``.
            sequence_length: Full sequence length ``S = P_valid_max + A``.
            device: Device for the created mask.

        Returns:
            Boolean attention mask with shape ``(B, 1, S, S)``, where ``True``
            means attention is allowed.
        """
        positions = torch.arange(sequence_length, device=device)
        query_positions = positions.view(1, sequence_length, 1)  # (1, S, 1)
        key_positions = positions.view(1, 1, sequence_length)  # (1, 1, S)
        prefix_lengths = prefix_lengths.to(device=device).view(-1, 1, 1)  # (B, 1, 1)
        sequence_lengths = prefix_lengths + action_token_length

        valid_queries = query_positions < sequence_lengths  # (B, S, 1)
        valid_keys = key_positions < sequence_lengths  # (B, 1, S)
        query_is_prefix = query_positions < prefix_lengths  # (B, S, 1)
        key_is_prefix = key_positions < prefix_lengths
        key_is_action = valid_keys & ~key_is_prefix

        prefix_query_attention = key_is_prefix
        action_query_attention = key_is_prefix | (
            key_is_action & (key_positions <= query_positions)
        )
        valid_query_attention = torch.where(
            query_is_prefix,
            prefix_query_attention,
            action_query_attention,
        )
        valid_query_attention = valid_query_attention & valid_keys
        padding_query_attention = valid_keys.expand_as(valid_query_attention)
        attention_mask = torch.where(
            valid_queries,
            valid_query_attention,
            padding_query_attention,
        )
        return attention_mask.unsqueeze(1)

    @staticmethod
    def _gather_next_token_logits(
        language_logits: torch.Tensor,
        prefix_lengths: torch.Tensor,
        action_token_length: int,
    ) -> torch.Tensor:
        """Gather next-token logits for each action token.

        Args:
            language_logits: LM logits with shape ``(B, S, V)``.
            prefix_lengths: Valid prefix lengths with shape ``(B,)``.
            action_token_length: Number of action tokens ``A``.

        Returns:
            Action-token logits with shape ``(B, A, V)``. Row ``b`` gathers
            positions ``prefix_lengths[b] - 1`` through
            ``prefix_lengths[b] + A - 2``.
        """
        action_positions = (
            prefix_lengths.to(device=language_logits.device).unsqueeze(1)
            - 1
            + torch.arange(action_token_length, device=language_logits.device)
        )  # (B, A)
        gather_indices = action_positions.unsqueeze(-1).expand(
            -1,
            -1,
            language_logits.shape[-1],
        )  # (B, A, V)
        return language_logits.gather(dim=1, index=gather_indices)

    def _decode_next_autoregressive_step(
        self,
        state: CachedAutoregressiveGenerationState,
    ) -> tuple[CausalLanguageModelOutput, PastKeyValues]:
        """Decode one cached VLM language-model action-token step."""
        output = self.vlm_backbone.forward_language_model(
            input_ids=state.next_inputs,
            attention_mask=state.attention_mask,
            past_key_values=state.past_key_values,
            use_cache=True,
            cache_position=state.cache_position,
            position_ids=state.position_ids,
        )
        if output.past_key_values is None:
            raise ValueError(
                "AutoregressiveVLADecoder VLM language model did not return "
                "past_key_values during cached generation."
            )
        return output, output.past_key_values

    def _sample_next_autoregressive_output(
        self,
        step_output: CausalLanguageModelOutput,
    ) -> torch.Tensor:
        """Sample the next action token from language logits."""
        language_logits = self._extract_language_model_logits(output=step_output)
        return self._sample_next_action_token(logits=language_logits[:, -1:, :])

    def _prepare_next_autoregressive_inputs(
        self,
        generated_output: torch.Tensor,
    ) -> torch.Tensor:
        """Feed sampled language-vocabulary token IDs into the next VLM step."""
        return generated_output

    def _get_completed_sequence_mask(
        self,
        generated_output: torch.Tensor,
        state: CachedAutoregressiveGenerationState,
    ) -> torch.Tensor | None:
        """Update the per-sample EOS mask for action-token generation."""
        completed = generated_output.squeeze(1) == self.eos_token_id
        if state.completed_sequence_mask is None:
            return completed
        return state.completed_sequence_mask | completed

    def _advance_autoregressive_attention_mask(
        self,
        state: CachedAutoregressiveGenerationState,
        generated_output: torch.Tensor,
    ) -> torch.Tensor | None:
        """Append an unmasked generated token to the cache attention mask."""
        if state.attention_mask is None:
            return None
        next_attention = torch.ones(
            state.attention_mask.shape[0],
            generated_output.shape[1],
            dtype=state.attention_mask.dtype,
            device=state.attention_mask.device,
        )
        return torch.cat([state.attention_mask, next_attention], dim=1)

    def _finalize_autoregressive_outputs(
        self,
        generated_outputs: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Pack generated action-token IDs into decoder outputs."""
        return {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.cat(
                generated_outputs,
                dim=1,
            )
        }

    def _forward_action_token_training(
        self,
        actions: dict[str, torch.Tensor],
        prefix_tokens: torch.Tensor,
        prefix_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict target action tokens after removing prefix padding slots.

        Args:
            actions: Action dict containing token IDs with shape ``(B, A)``.
            prefix_tokens: VLA prefix embeddings with shape ``(B, P, D)``.
            prefix_token_mask: Optional prefix padding mask with shape ``(B, P)``,
                where ``True`` marks padding.

        Returns:
            Dict containing action logits with shape ``(B, A, V)``.
        """
        prefix_len = prefix_tokens.shape[1]
        if prefix_len == 0:
            raise ValueError(
                "AutoregressiveVLADecoder requires a non-empty conditioning prefix."
            )
        target_token_ids = self._get_target_token_ids(
            actions=actions,
            batch_size=prefix_tokens.shape[0],
        )  # (B, A)
        action_token_embeddings = self.vlm_backbone.embed_input_ids(
            target_token_ids
        )  # (B, A, D)
        full_token_sequence, attention_mask, prefix_lengths = (
            self._build_training_inputs_without_prefix_padding(
                prefix_tokens=prefix_tokens,
                prefix_token_mask=prefix_token_mask,
                action_token_embeddings=action_token_embeddings,
            )
        )
        if full_token_sequence.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Input token length {full_token_sequence.shape[1]} > max_seq_len "
                f"{self.max_seq_len}. Consider increasing max_seq_len or reducing "
                "the text/action/feature token count."
            )
        language_logits = self._run_language_model_logits(
            tokens=full_token_sequence,
            attention_mask=attention_mask,
        )  # (B, P_valid_max+A, V)
        action_logits = self._gather_next_token_logits(
            language_logits=language_logits,
            prefix_lengths=prefix_lengths,
            action_token_length=target_token_ids.shape[1],
        )
        return {DecoderOutputKey.ACTION_LOGITS.value: action_logits}

    def _build_prefill_inputs(
        self,
        prefix_tokens: torch.Tensor,
        prefix_token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build right-aligned valid prefix tokens for cached generation.

        Args:
            prefix_tokens: VLA prefix embeddings with shape ``(B, P, D)``.
            prefix_token_mask: Optional prefix padding mask with shape ``(B, P)``,
                where ``True`` marks padding.

        Returns:
            Right-aligned prefix embeddings with shape ``(B, P_valid_max, D)``,
            padding mask with shape ``(B, P_valid_max)``, and valid prefix
            lengths with shape ``(B,)``. ``P_valid_max`` is the longest
            non-padding prefix length in this batch. Each row is laid out as
            ``[PAD, ..., PAD, valid prefix tokens]`` so the final column is the
            last real prefix token for every sample.
        """
        return self._remove_prefix_padding(
            prefix_tokens=prefix_tokens,
            prefix_token_mask=prefix_token_mask,
            right_align=True,
        )

    def _forward_action_token_inference(
        self,
        prefix_tokens: torch.Tensor,
        prefix_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate action tokens from a right-aligned VLA prefix.

        Args:
            prefix_tokens: VLA prefix embeddings with shape ``(B, P, D)``.
            prefix_token_mask: Optional prefix padding mask with shape ``(B, P)``,
                where ``True`` marks padding.

        Returns:
            Dict containing generated token IDs with shape ``(B, T_generated)``.
        """
        prefix_len = prefix_tokens.shape[1]
        if prefix_len == 0:
            raise ValueError(
                "AutoregressiveVLADecoder requires a non-empty conditioning prefix."
            )
        (
            right_aligned_prefix_tokens,
            right_aligned_prefix_padding_mask,
            prefix_lengths,
        ) = self._build_prefill_inputs(
            prefix_tokens=prefix_tokens,
            prefix_token_mask=prefix_token_mask,
        )
        max_valid_prefix_length = right_aligned_prefix_tokens.shape[1]
        if max_valid_prefix_length >= self.max_seq_len:
            raise ValueError(
                f"Input prefix token length {max_valid_prefix_length} >= max_seq_len "
                f"{self.max_seq_len}. No room for generated action tokens. "
                "Consider increasing max_seq_len or reducing feature count."
            )
        prefill_attention_mask = self._build_attention_mask(
            padding_mask=right_aligned_prefix_padding_mask,
            tokens=right_aligned_prefix_tokens,
            prefix_length=max_valid_prefix_length,
            causal_suffix=True,
        )
        prefill_position_ids = self._build_position_ids_from_padding_mask(
            padding_mask=right_aligned_prefix_padding_mask,
        )
        prefill_output = self.vlm_backbone.forward_language_model(
            inputs_embeds=right_aligned_prefix_tokens,
            attention_mask=prefill_attention_mask,
            use_cache=True,
            position_ids=prefill_position_ids,
        )
        if prefill_output.past_key_values is None:
            raise ValueError(
                "AutoregressiveVLADecoder VLM language model did not return "
                "past_key_values during prefix prefill."
            )
        initial_state = CachedAutoregressiveGenerationState(
            step_index=0,
            sequence_length=max_valid_prefix_length,
            past_key_values=prefill_output.past_key_values,
            next_inputs=torch.empty(
                right_aligned_prefix_tokens.shape[0],
                0,
                dtype=torch.long,
                device=right_aligned_prefix_tokens.device,
            ),
            attention_mask=self._build_cached_generation_attention_mask(
                prefix_token_mask=right_aligned_prefix_padding_mask,
                prefix_tokens=right_aligned_prefix_tokens,
            ),
            position_ids=prefix_lengths.unsqueeze(1) - 1,  # (B, 1)
        )
        return self._run_cached_autoregressive_generation(
            initial_state=initial_state,
            max_generation_steps=self._get_max_generation_steps(
                available_context_steps=self.max_seq_len - max_valid_prefix_length,
            ),
            initial_step_output=prefill_output,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run VLM-conditioned action-token prediction."""
        self._validate_action_tokenizer_is_set()
        prefix_tokens, prefix_token_mask = self._build_projected_prefix(
            features=features
        )
        if actions is not None:
            return self._forward_action_token_training(
                actions=actions,
                prefix_tokens=prefix_tokens,
                prefix_token_mask=prefix_token_mask,
            )
        else:
            return self._forward_action_token_inference(
                prefix_tokens=prefix_tokens,
                prefix_token_mask=prefix_token_mask,
            )
