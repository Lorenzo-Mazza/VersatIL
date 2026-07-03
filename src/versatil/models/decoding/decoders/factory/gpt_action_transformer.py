"""Action GPT Decoder for tokenized action prediction.

It uses a GPT-style autoregressive decoder (only self-attention) to generate sequences of tokenized actions.
"""

import torch

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_masking import make_attention_mask
from versatil.models.decoding.constants import ActionHeadLayout, DecoderOutputKey
from versatil.models.decoding.decoders.autoregressive_mixin import (
    AutoregressiveDecoderMixin,
    CachedAutoregressiveGenerationState,
    PastKeyValues,
)
from versatil.models.decoding.decoders.base import DecoderInput
from versatil.models.decoding.decoders.discrete import DiscreteDecoder
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)
from versatil.models.layers.transformer.autoregressive_decoder import GPTDecoder


class GPTActionTransformer(
    AutoregressiveDecoderMixin,
    DiscreteDecoder,
):
    """Autoregressive decoder for tokenized action prediction.

    Uses pure GPT-style transformer with self-attention only (no cross-attention).
    Observation features are concatenated as prefix tokens, followed by
    action token embeddings for autoregressive generation.
    This is similar to Pi0 FAST but adapted to work with any feature encoder.
    """

    action_head_layout: ActionHeadLayout = ActionHeadLayout.VOCABULARY

    def __init__(
        self,
        action_heads: dict[str, ActionHead],
        input_keys: list[str],
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        max_seq_len: int = 512,
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        number_of_layers: int = 6,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = PositionalEncodingType.ROPE.value,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        deterministic: bool = True,
    ) -> None:
        """Initialize GPTActionTransformer decoder.

        Args:
            action_heads: Action heads for different action components (only DecoderOutputKey.ACTION_LOGITS.value used here).
            input_keys: Feature keys expected from encoder pipeline
            action_space: Action space configuration
            observation_space: Observation space configuration
            observation_horizon: Number of observation timesteps
            prediction_horizon: Max action horizon for generation
            device: Device to run model on
            max_seq_len: Maximum sequence length for GPT (features + action tokens)
            embedding_dimension: Common embedding dimension to bring input tokens to, also Transformer hidden size
            number_of_heads: Number of query attention heads
            number_of_key_value_heads: Number of K/V heads for GQA (None = same as heads = MHA)
            feedforward_dimension: FFN hidden dimension (default: 4 * embedding_dimension)
            number_of_layers: Number of transformer layers
            activation: Activation function (swiglu, gelu, relu, silu)
            normalization_type: Normalization type (rmsnorm, layernorm)
            attention_type: Attention type (gqa, mha)
            dropout_rate: Dropout probability
            attention_dropout: Attention dropout probability
            positional_encoding_type: Type of positional encoding (sinusoidal, rope, None)
            temperature: Initial temperature for sampling (not used in greedy decoding)
            learnable_temperature: If True, make temperature a learnable parameter
            deterministic: If True, use greedy decoding during inference
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.observation_horizon = observation_horizon
        self.max_seq_len = max_seq_len
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads or number_of_heads
        self.feedforward_dimension = feedforward_dimension or (4 * embedding_dimension)
        self.number_of_layers = number_of_layers
        self.activation = activation
        self.normalization_type = normalization_type
        self.attention_type = attention_type
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.positional_encoding_type = positional_encoding_type
        decoder_input = DecoderInput(
            keys=input_keys,
            requires_actions=True,
        )
        super().__init__(
            decoder_input=decoder_input,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            deterministic=deterministic,
        )
        self._build_transformer_components()
        self._init_action_bos_embedding(
            embedding_dimension=self.embedding_dimension,
            initializer_range=self.gpt_decoder.initializer_range,
        )
        self.to(self.device)

    def _build_transformer_components(self) -> None:
        """Build core transformer encoder-decoder and positional encodings."""
        image_positional_encoding = SinusoidalPositionalEncoding2D(
            embedding_dimension=self.embedding_dimension, normalize=True
        )
        temporal_positional_encoding = None
        if self.observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension
            )

        # This layer transforms input features into a sequence of token embeddings + positional encodings
        self.input_sequence_builder = TransformerInputBuilder(
            embedding_dimension=self.embedding_dimension,
            has_time_dim=self.observation_horizon > 1,
            spatial_positional_encoding_layer=image_positional_encoding,
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
        )
        self.gpt_decoder = GPTDecoder(
            number_of_layers=self.number_of_layers,
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.number_of_heads,
            number_of_key_value_heads=self.number_of_key_value_heads,
            feedforward_dimension=self.feedforward_dimension,
            dropout=self.dropout_rate,
            attention_dropout=self.attention_dropout,
            activation=self.activation,
            normalization_type=self.normalization_type,
            attention_type=self.attention_type,
            use_cross_attention=False,  # Pure GPT - no cross-attention
            positional_encoding_type=self.positional_encoding_type,
            maximum_sequence_length=self.max_seq_len,
        )

    def _action_token_initializer_range(self) -> float:
        """Return the GPT decoder initializer std for token embeddings."""
        return self.gpt_decoder.initializer_range

    def _action_token_embedding_dimension(self) -> int:
        """Return the GPT token embedding dimension."""
        return self.embedding_dimension

    def _build_prefix_self_attention_mask(
        self,
        prefix_tokens: torch.Tensor,
        causal_prefix_suffix_length: int,
    ) -> torch.Tensor:
        """Build the prefix-only self-attention mask used during cache prefill."""
        prefix_len = prefix_tokens.shape[1]
        prefix_self_mask = torch.zeros(
            prefix_tokens.shape[0],
            1,
            prefix_len,
            prefix_len,
            dtype=torch.bool,
            device=prefix_tokens.device,
        )
        if causal_prefix_suffix_length > 0:
            if causal_prefix_suffix_length > prefix_len:
                raise ValueError(
                    "causal_prefix_suffix_length must be less than or equal to "
                    f"prefix length {prefix_len}, got {causal_prefix_suffix_length}."
                )
            causal_start = prefix_len - causal_prefix_suffix_length
            prefix_self_mask[:, :, :causal_start, causal_start:] = True
        return prefix_self_mask

    def _decode_next_autoregressive_step(
        self,
        state: CachedAutoregressiveGenerationState,
    ) -> tuple[torch.Tensor, PastKeyValues]:
        """Decode one GPT action-token step from cached context."""
        decoder_output, generation_cache = self.gpt_decoder(
            hidden_states=state.next_inputs,
            self_attention_mask=None,
            generation_cache=state.past_key_values,
        )
        return decoder_output, generation_cache

    def _sample_next_autoregressive_output(
        self,
        step_output: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the next action token from one GPT step output."""
        last_output = step_output[:, -1:, :]
        head = self.action_heads[DecoderOutputKey.ACTION_LOGITS.value]
        logits = head(last_output)
        return self._sample_next_action_token(logits=logits)

    def _prepare_next_autoregressive_inputs(
        self,
        generated_output: torch.Tensor,
    ) -> torch.Tensor:
        """Embed sampled action-token IDs for the next cached GPT step."""
        return self.token_embedding(generated_output)

    def _get_completed_sequence_mask(
        self,
        generated_output: torch.Tensor,
        state: CachedAutoregressiveGenerationState,
    ) -> torch.Tensor:
        """Update the per-sample EOS mask for discrete action generation."""
        completed = generated_output.squeeze(1) == self.tokenizer.eos_token_id
        if state.completed_sequence_mask is None:
            return completed
        return state.completed_sequence_mask | completed

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

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Training: Teacher forcing with ground truth tokens
        Inference: Autoregressive generation with KV caching

        Args:
            features: Encoded features from pipeline
            actions: Ground truth tokenized actions (training) or None (inference)

        Returns:
            Dict with DecoderOutputKey.ACTION_LOGITS.value (training) or DecoderOutputKey.PREDICTED_ACTION_TOKENS.value (inference)
        """
        self._validate_action_tokenizer_is_set()
        feature_tokens, pos_encodings, feature_token_mask = self.input_sequence_builder(
            features
        )  # (B, token_len, embedding_dimension)
        feature_tokens = (
            feature_tokens + pos_encodings
            if pos_encodings is not None
            else feature_tokens
        )
        if actions is not None:
            predictions = self._forward_training(
                feature_tokens=feature_tokens,
                feature_token_mask=feature_token_mask,
                actions=actions,
            )
        else:
            predictions = self._forward_inference(
                feature_tokens=feature_tokens, feature_token_mask=feature_token_mask
            )

        return predictions

    def _forward_training(
        self,
        actions: dict[str, torch.Tensor],
        feature_tokens: torch.Tensor,
        feature_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward with teacher forcing.

        Args:
            feature_tokens: Feature token embeddings (B, feat_token_len, emb_dim)
            feature_token_mask: Optional feature token mask (B, feat_token_len)
            actions: Ground truth actions

        Returns:
            Dict with DecoderOutputKey.ACTION_LOGITS.value and tokenized targets
        """
        prefix_len = feature_tokens.shape[1]
        target_token_ids = self._get_target_token_ids(
            actions=actions,
            batch_size=feature_tokens.shape[0],
        )
        action_token_embeddings = self.token_embedding(target_token_ids)
        action_bos_embedding = self._expand_action_bos_embedding(
            batch_size=feature_tokens.shape[0],
            device=action_token_embeddings.device,
            dtype=action_token_embeddings.dtype,
        )
        action_input_embeddings = torch.cat(
            [action_bos_embedding, action_token_embeddings],
            dim=1,
        )
        full_attention_mask, _ = make_attention_mask(
            feature_tokens=feature_tokens,
            action_tokens=action_input_embeddings,
            feature_token_mask=feature_token_mask,
        )
        full_token_sequence = torch.cat(
            [feature_tokens, action_input_embeddings],
            dim=1,
        )
        if full_token_sequence.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Input token length {full_token_sequence.shape[1]} > "
                f"max_seq_len {self.max_seq_len}. No room for any action tokens. "
                "Consider increasing max_seq_len or reducing feature token count."
            )

        decoder_output, _ = self.gpt_decoder(
            hidden_states=full_token_sequence,
            encoded_features=None,
            cross_attention_mask=None,
            self_attention_mask=full_attention_mask,
        )
        action_outputs = decoder_output[:, prefix_len:-1, :]
        logits = self.action_heads[DecoderOutputKey.ACTION_LOGITS.value](
            action_outputs,
        )
        return {
            DecoderOutputKey.ACTION_LOGITS.value: logits,
        }

    def _forward_inference(
        self,
        feature_tokens: torch.Tensor,
        feature_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inference with autoregressive generation and KV caching.

        Args:
            feature_tokens: Feature token embeddings (B, num_features, D) or None
            feature_token_mask: Feature token mask (B, num_features) or None

        Returns:
            Dict with tokenized action predictions
        """
        prefix_len = feature_tokens.shape[1]
        if prefix_len + 1 >= self.max_seq_len:
            raise ValueError(
                f"Input prefix token length {prefix_len} plus BOS token >= "
                f"max_seq_len {self.max_seq_len}. No room for generated action tokens. "
                "Consider increasing max_seq_len or reducing feature token count."
            )
        generation_cache = self.gpt_decoder.create_empty_generation_cache(
            batch_size=feature_tokens.shape[0],
            device=feature_tokens.device,
            dtype=feature_tokens.dtype,
        )
        _, generation_cache = self.gpt_decoder(
            hidden_states=feature_tokens,
            encoded_features=None,
            self_attention_mask=self._build_prefix_self_attention_mask(
                prefix_tokens=feature_tokens,
                causal_prefix_suffix_length=0,
            ),
            key_padding_mask=feature_token_mask,
            cross_attention_mask=None,
            generation_cache=generation_cache,
        )
        next_inputs = self._expand_action_bos_embedding(
            batch_size=feature_tokens.shape[0],
            device=feature_tokens.device,
            dtype=feature_tokens.dtype,
        )
        initial_state = CachedAutoregressiveGenerationState(
            step_index=0,
            sequence_length=prefix_len,
            past_key_values=generation_cache,
            next_inputs=next_inputs,
        )
        return self._run_cached_autoregressive_generation(
            initial_state=initial_state,
            max_generation_steps=self._get_max_generation_steps(
                available_context_steps=self.max_seq_len - prefix_len - 1,
            ),
        )
