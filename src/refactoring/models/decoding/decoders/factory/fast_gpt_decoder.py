"""Action GPT Decoder for tokenized action prediction.

Similarly to the FAST-pi0 model, it uses a GPT-style autoregressive decoder (only self-attention)
to generate sequences of tokenized actions.
"""
import logging

import torch
import torch.nn as nn
from tqdm.contrib.logging import logging_redirect_tqdm

from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.data.constants import (
    IS_PAD_ACTION_KEY, TOKENIZED_ACTIONS_KEY,
)
from refactoring.data.tokenization import ActionTokenizer, Tokenizer
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.decoding.constants import (
    ACTION_LOGITS_KEY,
    PREDICTED_ACTION_TOKENS_KEY,
    LOGVAR_KEY,
    MU_KEY,
)
from refactoring.models.decoding.decoders.base import ActionDecoder, DecoderInput
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.constants import AttentionType, PositionalEncodingType
from refactoring.models.layers.normalization.constants import NormalizationType
from refactoring.models.layers.gpt_transformer.gpt_decoder import GPTDecoder
from refactoring.models.layers.positional_encoding.learned import LearnedPositionalEncoding1D
from refactoring.models.layers.positional_encoding.sinusoidal import SinusoidalPositionalEncoding2D, SinusoidalPositionalEncoding1D
from refactoring.models.layers.transformer_input_builder import TransformerInputBuilder


class FASTGPTDecoder(ActionDecoder):
    """FAST GPT decoder for tokenized action prediction.

    Uses pure GPT-style transformer with self-attention only (no cross-attention).
    Observation features are concatenated as prefix tokens, followed by
    action token embeddings for autoregressive generation.
    This is similar to Pi0 FAST but adapted to work with any feature encoder.
    """

    supports_tokenized_actions: bool = True

    def __init__(
        self,
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
        action_heads: None = None,
    ):
        """Initialize FAST GPT decoder.

        Args:
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
            action_heads: Not used, placeholder for compatibility
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.observation_horizon = observation_horizon
        self.device = device
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
        self.temperature = temperature
        self.learnable_temperature = learnable_temperature
        self.deterministic = deterministic
        action_heads = {
            ACTION_LOGITS_KEY: nn.Linear(1,1),  # Placeholder, will be replaced in set_tokenizer
        }
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
        )
        self.temperature = nn.Parameter(
            torch.tensor(temperature, dtype=torch.float32),
            requires_grad=learnable_temperature,
        )
        self.token_embedding = None # Will be set in set_tokenizer
        self.vocab_size = None
        self._build_transformer_components()
        self.to(self.device)


    def _build_transformer_components(self):
        """Build core transformer encoder-decoder and positional encodings."""
        image_positional_encoding = SinusoidalPositionalEncoding2D(
            embedding_dimension=self.embedding_dimension,
            normalize=True
        )
        temporal_positional_encoding = None
        if self.observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(embedding_dimension=self.embedding_dimension)

        # This layer transforms input features into a sequence of token embeddings + positional encodings
        self.input_sequence_builder = TransformerInputBuilder(
            embedding_dim=self.embedding_dimension,
            has_time_dim=self.observation_horizon > 1,
            spatial_positional_encoding_layer=image_positional_encoding,
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension),
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


    def set_tokenizer(self, tokenizer: Tokenizer | None = None):
        """Set tokenizer and adjust vocabulary size accordingly."""
        if tokenizer is None or tokenizer.action_tokenizer is None:
            raise ValueError("FASTGPTDecoder requires a tokenizer for tokenized action prediction.")
        device = self.temperature.device
        self.vocab_size = tokenizer.action_tokenizer.vocab_size
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dimension).to(device)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.gpt_decoder.initializer_range)
        lm_head = nn.Linear(self.embedding_dimension, self.vocab_size, bias=False, device=device)
        lm_head.weight = self.token_embedding.weight  # tie output weights to input embedding weights, like in GPT-2
        self.action_heads[ACTION_LOGITS_KEY] = lm_head
        super().set_tokenizer(tokenizer)


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
            Dict with ACTION_LOGITS_KEY (training) or PREDICTED_ACTION_TOKENS_KEY (inference)
        """
        feature_tokens, pos_encodings, feature_token_mask = self.input_sequence_builder(features) # (B, token_len, embedding_dimension)
        feature_tokens = feature_tokens + pos_encodings if pos_encodings is not None else feature_tokens
        if actions is not None:
            predictions = self._forward_training(feature_tokens=feature_tokens, feature_token_mask=feature_token_mask,
                                                 actions=actions)
        else:
            predictions = self._forward_inference(feature_tokens=feature_tokens, feature_token_mask=feature_token_mask)

        for key in [MU_KEY, LOGVAR_KEY]:
            if key in features:
                predictions[key] = features[key]

        return predictions

    @staticmethod
    def _make_attention_mask(
                             action_tokens: torch.Tensor,
                             feature_tokens: torch.Tensor,
                             feature_token_mask: torch.Tensor| None = None,
                             ) -> torch.Tensor:
        """Compute attention mask with bidirectional prefix and causal actions.

        Args:
            feature_tokens: Feature token embeddings (B, feat_token_len, emb_dim)
            action_tokens: Action token embeddings (B, action_token_len, emb_dim)
            feature_token_mask: Optional feature token mask (B, feat_token_len)

        Note: True indicates masked tokens, False indicates valid tokens. Action padding is handled in loss computation.
        """
        batch_size = feature_tokens.shape[0]
        prefix_len = feature_tokens.shape[1]
        action_len = action_tokens.shape[1]
        total_len = prefix_len + action_len
        if feature_token_mask is None:
            feature_token_mask = torch.zeros(batch_size, prefix_len, dtype=torch.bool, device=feature_tokens.device)
        action_ar_pad_mask = torch.triu(
            torch.ones(action_tokens.shape[1], action_tokens.shape[1], device=action_tokens.device, dtype=torch.bool),
            diagonal=1 # `True`s start on the main diagonal, i.e. don't attend to current and future action tokens
        ).unsqueeze(0).unsqueeze(0) #(1, 1, action_len, action_len)
        action_ar_pad_mask = action_ar_pad_mask.expand(batch_size, 1, action_tokens.shape[1], action_tokens.shape[1])  # (B, 1, action_len, action_len)
        full_padding_mask = torch.zeros(batch_size, 1, total_len, total_len, dtype=torch.bool, device=feature_tokens.device)
        full_padding_mask[:, :, prefix_len:, prefix_len:] = action_ar_pad_mask
        key_padding_mask = torch.cat(
            (feature_token_mask, torch.zeros(batch_size, action_len, dtype=torch.bool, device=feature_tokens.device)),
            dim=1
        )  # (B, total_len)
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, total_len)
        full_padding_mask = full_padding_mask | key_padding_mask.expand(-1, -1, total_len, -1)  # (B, 1, total_len, total_len)
        full_padding_mask[:, :, :prefix_len, prefix_len:] = True  # Prefix tokens cannot attend to future action tokens
        return full_padding_mask



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
            Dict with ACTION_LOGITS_KEY and tokenized targets
        """
        prefix_len = feature_tokens.shape[1]
        target_token_ids = actions[TOKENIZED_ACTIONS_KEY]  # (B, action_token_len)
        input_ids = target_token_ids[:, :-1]  # (B, seq_len-1)
        action_token_embeddings = self.token_embedding(input_ids)  # (B, action_token_len, emb_dim)
        # query_len = prefix_len + action_token_len
        full_attention_mask = self._make_attention_mask(
            feature_tokens=feature_tokens,
            action_tokens=action_token_embeddings,
            feature_token_mask=feature_token_mask,
        )  # (B, query_len, query_len)
        full_token_sequence = torch.cat([feature_tokens, action_token_embeddings], dim=1) # (B, query_len, emb_dim)
        if full_token_sequence.shape[1]>self.max_seq_len:
            raise ValueError(f"Input token length {full_token_sequence.shape[1]} >= max_seq_len {self.max_seq_len}. "
                "No room for any action tokens. "
                "Consider increasing max_seq_len or reducing feature token count.")

        decoder_output, _ = self.gpt_decoder(
            hidden_states=full_token_sequence,
            encoded_features=None,
            cross_attention_mask=None,
            decoder_cache=None,
            use_cache=False,
            self_attention_mask=full_attention_mask,
        )  # (B, query_len, D)
        action_outputs = decoder_output[:, prefix_len:, :]  # (B, action_token_len, D)
        logits = self.action_heads[ACTION_LOGITS_KEY](action_outputs)  # (B, action_token_len, vocab_size)
        return {
            ACTION_LOGITS_KEY: logits,
        }

    def _forward_inference(
        self,
        feature_tokens: torch.Tensor,
        feature_token_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Inference with autoregressive generation and KV caching.

        Args:
            feature_tokens: Feature token embeddings (B, num_features, D) or None
            feature_token_mask: Feature token mask (B, num_features) or None

        Returns:
            Dict with continuous action predictions
        """
        batch_size = feature_tokens.shape[0]
        prefix_len = feature_tokens.shape[1]
        current_sequence = feature_tokens
        prefix_self_mask = torch.zeros(batch_size, 1, prefix_len, prefix_len, dtype=torch.bool, device=self.device)
        decoder_output, decoder_cache = self.gpt_decoder(
            hidden_states=current_sequence,
            encoded_features=None,
            self_attention_mask=prefix_self_mask, # First mask only to avoid a causal effect within prefix
            key_padding_mask=feature_token_mask, # (B, prefix_len) or None
            cross_attention_mask=None,
            decoder_cache=None,
            use_cache=True,
        )
        generated_tokens = []
        next_token_embedding = None
        for step in range(self.max_seq_len - prefix_len):
            if step > 0:
                decoder_output, decoder_cache = self.gpt_decoder(
                    hidden_states=next_token_embedding,
                    self_attention_mask=None, # Causal mask handled internally
                    decoder_cache=decoder_cache,
                    use_cache=True,
                )
            last_output = decoder_output[:, -1:, :]  # (B, 1, embedding_dimension)
            head = self.action_heads[ACTION_LOGITS_KEY]
            logits = head(last_output)  # (B, 1, vocab_size)
            logits_scaled = logits / self.temperature.clamp(min=0.01)
            if self.deterministic:
                next_token = torch.argmax(logits, dim=-1)  # (B, 1)
            else:
                probs = torch.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(probs.squeeze(-1), num_samples=1)  # (B, 1)
            next_token_embedding = self.token_embedding(next_token)  # (B, 1, embedding_dimension)
            generated_tokens.append(next_token)

        return {
            PREDICTED_ACTION_TOKENS_KEY: torch.cat(generated_tokens, dim=1)  # (B, max_seq_len)
        }

