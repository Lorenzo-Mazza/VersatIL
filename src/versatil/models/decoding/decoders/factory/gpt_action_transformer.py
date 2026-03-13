"""Action GPT Decoder for tokenized action prediction.

It uses a GPT-style autoregressive decoder (only self-attention) to generate sequences of tokenized actions.
"""

import torch
import torch.nn as nn

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import SampleKey
from versatil.data.tokenization import Tokenizer
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_masking import make_attention_mask
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.autoregressive_decoder import GPTDecoder
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding2D,
    SinusoidalPositionalEncoding1D,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder


class GPTActionTransformer(ActionDecoder):
    """Autoregressive decoder for tokenized action prediction.

    Uses pure GPT-style transformer with self-attention only (no cross-attention).
    Observation features are concatenated as prefix tokens, followed by
    action token embeddings for autoregressive generation.
    This is similar to Pi0 FAST but adapted to work with any feature encoder.
    """

    supports_tokenized_actions: bool = True

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
    ):
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
        if action_heads.keys() != {DecoderOutputKey.ACTION_LOGITS.value}:
            raise ValueError(
                f"GPTActionTransformer only supports DecoderOutputKey.ACTION_LOGITS.value in action_heads. Make sure to use key {DecoderOutputKey.ACTION_LOGITS.value}"
                " in your hydra config."
            )
        self.action_heads = action_heads
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
        self.token_embedding = None  # Will be set in set_tokenizer
        self.vocab_size = None
        self._build_transformer_components()
        self.to(self.device)

    def _build_transformer_components(self):
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
            embedding_dim=self.embedding_dimension,
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

    def set_tokenizer(self, tokenizer: Tokenizer | None = None):
        """Set tokenizer and adjust vocabulary size accordingly."""
        if tokenizer is None or tokenizer.action_tokenizer is None:
            raise ValueError(
                "GPTActionTransformer requires a tokenizer for tokenized action prediction."
            )
        device = self.temperature.device
        self.vocab_size = tokenizer.action_tokenizer.vocab_size
        output_block_in_features = self.action_heads[
            DecoderOutputKey.ACTION_LOGITS.value
        ].output_proj.in_features
        if output_block_in_features != self.embedding_dimension:
            token_input_embedding = nn.Embedding(
                self.vocab_size, output_block_in_features
            ).to(device)
            token_projection = nn.Linear(
                output_block_in_features, self.embedding_dimension
            ).to(device)
            self.token_embedding = nn.Sequential(
                token_input_embedding, token_projection
            ).to(device)
            nn.init.normal_(
                token_input_embedding.weight,
                mean=0.0,
                std=self.gpt_decoder.initializer_range,
            )
            nn.init.normal_(
                token_projection.weight,
                mean=0.0,
                std=self.gpt_decoder.initializer_range,
            )
        else:
            token_input_embedding = nn.Embedding(
                self.vocab_size, self.embedding_dimension
            ).to(device)
            self.token_embedding = token_input_embedding
            nn.init.normal_(
                token_input_embedding.weight,
                mean=0.0,
                std=self.gpt_decoder.initializer_range,
            )
        lm_head = nn.Linear(
            output_block_in_features, self.vocab_size, bias=False, device=device
        )
        lm_head.weight = (
            token_input_embedding.weight
        )  # tie output weights to input embedding weights, like in GPT-2
        self.action_heads[
            DecoderOutputKey.ACTION_LOGITS.value
        ].output_dim = self.vocab_size
        self.action_heads[
            DecoderOutputKey.ACTION_LOGITS.value
        ].output_proj = lm_head  # Replace final projection with tied head
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
            Dict with DecoderOutputKey.ACTION_LOGITS.value (training) or DecoderOutputKey.PREDICTED_ACTION_TOKENS.value (inference)
        """
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

        for key in [LatentKey.POSTERIOR_MU.value, LatentKey.POSTERIOR_LOGVAR.value]:
            if key in features:
                predictions[key] = features[key]

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
        target_token_ids = actions[
            SampleKey.TOKENIZED_ACTIONS.value
        ]  # (B, action_token_len)
        action_token_embeddings = self.token_embedding(
            target_token_ids
        )  # (B, action_token_len, emb_dim)
        # query_len = prefix_len + action_token_len
        full_attention_mask, _ = make_attention_mask(
            feature_tokens=feature_tokens,
            action_tokens=action_token_embeddings,
            feature_token_mask=feature_token_mask,
        )  # (B, query_len, query_len)
        full_token_sequence = torch.cat(
            [feature_tokens, action_token_embeddings], dim=1
        )  # (B, query_len, emb_dim)
        if full_token_sequence.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Input token length {full_token_sequence.shape[1]} >= max_seq_len {self.max_seq_len}. "
                "No room for any action tokens. "
                "Consider increasing max_seq_len or reducing feature token count."
            )

        decoder_output, _ = self.gpt_decoder(
            hidden_states=full_token_sequence,
            encoded_features=None,
            cross_attention_mask=None,
            decoder_cache=None,
            use_cache=False,
            self_attention_mask=full_attention_mask,
        )  # (B, query_len, D)
        # Shift alignment: grabs outputs from the last feature to the penultimate action so step t predicts target t+1.
        # NB: This is crucial for correct teacher forcing without information leakage from future tokens.
        # The first action token attends to all feature tokens but not future action tokens, etc.
        action_outputs = decoder_output[:, prefix_len - 1 : -1, :] # (B, action_token_len, D)
        logits = self.action_heads[DecoderOutputKey.ACTION_LOGITS.value](
            action_outputs
        )  # (B, action_token_len, vocab_size)
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
        batch_size = feature_tokens.shape[0]
        prefix_len = feature_tokens.shape[1]
        current_sequence = feature_tokens
        prefix_self_mask = torch.zeros(
            batch_size, 1, prefix_len, prefix_len, dtype=torch.bool, device=self.device
        )
        decoder_output, decoder_cache = self.gpt_decoder(
            hidden_states=current_sequence,
            encoded_features=None,
            self_attention_mask=prefix_self_mask,  # First mask only to avoid a causal effect within prefix
            key_padding_mask=feature_token_mask,  # (B, prefix_len) or None
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
                    self_attention_mask=None,  # Causal mask handled internally
                    decoder_cache=decoder_cache,
                    use_cache=True,
                )
            last_output = decoder_output[:, -1:, :]  # (B, 1, embedding_dimension)
            head = self.action_heads[DecoderOutputKey.ACTION_LOGITS.value]
            logits = head(last_output)  # (B, 1, vocab_size)
            logits_scaled = logits / self.temperature.clamp(min=0.01)
            if self.deterministic:
                next_token = torch.argmax(logits, dim=-1)  # (B, 1)
            else:
                probs = torch.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(
                    probs.squeeze(-1), num_samples=1
                )  # (B, 1)
            generated_tokens.append(next_token)
            if (next_token == self.tokenizer.pad_token_id).all(): # Check across batch
                break
            next_token_embedding = self.token_embedding(
                next_token
            )  # (B, 1, embedding_dimension)

        return {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.cat(
                generated_tokens, dim=1
            )  # (B, num_generated_tokens)
        }
