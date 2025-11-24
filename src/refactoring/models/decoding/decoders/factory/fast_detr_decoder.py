"""DETR-style transformer for non-autoregressive tokenized action prediction."""
import torch
import torch.nn as nn

from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.data.tokenization.tokenizer import Tokenizer
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.decoding.constants import (
    ACTION_LOGITS_KEY,
    PREDICTED_ACTION_TOKENS_KEY,
)
from refactoring.models.decoding.decoders.base import ActionDecoder, DecoderInput, FeatureType
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.detr_transformer import Transformer
from refactoring.models.layers.positional_encoding.learned import LearnedPositionalEncoding1D
from refactoring.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding2D,
)
from refactoring.models.layers.transformer_input_builder import TransformerInputBuilder


class FASTDETRDecoder(ActionDecoder):
    """FAST DETR decoder for tokenized action prediction.

    Uses DETR-style non-autoregressive transformer to generate
     sequences of discrete action tokens.
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
        feedforward_dimension: int = 512,
        number_of_encoder_layers: int = 6,
        number_of_decoder_layers: int = 6,
        activation: str = ActivationFunction.RELU.value,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        deterministic: bool = True,
    ):
        """Initialize FAST decoder.

        Args:
            action_heads: Action heads for different action components (only ACTION_LOGITS_KEY used here).
            input_keys: Feature keys expected from encoder pipeline
            action_space: Action space configuration
            observation_space: Observation space configuration
            observation_horizon: Number of observation timesteps
            prediction_horizon: Max sequence length for generation
            device: Device to run model on
            max_seq_len: Maximum sequence length for positional encoding
            embedding_dimension: Transformer hidden dimension
            number_of_heads: Number of attention heads
            feedforward_dimension: FFN hidden dimension
            number_of_encoder_layers: Number of encoder layers for visual features
            number_of_decoder_layers: Number of decoder layers
            activation: Activation function
            dropout_rate: Dropout probability
            normalize_before: Use pre-normalization
            temperature: Initial temperature for sampling (not used in greedy decoding)
            learnable_temperature: If True, make temperature a learnable parameter
            deterministic: If True, use greedy decoding during inference
            action_heads: Not used, placeholder for compatibility
        """
        self.max_seq_len = max_seq_len
        self.embedding_dimension = embedding_dimension
        self.deterministic = deterministic

        if action_heads.keys() !={ACTION_LOGITS_KEY}:
            raise ValueError(f"FASTDETRDecoder only supports ACTION_LOGITS_KEY in action_heads. Make sure to use key {ACTION_LOGITS_KEY}"
                             " in your hydra config.")
        self.action_heads = action_heads
        decoder_input = DecoderInput(
            keys=input_keys,
            required_types=[FeatureType.SPATIAL.value],
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

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.feedforward_dimension = feedforward_dimension
        self.number_of_encoder_layers = number_of_encoder_layers
        self.number_of_decoder_layers = number_of_decoder_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.normalize_before = normalize_before
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
            flat_positional_encoding_layer=LearnedPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension,
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
        )
        self.action_decoder = Transformer(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.number_of_heads,
            number_of_encoder_layers=self.number_of_encoder_layers,
            number_of_decoder_layers=self.number_of_decoder_layers,
            activation=self.activation,
            dropout=self.dropout_rate,
            normalize_before=self.normalize_before,
            feedforward_dimension=self.feedforward_dimension,
        )
        # Learnable queries for action prediction
        self.learnable_query = nn.Embedding(self.max_seq_len, self.embedding_dimension)  # (max_seq_len, emb)


    def set_tokenizer(self, tokenizer: Tokenizer | None = None):
        """Set tokenizer and adjust vocabulary size accordingly."""
        if tokenizer is None or tokenizer.action_tokenizer is None:
            raise ValueError("FASTDETR Decoder requires a tokenizer for tokenized action prediction.")
        device = self.temperature.device
        self.vocab_size = tokenizer.action_tokenizer.vocab_size
        output_block_in_features = self.action_heads[ACTION_LOGITS_KEY].output_proj.in_features
        if output_block_in_features != self.embedding_dimension:
            token_input_embedding = nn.Embedding(self.vocab_size, output_block_in_features).to(device)
            token_projection = nn.Linear(output_block_in_features, self.embedding_dimension).to(device)
            self.token_embedding = nn.Sequential(
                token_input_embedding,
                token_projection
            ).to(device)
            nn.init.normal_(token_input_embedding.weight, mean=0.0, std=self.action_decoder.initializer_range)
            nn.init.normal_(token_projection.weight, mean=0.0, std=self.action_decoder.initializer_range)
        else:
            token_input_embedding = nn.Embedding(self.vocab_size, self.embedding_dimension).to(device)
            self.token_embedding = token_input_embedding
            nn.init.normal_(token_input_embedding.weight, mean=0.0, std=self.action_decoder.initializer_range)

        lm_head = nn.Linear(output_block_in_features, self.vocab_size, bias=False, device=device)
        lm_head.weight = token_input_embedding.weight  # tie output weights to input embedding weights, like in GPT-2
        self.action_heads[ACTION_LOGITS_KEY].output_dim = self.vocab_size
        self.action_heads[ACTION_LOGITS_KEY].output_proj = lm_head  # Replace final projection with tied head
        super().set_tokenizer(tokenizer)


    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the non-Autoregressive DETR action transformer (ACT style).

        Args:
            features: Encoded features from encoding pipeline.
            actions: Not used, here for compatibility reasons.

        Returns:
            Dict with logits over vocabulary (training) or predicted tokens (inference)
        """
        input_tokens, pos_encodings, padding_mask = self.input_sequence_builder(features) # (B, input_token_len, embedding_dimension)
        action_embeddings = self._decode_actions(input_tokens=input_tokens, positional_encodings=pos_encodings,
                                                 padding_mask=padding_mask)
        if self.train():
            head = self.action_heads[ACTION_LOGITS_KEY]
            logits = head(action_embeddings)  # (B, max_seq_len, vocab_size)
            return {
                ACTION_LOGITS_KEY: logits,
            }
        else:
            head = self.action_heads[ACTION_LOGITS_KEY]
            logits = head(action_embeddings)  # (B, max_seq_len, vocab_size)
            logits_scaled = logits / self.temperature.clamp(min=0.01)  # Prevent division by zero
            probs = torch.softmax(logits_scaled, dim=-1)  # (B, max_seq_len, vocab_size)
            if self.deterministic:
                pred_tokens = torch.argmax(logits, dim=-1)  # (B, max_seq_len) # Deterministic greedy decoding
            else:
                pred_tokens = torch.distributions.Categorical(probs).sample()  # (B, max_seq_len) - stochastic sampling
            return {
                PREDICTED_ACTION_TOKENS_KEY: pred_tokens  # (B, max_seq_len)
            }


    def _decode_actions(
            self,
            input_tokens: torch.Tensor,
            positional_encodings: torch.Tensor,
            padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run DETR non-causal transformer encoder-decoder to predict chunks of action token embeddings in parallel.

        Args:
            input_tokens: Input tokens to the action decoder, shape (B, obs_token_sequence_len, embedding_dimension)
            positional_encodings: Positional encodings, shape (B, obs_token_sequence_len, embedding_dimension)
            padding_mask: Optional padding mask for encoder tokens, shape (B, obs_token_sequence_len), where True indicates padding tokens.

        Returns:
            Predicted action token embeddings (B, max_seq_len, embedding_dimension)
        """
        batch_size = input_tokens.shape[0]
        query_positional_encoding = self.learnable_query.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (B, max_seq_len, emb)
        target = torch.zeros_like(query_positional_encoding)
        return self.action_decoder(
            source=input_tokens,
            target=target,
            source_positional_encoding=positional_encodings,
            source_key_padding_mask=padding_mask,
            target_positional_encoding=query_positional_encoding
        )[0]  # (B, max_seq_len, embedding_dimension)  type: ignore[no-any-return]



