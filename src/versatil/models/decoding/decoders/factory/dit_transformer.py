"""DiT (Diffusion Transformer) Decoder for action generation.

Reference implementation based on DiT architecture with modulation-based conditioning.
"""

import logging
from typing import Optional

import torch
from torch import nn

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.constants import FeatureType, TIMESTEP_KEY
from versatil.models.decoding.decoders.base import DecoderInput
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.layers.dit.dit_decoder import DiTDecoder
from versatil.models.layers.dit.dit_encoder import DiTEncoder, DiTEncoderLayer
from versatil.models.layers.dit.dit_decoder_layer import DiTDecoderLayer
from versatil.models.layers.dit.timestep_embedding import TimestepEmbeddingNetwork
from versatil.models.layers.feature_projection import FeatureProjection
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding2D,
    SinusoidalPositionalEncoding1D,
)
from versatil.models.layers.transformer_input_builder import TransformerInputBuilder


class DiTTransformerDecoder(ActionDecoder):
    """DiT (Diffusion Transformer) decoder for generative action generation.

    This architecture:
    - Uses encoder-decoder transformer with DiT-style modulation
    - Conditions decoder layers on encoder outputs via modulation (not cross-attention)
    - Processes observation tokens through encoder, action tokens through decoder
    - Designed for use with Diffusion algorithm

    The decoder expects:
    - Noisy actions as input (via actions parameter during forward)
    - Timesteps injected by the algorithm (via features[TIMESTEP_KEY])
    - Observation features for encoding (via features dict)

    Note: This decoder is specifically designed for diffusion algorithms
    and expects the algorithm to handle noise scheduling and timestep injection.
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
        embedding_dimension: int = 512,
        num_heads: int = 8,
        num_blocks: int = 6,
        feedforward_dimension: int = 2048,
        dropout: float = 0.1,
        activation_name: str = "gelu",
        timestep_embedding_dim: int = 256,
        normalization_type: str = NormalizationType.ADALN.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
    ):
        """Initialize DiT Transformer decoder.

        Args:
            input_keys: List of feature keys expected from encoder pipeline
            action_space: Action space configuration
            action_heads: Dictionary of action head modules
            observation_space: Observation space configuration
            observation_horizon: Number of observation timesteps (for history)
            prediction_horizon: Number of actions to predict (horizon)
            device: Device to run the model on
            embedding_dimension: Transformer hidden dimension
            num_heads: Number of attention heads
            num_blocks: Number of encoder/decoder blocks
            feedforward_dimension: Feedforward network dimension
            dropout: Dropout probability
            activation_name: Activation function name
            timestep_embedding_dim: Diffusion timestep embedding dimension
            normalization_type: Normalization type for transformer layers
            attention_type: Attention type for custom attention layer
        """
        decoder_input = DecoderInput(
            keys=input_keys,
            raises_for_types=[FeatureType.SPATIAL.value],
            requires_actions=True,
        )

        # Action heads are not used by DiT (it handles all processing internally)
        for k, head in action_heads.items():
            if len(head.blocks) > 0:
                logging.warning(
                    f"Action heads are ignored by DiTTransformerDecoder, but one was provided for action '{k}'. Skipping."
                )
                action_heads[k].blocks = nn.ModuleList()

        super().__init__(
            decoder_input=decoder_input,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            device=device,
        )

        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.feedforward_dimension = feedforward_dimension
        self.dropout = dropout
        self.activation_name = activation_name
        self.timestep_embedding_dim = timestep_embedding_dim
        self.normalization_type = normalization_type
        self.attention_type = attention_type

        # Feature input builder: concatenates features into token sequence
        image_positional_encoding = SinusoidalPositionalEncoding2D(
            embedding_dimension=embedding_dimension, normalize=True
        )
        temporal_positional_encoding = None
        if self.observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(
                embedding_dimension=embedding_dimension
            )
        self.input_builder = TransformerInputBuilder(
            embedding_dim=embedding_dimension,
            has_time_dim=self.observation_horizon > 1,
            spatial_positional_encoding_layer=image_positional_encoding,
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=embedding_dimension
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
        )

        # Positional encodings for decoder (learnable)
        self.register_parameter(
            "decoder_positional_encoding",
            nn.Parameter(
                torch.empty(prediction_horizon, 1, embedding_dimension), requires_grad=True
            ),
        )
        nn.init.xavier_uniform_(self.decoder_positional_encoding.data)

        # Timestep embedding network
        self.timestep_embedding_network = TimestepEmbeddingNetwork(
            timestep_embedding_dim=timestep_embedding_dim,
            output_dim=embedding_dimension,
        )

        # Action projection: projects actions to embedding dimension
        self.action_projection = nn.Sequential(
            nn.Linear(self.action_dim, self.action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.action_dim, embedding_dimension),
        )

        # Encoder: stack of self-attention blocks for observation tokens
        encoder_base_block = DiTEncoderLayer(
            embedding_dim=embedding_dimension,
            num_heads=num_heads,
            feedforward_dim=feedforward_dimension,
            dropout=dropout,
            activation_name=activation_name,
            normalization_type=normalization_type,
            attention_type=attention_type,
        )
        self.encoder = DiTEncoder(encoder_base_block, num_blocks)

        # Decoder: stack of modulated self-attention blocks for action tokens
        decoder_base_block = DiTDecoderLayer(
            embedding_dim=embedding_dimension,
            num_heads=num_heads,
            feedforward_dim=feedforward_dimension,
            dropout=dropout,
            activation_name=activation_name,
            normalization_type=normalization_type,
            attention_type=attention_type,
        )
        self.decoder = DiTDecoder(
            base_block=decoder_base_block,
            num_layers=num_blocks,
            action_dim=self.action_dim,
            hidden_dim=embedding_dimension,
        )

        # Encoder cache for inference optimization
        self._encoder_cache: Optional[list[torch.Tensor]] = None

        self.to(self.device)
        logging.info(
            f"Initialized DiTTransformerDecoder with {sum(p.numel() for p in self.parameters()):e} parameters"
        )

    def _prepare_observation_tokens(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Concatenate and prepare observation features as token sequence.

        Args:
            features: Dictionary of encoded features from the encoding pipeline

        Returns:
            Tuple of (observation_tokens, positional_encodings) where:
                - observation_tokens: (batch_size, num_tokens, embedding_dimension)
                - positional_encodings: (batch_size, num_tokens, embedding_dimension) or None
        """
        # Use TransformerInputBuilder to concatenate features
        # Returns (B, Total_Seq, Emb)
        observation_tokens, positional_encodings, _ = self.input_builder(features)

        if observation_tokens is None:
            raise ValueError("No valid observation features provided to DiT decoder")

        return observation_tokens, positional_encodings

    def _forward_encoder(
        self,
        observation_tokens: torch.Tensor,
        positional_encodings: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Encode observation tokens and return intermediate outputs.

        Args:
            observation_tokens: Observation tokens (batch_size, num_tokens, embedding_dimension)

        Returns:
            List of encoder layer outputs, each (num_tokens, batch_size, embedding_dimension)
        """
        if positional_encodings is None:
            raise ValueError(
                "Missing positional encodings from TransformerInputBuilder. "
                "Ensure positional encodings are configured for DiT inputs."
            )

        # Transpose to sequence-first for attention: (B, S, D) -> (S, B, D)
        observation_tokens_seq = observation_tokens.transpose(0, 1)

        positional_embedding = positional_encodings.transpose(0, 1)  # (S, B, D)

        # Encode and get intermediate outputs
        encoder_cache = self.encoder(observation_tokens_seq, positional_embedding)
        return encoder_cache

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the DiT transformer.

        This method is called by the decoding algorithm (Diffusion)
        which provides:
        - Noisy actions
        - Observation features dictionary containing the timestep key

        Args:
            features: Dictionary of encoded features from the encoding pipeline.
            actions: Dictionary of noise-injected actions (provided by algorithm during training)

        Returns:
            Dictionary containing denoised predictions for each action head

        Raises:
            ValueError: If timesteps or actions are missing.
        """
        if actions is None:
            raise ValueError(
                "DiTTransformerDecoder requires 'actions' parameter. "
                "The algorithm should provide noisy actions during forward pass."
            )

        if TIMESTEP_KEY not in features:
            raise ValueError(
                f"Missing '{TIMESTEP_KEY}' in features dict. "
                "The algorithm should inject timesteps into features."
            )

        timesteps = features.pop(TIMESTEP_KEY)  # (B,) or (B, 1)
        if len(timesteps.shape) == 2:
            timesteps = timesteps.squeeze(-1)

        # Prepare observation tokens
        observation_tokens, positional_encodings = self._prepare_observation_tokens(
            features
        )  # (B, S, D)

        # Encode observations (get intermediate outputs for decoder conditioning)
        encoder_cache = self._forward_encoder(
            observation_tokens, positional_encodings
        )

        # Concatenate all action modalities into single tensor
        # Shape: (B, T, action_dimension) where T = prediction_horizon
        action_tensors = []
        for action_key in sorted(actions.keys()):
            action_tensors.append(actions[action_key])
        noisy_actions = torch.cat(action_tensors, dim=-1)  # (B, T, total_action_dimension)

        # Project noisy actions to tokens
        action_tokens_bf = self.action_projection(noisy_actions)  # (B, T, D)

        # Transpose to sequence-first and add positional encoding
        action_tokens_seq = action_tokens_bf.transpose(0, 1)  # (T, B, D)
        decoder_input = action_tokens_seq + self.decoder_positional_encoding  # (T, B, D)

        # Get timestep embedding
        timestep_embedding = self.timestep_embedding_network(timesteps)  # (B, D)

        # Apply decoder blocks conditioned on encoder layers
        noise_prediction_bf = self.decoder(
            decoder_input, timestep_embedding, encoder_cache
        )  # (B, T, action_dim)

        # Split denoised output through action heads
        outputs = {}
        start_index = 0
        for action_key in sorted(actions.keys()):
            head = self.action_heads[action_key]
            end_index = start_index + head.output_dim
            action_slice = noise_prediction_bf[..., start_index:end_index]  # (B, T, action_dimension_i)
            outputs[action_key] = action_slice
            start_index = end_index

        return outputs

