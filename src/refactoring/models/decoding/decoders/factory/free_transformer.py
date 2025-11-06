"""Free Transformer architecture for action decoding with variational latent codes.

Based on "The Free Transformer" (Fleuret, 2025) - arXiv:2510.17558
https://arxiv.org/abs/2510.17558

The Free Transformer encodes trajectory style/mode in a discrete latent variable
and generates action chunks non-autoregressively.
"""

import torch
from torch import nn

from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.data.constants import IS_PAD_KEY
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.decoding.constants import FeatureType, LATENT_KEY, BINARY_LOGITS_KEY
from refactoring.models.decoding.decoders import ActionDecoder, DecoderInput
from refactoring.models.layers.feature_projection import FeatureProjection
from refactoring.models.layers.free_transformer import (
    FreeTransformerEncoder,
    FreeTransformerDecoder,
)


class FreeTransformer(ActionDecoder):
    """Free Transformer for action decoding with discrete latent codes.

    Architecture:
    - **Encoder** (training only): Processes [obs, GT actions] → extracts latent encoding trajectory style/mode
    - **Decoder** (always): Processes [obs, action queries] with latent injection → generates actions non-autoregressively

    During training:
    - Encoder sees [obs, GT actions] to infer posterior latent p(z|obs,actions)
    - Decoder generates [obs, queries] conditioned on encoder's latent

    During inference:
    - Sample latent from uniform prior p(z)
    - Decoder generates [obs, queries] conditioned on sampled latent

    Args:
        input_keys: List of feature keys required from encoding pipeline
        action_space: Action space configuration
        action_heads: Dictionary of action prediction heads
        observation_space: Observation space configuration
        observation_horizon: Number of observation timesteps
        prediction_horizon: Number of action timesteps to predict
        device: Device for computation
        embedding_dimension: Model embedding dimension
        number_of_heads: Number of attention heads
        feedforward_dimension: FFN hidden dimension
        number_of_decoder_layers: Total decoder layers (must be even for latent injection at midpoint)
        number_of_encoder_layers: Number of encoder layers (training only)
        latent_bits: Number of bits for latent codes (2^bits total codes, default 16 → 65536)
        dropout_rate: Dropout probability
        use_rope: Whether to use RoPE
        rope_base: Base frequency for RoPE
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
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        feedforward_dimension: int = 1024,
        number_of_decoder_layers: int = 6,
        number_of_encoder_layers: int = 1,
        latent_bits: int = 16,
        dropout_rate: float = 0.1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    ):
        decoder_input = DecoderInput(
            keys=input_keys,
            requires_actions=True,
            raises_for_types=[FeatureType.SPATIAL.value]
        )
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
        self.prediction_horizon = prediction_horizon
        self.latent_bits = latent_bits
        self.latent_dim = 2**latent_bits

        self.flat_feature_projection = FeatureProjection(
            embedding_dim=embedding_dimension,
            warn_on_projection=True,
            raise_on_mismatch=False,
        )

        self.embedding_projection = nn.LazyLinear(embedding_dimension)
        self.action_embedding = nn.Linear(action_space.get_total_action_dim(), embedding_dimension)

        self.encoder = FreeTransformerEncoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_encoder_layers,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            latent_bits=latent_bits,
            dropout=dropout_rate,
            use_rope=use_rope,
            rope_base=rope_base,
        )

        self.decoder = FreeTransformerDecoder(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_decoder_layers,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            latent_dim=self.latent_dim,
            dropout=dropout_rate,
            use_rope=use_rope,
            rope_base=rope_base,
            causal=False,
        )

        self.action_queries = nn.Embedding(prediction_horizon, embedding_dimension)

    def _prepare_sequential_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and project sequential features (proprioceptive, language, etc.).

        Args:
            features: Dictionary of encoded features

        Returns:
            Concatenated sequential features (B, T_observation, total_embedding_dimension)
        """
        sequential_features_dict = {}
        for key, feature in features.items():
            if len(feature.shape) == 3:
                sequential_features_dict[key] = feature  # Keep full T if history
            elif len(feature.shape) == 2:
                sequential_features_dict[key] = feature.unsqueeze(1)  # (B,1,D)

        if len(sequential_features_dict) == 0:
            raise ValueError("No flat or sequential features found. Free Transformer requires at least 1 observation feature as input.")

        return self.flat_feature_projection.project_and_concatenate(
            sequential_features_dict,
            concatenation_dimension=-1,
        )

    def _encode_latent(
        self,
        gt_actions: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode latent from observations and ground-truth actions (training only).

        Args:
            gt_actions: Dictionary of ground-truth actions

        Returns:
            Tuple of (latent_codes, binary_logits)
        """
        action_keys = [key for key in sorted(gt_actions.keys()) if key != IS_PAD_KEY]
        gt_actions_concat = torch.cat([gt_actions[key] for key in action_keys], dim=-1)
        action_embeds = self.action_embedding(gt_actions_concat)
        # Self-attend action embeddings to get encoder mid-features
        encoder_mid = self.decoder.forward_to_mid(source=action_embeds, memory=action_embeds, key_padding_mask=None)
        latent_codes, binary_logits = self.encoder(
            mid_decoder_features=encoder_mid,
            key_padding_mask=None,
            deterministic=False
        )
        return latent_codes, binary_logits

    def _sample_prior_latent(self, batch_size: int) -> torch.Tensor:
        """Sample latent from uniform prior (inference only).

        Args:
            batch_size: Batch size

        Returns:
            Sampled latent codes (B, sequence_length, latent_dim)
        """
        latent_codes = torch.zeros(batch_size, self.prediction_horizon, self.latent_dim, device=self.device)
        random_indices = torch.randint(0, self.latent_dim, (batch_size, self.prediction_horizon), device=self.device)
        latent_codes.scatter_(2, random_indices.unsqueeze(-1), 1.0)
        return latent_codes

    def _decode_actions(
        self,
        latent_codes: torch.Tensor,
        observation_embeddings: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Decode actions from observations conditioned on latent.

        Args:
            latent_codes: Latent codes (B, prediction_horizon, latent_dim)
            observation_embeddings: Observation embeddings (B, T, embedding_dimension)
            batch_size: Batch size

        Returns:
            Action embeddings (B, prediction_horizon, embedding_dimension)
        """
        action_queries = self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        action_embeddings, _ = self.decoder(
            x=action_queries,
            latent=latent_codes,
            memory=observation_embeddings,
            key_padding_mask=None,
            return_mid_features=False,
        )
        return action_embeddings

    def _apply_action_heads(self, action_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply prediction heads to action embeddings.

        Args:
            action_embeddings: Action embeddings (B, horizon, embedding_dimension)

        Returns:
            Dictionary of predicted actions
        """
        predictions = {}
        for action_key, head in self.action_heads.items():
            predictions[action_key] = head(action_embeddings)
        return predictions

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass of Free Transformer.

        Args:
            features: Dictionary of encoded features from EncodingPipeline
                Expected to contain flat features (B, D) or (B, T, D)
            actions: Ground-truth actions (required during training)

        Returns:
            Dictionary containing:
                - Action head predictions (e.g. position, orientation, gripper)
                - latent: Latent codes used for generation
                - binary_logits: Raw logits for KL divergence (training only)
        """
        for key, feature in features.items():
            if (len(feature.shape) == 4 and not self.has_history) or (len(feature.shape) == 5 and self.has_history):
                raise ValueError(
                    "Free Transformer does not support spatial features. "
                    "Please flatten your features before passing them to the decoder."
                )

        feature_vector = self._prepare_sequential_features(features=features)
        obs_embedding = self.embedding_projection(feature_vector)
        batch_size = obs_embedding.size(0)
        if self.training:
            if actions is None:
                raise ValueError("Ground-truth actions required during training for posterior computation.")

            latent_codes, binary_logits = self._encode_latent(actions)
            action_embeddings = self._decode_actions(latent_codes=latent_codes, batch_size=batch_size, observation_embeddings=obs_embedding)
        else:
            latent_codes = self._sample_prior_latent(batch_size)
            action_embeddings = self._decode_actions(latent_codes=latent_codes, batch_size=batch_size, observation_embeddings=obs_embedding)
            binary_logits = None

        predictions = self._apply_action_heads(action_embeddings)
        predictions[LATENT_KEY] = latent_codes
        if binary_logits is not None:
            predictions[BINARY_LOGITS_KEY] = binary_logits

        return predictions
