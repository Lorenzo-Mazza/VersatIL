"""Transformer-based VAE latent action encoder."""

import torch
from torch import nn

from refactoring.models.decoding.constants import LATENT_KEY, LOGVAR_KEY, MU_KEY, STATE_FEATURE_KEYS
from refactoring.models.decoding.latent.base_posterior import LatentActionEncoder
from refactoring.models.layers.detr_transformer.vae_transformer import VAE


class VAETransformerEncoder(LatentActionEncoder):
    """Transformer-based Variational Autoencoder for encoding actions into latent space.

    Args:
        output_dim: Transformer hidden dimension
        number_of_heads: Number of attention heads
        feedforward_dimension: Feedforward network dimension
        number_of_encoder_layers: Number of transformer encoder layers
        activation: Activation function name
        dropout_rate: Dropout probability
        normalize_before: Use pre-normalization
        latent_dim: Dimension of VAE latent space (z)
        use_proprioceptive: Whether to condition on proprioceptive observations
        prediction_horizon: Number of action timesteps
        device: Device to place encoder on
    """

    def __init__(
        self,
        output_dim: int,
        latent_dim: int,
        prediction_horizon: int,
        device: str,
        number_of_heads: int = 8,
        feedforward_dimension: int = 512,
        number_of_encoder_layers: int = 4,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        use_proprioceptive: bool = False,
    ):
        """Initialize VAE latent action encoder.

        Args:
            output_dim: Dimension of the output embedding
            latent_dim: Dimension of VAE latent space, i.e. the dimension of the output z
            prediction_horizon: Number of action timesteps
            device: Device to place encoder on
            number_of_heads: Number of attention heads
            feedforward_dimension: Feedforward network dimension
            number_of_encoder_layers: Number of transformer encoder layers
            activation: Activation function name
            dropout_rate: Dropout probability
            normalize_before: Use pre-normalization
            use_proprioceptive: Whether to condition on proprioceptive observations
        """
        super().__init__(latent_dim=latent_dim, device=device, output_dim=output_dim)

        self.embedding_dimension = output_dim
        self.use_proprioceptive = use_proprioceptive
        self.prediction_horizon = prediction_horizon
        self.vae = VAE(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_encoder_layers=number_of_encoder_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            vae_latent_dimension=latent_dim,
            use_state=use_proprioceptive,
            prediction_horizon=prediction_horizon,
            device=device,
        )
        # Latent to embedding projection, output dimension matches embedding_dimension
        self.latent_output_projection = nn.Linear(
            latent_dim,
            self.embedding_dimension
        )
        self.to(device)


    def encode(
        self,
        actions: dict[str, torch.Tensor],
        observations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode actions into latent space using VAE.

        Args:
            actions: Dictionary of action tensors
                Shape: (B, prediction_horizon, action_dim) for each component
            observations: Optional observation features (proprioceptive state)
                Should contain flattened proprioceptive features if use_proprioceptive=True

        Returns:
            Dictionary containing:
                - LATENT_KEY: Latent embedding (B, embedding_dimension)
                - MU_KEY: Latent distribution mean (B, vae_latent_dimension)
                - LOGVAR_KEY: Latent distribution log variance (B, vae_latent_dimension)
        """
        # Extract proprioceptive features if needed
        state_features = None
        if self.use_proprioceptive and observations is not None:
            # Concatenate all flat observation features
            # Assume observations contains flattened features from encoding pipeline
            flat_obs_list = [
                feat for feat in observations.values()
                if isinstance(feat, torch.Tensor) and feat.ndim == 2
            ]
            if flat_obs_list:
                state_features = torch.cat(flat_obs_list, dim=-1)

        z, mu, logvar = self.vae(
            inputs=actions,
            state_features=state_features,
        )
        latent_embedding = self.latent_output_projection(z)

        return {
            LATENT_KEY: latent_embedding,
            MU_KEY: mu,
            LOGVAR_KEY: logvar,
            STATE_FEATURE_KEYS: state_features
        }
