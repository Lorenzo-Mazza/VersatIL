"""Transformer-based VAE latent action encoder."""

import torch

from refactoring.models.decoding.constants import LOGVAR_KEY, MU_KEY, STATE_FEATURE_KEYS, LATENT_KEY
from refactoring.models.decoding.latent.base_posterior import LatentActionEncoder
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.detr_transformer.vae_transformer import VAE


class VAETransformerEncoder(LatentActionEncoder):
    """Transformer-based Variational Autoencoder for encoding actions into latent space.

    Args:
        embedding_dimension: Transformer hidden dimension
        number_of_heads: Number of attention heads
        feedforward_dimension: Feedforward network dimension
        number_of_encoder_layers: Number of transformer encoder layers
        activation: Activation function name
        dropout_rate: Dropout probability
        normalize_before: Use pre-normalization
        latent_dimension: Dimension of VAE latent space (z)
        use_proprioceptive: Whether to condition on proprioceptive observations
        prediction_horizon: Number of action timesteps
        device: Device to place encoder on
    """

    def __init__(
        self,
        embedding_dimension: int,
        latent_dimension: int,
        prediction_horizon: int,
        observation_horizon: int,
        device: str,
        number_of_heads: int = 8,
        feedforward_dimension: int = 512,
        number_of_encoder_layers: int = 4,
        activation: str = ActivationFunction.SWIGLU.value,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        use_proprioceptive: bool = False,
        exclude_keys: list[str] = None,
    ):
        """Initialize VAE latent action encoder.

        Args:
            embedding_dimension: Dimension of the output embedding
            latent_dimension: Dimension of VAE latent space, i.e. the dimension of the output z
            prediction_horizon: Number of action timesteps
            observation_horizon: Number of observation timesteps
            device: Device to place encoder on
            number_of_heads: Number of attention heads
            feedforward_dimension: Feedforward network dimension
            number_of_encoder_layers: Number of transformer encoder layers
            activation: Activation function name
            dropout_rate: Dropout probability
            normalize_before: Use pre-normalization
            use_proprioceptive: Whether to condition on proprioceptive observations
            exclude_keys: List of keys to exclude from encoding
        """
        super().__init__(latent_dimension=latent_dimension, device=device, )
        self.exclude_keys = exclude_keys if exclude_keys is not None else []
        self.embedding_dimension = embedding_dimension
        self.use_proprioceptive = use_proprioceptive
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon
        self.vae = VAE(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_encoder_layers=number_of_encoder_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            latent_dimension=latent_dimension,
            use_proprioceptive=use_proprioceptive,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            device=device,
        )
        self.to(device)


    def encode(
        self,
        actions: dict[str, torch.Tensor],
        observations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        """Encode actions into latent space using VAE.

        Args:
            actions: Dictionary of action tensors
                Shape: (B, prediction_horizon, action_dim) for each component
            observations: Optional observation features to condition encoding

        Note:
            Image observations are automatically excluded from encoding, plus any additional custom key.

        Returns:
            Dictionary containing:
                - LATENT_KEY: Latent embedding z (B, vae_latent_dimension)
                - MU_KEY: Latent distribution mean (B, vae_latent_dimension)
                - LOGVAR_KEY: Latent distribution log variance (B, vae_latent_dimension)
                - STATE_FEATURE_KEYS: Input observations used for encoding (dict or None)
        """
        if observations is not None:
            input_observations = {
                k: v for k, v in observations.items()
                if not (
                        k in self.exclude_keys # Custom excluded keys
                        or (v.ndim == 4 and self.observation_horizon == 1) # Image observation are excluded by default
                        or v.ndim == 5
                )
            }
        else:
            input_observations = None
        z, mu, logvar = self.vae(
            inputs=actions,
            observations=input_observations,
        )
        return {
            LATENT_KEY: z,
            MU_KEY: mu,
            LOGVAR_KEY: logvar,
            STATE_FEATURE_KEYS: input_observations
        }
