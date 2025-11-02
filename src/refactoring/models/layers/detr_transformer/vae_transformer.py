import logging

import torch
from torch import nn

from refactoring.data.constants import IS_PAD_KEY
from refactoring.models.layers.detr_transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from refactoring.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparametrization trick for VAE: sample from N(mu, var).

    Args:
        mu: Mean of the latent distribution, shape (batch, latent_dim).
        logvar: Log variance of the latent distribution, shape (batch, latent_dim).

    Returns:
        Sampled latent vector, shape (batch, latent_dim).
    """
    std = (logvar / 2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


class VAE(nn.Module):
    """VAE transformer architecture for encoding inputs into latent space.

    Handles input projection, transformer encoding with CLS token, and latent sampling and reparametrization.
    """
    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        feedforward_dimension: int,
        number_of_encoder_layers: int,
        activation: str,
        dropout_rate: float,
        normalize_before: bool,
        vae_latent_dimension: int,
        use_state: bool,
        prediction_horizon: int,
        device: str,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.feedforward_dimension = feedforward_dimension
        self.number_of_encoder_layers = number_of_encoder_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.normalize_before = normalize_before
        self.vae_latent_dimension = vae_latent_dimension
        self.use_state = use_state
        self.prediction_horizon = prediction_horizon
        self.device = device

        # Input projection - uses LazyLinear to infer dimension
        self.vae_input_projection = nn.LazyLinear(self.embedding_dimension)
        # State projection (if needed)
        if self.use_state:
            self.vae_state_projection = nn.LazyLinear(self.embedding_dimension)

        # VAE transformer encoder
        self.vae_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                embedding_dimension=self.embedding_dimension,
                number_of_heads=self.number_of_heads,
                feedforward_dimension=self.feedforward_dimension,
                activation=self.activation,
                dropout=self.dropout_rate,
                normalize_before=self.normalize_before,
            ),
            number_of_layers=self.number_of_encoder_layers,
            normalization=nn.LayerNorm(self.embedding_dimension) if self.normalize_before else None
        )
        # CLS token for VAE encoder
        self.cls_token = nn.Embedding(1, self.embedding_dimension)
        # Latent distribution parameters
        self.latent_stats_projection = nn.Linear(
            self.embedding_dimension,
            self.vae_latent_dimension * 2  # mu and logvar
        )
        # Positional encoding table (lazily initialized on first forward pass)
        self.register_buffer('vae_positional_encoding_table', None, persistent=False)
        self.to(device)  # Need to set device for Lazy modules

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        state_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode actions to latent embedding, or sample from prior if no actions.

        Args:
            inputs: Ground-truth input for VAE encoding
            state_features: Optional state observations

        Returns:
            Tuple of (z, mu, logvar)
            - mu and logvar are None during inference
        """
        return self._encode_actions_to_latent(inputs, state_features)  # type: ignore[arg-type]


    def _encode_actions_to_latent(
        self,
        inputs: dict[str, torch.Tensor],
        state_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode inputs to latent representation using VAE encoder."""
        # Extract padding mask and action embeddings
        batch_size = state_features.size(0) if state_features is not None else list(inputs.values())[0].size(0)
        is_pad = inputs.get(IS_PAD_KEY)
        if is_pad is None:
            logging.warning("No padding key found in actions; assuming no padding.")
            is_pad = torch.zeros(
                batch_size,
                self.prediction_horizon,
                dtype=torch.bool,
                device=self.cls_token.weight.device
            )
        else:
            is_pad = is_pad.to(self.cls_token.weight.device)

        input_tensors_list = []
        for key, input_tensor in sorted(inputs.items()):
            if key == IS_PAD_KEY:
                continue
            input_tensors_list.append(input_tensor.to(self.cls_token.weight.device))
        all_inputs = torch.cat(input_tensors_list, dim=-1)  # (B, horizon, total_input_dim)

        # Project concatenated input to embedding dimension
        input_embeddings = self.vae_input_projection(all_inputs)  # (B, horizon, embedding_dim)
        # Prepare encoder input: CLS token + [optional state] + input embeddings
        cls_embedding = self.cls_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.use_state and state_features is not None:
            state_embedding = self.vae_state_projection(
                state_features
            ).unsqueeze(1)
            encoder_input = torch.cat(
                [cls_embedding, state_embedding, input_embeddings], dim=1
            )
            non_action_mask = torch.full((batch_size, 2), False, device=self.cls_token.weight.device)
        else:
            encoder_input = torch.cat([cls_embedding, input_embeddings], dim=1)
            non_action_mask = torch.full((batch_size, 1), False, device=self.cls_token.weight.device)

        # Transpose to (sequence_length, batch, embedding dimension)
        encoder_input = encoder_input.permute(1, 0, 2)
        key_padding_mask = torch.cat([non_action_mask, is_pad], dim=1)
        # Get positional encodings for the input
        if self.vae_positional_encoding_table is None:
            num_positions = (
                1  # CLS
                + (1 if self.use_state else 0)
                + self.prediction_horizon
            )
            positional_encoding_table = SinusoidalPositionalEncoding1D.create_encoding_table(
                number_of_positions=num_positions,
                embedding_dimension=self.embedding_dimension,
            ).to(self.cls_token.weight.device)
            self.register_buffer('vae_positional_encoding_table', positional_encoding_table)

        positional_encoding = self.vae_positional_encoding_table.clone().permute(1, 0, 2)

        encoder_output = self.vae_encoder(
            encoder_input,
            positional_encoding=positional_encoding,
            source_key_padding_mask=key_padding_mask
        )[0]  # CLS token output
        # Get latent distribution parameters
        latent_stats = self.latent_stats_projection(encoder_output)
        mu, logvar = latent_stats.chunk(2, dim=1)
        # Sample using reparametrization trick
        z = reparametrize(mu, logvar)
        return z, mu, logvar
