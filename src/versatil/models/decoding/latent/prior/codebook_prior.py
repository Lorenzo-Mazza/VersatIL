"""Learned prior over discrete codebook indices for VQ latent variable models.

Predicts a categorical distribution over codebook entries conditioned on
observations. At inference, samples codebook indices and returns the
corresponding quantized embedding for the decoder. Shares the codebook
with the VQ posterior encoder — the codebook is set after construction
via wire_posterior().
"""

from typing import Any

import torch
from torch import nn

from versatil.models.decoding.constants import (
    AlgorithmContextKey,
    LatentKey,
)
from versatil.models.decoding.latent import PriorLatentEncoder
from versatil.models.decoding.latent.posterior.vq_encoder import VQPosteriorEncoder
from versatil.models.decoding.latent.prior.state_condition_pool import (
    StateConditionPool,
)
from versatil.models.decoding.latent.vq.residual_vq import ResidualVQ
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)
from versatil.models.layers.transformer.encoder import TransformerEncoder


class CodebookPrior(PriorLatentEncoder):
    """Learned categorical prior over VQ codebook indices.

    Encodes observations via a transformer, then predicts logits over
    the K codebook entries for each residual VQ layer. At inference,
    samples indices from the predicted categorical and decodes them
    to a quantized embedding via the shared codebook.

    The codebook is owned by the VQ posterior encoder and shared via
    wire_posterior(). This must be called before the first forward pass.

    Args:
        latent_dimension: Dimension of each codebook vector. Must match
            the posterior encoder's latent dimension.
        num_codes: Number of codebook entries per layer (K).
        num_residual_layers: Number of residual VQ layers.
        embedding_dimension: Transformer hidden dimension.
        observation_horizon: Number of observation timesteps.
        device: Device string.
        number_of_heads: Number of attention heads.
        feedforward_dimension: Feedforward network dimension.
        number_of_encoder_layers: Number of transformer encoder layers.
        activation: Activation function name.
        dropout_rate: Dropout probability.
        attention_type: Attention mechanism type (use AttentionType enum values).
        exclude_keys: Observation keys to exclude from encoding.
        temperature: Softmax temperature for sampling. Lower values
            produce sharper categorical distributions.
    """

    def __init__(
        self,
        latent_dimension: int,
        num_codes: int,
        num_residual_layers: int,
        embedding_dimension: int,
        observation_horizon: int,
        device: str,
        number_of_heads: int = 4,
        feedforward_dimension: int = 128,
        number_of_encoder_layers: int = 1,
        activation: str = ActivationFunction.SWIGLU.value,
        dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        positional_encoding_type: str | None = None,
        exclude_keys: list[str] | None = None,
        temperature: float = 1.0,
    ):
        super().__init__(
            latent_dimension=latent_dimension,
            device=device,
        )
        if latent_dimension <= 0:
            raise ValueError(
                f"latent_dimension must be positive, got {latent_dimension}."
            )
        if num_codes <= 0:
            raise ValueError(f"num_codes must be positive, got {num_codes}.")
        if num_residual_layers <= 0:
            raise ValueError(
                f"num_residual_layers must be positive, got {num_residual_layers}."
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.code_dim = latent_dimension
        self.num_codes = num_codes
        self.num_residual_layers = num_residual_layers
        self.embedding_dimension = embedding_dimension
        self.exclude_keys = exclude_keys if exclude_keys is not None else []
        self.temperature = temperature
        self._residual_vq_reference: tuple[ResidualVQ, ...] = ()
        self.state_condition_pool = StateConditionPool(
            embedding_dimension=embedding_dimension
        )

        self.encoder = TransformerEncoder(
            number_of_layers=number_of_encoder_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            activation=activation,
            dropout=dropout_rate,
            attention_dropout=attention_dropout,
            normalization_type=normalization_type,
            attention_type=attention_type,
            positional_encoding_type=positional_encoding_type,
        )

        temporal_positional_encoding = None
        if observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(
                embedding_dimension=embedding_dimension
            )

        self.input_sequence_builder = TransformerInputBuilder(
            embedding_dimension=embedding_dimension,
            has_time_dim=observation_horizon > 1,
            spatial_positional_encoding_layer=SinusoidalPositionalEncoding2D(
                embedding_dimension=embedding_dimension, normalize=True
            ),
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=embedding_dimension,
                maximum_sequence_length=1000,
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
        )

        self.cls_token = nn.Embedding(1, embedding_dimension)  # (1, emb_dim)

        # One classification head per residual VQ layer
        self.code_heads = nn.ModuleList(
            [
                nn.Linear(embedding_dimension, num_codes)
                for _ in range(num_residual_layers)
            ]
        )

        self.to(device)

    @property
    def residual_vq(self) -> ResidualVQ | None:
        """Return the posterior-owned VQ module without registering it as a child."""
        if not self._residual_vq_reference:
            return None
        return self._residual_vq_reference[0]

    def __getstate__(self) -> dict[str, Any]:
        """Return copy-safe state without the posterior VQ reference.

        Returns:
            Module state with runtime wiring cleared. The owning
            VariationalAlgorithm reconnects the copied prior to the copied
            posterior after deepcopy or unpickling.
        """
        state = super().__getstate__()
        state["_residual_vq_reference"] = ()
        return state

    def get_auxiliary_output_keys(self) -> set[str]:
        """Codebook prior outputs quantized latent, sampled indices, and logits."""
        return {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_CONDITION.value,
            LatentKey.VQ_PRIOR_INDICES.value,
            LatentKey.PRIOR_CODE_LOGITS.value,
        }

    def wire_posterior(self, posterior: VQPosteriorEncoder) -> None:
        """Wire shared codebook from the VQ posterior encoder.

        Extracts the ResidualVQ reference needed to decode sampled
        indices into quantized embeddings at inference.

        Args:
            posterior: VQ posterior encoder with a residual_vq attribute.

        Raises:
            AttributeError: If the posterior does not expose ResidualVQ state.
            ValueError: If the posterior's VQ configuration does not match
                this prior's configuration.
        """
        residual_vq = getattr(posterior, "residual_vq", None)
        if residual_vq is None:
            raise AttributeError(
                f"Posterior {type(posterior).__name__} does not expose a "
                f"residual_vq attribute required by CodebookPrior."
            )
        if residual_vq.code_dim != self.code_dim:
            raise ValueError(
                f"ResidualVQ code_dim ({residual_vq.code_dim}) does not match "
                f"CodebookPrior code_dim ({self.code_dim})"
            )
        if residual_vq.num_codes != self.num_codes:
            raise ValueError(
                f"ResidualVQ num_codes ({residual_vq.num_codes}) does not match "
                f"CodebookPrior num_codes ({self.num_codes})"
            )
        if residual_vq.num_layers != self.num_residual_layers:
            raise ValueError(
                f"ResidualVQ num_layers ({residual_vq.num_layers}) does not match "
                f"CodebookPrior num_residual_layers ({self.num_residual_layers})"
            )
        self._residual_vq_reference = (residual_vq,)

    def forward(
        self,
        target_latents: torch.Tensor | None,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Predict codebook index distribution from observations.

        Args:
            target_latents: Unused (batch size derived from observations).
            observations: Dictionary of observation features.

        Returns:
            Dictionary containing:
                - LatentKey.PRIOR_LATENT: Quantized embedding from sampled
                    indices, shape (B, code_dim).
                - LatentKey.VQ_PRIOR_INDICES: List of sampled per-layer
                    indices, each shape (B,). Emitted under a distinct key
                    from the posterior's VQ_INDICES to avoid collision.
                - LatentKey.PRIOR_CODE_LOGITS: List of per-layer logits
                    over the K codebook entries, each shape (B, K).
                    Consumed by VQPriorCrossEntropyLoss.
        """
        residual_vq = self.residual_vq
        if residual_vq is None:
            raise RuntimeError(
                "CodebookPrior.residual_vq is not set. "
                "Call wire_posterior() before forward()."
            )

        input_observations = {
            k: v for k, v in observations.items() if k not in self.exclude_keys
        }

        batch_size = list(input_observations.values())[0].size(0)
        cls_embedding = self.cls_token.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, 1, emb_dim)
        input_observations[AlgorithmContextKey.CLASS_TOKEN.value] = cls_embedding

        input_tokens, pos_encodings, padding_mask = self.input_sequence_builder(
            input_observations
        )  # (B, seq_len, emb_dim)

        hidden_states = input_tokens + pos_encodings
        # TransformerInputBuilder appends the CLS token as the final token.
        # The conditional loss state must come from observation tokens only,
        # so exclude that final CLS token from the pooled state vector.
        condition_tokens = hidden_states[:, :-1, :]
        condition_padding_mask = (
            padding_mask[:, :-1] if padding_mask is not None else None
        )
        prior_condition = self.state_condition_pool(
            tokens=condition_tokens,
            padding_mask=condition_padding_mask,
        ).detach()
        encoder_output = self.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
        )[:, -1, :]  # (B, emb_dim) — CLS token

        all_indices = []
        all_logits = []

        for head in self.code_heads:
            logits = head(encoder_output)  # (B, K)
            all_logits.append(logits)
            probs = torch.softmax(logits / self.temperature, dim=-1)  # (B, K)
            indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
            all_indices.append(indices)

        z_q = residual_vq.decode_from_indices(all_indices)  # (B, code_dim)

        return {
            LatentKey.PRIOR_LATENT.value: z_q,
            LatentKey.PRIOR_CONDITION.value: prior_condition,
            LatentKey.VQ_PRIOR_INDICES.value: all_indices,
            LatentKey.PRIOR_CODE_LOGITS.value: all_logits,
        }

    def sample_prior(
        self,
        batch_size: int,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample latent from learned categorical prior p(k|s).

        Args:
            batch_size: Number of samples.
            observations: Observation features for conditioning.

        Returns:
            Quantized latent embedding, shape (batch_size, code_dim).
        """
        if observations is None:
            raise ValueError(
                "CodebookPrior requires observations for conditional sampling."
            )
        return self.forward(
            target_latents=None,
            observations=observations,
        )[LatentKey.PRIOR_LATENT.value]
