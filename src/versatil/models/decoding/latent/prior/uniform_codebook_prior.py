"""Fixed uniform prior over discrete codebook indices.

The VQ-VAE equivalent of GaussianPrior: samples a random codebook index
uniformly and returns the corresponding embedding. No learnable parameters.
The codebook is shared from the VQ posterior encoder via wire_posterior().
"""

import torch

from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder
from versatil.models.decoding.latent.vq.residual_vq import ResidualVQ


class UniformCodebookPrior(PriorLatentEncoder):
    """Fixed uniform prior over VQ codebook indices.

    Samples each residual VQ layer's index uniformly from {0, ..., K-1},
    then decodes to a quantized embedding via the shared codebook. No
    trainable parameters — the VQ equivalent of GaussianPrior.

    Args:
        latent_dimension: Dimension of each codebook vector.
        num_codes: Number of codebook entries per layer (K).
        num_residual_layers: Number of residual VQ layers.
        device: Device string.
    """

    def __init__(
        self,
        latent_dimension: int,
        num_codes: int,
        num_residual_layers: int,
        device: str,
    ):
        super().__init__(latent_dimension=latent_dimension, device=device)
        self.code_dim = latent_dimension
        self.num_codes = num_codes
        self.num_residual_layers = num_residual_layers
        self.residual_vq: ResidualVQ | None = None
        self.register_buffer("_device_tracker", torch.zeros(1))
        self.to(torch.device(device))

    def get_auxiliary_output_keys(self) -> set[str]:
        """Uniform prior outputs only the quantized latent and sampled indices."""
        return {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.VQ_PRIOR_INDICES.value,
        }

    def wire_posterior(self, posterior: PosteriorLatentEncoder) -> None:
        """Wire shared codebook from the VQ posterior encoder.

        Args:
            posterior: VQ posterior encoder with a residual_vq attribute.

        Raises:
            ValueError: If the posterior's code_dim does not match.
        """
        residual_vq = getattr(posterior, "residual_vq", None)
        if residual_vq is None:
            raise AttributeError(
                f"Posterior {type(posterior).__name__} does not expose a "
                f"residual_vq attribute required by UniformCodebookPrior."
            )
        if residual_vq.code_dim != self.code_dim:
            raise ValueError(
                f"ResidualVQ code_dim ({residual_vq.code_dim}) does not match "
                f"UniformCodebookPrior code_dim ({self.code_dim})"
            )
        self.residual_vq = residual_vq

    def forward(
        self,
        target_latents: torch.Tensor | None,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample uniform codebook indices and decode to embedding.

        Args:
            target_latents: Posterior latent used only to infer batch size.
                None is accepted; batch size is then derived from observations.
            observations: Observation features used only to infer batch size.

        Returns:
            Dictionary containing:
                - LatentKey.PRIOR_LATENT: Quantized embedding, shape (B, code_dim).
                - LatentKey.VQ_PRIOR_INDICES: List of per-layer indices
                    sampled by the prior, each shape (B,). Emitted under a
                    distinct key from the posterior's VQ_INDICES so the two
                    do not collide when merged into the predictions dict.
        """
        if self.residual_vq is None:
            raise RuntimeError(
                "UniformCodebookPrior.residual_vq is not set. "
                "Call wire_posterior() before forward()."
            )
        if target_latents is not None:
            batch_size = target_latents.shape[0]
        else:
            batch_size = next(iter(observations.values())).shape[0]
        device = self._device_tracker.device

        all_indices = [
            torch.randint(0, self.num_codes, (batch_size,), device=device)  # (B,)
            for _ in range(self.num_residual_layers)
        ]

        z_q = self.residual_vq.decode_from_indices(all_indices)  # (B, code_dim)

        return {
            LatentKey.PRIOR_LATENT.value: z_q,
            LatentKey.VQ_PRIOR_INDICES.value: all_indices,
        }

    def sample_prior(
        self,
        batch_size: int,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample latent from uniform categorical prior.

        Args:
            batch_size: Number of samples.
            observations: Unused (uniform prior is unconditional).

        Returns:
            Quantized latent embedding, shape (batch_size, code_dim).
        """
        if self.residual_vq is None:
            raise RuntimeError(
                "UniformCodebookPrior.residual_vq is not set. "
                "Call wire_posterior() before sample_prior()."
            )
        device = self._device_tracker.device
        all_indices = [
            torch.randint(0, self.num_codes, (batch_size,), device=device)  # (B,)
            for _ in range(self.num_residual_layers)
        ]
        return self.residual_vq.decode_from_indices(all_indices)  # (B, code_dim)
