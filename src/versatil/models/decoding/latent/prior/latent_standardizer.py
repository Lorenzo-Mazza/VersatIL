"""Latent-space standardization module for learned denoising priors."""

import torch
from torch import nn


class LatentStandardizer(nn.Module):
    """Optional affine standardization for DiT prior latent targets."""

    def __init__(
        self,
        latent_dimension: int,
        enabled: bool = True,
        epsilon: float = 1e-6,
        require_fitted: bool = False,
    ):
        """Initialize standardizer buffers.

        Args:
            latent_dimension: Size of the latent vector to standardize.
            enabled: Whether standardization should be applied after fitting.
            epsilon: Numerical epsilon used for clamping and division.
            require_fitted: Whether transform calls should raise before stats exist.
        """
        super().__init__()
        self.latent_dimension = latent_dimension
        self.enabled = enabled
        self.epsilon = epsilon
        self.require_fitted = require_fitted
        self.register_buffer("mean", torch.zeros(latent_dimension))
        self.register_buffer("std", torch.ones(latent_dimension))
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))

    def _check_ready(self) -> None:
        """Validate that required latent statistics are available.

        Raises:
            RuntimeError: If standardization is enabled, required, and unfitted.
        """
        if self.enabled and self.require_fitted and not bool(self.is_fitted.item()):
            raise RuntimeError(
                "LatentStandardizer requires fitted latent statistics before use."
            )

    def _should_transform(self) -> bool:
        """Return whether standardization should be applied.

        Returns:
            True when the module is enabled and fitted.
        """
        self._check_ready()
        return self.enabled and bool(self.is_fitted.item())

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set precomputed latent mean and standard deviation.

        Args:
            mean: Latent mean tensor with shape ``(latent_dimension,)``.
            std: Latent standard-deviation tensor with shape ``(latent_dimension,)``.

        Raises:
            ValueError: If shapes are wrong, values are non-finite, or std is invalid.
        """
        expected_shape = (self.latent_dimension,)
        if mean.shape != expected_shape or std.shape != expected_shape:
            raise ValueError(
                f"Latent stats must have shape {expected_shape}, got "
                f"mean={tuple(mean.shape)}, std={tuple(std.shape)}."
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Latent stats must be finite.")
        if torch.any(std <= 0):
            raise ValueError("Latent std must be strictly positive.")
        self.mean.copy_(
            mean.detach().to(device=self.mean.device, dtype=self.mean.dtype)
        )
        self.std.copy_(std.detach().to(device=self.std.device, dtype=self.std.dtype))
        self.is_fitted.fill_(True)

    def fit(self, latents: torch.Tensor) -> None:
        """Fit statistics from latent samples.

        Args:
            latents: Tensor whose trailing dimension is ``latent_dimension``.

        Raises:
            ValueError: If the trailing dimension does not match the latent size.
        """
        if latents.shape[-1] != self.latent_dimension:
            raise ValueError(
                f"Latents must have trailing dimension {self.latent_dimension}, "
                f"got {latents.shape[-1]}."
            )
        flattened_latents = latents.detach().reshape(-1, self.latent_dimension).float()
        mean = flattened_latents.mean(dim=0)
        std = flattened_latents.std(dim=0, unbiased=False).clamp(min=self.epsilon)
        self.set_stats(mean=mean, std=std)

    def standardize(self, latents: torch.Tensor) -> torch.Tensor:
        """Map raw posterior latents to DiT prior training space.

        Args:
            latents: Raw latent tensor with trailing latent dimension.

        Returns:
            Standardized latents, or the input unchanged when disabled/unfitted.
        """
        if not self._should_transform():
            return latents
        mean = self.mean.to(device=latents.device, dtype=latents.dtype)
        std = self.std.to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) / (std + self.epsilon)

    def unstandardize(self, latents: torch.Tensor) -> torch.Tensor:
        """Map DiT prior samples back to decoder latent space.

        Args:
            latents: Standardized latent tensor with trailing latent dimension.

        Returns:
            Raw-space latents, or the input unchanged when disabled/unfitted.
        """
        if not self._should_transform():
            return latents
        mean = self.mean.to(device=latents.device, dtype=latents.dtype)
        std = self.std.to(device=latents.device, dtype=latents.dtype)
        return latents * (std + self.epsilon) + mean
