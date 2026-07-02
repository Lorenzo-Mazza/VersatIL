"""Divergence and entropy losses for latent distributions."""

import logging
import math

import torch

from versatil.metrics.base import BaseLoss, LossOutput, WeightsDictionary
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey


class GaussianEntropyLoss(BaseLoss):
    """Entropy regularization for Gaussian distributions.

    Maximizes entropy H(N(μ, σ²)) = 0.5 * sum(1 + log(2π) + logvar) to prevent
    distribution collapse.

    Since we maximize entropy, this loss contributes negatively to the total.
    """

    def __init__(
        self,
        key: str = LatentKey.PRIOR_LOGVAR.value,
        weight: float = 0.01,
        logvar_min: float = -4.0,  # σ² ≈ 0.018
        logvar_max: float = 2.0,  # σ² ≈ 7.4
        bound_weight: float = 1.0,
    ):
        """Initialize Gaussian entropy loss.

        Args:
            key: Prediction key for logvar tensor to compute entropy over.
            weight: Loss weight. Positive values encourage higher entropy.
            logvar_min: Minimum logvar value.
            logvar_max: Maximum logvar value.
            bound_weight: Weight for the bound entropy loss.
        """
        super().__init__()
        if "logvar" not in key:
            raise ValueError(f"GaussianEntropyLoss expects a logvar key, got '{key}'.")
        self.key = key
        self.weight = weight
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.bound_weight = bound_weight

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {"weight": self.weight, "bound_weight": self.bound_weight}

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        self.weight = new_weights["weight"]
        self.bound_weight = new_weights["bound_weight"]

    def get_required_keys(self) -> set[str]:
        """Returns required prediction keys."""
        return {self.key}

    @staticmethod
    def compute_entropy(logvar: torch.Tensor) -> torch.Tensor:
        """Compute entropy of a diagonal Gaussian.

        H(N(μ, σ²)) = 0.5 * sum(1 + log(2π) + logvar)

        Args:
            logvar: Log variance tensor (..., latent_dim).

        Returns:
            Entropy summed over latent dimensions, shape (...).
        """
        return 0.5 * (1 + math.log(2 * math.pi) + logvar).sum(dim=-1)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute negative entropy loss (to maximize entropy via minimization).

        Args:
            predictions: Must contain the logvar key.
            targets: Unused.
            is_pad: Unused.

        Returns:
            LossOutput with negative weighted entropy.
        """
        if self.key not in predictions:
            raise ValueError(
                f"Predictions must contain '{self.key}' for GaussianEntropyLoss."
            )
        logvar = predictions[self.key].float()
        bound_violation = (
            torch.relu(logvar - self.logvar_max).pow(2).mean()
            + torch.relu(self.logvar_min - logvar).pow(2).mean()
        )
        entropy = self.compute_entropy(logvar).mean()
        total_loss = -self.weight * entropy + self.bound_weight * bound_violation
        return LossOutput(
            total_loss=total_loss,
            component_losses={f"{self.key}_{MetricKey.ENTROPY.value}": entropy},
        )


class KLDivergenceLoss(BaseLoss):
    """KL divergence loss for VAE latent distributions."""

    def __init__(
        self,
        weight: float = 10.0,
        prior_regularization_weight: float = 0.0,
    ):
        """Initialize KL divergence loss.

        Args:
            weight: Weight for KL divergence loss KL(posterior || prior)
            prior_regularization_weight: Weight for KL(prior || N(0,I)) regularization.
                Only meaningful for learned priors. Pushes the learned prior towards
                a standard Gaussian.
        """
        super().__init__()
        self.weight = weight
        self.prior_regularization_weight = prior_regularization_weight

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {
            "weight": self.weight,
            "prior_regularization_weight": self.prior_regularization_weight,
        }

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        self.weight = new_weights["weight"]
        self.prior_regularization_weight = new_weights["prior_regularization_weight"]

    def get_required_keys(self) -> set[str]:
        """Get required keys for KL divergence loss."""
        return {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.PRIOR_LATENT.value,
            LatentKey.POSTERIOR_MU.value,
            LatentKey.POSTERIOR_LOGVAR.value,
        }

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute KL divergence loss.

        Args:
            predictions: Dictionary with 'mu' and 'logvar' keys
            targets: Not used for KL divergence
            is_pad: Not used for KL divergence

        Returns:
            LossOutput with KL divergence loss
        """
        if LatentKey.PRIOR_LOG_PROB.value in predictions:
            mu_q = predictions[LatentKey.POSTERIOR_MU.value].float()
            logvar_q = predictions[LatentKey.POSTERIOR_LOGVAR.value].float()
            std_q = (0.5 * logvar_q).exp()
            z = predictions[LatentKey.POSTERIOR_LATENT.value]
            log_p_z = predictions[LatentKey.PRIOR_LOG_PROB.value]  # (B,)
            posterior = torch.distributions.Normal(mu_q, std_q)
            log_q_z = posterior.log_prob(z).sum(dim=-1)  # (B,)
            # KL(q || p) = E_q[log q - log p] ≈ log q(z) - log p(z)
            kld = log_q_z - log_p_z
            kld_mean = kld.mean()

            component_losses = {MetricKey.KL_DIVERGENCE.value: kld_mean}
            total_loss = self.weight * kld_mean

            metadata = {
                MetadataKey.POSTERIOR_Z.value: z,
                MetadataKey.POSTERIOR_MU.value: mu_q,
                MetadataKey.POSTERIOR_LOGVAR.value: logvar_q,
            }
            prior_latent = predictions.get(LatentKey.PRIOR_LATENT.value)
            if prior_latent is not None:
                metadata[MetadataKey.PRIOR_Z.value] = prior_latent

            return LossOutput(
                total_loss=total_loss,
                component_losses=component_losses,
                metadata=metadata,
            )

        # Standard Gaussian prior - uses closed-form KL
        required_keys = self.get_required_keys()
        required_keys.update({LatentKey.PRIOR_MU.value, LatentKey.PRIOR_LOGVAR.value})
        if not all(k in predictions for k in required_keys):
            raise ValueError(
                f"Predictions must contain '{required_keys}' for KLDivergenceLoss."
            )
        mu_posterior = predictions[
            LatentKey.POSTERIOR_MU.value
        ].float()  # Using fp32 float for stability
        logvar_posterior = predictions[LatentKey.POSTERIOR_LOGVAR.value].float()
        mu_prior = predictions[LatentKey.PRIOR_MU.value].float()
        logvar_prior = predictions[LatentKey.PRIOR_LOGVAR.value].float()
        std_posterior = (0.5 * logvar_posterior).exp()
        std_prior = (0.5 * logvar_prior).exp()
        posterior = torch.distributions.Normal(mu_posterior, std_posterior)
        prior = torch.distributions.Normal(mu_prior, std_prior)
        kld = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1)
        if kld.min() < 0:
            logging.warning(
                msg=f"Warning: Negative KL divergence encountered: min={kld.min().item():.4f}"
                f"per_dim_kl: min={kld.min().item():.4f}, max={kld.max().item():.4f}"
            )
        kld_mean = kld.mean()
        component_losses = {MetricKey.KL_DIVERGENCE.value: kld_mean}
        total_loss = self.weight * kld_mean
        if self.prior_regularization_weight > 0.0:
            # KL(N(μ, σ²) || N(0, I)) = 0.5 * sum(μ² + σ² - log(σ²) - 1)
            prior_kl = 0.5 * (
                mu_prior.pow(2) + logvar_prior.exp() - logvar_prior - 1
            ).sum(dim=-1)
            prior_kl_mean = prior_kl.mean()
            component_losses[MetricKey.HYPERPRIOR_KL_REGULARIZATION.value] = (
                prior_kl_mean
            )
            total_loss = total_loss + self.prior_regularization_weight * prior_kl_mean

        metadata = {
            MetadataKey.POSTERIOR_Z.value: predictions[
                LatentKey.POSTERIOR_LATENT.value
            ],
            MetadataKey.POSTERIOR_MU.value: mu_posterior,
            MetadataKey.POSTERIOR_LOGVAR.value: logvar_posterior,
            MetadataKey.PRIOR_Z.value: predictions[LatentKey.PRIOR_LATENT.value],
            MetadataKey.PRIOR_MU.value: mu_prior,
            MetadataKey.PRIOR_LOGVAR.value: logvar_prior,
        }

        return LossOutput(
            total_loss=total_loss,
            component_losses=component_losses,
            metadata=metadata,
        )


class BinaryKLDivergenceLoss(BaseLoss):
    """KL divergence loss for Free Transformer binary latent distributions.

    Computes KL divergence between learned binary distributions and uniform prior.
    Used with Free Transformer's binary mapper output.

    Based on "The Free Transformer" (Fleuret, 2025) - arXiv:2510.17558
    """

    def __init__(
        self,
        weight: float = 5.0,
        entropy_weight: float = 0.01,
        latent_bits: float = 64,
        free_bits: float = 2 * math.log(2),
    ):
        """Initialize binary KL divergence loss.

        Args:
            weight: Weight for KL divergence loss
            entropy_weight: Weight for the entropy regularization term
            latent_bits: Number of bits of the latent codes.
            free_bits: Free bits threshold (only penalize KL above this value)
        """
        super().__init__()
        self.weight = weight
        self.entropy_weight = entropy_weight
        self.free_bits = free_bits
        self.latent_bits = latent_bits

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {"weight": self.weight, "entropy_weight": self.entropy_weight}

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        self.weight = new_weights["weight"]
        self.entropy_weight = new_weights["entropy_weight"]

    def get_required_keys(self) -> set[str]:
        """Get required keys for binary KL divergence loss.

        Returns:
            Set containing binary_logits key from Free Transformer
        """
        return {DecoderOutputKey.BINARY_LOGITS.value}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute binary KL divergence loss.

        Args:
            predictions: Dictionary with 'binary_logits' key (B, T, H) or (B, H)
            targets: Not used for KL divergence
            is_pad: Optional padding mask (B, T) or (B,)

        Returns:
            LossOutput with KL divergence loss
        """
        if DecoderOutputKey.BINARY_LOGITS.value not in predictions:
            raise ValueError(
                f"Predictions must contain key '{DecoderOutputKey.BINARY_LOGITS.value}' for BinaryKLDivergenceLoss."
            )
        all_component_losses = {}
        if DecoderOutputKey.LATENT_CODES.value in predictions:
            latent_codes = predictions[
                DecoderOutputKey.LATENT_CODES.value
            ]  # (B, token_len, 2^H)
            code_indices = torch.argmax(
                latent_codes, dim=-1
            ).flatten()  # (B*token_len,)
            unique_codes = torch.unique(code_indices).numel()
            total_codes = 2**self.latent_bits
            usage_pct = unique_codes / total_codes
            all_component_losses[MetricKey.LATENT_CODE_USAGE.value] = usage_pct

        logits = predictions[
            DecoderOutputKey.BINARY_LOGITS.value
        ]  # (B, T, H) or (B, H)
        if logits is None:  # Inference, zero loss
            return LossOutput(
                total_loss=torch.tensor(
                    0.0, device=next(iter(predictions.values())).device
                ),
                component_losses=all_component_losses,
            )

        # P(B_h=1) = sigmoid(L_h) for each bit
        probs = torch.sigmoid(
            logits.float()
        )  # (B, T, H) or (B, H), cast to fp32 for stability
        # KL divergence for independent Bernoulli vs uniform Bernoulli(0.5)
        # KL(Bernoulli(p) || Bernoulli(0.5)) = p*log(2p) + (1-p)*log(2(1-p))
        eps = 1e-8  # For numerical stability
        kl_per_bit = probs * torch.log(2 * probs + eps) + (1 - probs) * torch.log(
            2 * (1 - probs) + eps
        )
        # Sum over bits to get total KL per token
        kl_per_token = kl_per_bit.sum(dim=-1)  # (B, T)
        raw_kl_mean = kl_per_token.mean()  # Scalar (mean over B,T)
        all_component_losses[MetricKey.RAW_KL_DIVERGENCE.value] = raw_kl_mean

        # Apply free bits threshold: max(0, KL - κ)
        if self.free_bits > 0:
            clamped_kl_per_token = torch.clamp(
                kl_per_token - self.free_bits, min=0.0
            )  # (B, T)
            clamped_kl_mean = clamped_kl_per_token.mean()  # Scalar
        else:
            clamped_kl_mean = raw_kl_mean

        all_component_losses[MetricKey.CLAMPED_KL_DIVERGENCE.value] = clamped_kl_mean
        entropy = -(
            probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps)
        )  # (B,token_len,H)
        regularized_kl = (
            clamped_kl_mean - self.entropy_weight * entropy.mean()
        )  # Scalar (avg over B,T,H)
        all_component_losses[MetricKey.POSTERIOR_ENTROPY.value] = entropy.mean()
        all_component_losses[MetricKey.KL_DIVERGENCE.value] = regularized_kl
        metadata = {
            MetadataKey.POSTERIOR_Z.value: torch.bernoulli(probs),
        }
        return LossOutput(
            total_loss=self.weight * regularized_kl,
            component_losses=all_component_losses,
            metadata=metadata,
        )
