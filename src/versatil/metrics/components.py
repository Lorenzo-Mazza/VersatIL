"""Individual loss components for action prediction tasks."""

import logging
import math

import torch
import torch.nn.functional as F

from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.configs.experiment import ExperimentConfig
from versatil.data.constants import (
    BinaryGripperRange,
    GripperType,
    SampleKey,
)
from versatil.data.metadata import (
    ActionMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
)
from versatil.metrics.base import BaseLoss, LossOutput, reduce_loss_with_padding
from versatil.metrics.constants import (
    MetadataKey,
    MetricKey,
)
from versatil.metrics.kernels import KernelType
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey
from versatil.training.callbacks import ExpertUsageCallback


class RegressionLoss(BaseLoss):
    """Regression loss for continuous action predictions (position, orientation).

    Supports MSE, L1, and Huber loss functions with optional per-modality weighting.
    """

    def __init__(
        self,
        action_keys: list[str],
        mse_weight: float = 1.0,
        l1_weight: float = 0.0,
        huber_weight: float = 0.0,
        huber_delta: float = 1.0,
        per_key_weights: dict[str, float] | None = None,
    ):
        """Initialize regression loss.

        Args:
            action_keys: List of action keys to compute loss for (e.g., ['position', 'orientation'])
            mse_weight: Weight for MSE loss
            l1_weight: Weight for L1 loss
            huber_weight: Weight for Huber loss
            huber_delta: Delta parameter for Huber loss
            per_key_weights: Optional dictionary of per-key weights
        """
        super().__init__()
        self.action_keys = action_keys
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.huber_weight = huber_weight
        self.huber_delta = huber_delta
        self.per_key_weights = per_key_weights or {}

    def get_required_keys(self) -> set[str]:
        """Get required target keys for regression loss.

        Returns:
            Set of action keys this loss operates on
        """
        return set(self.action_keys)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute regression loss.

        Args:
            predictions: Dictionary with predicted actions
            targets: Dictionary with ground truth actions
            is_pad: Optional padding mask (B, horizon)

        Returns:
            LossOutput with regression loss components
        """
        component_losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for action_key in self.action_keys:
            if action_key not in predictions or action_key not in targets:
                raise ValueError(
                    f"Predictions and targets must contain key '{action_key}' for RegressionLoss."
                )

            pred = predictions[action_key]
            target = targets[action_key]
            key_weight = self.per_key_weights.get(action_key, 1.0)

            if self.mse_weight > 0:
                mse = F.mse_loss(pred, target, reduction="none")
                mse_reduced = reduce_loss_with_padding(mse, is_pad, reduction="mean")
                loss_key = f"{action_key}_{MetricKey.MSE_LOSS.value}"
                component_losses[loss_key] = mse_reduced
                total_loss = total_loss + self.mse_weight * key_weight * mse_reduced

            if self.l1_weight > 0:
                l1 = F.l1_loss(pred, target, reduction="none")
                l1_reduced = reduce_loss_with_padding(l1, is_pad, reduction="mean")
                loss_key = f"{action_key}_{MetricKey.L1_LOSS.value}"
                component_losses[loss_key] = l1_reduced
                total_loss = total_loss + self.l1_weight * key_weight * l1_reduced

            if self.huber_weight > 0:
                huber = F.huber_loss(
                    pred, target, delta=self.huber_delta, reduction="none"
                )
                huber_reduced = reduce_loss_with_padding(
                    huber, is_pad, reduction="mean"
                )
                loss_key = f"{action_key}_{MetricKey.HUBER_LOSS.value}"
                component_losses[loss_key] = huber_reduced
                total_loss = total_loss + self.huber_weight * key_weight * huber_reduced

        return LossOutput(total_loss=total_loss, component_losses=component_losses)


class GripperLoss(BaseLoss):
    """Loss for gripper action prediction (binary or continuous)."""

    def __init__(
        self,
        key: str,
        actions_metadata: dict[str, ActionMetadata],
        bce_weight: float = 0.005,
        mse_weight: float = 0.0,
        pos_weight: torch.Tensor | None = None,
    ):
        """Initialize gripper loss.

        Args:
            key: Action key for gripper
            actions_metadata: Dict of metadata of the action space
            bce_weight: Weight for binary cross entropy (binary gripper)
            mse_weight: Weight for MSE loss (continuous gripper)
            pos_weight: Optional positive class weight for BCE
        """
        super().__init__()
        self.key = key
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.register_buffer("pos_weight", pos_weight)
        resolved_metadata = resolve_dict_keys(dict(actions_metadata))
        if key not in resolved_metadata:
            raise ValueError(
                f"{key} is not available to the action space. Can't compute gripper loss. "
                f"Available keys: {list(resolved_metadata)}"
            )
        meta = resolved_metadata[key]
        if isinstance(meta, GripperActionMetadata):
            self.gripper_type = meta.gripper_type
            self.binary_gripper_range = meta.binary_gripper_range
        elif isinstance(meta, OnTheFlyActionMetadata):
            source = meta.source_metadata
            if isinstance(source, GripperObservationMetadata):
                self.gripper_type = source.gripper_type
                self.binary_gripper_range = source.binary_gripper_range
            else:
                raise ValueError(
                    f"Expected GripperObservationMetadata for key '{key}', got {type(source).__name__}"
                )
        else:
            raise ValueError(
                f"Expected gripper metadata for key '{key}', got {type(meta).__name__}"
            )

    @property
    def requires_action_space_targets(self) -> bool:
        return self.bce_weight > 0

    def get_required_keys(self) -> set[str]:
        """Get required target keys for gripper loss.

        Returns:
            Set containing the gripper action key
        """
        return {self.key}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute gripper loss.

        Args:
            predictions: Dictionary with 'gripper_action' key
            targets: Dictionary with ground truth gripper actions
            is_pad: Optional padding mask

        Returns:
            LossOutput with gripper loss
        """
        if self.key not in predictions or self.key not in targets:
            raise ValueError(
                f"Predictions and targets must contain key '{self.key}' for GripperLoss."
            )
        pred_gripper = predictions[self.key]
        target_gripper = targets[self.key]

        if self.gripper_type == GripperType.BINARY.value:
            if self.binary_gripper_range == BinaryGripperRange.MINUS_ONE_ONE.value:
                target_gripper = (target_gripper.float() + 1.0) / 2.0
            bce = F.binary_cross_entropy_with_logits(
                pred_gripper,
                target_gripper.float(),
                pos_weight=self.pos_weight,
                reduction="none",
            )
            bce_reduced = reduce_loss_with_padding(bce, is_pad, reduction="mean")
            return LossOutput(
                total_loss=self.bce_weight * bce_reduced,
                component_losses={MetricKey.GRIPPER_BCE.value: bce_reduced},
            )
        else:
            mse = F.mse_loss(pred_gripper, target_gripper, reduction="none")
            mse_reduced = reduce_loss_with_padding(mse, is_pad, reduction="mean")
            return LossOutput(
                total_loss=self.mse_weight * mse_reduced,
                component_losses={MetricKey.GRIPPER_MSE.value: mse_reduced},
            )


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
        prior_entropy_weight: float = 0.0,
        prior_regularization_weight: float = 0.0,
    ):
        """Initialize KL divergence loss.

        Args:
            weight: Weight for KL divergence loss KL(posterior || prior)
            prior_entropy_weight: Weight for prior entropy regularization
            prior_regularization_weight: Weight for KL(prior || N(0,I)) regularization.
                Only meaningful for learned priors. Pushes the learned prior towards
                a standard Gaussian.
        """
        super().__init__()
        self.weight = weight
        self.prior_entropy_weight = prior_entropy_weight
        self.prior_regularization_weight = prior_regularization_weight

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
                MetadataKey.PRIOR_Z.value: predictions.get(
                    LatentKey.PRIOR_LATENT.value
                ),
            }

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


class MaximumMeanDiscrepancyLoss(BaseLoss):
    """MMD loss for regularizing latent distributions toward a prior.

    Ref: [Info-VAE / MMD-VAE](https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/)
    """

    def __init__(
        self,
        weight: float = 1.0,
        prior_regularization_weight: float = 0.0,
        kernel_type: str = KernelType.RBF.value,
        bandwidth_multipliers: list[float] | None = None,
        use_fixed_gaussian_as_prior: bool = False,
    ):
        """Initialize MMD loss.

        Args:
            weight: Loss weight for MMD(posterior, prior).
            prior_regularization_weight: Weight for MMD(prior, N(0,I)) regularization.
                Only meaningful for learned priors.
            kernel_type: Kernel type for MMD computation (see KernelType enum).
            bandwidth_multipliers: Scale factors for the median heuristic bandwidth.
            use_fixed_gaussian_as_prior: If True, always use standard Gaussian as prior.
        """
        super().__init__()
        self.weight = weight
        self.prior_regularization_weight = prior_regularization_weight
        self.kernel = KernelType(kernel_type).to_kernel(
            bandwidth_multipliers=bandwidth_multipliers
        )
        self.use_fixed_gaussian_as_prior = use_fixed_gaussian_as_prior

    def get_required_keys(self) -> set[str]:
        """Get required keys for MMD loss."""
        keys = {LatentKey.POSTERIOR_LATENT.value}
        if not self.use_fixed_gaussian_as_prior:
            keys.add(LatentKey.PRIOR_LATENT.value)
        return keys

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute MMD between latent samples and standard Gaussian prior.
        Args:
            predictions: Must contain LatentKey.POSTERIOR_LATENT.value with shape (B, latent_dim).
            targets: Unused (prior is implicit).
            is_pad: Unused.
        Returns:
            LossOutput with MMD loss.
        """
        required_keys = self.get_required_keys()
        if not all(k in predictions for k in required_keys):
            raise ValueError(
                f"Predictions must contain '{required_keys}' for MaximumMeanDiscrepancyLoss."
            )

        z_posterior = predictions[LatentKey.POSTERIOR_LATENT.value]  # (B, latent_dim)
        original_z_prior = predictions.get(
            LatentKey.PRIOR_LATENT.value
        )  # (B, latent_dim) or None
        if self.use_fixed_gaussian_as_prior:
            z_prior = torch.randn_like(z_posterior)  # (B, latent_dim)
        else:
            if original_z_prior is None:
                raise ValueError(
                    "Prior latent is required when use_fixed_gaussian_as_prior=False."
                )
            z_prior = original_z_prior  # (B, latent_dim)

        k_zz = self.kernel(z_posterior, z_posterior)
        k_pp = self.kernel(z_prior, z_prior)
        k_zp = self.kernel(z_posterior, z_prior)

        mmd_sq = k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()
        mmd_sq = torch.clamp(mmd_sq, min=0.0)

        component_losses = {MetricKey.MMD_LOSS.value: mmd_sq}
        total_loss = self.weight * mmd_sq

        if self.prior_regularization_weight > 0.0:
            z_standard = torch.randn_like(z_prior)  # (B, latent_dim)

            k_pp_reg = self.kernel(z_prior, z_prior)
            k_ss = self.kernel(z_standard, z_standard)
            k_ps = self.kernel(z_prior, z_standard)

            prior_mmd_sq = k_pp_reg.mean() + k_ss.mean() - 2 * k_ps.mean()
            prior_mmd_sq = torch.clamp(prior_mmd_sq, min=0.0)

            component_losses[MetricKey.HYPERPRIOR_MMD_REGULARIZATION.value] = (
                prior_mmd_sq
            )
            total_loss = total_loss + self.prior_regularization_weight * prior_mmd_sq

        metadata = {MetadataKey.POSTERIOR_Z.value: z_posterior}
        posterior_mu = predictions.get(LatentKey.POSTERIOR_MU.value)
        if posterior_mu is not None:
            metadata[MetadataKey.POSTERIOR_MU.value] = posterior_mu
        posterior_logvar = predictions.get(LatentKey.POSTERIOR_LOGVAR.value)
        if posterior_logvar is not None:
            metadata[MetadataKey.POSTERIOR_LOGVAR.value] = posterior_logvar
        if self.use_fixed_gaussian_as_prior:
            metadata[MetadataKey.HYPERPRIOR_Z.value] = z_prior
        if original_z_prior is not None:
            metadata[MetadataKey.PRIOR_Z.value] = original_z_prior
        prior_mu = predictions.get(LatentKey.PRIOR_MU.value)
        if prior_mu is not None:
            metadata[MetadataKey.PRIOR_MU.value] = prior_mu
        prior_logvar = predictions.get(LatentKey.PRIOR_LOGVAR.value)
        if prior_logvar is not None:
            metadata[MetadataKey.PRIOR_LOGVAR.value] = prior_logvar

        return LossOutput(
            total_loss=total_loss,
            component_losses=component_losses,
            metadata=metadata,
        )


class BinaryMaximumMeanDiscrepancyLoss(BaseLoss):
    """MMD loss for regularizing binary latent distributions toward a uniform prior.

    Encourages q(b|x) ≈ p(b) where p(b) = Bernoulli(0.5) independent for each bit.
    """

    def __init__(
        self,
        weight: float = 1.0,
        kernel_type: str = KernelType.RBF.value,
        bandwidth_multipliers: list[float] | None = None,
    ):
        """Initialize binary MMD loss.

        Args:
            weight: Loss weight.
            kernel_type: Kernel type for MMD computation (see KernelType enum).
            bandwidth_multipliers: Scale factors for the median heuristic bandwidth.
        """
        super().__init__()
        self.weight = weight
        self.kernel = KernelType(kernel_type).to_kernel(
            bandwidth_multipliers=bandwidth_multipliers
        )

    def get_required_keys(self) -> set[str]:
        """Returns required prediction keys: {DecoderOutputKey.BINARY_LOGITS.value}."""
        return {DecoderOutputKey.BINARY_LOGITS.value}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute MMD between binary latent samples and uniform Bernoulli prior.

        Args:
            predictions: Must contain DecoderOutputKey.BINARY_LOGITS.value with shape (B, H).
            targets: Unused (prior is implicit).
            is_pad: Unused.

        Returns:
            LossOutput with MMD loss.
        """
        if DecoderOutputKey.BINARY_LOGITS.value not in predictions:
            raise ValueError(
                f"Predictions must contain '{DecoderOutputKey.BINARY_LOGITS.value}' for BinaryMaximumMeanDiscrepancyLoss."
            )

        logits = predictions[DecoderOutputKey.BINARY_LOGITS.value]  # (B, T, H)
        probs = torch.sigmoid(logits.float())  # Cast to fp32 for stability
        z_hard = torch.bernoulli(probs)
        z = (
            z_hard - probs.detach() + probs
        )  # Straight-through: forward=hard, backward=soft
        z_prior = torch.bernoulli(
            0.5 * torch.ones_like(z)
        )  # samples from Bernoulli(0.5)
        k_zz = self.kernel(z, z)
        k_pp = self.kernel(z_prior, z_prior)
        k_zp = self.kernel(z, z_prior)
        # MMD² = E[k(z,z')] + E[k(p,p')] - 2E[k(z,p)]
        mmd = k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()
        metadata = {
            MetadataKey.POSTERIOR_Z.value: z,
        }
        return LossOutput(
            total_loss=self.weight * mmd,
            component_losses={MetricKey.BINARY_MMD_LOSS.value: mmd},
            metadata=metadata,
        )


class TrajectoryLengthLoss(BaseLoss):
    """Loss for trajectory length consistency.

    Penalizes differences between predicted and ground truth trajectory lengths.
    """

    def __init__(self, action_key: str, weight: float = 0.001):
        """Initialize trajectory length loss.

        Args:
            weight: Weight for length loss
            action_key: Action key to compute length for
        """
        super().__init__()
        self.weight = weight
        self.action_key = action_key

    def get_required_keys(self) -> set[str]:
        """Get required target keys for trajectory length loss.

        Returns:
            Set containing the action key this loss operates on
        """
        return {self.action_key}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute trajectory length loss.

        Args:
            predictions: Dictionary with predicted actions
            targets: Dictionary with ground truth actions
            is_pad: Optional padding mask

        Returns:
            LossOutput with length loss
        """
        if self.action_key not in predictions or self.action_key not in targets:
            raise ValueError(
                f"Predictions and targets must contain key '{self.action_key}' for TrajectoryLengthLoss."
            )
        pred = predictions[self.action_key]
        target = targets[self.action_key]

        if is_pad is not None:
            mask = (~is_pad).unsqueeze(-1).float()
            pred_masked = pred * mask
            target_masked = target * mask
        else:
            pred_masked = pred
            target_masked = target

        pred_length = torch.norm(
            pred_masked[:, 1:] - pred_masked[:, :-1], dim=-1
        ).mean()
        target_length = torch.norm(
            target_masked[:, 1:] - target_masked[:, :-1], dim=-1
        ).mean()

        length_loss = (pred_length - target_length) ** 2

        return LossOutput(
            total_loss=self.weight * length_loss,
            component_losses={MetricKey.LENGTH_LOSS.value: length_loss},
        )


class TrajectorySmoothness(BaseLoss):
    """Loss for trajectory smoothness (acceleration regularization)."""

    def __init__(self, action_key: str, weight: float = 0.001):
        """Initialize smoothness loss.

        Args:
            weight: Weight for smoothness loss
            action_key: Action key to compute smoothness for
        """
        super().__init__()
        self.weight = weight
        self.action_key = action_key

    def get_required_keys(self) -> set[str]:
        """Get required target keys for trajectory smoothness loss.

        Note: This loss only uses predictions, not targets, but we return
        the action key for consistency with other trajectory losses.

        Returns:
            Empty set since this loss doesn't use targets
        """
        return set()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute trajectory smoothness loss.

        Args:
            predictions: Dictionary with predicted actions
            targets: Not used for smoothness
            is_pad: Optional padding mask

        Returns:
            LossOutput with smoothness loss
        """
        if self.action_key not in predictions:
            raise ValueError(
                f"Predictions must contain key '{self.action_key}' for TrajectorySmoothness loss."
            )
        pred = predictions[self.action_key]
        if (
            pred.shape[1] < 3
        ):  # If trajectory too short, no acceleration can be computed
            return LossOutput(
                total_loss=torch.tensor(0.0, device=pred.device),
                component_losses={MetricKey.SMOOTHNESS_LOSS.value: torch.tensor(0.0)},
            )
        velocities = pred[:, 1:] - pred[:, :-1]
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        smoothness = torch.norm(accelerations, dim=-1)
        if is_pad is not None:
            pad_mask_accel = is_pad[:, 2:]
            smoothness = reduce_loss_with_padding(
                smoothness, pad_mask_accel, reduction="mean"
            )
        else:
            smoothness = smoothness.mean()

        return LossOutput(
            total_loss=self.weight * smoothness,
            component_losses={MetricKey.SMOOTHNESS_LOSS.value: smoothness},
        )


class PhaseClassificationLoss(BaseLoss):
    """Loss for phase classification in PhaseACT models.

    Includes cross-entropy loss and optional entropy regularization.
    """

    def __init__(
        self,
        key: str,
        cross_entropy_weight: float = 0.1,
        entropy_weight: float = 0.01,
        label_smoothing: float = 0.2,
    ):
        """Initialize phase classification loss.

        Args:
            key: Key for phase labels
            cross_entropy_weight: Weight for cross-entropy loss
            entropy_weight: Weight for entropy regularization (Entropy maximization avoids experts collapse)
            label_smoothing: Label smoothing factor for cross-entropy
        """
        super().__init__()
        self.key = key
        self.cross_entropy_weight = cross_entropy_weight
        self.entropy_weight = entropy_weight
        self.label_smoothing = label_smoothing

    def get_required_keys(self) -> set[str]:
        """Get required target keys for phase classification loss.

        Returns:
            Set containing the phase label key
        """
        return {self.key}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute phase classification loss.

        Args:
            predictions: Dictionary with 'phase_label' logits (B, horizon, n_phases)
            targets: Dictionary with 'phase_label' ground truth (B, horizon) or (B, horizon, 1)
            is_pad: Optional padding mask

        Returns:
            LossOutput with cross-entropy and optional entropy loss
        """
        if self.key not in predictions or self.key not in targets:
            raise ValueError(
                f"Predictions and targets must contain key '{self.key}' for PhaseClassificationLoss."
            )

        pred_logits = predictions[self.key]
        target_labels = targets[self.key]

        if target_labels.dim() == 3 and target_labels.shape[-1] == 1:
            target_labels = target_labels.squeeze(-1)

        batch_size, horizon, n_phases = pred_logits.shape

        pred_flat = pred_logits.reshape(-1, n_phases)
        target_flat = target_labels.reshape(-1)

        if is_pad is not None:
            is_pad_flat = is_pad.reshape(-1)
            pred_flat = pred_flat[~is_pad_flat]
            target_flat = target_flat[~is_pad_flat]

        ce_loss = F.cross_entropy(
            pred_flat, target_flat, label_smoothing=self.label_smoothing
        )

        component_losses = {MetricKey.PHASE_CROSS_ENTROPY.value: ce_loss}
        total_loss = self.cross_entropy_weight * ce_loss

        if self.entropy_weight != 0.0:
            probs = F.softmax(pred_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            entropy_reduced = reduce_loss_with_padding(
                entropy, is_pad, reduction="mean"
            )
            component_losses[MetricKey.PHASE_ENTROPY.value] = entropy_reduced
            # Entropy is always positive
            # We want to maximize entropy so we need to subtract it from the loss.
            total_loss = total_loss - self.entropy_weight * entropy_reduced

        metadata = {
            MetadataKey.PHASE_LOGITS.value: pred_logits.detach(),
            MetadataKey.PHASE_LABEL.value: target_labels.detach(),
        }

        return LossOutput(
            total_loss=total_loss,
            component_losses=component_losses,
            metadata=metadata,
        )


class ActionTokenLoss(BaseLoss):
    """Cross-entropy loss for tokenized actions."""

    def __init__(
        self,
        label_smoothing: float = 0.2,
    ):
        """Initialize action token loss.

        Args:
            label_smoothing: Label smoothing factor [0, 1]
        """
        super().__init__()
        self.label_smoothing = label_smoothing

    def get_required_keys(self) -> set[str]:
        """Get required keys from predictions.

        Returns:
            Empty set since target ground-truth tokens are in predictions
        """
        return {DecoderOutputKey.ACTION_LOGITS.value}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute cross-entropy loss for tokenized actions.

        Args:
            predictions: Dictionary containing:
                - '{DecoderOutputKey.ACTION_LOGITS.value}': logits (B, horizon, vocab_size)
                - '{SampleKey.TOKENIZED_ACTIONS.value}': ground truth token IDs (B, horizon)
                - 'is_pad': optional padding mask (B, horizon)
            targets: Dictionary containing ground truth tokens
            is_pad: Optional padding mask

        Returns:
            LossOutput with per-key cross-entropy losses
        """
        if DecoderOutputKey.ACTION_LOGITS.value not in predictions:
            raise ValueError(
                f"Predictions must contain keys '{DecoderOutputKey.ACTION_LOGITS.value}' for ActionTokenLoss."
            )
        pred_logits = predictions[
            DecoderOutputKey.ACTION_LOGITS.value
        ]  # (B, num_tokens, vocab_size)
        target_tokens = targets[SampleKey.TOKENIZED_ACTIONS.value]  # (B, num_tokens)
        token_sequence_dim = 1
        vocabulary_size_dim = 2
        logits = pred_logits.transpose(
            token_sequence_dim, vocabulary_size_dim
        )  # (B, vocab_size, num_tokens)
        ce_loss = F.cross_entropy(
            logits,
            target_tokens,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        ce_loss = reduce_loss_with_padding(ce_loss, is_pad, reduction="mean")
        predicted_tokens = torch.argmax(
            pred_logits, dim=-1
        )  # (B, seq) over C=dim=-1 (no view needed)
        correct = (predicted_tokens == target_tokens).float()  # (B, seq)
        accuracy = reduce_loss_with_padding(
            correct, is_pad, reduction="mean"
        )  # Scalar %
        perplexity = torch.exp(ce_loss)  # Scalar
        return LossOutput(
            total_loss=ce_loss,
            component_losses={
                MetricKey.ACTION_TOKEN_CROSS_ENTROPY.value: ce_loss,
                MetricKey.PERPLEXITY.value: perplexity,
                MetricKey.TOKEN_ACCURACY.value: accuracy,
            },
        )


class PriorDenoisingLoss(BaseLoss):
    """Denoising loss for learned diffusion prior.

    Computes MSE loss between predicted noise and target noise from the
    diffusion prior. Used in variational models to train the prior p(z|s)
    to match the posterior q(z|a,s).
    """

    def __init__(self, weight: float = 1.0):
        """Initialize prior denoising loss.

        Args:
            weight: Weight for this loss component
        """
        super().__init__()
        self.weight = weight

    def get_required_keys(self) -> set[str]:
        """Return required prediction keys."""
        return {LatentKey.PRIOR_PREDICTION.value, LatentKey.PRIOR_TARGET.value}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute prior denoising loss.

        Args:
            predictions: Dictionary containing LatentKey.PRIOR_PREDICTION.value and LatentKey.PRIOR_TARGET.value
            targets: Not used (targets are in predictions dict)
            is_pad: Not used (prior loss doesn't need padding)

        Returns:
            LossOutput with weighted MSE loss

        Raises:
            ValueError: If required keys are missing from predictions
        """
        if LatentKey.PRIOR_PREDICTION.value not in predictions:
            raise ValueError(
                f"Predictions must contain '{LatentKey.PRIOR_PREDICTION.value}' for PriorDenoisingLoss."
            )
        if LatentKey.PRIOR_TARGET.value not in predictions:
            raise ValueError(
                f"Predictions must contain '{LatentKey.PRIOR_TARGET.value}' for PriorDenoisingLoss."
            )
        prior_loss = F.mse_loss(
            predictions[LatentKey.PRIOR_PREDICTION.value],
            predictions[LatentKey.PRIOR_TARGET.value],
        )
        metadata: dict[str, torch.Tensor] = {}
        if LatentKey.POSTERIOR_LATENT.value in predictions:
            metadata[MetadataKey.POSTERIOR_Z.value] = predictions[
                LatentKey.POSTERIOR_LATENT.value
            ]
        if LatentKey.POSTERIOR_MU.value in predictions:
            metadata[MetadataKey.POSTERIOR_MU.value] = predictions[
                LatentKey.POSTERIOR_MU.value
            ]
        if LatentKey.POSTERIOR_LOGVAR.value in predictions:
            metadata[MetadataKey.POSTERIOR_LOGVAR.value] = predictions[
                LatentKey.POSTERIOR_LOGVAR.value
            ]
        if LatentKey.PRIOR_LATENT.value in predictions:
            metadata[MetadataKey.PRIOR_Z.value] = predictions[
                LatentKey.PRIOR_LATENT.value
            ]
        if LatentKey.PRIOR_MU.value in predictions:
            metadata[MetadataKey.PRIOR_MU.value] = predictions[LatentKey.PRIOR_MU.value]
        if LatentKey.PRIOR_LOGVAR.value in predictions:
            metadata[MetadataKey.PRIOR_LOGVAR.value] = predictions[
                LatentKey.PRIOR_LOGVAR.value
            ]

        return LossOutput(
            total_loss=self.weight * prior_loss,
            component_losses={MetricKey.PRIOR_DENOISING_LOSS.value: prior_loss},
            metadata=metadata,
        )


class GaussianMixtureNLLoss(BaseLoss):
    """Negative Log-Likelihood loss for Gaussian Mixture Model.

    Supports both learned variance (from logvar predictions) and fixed variance (sigma parameter).

    The loss computes: NLL = -log Σ_k π_k · N(a | μ_k, σ_k²)
    """

    def __init__(
        self,
        action_keys: list[str],
        weight: float = 1.0,
        per_key_weights: dict[str, float] | None = None,
        learned_variance: bool = True,
        sigmas: dict[str, float] | None = None,
        min_variance: float = 1e-4,
    ):
        """Initialize Gaussian mixture NLL loss.

        Args:
            action_keys: List of continuous action keys.
            weight: Overall loss weight.
            per_key_weights: Optional per-key weights.
            learned_variance: If True, expects {action_key}_mean and {action_key}_logvar.
                If False, expects {action_key} (stacked means) and uses sigmas.
            sigmas: Fixed stddev per action key (only used when learned_variance=False).
            min_variance: Minimum variance for numerical stability (learned_variance=True).
        """
        super().__init__()
        self.action_keys = action_keys
        self.weight = weight
        self.per_key_weights = per_key_weights or dict.fromkeys(action_keys, 1.0)
        self.learned_variance = learned_variance
        self.min_variance = min_variance
        if not learned_variance:
            self.sigmas = sigmas or dict.fromkeys(action_keys, 0.5)

    def get_required_keys(self) -> set[str]:
        """Get required target keys."""
        return set(self.action_keys)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute Gaussian mixture NLL loss.

        Args:
            predictions: Dictionary containing:
                - routing_weights: (B, K) or (B, T, K)
                - If learned_variance: {action_key}_mean (B, T, K, D), {action_key}_logvar (B, T, K, D)
                - If fixed variance: {action_key} (B, T, K, D) stacked expert means
            targets: Dictionary with action_key targets (B, T, D).
            is_pad: Optional padding mask (B, T).

        Returns:
            LossOutput with Gaussian mixture NLL.
        """
        component_losses = {}
        total_loss = 0.0
        mixing_probs = predictions[DecoderOutputKey.ROUTING_WEIGHTS.value]
        for action_key in self.action_keys:
            target = targets[action_key]  # (B, T, D)
            if self.learned_variance:
                mean_key = f"{action_key}_{DecoderOutputKey.MEAN.value}"
                logvar_key = f"{action_key}_{DecoderOutputKey.LOGVAR.value}"
                means = predictions[mean_key]  # (B, T, K, D)
                logvars = predictions[logvar_key]  # (B, T, K, D)
                nll = self._compute_learned_variance_nll(
                    target, mixing_probs, means, logvars
                )
            else:
                means = predictions[action_key]  # (B, T, K, D)
                sigma = self.sigmas.get(action_key, 0.5)
                nll = self._compute_fixed_variance_nll(
                    target, mixing_probs, means, sigma
                )
            nll_reduced = reduce_loss_with_padding(nll, is_pad, reduction="mean")
            key_weight = self.per_key_weights.get(action_key, 1.0)
            component_losses[f"{action_key}_{MetricKey.GAUSSIAN_MIXTURE_NLL.value}"] = (
                nll_reduced
            )
            total_loss = total_loss + key_weight * nll_reduced
        return LossOutput(
            total_loss=self.weight * total_loss, component_losses=component_losses
        )

    def _compute_learned_variance_nll(
        self,
        target: torch.Tensor,
        mixing_probs: torch.Tensor,
        means: torch.Tensor,
        logvars: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NLL with learned variance from logvar predictions."""
        action_dimension = target.shape[-1]
        logvars = logvars.clamp(min=math.log(self.min_variance))
        target = target.unsqueeze(2)  # (B, T, 1, D)
        difference = target - means  # (B, T, K, D)
        scaled_squared_error = (difference**2) * torch.exp(-logvars)  # (B, T, K, D)
        log_normalization = -0.5 * action_dimension * math.log(2 * math.pi)
        log_gaussian = log_normalization - 0.5 * (logvars + scaled_squared_error).sum(
            dim=-1
        )
        if mixing_probs.dim() == 2:
            mixing_probs = mixing_probs.unsqueeze(1)  # (B, 1, K)
        log_mixing_weights = torch.log(mixing_probs + 1e-8)  # (B, T, K)
        log_mixture_prob = torch.logsumexp(log_mixing_weights + log_gaussian, dim=-1)
        return -log_mixture_prob

    def _compute_fixed_variance_nll(
        self,
        target: torch.Tensor,
        mixing_probs: torch.Tensor,
        means: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Compute NLL with fixed variance (sigma parameter)."""
        target = target.unsqueeze(2)  # (B, T, 1, D)
        difference = target - means  # (B, T, K, D)
        log_gaussian = -0.5 * (difference**2).sum(-1) / (sigma**2)  # (B, T, K)
        if mixing_probs.dim() == 2:
            mixing_probs = mixing_probs.unsqueeze(1)  # (B, 1, K)
        log_mixing_weights = torch.log(mixing_probs + 1e-8)  # (B, T, K)
        log_mixture_prob = torch.logsumexp(log_mixing_weights + log_gaussian, dim=-1)
        return -log_mixture_prob


class GripperMixtureNLLoss(BaseLoss):
    """Negative Log-Likelihood loss for gripper with mixture distribution.

    Binary gripper: p(a|z) = Σ_k π_k(z) · Bernoulli(a | p_k(z))
    Continuous gripper: p(a|z) = Σ_k π_k(z) · N(a | μ_k(z), σ_k²)

    Supports both fixed and learned variance for continuous gripper.
    """

    def __init__(
        self,
        key: str,
        actions_metadata: dict[str, ActionMetadata],
        weight: float = 1.0,
        learned_variance: bool = False,
        sigma: float = 0.5,
        min_variance: float = 1e-4,
    ):
        """Initialize gripper mixture NLL loss.

        Args:
            key: Key for gripper actions.
            actions_metadata: Dict of metadata of the action space.
            weight: Loss weight.
            learned_variance: If True, expects {key}_mean and {key}_logvar for continuous.
                If False, expects {key} (stacked means) and uses sigma.
            sigma: Fixed std for continuous gripper (only used when learned_variance=False).
            min_variance: Minimum variance for numerical stability (learned_variance=True).
        """
        super().__init__()
        self.key = key
        self.weight = weight
        self.learned_variance = learned_variance
        self.sigma = sigma
        self.min_variance = min_variance
        resolved_metadata = resolve_dict_keys(dict(actions_metadata))
        if key not in resolved_metadata:
            raise ValueError(
                f"{key} is not available to the action space. Can't compute gripper NLL loss. "
                f"Available keys: {list(resolved_metadata)}"
            )
        meta = resolved_metadata[key]
        if isinstance(meta, GripperActionMetadata):
            self.gripper_type = meta.gripper_type
            self.binary_gripper_range = meta.binary_gripper_range
        elif isinstance(meta, OnTheFlyActionMetadata):
            source = meta.source_metadata
            if isinstance(source, GripperObservationMetadata):
                self.gripper_type = source.gripper_type
                self.binary_gripper_range = source.binary_gripper_range
            else:
                raise ValueError(
                    f"Expected GripperObservationMetadata for key '{key}', got {type(source).__name__}"
                )
        else:
            raise ValueError(
                f"Expected gripper metadata for key '{key}', got {type(meta).__name__}"
            )

    def get_required_keys(self) -> set[str]:
        return {self.key}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute gripper mixture NLL.

        Args:
            predictions: Dictionary containing:
                - routing_weights: (B, K) or (B, T, K) mixing probabilities
                - For binary: {key} stacked logits (B, T, K) or (B, T, K, 1)
                - For continuous with learned_variance: {key}_mean (B, T, K, D), {key}_logvar (B, T, K, D)
                - For continuous with fixed variance: {key} stacked means (B, T, K, D)
            targets: Dictionary with gripper targets.
            is_pad: Optional padding mask (B, T).

        Returns:
            LossOutput with gripper NLL.
        """
        if self.key not in targets:
            raise ValueError(
                f"Targets must contain '{self.key}' for GripperMixtureNLLoss."
            )
        target = targets[self.key]
        mixing_probs = predictions[DecoderOutputKey.ROUTING_WEIGHTS.value]
        if mixing_probs.dim() == 2:
            mixing_probs = mixing_probs.unsqueeze(1)  # (B, 1, K)
        log_mixing_weights = torch.log(mixing_probs + 1e-8)  # (B, T, K)
        if self.gripper_type == GripperType.BINARY.value:
            log_component = self._compute_binary_log_component(predictions, target)
        else:
            log_component = self._compute_continuous_log_component(predictions, target)

        log_mixture_prob = torch.logsumexp(log_mixing_weights + log_component, dim=-1)
        nll = -log_mixture_prob
        nll_reduced = reduce_loss_with_padding(nll, is_pad, reduction="mean")
        return LossOutput(
            total_loss=self.weight * nll_reduced,
            component_losses={MetricKey.GRIPPER_NLL.value: nll_reduced},
        )

    def _compute_binary_log_component(
        self, predictions: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Compute log Bernoulli component for binary gripper."""
        expert_logits = predictions[self.key]  # (B, T, K) or (B, T, K, 1)
        if target.dim() == 3:
            target = target.squeeze(-1)
        if expert_logits.dim() == 4:
            expert_logits = expert_logits.squeeze(-1)
        if self.binary_gripper_range == BinaryGripperRange.MINUS_ONE_ONE.value:
            target = (target.float() + 1.0) / 2.0
        log_probability = F.logsigmoid(expert_logits)  # (B, T, K)
        log_one_minus_probability = F.logsigmoid(-expert_logits)  # (B, T, K)
        target_expanded = target.unsqueeze(-1)  # (B, T, 1)
        log_bernoulli = (
            target_expanded * log_probability
            + (1 - target_expanded) * log_one_minus_probability
        )
        return log_bernoulli  # (B, T, K)

    def _compute_continuous_log_component(
        self, predictions: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Compute log Gaussian component for continuous gripper."""
        target_expanded = target.unsqueeze(2)  # (B, T, 1, D)
        if self.learned_variance:
            mean_key = f"{self.key}_{DecoderOutputKey.MEAN.value}"
            logvar_key = f"{self.key}_{DecoderOutputKey.LOGVAR.value}"
            means = predictions[mean_key]  # (B, T, K, D)
            logvars = predictions[logvar_key].clamp(min=math.log(self.min_variance))
            difference = target_expanded - means
            scaled_squared_error = (difference**2) * torch.exp(-logvars)
            action_dimension = target.shape[-1]
            log_normalization = -0.5 * action_dimension * math.log(2 * math.pi)
            return log_normalization - 0.5 * (logvars + scaled_squared_error).sum(
                dim=-1
            )
        else:
            means = predictions[self.key]  # (B, T, K, D)
            difference = target_expanded - means
            return -0.5 * (difference**2).sum(dim=-1) / (self.sigma**2)  # (B, T, K)


class MoELoss(BaseLoss):
    """Wrapper for any BaseLoss to add MoE expert usage metric from routing weights."""

    def __init__(
        self,
        base_loss: BaseLoss,
        entropy_weight: float = 0.0,
    ):
        """Initialize MoE wrapper.

        Args:
            base_loss: Any BaseLoss instance to wrap (e.g., RegressionLoss(...))
            entropy_weight: Weight for entropy regularization on global routing weights.

        Note: The entropy term enforces the use of multiple experts and tries to prevent the model
         to only select a few of the available experts.
        """
        super().__init__()
        self.base_loss = base_loss
        self.entropy_weight = entropy_weight

    def get_callbacks(self, experiment_config: ExperimentConfig) -> list:
        """Provide expert usage monitoring callback."""
        return [ExpertUsageCallback(log_every_n_epochs=1)]

    def get_required_keys(self) -> set[str]:
        """Union of base loss keys plus routing weight."""
        return self.base_loss.get_required_keys() | {
            DecoderOutputKey.ROUTING_WEIGHTS.value
        }

    def _add_weighted_mean_predictions(
        self, predictions: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Add weighted mean predictions for GMM outputs.

        For each {key}_mean (B, T, K, D), computes weighted mean using routing_weights
        and adds it as {key} (B, T, D). This allows base losses to use standard action keys.
        """
        routing_weights = predictions[DecoderOutputKey.ROUTING_WEIGHTS.value]
        mean_suffix = f"_{DecoderOutputKey.MEAN.value}"
        augmented = dict(predictions)

        for key, value in predictions.items():
            if not key.endswith(mean_suffix):
                continue
            base_key = key[: -len(mean_suffix)]
            if base_key in predictions:
                continue
            # value: (B, T, K, D), routing_weights: (B, K) or (B, T, K)
            weights = routing_weights
            if weights.dim() == 2:
                weights = weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, K, 1)
            elif weights.dim() == 3:
                weights = weights.unsqueeze(-1)  # (B, T, K, 1)
            weighted_mean = (value * weights).sum(dim=2)  # (B, T, D)
            augmented[base_key] = weighted_mean
        return augmented

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Passthrough base loss, then add expert_usage from routing weights and optionally add entropy term."""
        predictions = self._add_weighted_mean_predictions(predictions)
        base_output: LossOutput = self.base_loss(predictions, targets, is_pad)
        metadata = base_output.metadata if base_output.metadata is not None else {}
        component_losses = dict(base_output.component_losses)
        entropy_loss = 0.0
        pi = predictions[DecoderOutputKey.ROUTING_WEIGHTS.value]  # (B, K) or (B, T, K)
        if self.entropy_weight != 0.0:
            entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=-1)  # (B, T)
            if entropy.dim() == 2:
                # (B, T) -> reduce with padding
                entropy_mean = reduce_loss_with_padding(
                    entropy, is_pad, reduction="mean"
                )
            else:
                # if we have B only (when the experts are not chunk-dependent)
                entropy_mean = entropy.mean()
            component_losses[f"{MetricKey.EXPERTS_ENTROPY.value}"] = entropy_mean
            # We want to maximize the entropy of the expert usage distribution
            # Entropy is always positive, so we subtract it to the loss to maximize it.
            entropy_loss = -self.entropy_weight * entropy_mean
        expert_usage = pi.mean(
            dim=list(range(pi.ndim - 1))
        )  # Mean over all but last dim, which is num_experts
        metadata[MetadataKey.EXPERT_USAGE.value] = expert_usage
        return LossOutput(
            total_loss=base_output.total_loss + entropy_loss,
            component_losses=component_losses,
            metadata=metadata,
        )


class MetadataPassthrough(BaseLoss):
    """Passthrough to add target keys to metadata without computing loss.

    Useful for adding auxiliary data to metadata for
    visualization/analysis without affecting training.
    """

    def __init__(self, keys_mapping: dict[str, str]):
        """Initialize metadata passthrough.

        Args:
            keys_mapping: Mapping from target keys to metadata keys.
                Example: {"phase_label": "phase_label"} extracts targets["phase_label"]
                and stores it in metadata["phase_label"].
        """
        super().__init__()
        self.keys_mapping = resolve_dict_keys(dict(keys_mapping))

    def get_required_keys(self) -> set[str]:
        """Get required target keys."""
        return set(self.keys_mapping.keys())

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Extract keys from targets and add to metadata."""
        device = next(iter(predictions.values())).device
        metadata = {}
        for target_key, metadata_key in self.keys_mapping.items():
            if target_key in targets:
                metadata[metadata_key] = targets[target_key].detach()
        return LossOutput(
            total_loss=torch.tensor(0.0, device=device),
            component_losses={},
            metadata=metadata,
        )


class VICLatentLoss(BaseLoss):
    """VICReg-style covariance + variance loss for latent decorrelation and anti-collapse.

    Note:
        Combines two regularization terms:
        - Covariance: Penalizes off-diagonal covariance to encourage independent dimensions
        - Variance: Hinge loss forcing std >= gamma per dimension to prevent collapse
        Ref. https://arxiv.org/pdf/2105.04906
    """

    def __init__(
        self,
        key: str = LatentKey.POSTERIOR_MU.value,
        covariance_weight: float = 3.0,
        variance_weight: float = 10.0,
        gamma: float = 0.3,
    ):
        """Initialize VICReg latent loss.

        Args:
            key: Prediction key for latent mu tensor.
            covariance_weight: Weight for off-diagonal covariance penalty.
            variance_weight: Weight for variance hinge loss.
            gamma: Hinge threshold for per-dimension standard deviation.
        """
        super().__init__()
        self.key = key
        self.covariance_weight = covariance_weight
        self.variance_weight = variance_weight
        self.gamma = gamma

    def get_required_keys(self) -> set[str]:
        """Get required prediction keys."""
        return {self.key}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute VICReg loss combining covariance and variance terms.

        Args:
            predictions: Must contain self.key with shape (B, latent_dim).
            targets: Unused.
            is_pad: Unused.

        Returns:
            LossOutput with weighted covariance and variance penalties.
        """
        if self.key not in predictions:
            raise ValueError(
                f"Predictions must contain '{self.key}' for VICLatentLoss."
            )
        latent_vectors = predictions[self.key].float()
        batch_size, latent_dimension = latent_vectors.shape
        centered = latent_vectors - latent_vectors.mean(dim=0)
        standard_deviation = torch.sqrt(centered.var(dim=0) + 1e-6)
        variance_loss = torch.mean(F.relu(self.gamma - standard_deviation))
        covariance = (centered.T @ centered) / (batch_size - 1)
        diagonal_mask = torch.eye(latent_dimension, device=latent_vectors.device)
        off_diagonal_covariance = covariance * (1 - diagonal_mask)
        covariance_loss = off_diagonal_covariance.pow(2).sum() / latent_dimension
        total_loss = (
            self.covariance_weight * covariance_loss
            + self.variance_weight * variance_loss
        )
        return LossOutput(
            total_loss=total_loss,
            component_losses={
                MetricKey.COVARIANCE_LOSS.value: self.covariance_weight
                * covariance_loss,
                MetricKey.VARIANCE_LOSS.value: self.variance_weight * variance_loss,
            },
        )
