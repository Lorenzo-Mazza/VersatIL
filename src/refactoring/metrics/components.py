"""Individual loss components for action prediction tasks."""
import math

import torch
import torch.nn.functional as F

from refactoring.common.omegaconf_ops import resolve_dict_keys
from refactoring.data.constants import (
    BinaryGripperRange,
    GripperType,
    TOKENIZED_ACTIONS_KEY,
)
from refactoring.data.metadata import (
    ActionMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
)
from refactoring.metrics.base import BaseLoss, LossOutput, reduce_loss_with_padding
from refactoring.metrics.constants import (
    MetadataKey,
    MetricKey,
)
from refactoring.models.decoding.constants import (
    PRIOR_PREDICTION_KEY,
    PRIOR_TARGET_KEY,
    BINARY_LOGITS_KEY,
    MU_KEY,
    LOGVAR_KEY,
    ACTION_LOGITS_KEY,
    LATENT_CODES,
    ROUTING_WEIGHT,
    LATENT_KEY,
    EXPERT_OUTPUTS,
    PRIOR_MU_KEY,
    PRIOR_LOGVAR_KEY,
    PRIOR_LATENT_KEY,
    PRIOR_LOG_PROB_KEY,
)


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
        if key not in resolved_metadata.keys():
            raise ValueError(
                f"{key} is not available to the action space. Can't compute gripper loss. "
                f"Available keys: {list(resolved_metadata.keys())}"
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
        key: str = PRIOR_LOGVAR_KEY,
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
        if 'logvar' not in key:
            raise ValueError(
                f"GaussianEntropyLoss expects a logvar key, got '{key}'."
            )
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
            LATENT_KEY,
            PRIOR_LATENT_KEY,
            MU_KEY,
            LOGVAR_KEY,
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
        if PRIOR_LOG_PROB_KEY in predictions:
            mu_q = predictions[MU_KEY].float()
            logvar_q = predictions[LOGVAR_KEY].float()
            std_q = (0.5 * logvar_q).exp()
            z = predictions[LATENT_KEY]
            log_p_z = predictions[PRIOR_LOG_PROB_KEY]  # (B,)
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
                MetadataKey.PRIOR_Z.value: predictions.get(PRIOR_LATENT_KEY),
            }

            return LossOutput(
                total_loss=total_loss,
                component_losses=component_losses,
                metadata=metadata,
            )

        # Standard Gaussian prior - uses closed-form KL
        required_keys = self.get_required_keys()
        required_keys.update({PRIOR_MU_KEY, PRIOR_LOGVAR_KEY})
        if not all(k in predictions for k in required_keys):
            raise ValueError(
                f"Predictions must contain '{required_keys}' for KLDivergenceLoss."
            )
        mu_posterior = predictions[MU_KEY].float()  # Using fp32 float for stability
        logvar_posterior = predictions[LOGVAR_KEY].float()
        mu_prior = predictions[PRIOR_MU_KEY].float()
        logvar_prior = predictions[PRIOR_LOGVAR_KEY].float()
        std_posterior = (0.5 * logvar_posterior).exp()
        std_prior = (0.5 * logvar_prior).exp()
        posterior = torch.distributions.Normal(mu_posterior, std_posterior)
        prior = torch.distributions.Normal(mu_prior, std_prior)
        kld = torch.distributions.kl_divergence(posterior, prior).sum(dim=-1)
        if kld.min() < 0:
            print(
                f"Warning: Negative KL divergence encountered: min={kld.min().item():.4f}"
            )
            print(f"per_dim_kl: min={kld.min().item():.4f}, max={kld.max().item():.4f}")
        kld_mean = kld.mean()
        component_losses = {MetricKey.KL_DIVERGENCE.value: kld_mean}
        total_loss = self.weight * kld_mean
        if self.prior_regularization_weight > 0.0:
            # KL(N(μ, σ²) || N(0, I)) = 0.5 * sum(μ² + σ² - log(σ²) - 1)
            prior_kl = 0.5 * (
                mu_prior.pow(2) + logvar_prior.exp() - logvar_prior - 1
            ).sum(dim=-1)
            prior_kl_mean = prior_kl.mean()
            component_losses[MetricKey.HYPERPRIOR_KL_REGULARIZATION.value] = prior_kl_mean
            total_loss = total_loss + self.prior_regularization_weight * prior_kl_mean

        metadata = {
            MetadataKey.POSTERIOR_Z.value: predictions[LATENT_KEY],
            MetadataKey.POSTERIOR_MU.value: mu_posterior,
            MetadataKey.POSTERIOR_LOGVAR.value: logvar_posterior,
            MetadataKey.PRIOR_Z.value: predictions[PRIOR_LATENT_KEY],
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
        return {BINARY_LOGITS_KEY}

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
        if BINARY_LOGITS_KEY not in predictions:
            raise ValueError(
                f"Predictions must contain key '{BINARY_LOGITS_KEY}' for BinaryKLDivergenceLoss."
            )
        all_component_losses = {}
        if LATENT_CODES in predictions:
            latent_codes = predictions[LATENT_CODES]  # (B, token_len, 2^H)
            code_indices = torch.argmax(
                latent_codes, dim=-1
            ).flatten()  # (B*token_len,)
            unique_codes = torch.unique(code_indices).numel()
            total_codes = 2 ** self.latent_bits
            usage_pct = unique_codes / total_codes
            all_component_losses[MetricKey.LATENT_CODE_USAGE.value] = usage_pct

        logits = predictions[BINARY_LOGITS_KEY]  # (B, T, H) or (B, H)
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

    Note:
        From Info-VAE/MMD-VAE (https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/)
         Uses RBF kernel for robust distribution matching, to encourage q(z|x) ≈ p(z)
         where p(z) = N(0, I).
    """

    def __init__(
        self,
        weight: float = 1.0,
        prior_regularization_weight: float = 0.0,
        kernel_bandwidths: list[float] | None = None,
    ):
        """Initialize MMD loss.

        Args:
            weight: Loss weight for MMD(posterior, prior).
            prior_regularization_weight: Weight for MMD(prior, N(0,I)) regularization.
                Only meaningful for learned priors. Pushes the learned prior towards
                a standard Gaussian.
            kernel_bandwidths: Optional list of bandwidths for multi-scale RBF kernel.
        """
        super().__init__()
        self.weight = weight
        self.prior_regularization_weight = prior_regularization_weight
        if kernel_bandwidths is None:
            kernel_bandwidths = [0.1, 1.0, 10.0]
        self.kernel_bandwidths = kernel_bandwidths

    def get_required_keys(self) -> set[str]:
        """Get required keys for MMD loss."""
        return {
            LATENT_KEY,
            PRIOR_LATENT_KEY,
            MU_KEY,
            LOGVAR_KEY,
        }

    def _compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Multi-Scale RBF kernel with Median Heuristic.

        Uses implicit bandwidth σ² ∝ dim for scale invariance across
        different latent dimensionalities.

        Args:
            x: First set of points (N, D).
            y: Second set of points (M, D).

        Returns:
            Kernel matrix (N, M).
        """
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))
        if y.dim() > 2:
            y = y.reshape(-1, y.size(-1))
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        dist_sq = torch.clamp(dist_sq, min=1e-6)
        # We want sigma^2 to be the median squared distance in the combined batch.
        median_dist = torch.median(dist_sq.detach())
        sigma_sq = median_dist
        if sigma_sq < 1e-6:
            sigma_sq = torch.tensor(1.0, device=x.device)
        kernel_val = torch.zeros_like(dist_sq)
        for multiplier in self.kernel_bandwidths:
            bandwidth = 2.0 * multiplier * sigma_sq
            kernel_val += torch.exp(-dist_sq / bandwidth)
        return kernel_val / len(self.kernel_bandwidths)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute MMD between latent samples and standard Gaussian prior.

        Args:
            predictions: Must contain LATENT_KEY with shape (B, latent_dim).
            targets: Unused (prior is implicit).
            is_pad: Unused.

        Returns:
            LossOutput with MMD loss.
        """
        is_mixture_prior = PRIOR_LOG_PROB_KEY in predictions
        required_keys = {LATENT_KEY, PRIOR_LATENT_KEY, MU_KEY, LOGVAR_KEY}
        if not is_mixture_prior:
            required_keys.update({PRIOR_MU_KEY, PRIOR_LOGVAR_KEY})

        if not all(k in predictions for k in required_keys):
            raise ValueError(
                f"Predictions must contain '{required_keys}' for MaximumMeanDiscrepancyLoss."
            )

        z_posterior = predictions[LATENT_KEY]  # (B, latent_dim)
        z_prior = predictions[PRIOR_LATENT_KEY]
        k_zz = self._compute_kernel(z_posterior, z_posterior)
        k_pp = self._compute_kernel(z_prior, z_prior)
        k_zp = self._compute_kernel(z_posterior, z_prior)
        # MMD² = E[k(z,z')] + E[k(p,p')] - 2E[k(z,p)]
        mmd = k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()

        component_losses = {MetricKey.MMD_LOSS.value: mmd}
        total_loss = self.weight * mmd

        if self.prior_regularization_weight > 0.0:
            z_standard = torch.randn_like(z_prior)  # Samples from N(0, I)
            k_pp_standard = self._compute_kernel(z_prior, z_prior)
            k_ss = self._compute_kernel(z_standard, z_standard)
            k_ps = self._compute_kernel(z_prior, z_standard)
            prior_mmd = k_pp_standard.mean() + k_ss.mean() - 2 * k_ps.mean()
            component_losses[MetricKey.HYPERPRIOR_MMD_REGULARIZATION.value] = prior_mmd
            total_loss = total_loss + self.prior_regularization_weight * prior_mmd

        metadata = {
            MetadataKey.POSTERIOR_Z.value: z_posterior,
            MetadataKey.POSTERIOR_MU.value: predictions[MU_KEY],
            MetadataKey.POSTERIOR_LOGVAR.value: predictions[LOGVAR_KEY],
            MetadataKey.PRIOR_Z.value: z_prior,
        }
        # Only include prior mu/logvar for Gaussian priors
        if not is_mixture_prior:
            metadata[MetadataKey.PRIOR_MU.value] = predictions[PRIOR_MU_KEY]
            metadata[MetadataKey.PRIOR_LOGVAR.value] = predictions[PRIOR_LOGVAR_KEY]

        return LossOutput(
            total_loss=total_loss,
            component_losses=component_losses,
            metadata=metadata,
        )


class BinaryMaximumMeanDiscrepancyLoss(BaseLoss):
    """MMD loss for regularizing binary latent distributions toward a uniform prior.

    Uses RBF kernel for robust distribution matching, to encourage q(b|x) ≈ p(b)
    where p(b) = Bernoulli(0.5) independent for each bit.
    """

    def __init__(
        self,
        weight: float = 1.0,
    ):
        """Initialize binary MMD loss.

        Args:
            weight: Loss weight.
        """
        super().__init__()
        self.weight = weight

    def get_required_keys(self) -> set[str]:
        """Returns required prediction keys: {BINARY_LOGITS_KEY}."""
        return {BINARY_LOGITS_KEY}

    def _compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Multi-Scale RBF kernel with Median Heuristic.

        Uses implicit bandwidth σ² ∝ dim for scale invariance across
        different latent dimensionalities.

        Args:
            x: First set of points (N, D).
            y: Second set of points (M, D).

        Returns:
            Kernel matrix (N, M).
        """
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))
        if y.dim() > 2:
            y = y.reshape(-1, y.size(-1))
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        dist_sq = torch.clamp(dist_sq, min=1e-6)
        # We want sigma^2 to be the median squared distance in the combined batch.
        median_dist = torch.median(dist_sq.detach())
        sigma_sq = median_dist
        if sigma_sq < 1e-6:
            sigma_sq = torch.tensor(1.0, device=x.device)
        kernel_val = torch.zeros_like(dist_sq)
        for multiplier in self.kernel_bandwidths:
            bandwidth = 2.0 * multiplier * sigma_sq
            kernel_val += torch.exp(-dist_sq / bandwidth)
        return kernel_val / len(self.kernel_bandwidths)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute MMD between binary latent samples and uniform Bernoulli prior.

        Args:
            predictions: Must contain BINARY_LOGITS_KEY with shape (B, H).
            targets: Unused (prior is implicit).
            is_pad: Unused.

        Returns:
            LossOutput with MMD loss.
        """
        if BINARY_LOGITS_KEY not in predictions:
            raise ValueError(
                f"Predictions must contain '{BINARY_LOGITS_KEY}'for BinaryMaximumMeanDiscrepancyLoss."
            )

        logits = predictions[BINARY_LOGITS_KEY]  # (B, T, H)
        probs = torch.sigmoid(logits.float())  # Cast to fp32 for stability
        z_hard = torch.bernoulli(probs)
        z = (
            z_hard - probs.detach() + probs
        )  # Straight-through: forward=hard, backward=soft
        z_prior = torch.bernoulli(
            0.5 * torch.ones_like(z)
        )  # samples from Bernoulli(0.5)
        k_zz = self._compute_kernel(z, z)
        k_pp = self._compute_kernel(z_prior, z_prior)
        k_zp = self._compute_kernel(z, z_prior)
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
            MetadataKey.PHASE_LABELS.value: target_labels.detach(),
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
        return {ACTION_LOGITS_KEY}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute cross-entropy loss for tokenized actions.

        Args:
            predictions: Dictionary containing:
                - '{ACTION_LOGITS_KEY}': logits (B, horizon, vocab_size)
                - '{TOKENIZED_ACTIONS_KEY}': ground truth token IDs (B, horizon)
                - 'is_pad': optional padding mask (B, horizon)
            targets: Dictionary containing ground truth tokens
            is_pad: Optional padding mask

        Returns:
            LossOutput with per-key cross-entropy losses
        """
        if ACTION_LOGITS_KEY not in predictions:
            raise ValueError(
                f"Predictions must contain keys '{ACTION_LOGITS_KEY}' for ActionTokenLoss."
            )
        pred_logits = predictions[ACTION_LOGITS_KEY]  # (B, num_tokens, vocab_size)
        target_tokens = targets[TOKENIZED_ACTIONS_KEY]  # (B, num_tokens)
        vocab_size = pred_logits.shape[-1]
        num_tokens = pred_logits.shape[1]
        logits = pred_logits.view(
            -1, vocab_size, num_tokens
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
        return {PRIOR_PREDICTION_KEY, PRIOR_TARGET_KEY}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute prior denoising loss.

        Args:
            predictions: Dictionary containing PRIOR_PREDICTION_KEY and PRIOR_TARGET_KEY
            targets: Not used (targets are in predictions dict)
            is_pad: Not used (prior loss doesn't need padding)

        Returns:
            LossOutput with weighted MSE loss

        Raises:
            ValueError: If required keys are missing from predictions
        """
        if PRIOR_PREDICTION_KEY not in predictions:
            raise ValueError(
                f"Predictions must contain '{PRIOR_PREDICTION_KEY}' for PriorDenoisingLoss."
            )
        if PRIOR_TARGET_KEY not in predictions:
            raise ValueError(
                f"Predictions must contain '{PRIOR_TARGET_KEY}' for PriorDenoisingLoss."
            )
        prior_loss = F.mse_loss(
            predictions[PRIOR_PREDICTION_KEY],
            predictions[PRIOR_TARGET_KEY],
        )
        return LossOutput(
            total_loss=self.weight * prior_loss,
            component_losses={MetricKey.PRIOR_DENOISING_LOSS.value: prior_loss},
        )


class FixedVarianceGaussianNLLoss(BaseLoss):
    """Negative Log-Likelihood loss for Gaussian Mixture Model with fixed variance.

    This loss assumes the action distribution is a mixture of Gaussians:
        p(a | z) = Σ_k π_k(z) · N(a | μ_k(z), σ²I)

    where:
        - K is the number of mixture components (experts)
        - π_k(z) are the mixing probabilities (must sum to 1), predicted by a gating network
        - μ_k(z) are the component means, predicted by expert networks
        - σ² is a fixed (not learned) isotropic variance, specified as a hyperparameter

    The negative log-likelihood is:
        NLL = -log p(a | z) = -log Σ_k π_k · N(a | μ_k, σ²I)

    Unlike MSE on blended predictions, this loss rewards having at least one expert
    close to the target, enabling true multimodal action distributions. Gradients are
    weighted by posterior responsibility γ_k, causing experts to specialize naturally.

    The fixed standard deviation σ controls the "softness" of expert assignment:
        - Small σ: Sharp assignments, experts must be very close to claim a sample
        - Large σ: Soft assignments, multiple experts can share responsibility

    Typical values depend on your action scale. Start with σ ≈ 0.1 * action_range/ 0.5*action_std.

    Note: The Gaussian normalization constant -0.5 * log(2πσ²) is omitted since
    it's constant w.r.t. parameters and doesn't affect optimization.
    """

    def __init__(
        self,
        action_keys: list[str],
        sigmas: dict[str, float] | None = None,
        weight: float = 1.0,
        per_key_weights: dict[str, float] | None = None,
    ):
        """Initialize NLL loss with fixed variance.

        Args:
            action_keys: List of action keys this loss applies to
            sigmas: Optional dict of fixed stddev per action key; if None, defaults to 1.0
            weight: Weight for NLL loss
            per_key_weights: Optional dict of per-key weights for loss components
        """
        super().__init__()
        self.action_keys = action_keys
        self.weight = weight
        self.per_key_weights = (
            per_key_weights
            if per_key_weights is not None
            else {key: 1.0 for key in action_keys}
        )
        if sigmas is None:
            self.sigmas = {key: 0.5 for key in action_keys}
        else:
            # Fill missing keys with default
            self.sigmas = {key: sigmas.get(key, 0.5) for key in action_keys}

    def get_required_keys(self) -> set[str]:
        """Get required target keys for FV-NLL.

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
        """Compute NLL loss.

        Args:
            predictions: Dictionary containing for each action_key:
                - '{action_key}_{ROUTING_WEIGHT}': Mixing probabilities π_k from softmax
                  Shape: (batch, chunk_size, num_experts)
                  Must sum to 1 along last dimension
                - '{action_key}_{EXPERT_OUTPUTS}': Expert mean predictions μ_k
                  Shape: (batch, chunk_size, num_experts, action_dim)
            targets: Dictionary containing for each action_key:
                - '{action_key}': Ground truth actions
                  Shape: (batch, chunk_size, action_dim)
            is_pad: Optional boolean mask indicating padded timesteps
                Shape: (batch, chunk_size)
                True = padded (excluded from loss), False = valid

        Returns:
            LossOutput with:
                - total_loss: Weighted sum of NLL across all action keys
                - component_losses: Dict of per-key NLL values for logging
        """
        component_losses = {}
        total_loss = 0.0
        for action_key in self.action_keys:
            routing_key = f"{action_key}_{ROUTING_WEIGHT}"
            expert_key = f"{action_key}_{EXPERT_OUTPUTS}"
            if routing_key not in predictions or expert_key not in predictions:
                raise ValueError(
                    f"Predictions must contain '{routing_key}' and '{expert_key}' "
                    f"for FixedVarianceNLLoss. Got keys: {list(predictions.keys())}"
                )
            if action_key not in targets:
                raise ValueError(
                    f"Targets must contain '{action_key}' for FixedVarianceNLLoss. "
                    f"Got keys: {list(targets.keys())}"
                )

            key_weight = self.per_key_weights.get(action_key, 1.0)
            sigma = self.sigmas[action_key]

            target = targets[action_key].unsqueeze(
                2
            )  # (B, T, 1, action_dim) -- broadcasts with (B, T, num_experts, action_dim)
            mixing_probs = predictions[
                f"{action_key}_{ROUTING_WEIGHT}"
            ]  # (B, T, num_experts)
            expert_outs = predictions[
                f"{action_key}_{EXPERT_OUTPUTS}"
            ]  # (B, T, num_experts, action_dim)

            log_pi = torch.log(mixing_probs + 1e-8)  # (B, T, num_experts)

            # Log Gaussian component (up to constant): log N(a | μ_k, σ²) ∝ -||a - μ_k||² / (2σ²)
            diff = target - expert_outs  # (B, T, 1, D) - (B, T, K, D) = (B, T, K, D)
            log_component = (
                -0.5 * (diff ** 2).sum(-1) / (sigma ** 2)
            )  # (B, T, num_experts)
            # Log mixture probability via logsumexp: log Σ_k π_k N(a | μ_k)
            log_prob = torch.logsumexp(log_pi + log_component, dim=-1)  # (B, T)
            nll = -log_prob  # (B, T)
            nll_reduced = reduce_loss_with_padding(nll, is_pad, reduction="mean")
            component_losses[action_key] = nll_reduced
            total_loss = total_loss + key_weight * nll_reduced

        return LossOutput(
            total_loss=self.weight * total_loss,
            component_losses=component_losses,
        )


class FixedVarianceGripperMixtureNLLoss(BaseLoss):
    """Negative Log-Likelihood loss for gripper with mixture distribution and shared expert routing.

    Binary gripper: p(a|z) = Σ_k π_k(z) · Bernoulli(a | p_k(z))
    Continuous gripper: p(a|z) = Σ_k π_k(z) · N(a | μ_k(z), σ²I) with fixed variance

    Uses same routing weights as continuous action experts, ensuring gripper
    behavior is coupled with the selected manipulation strategy.
    """

    def __init__(
        self,
        key: str,
        actions_metadata: dict[str, ActionMetadata],
        weight: float = 1.0,
        sigma: float = 0.5,
    ):
        """Initialize gripper mixture NLL loss.

        Args:
            key: Key for gripper actions
            actions_metadata: Dict of metadata of the action space
            weight: Loss weight
            sigma: Fixed std for continuous gripper (ignored for binary)
        """
        super().__init__()
        self.key = key
        self.weight = weight
        self.sigma = sigma
        resolved_metadata = resolve_dict_keys(dict(actions_metadata))
        if key not in resolved_metadata.keys():
            raise ValueError(
                f"{key} is not available to the action space. Can't compute gripper NLL loss. "
                f"Available keys: {list(resolved_metadata.keys())}"
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
                - '{GRIPPER_ACTION_KEY}_{ROUTING_WEIGHT}': Mixing probs π_k, shape (B, T, K)
                - '{GRIPPER_ACTION_KEY}_{EXPERT_OUTPUTS}': Expert predictions
                  Binary: (B, T, K) logits
                  Continuous: (B, T, K, D) means
            targets: Dictionary with '{GRIPPER_ACTION_KEY}'
                Binary: (B, T) or (B, T, 1)
                Continuous: (B, T, D)
            is_pad: Optional padding mask (B, T)

        Returns:
            LossOutput with gripper NLL
        """
        routing_key = f"{self.key}_{ROUTING_WEIGHT}"
        expert_key = f"{self.key}_{EXPERT_OUTPUTS}"
        if routing_key not in predictions or expert_key not in predictions:
            raise ValueError(
                f"Predictions must contain '{routing_key}' and '{expert_key}' for GripperMixtureNLLoss."
            )
        if self.key not in targets:
            raise ValueError(
                f"Targets must contain '{self.key}' for GripperMixtureNLLoss."
            )

        target = targets[self.key]
        mixing_probs = predictions[routing_key]  # (B, T, K)
        expert_outs = predictions[expert_key]
        log_pi = torch.log(mixing_probs + 1e-8)  # (B, T, K)
        if self.gripper_type == GripperType.BINARY.value:
            # target: (B, T) or (B, T, 1) -> (B, T)
            # expert_outs: (B, T, K) logits
            if target.dim() == 3:
                target = target.squeeze(-1)
            if expert_outs.dim() == 4:
                expert_outs = expert_outs.squeeze(-1)
            if self.binary_gripper_range == BinaryGripperRange.MINUS_ONE_ONE.value:
                target = (target.float() + 1.0) / 2.0

            expert_probs = torch.sigmoid(expert_outs).clamp(1e-8, 1 - 1e-8)
            target_expanded = target.unsqueeze(-1)  # (B, T, 1)
            log_component = target_expanded * torch.log(expert_probs) + (
                1 - target_expanded
            ) * torch.log(
                1 - expert_probs
            )  # (B, T, K)
        else:
            # target: (B, T, D) -> (B, T, 1, D)
            # expert_outs: (B, T, K, D)
            target_expanded = target.unsqueeze(2)  # (B, T, 1, D)
            diff = target_expanded - expert_outs  # (B, T, K, D)
            log_component = (
                -0.5 * (diff ** 2).sum(dim=-1) / (self.sigma ** 2)
            )  # (B, T, K)

        log_prob = torch.logsumexp(log_pi + log_component, dim=-1)  # (B, T)
        nll = -log_prob
        nll_reduced = reduce_loss_with_padding(nll, is_pad, reduction="mean")
        metric_key = MetricKey.GRIPPER_NLL.value
        return LossOutput(
            total_loss=self.weight * nll_reduced,
            component_losses={metric_key: nll_reduced},
        )


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

    def get_required_keys(self) -> set[str]:
        """Union of base loss keys plus routing weight."""
        return self.base_loss.get_required_keys() | {ROUTING_WEIGHT}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Passthrough base loss, then add expert_usage from routing weights and optionally add entropy term."""
        base_output: LossOutput = self.base_loss(predictions, targets, is_pad)
        metadata = base_output.metadata if base_output.metadata is not None else {}
        component_losses = dict(base_output.component_losses)
        entropy_loss = 0.0
        pi = predictions[ROUTING_WEIGHT]  # (B, T, num_experts)
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
