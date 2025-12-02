"""Individual loss components for action prediction tasks."""
import math

import torch
import torch.nn.functional as F

from refactoring.data.constants import (
    GRIPPER_ACTION_KEY,
    PHASE_LABEL_KEY,
    POSITION_ACTION_KEY,
    GripperType, TOKENIZED_ACTIONS_KEY,
)
from refactoring.metrics.base import BaseLoss, LossOutput, reduce_loss_with_padding
from refactoring.metrics.constants import (
    MetadataKey,
    MetricKey,
)
from refactoring.models.decoding.constants import (
    PRIOR_PREDICTION_KEY,
    PRIOR_TARGET_KEY,
    BINARY_LOGITS_KEY, MU_KEY, LOGVAR_KEY, ACTION_LOGITS_KEY, LATENT_CODES, ROUTING_WEIGHT,
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
                raise ValueError(f"Predictions and targets must contain key '{action_key}' for RegressionLoss.")

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
        gripper_type: str = GripperType.BINARY.value,
        bce_weight: float = 0.005,
        mse_weight: float = 0.0,
        pos_weight: torch.Tensor | None = None,
    ):
        """Initialize gripper loss.

        Args:
            gripper_type: Type of gripper ('binary' or 'continuous')
            bce_weight: Weight for binary cross entropy (binary gripper)
            mse_weight: Weight for MSE loss (continuous gripper)
            pos_weight: Optional positive class weight for BCE
        """
        super().__init__()
        self.gripper_type = gripper_type
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.register_buffer("pos_weight", pos_weight)

    def get_required_keys(self) -> set[str]:
        """Get required target keys for gripper loss.

        Returns:
            Set containing the gripper action key
        """
        return {GRIPPER_ACTION_KEY}

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
        if GRIPPER_ACTION_KEY not in predictions or GRIPPER_ACTION_KEY not in targets:
            raise ValueError(f"Predictions and targets must contain key '{GRIPPER_ACTION_KEY}' for GripperLoss.")
        pred_gripper = predictions[GRIPPER_ACTION_KEY]
        target_gripper = targets[GRIPPER_ACTION_KEY]

        if self.gripper_type == GripperType.BINARY.value:
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


class KLDivergenceLoss(BaseLoss):
    """KL divergence loss for VAE latent distributions."""

    def __init__(self, weight: float = 10.0):
        """Initialize KL divergence loss.

        Args:
            weight: Weight for KL divergence loss
        """
        super().__init__()
        self.weight = weight

    def get_required_keys(self) -> set[str]:
        """Get required keys for KL divergence loss.

        Returns:
            Set containing VAE latent distribution keys (mu, logvar)
        """
        return {MU_KEY, LOGVAR_KEY}

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
        if MU_KEY not in predictions or LOGVAR_KEY not in predictions:
            raise ValueError(f"Predictions must contain keys '{MU_KEY}' and '{LOGVAR_KEY}' for KLDivergenceLoss.")
        mu = predictions[MU_KEY].float() # Using fp32 float for stability
        logvar = predictions[LOGVAR_KEY].float()
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        if kld.min() < 0:
            print(f"Warning: Negative KL divergence encountered: min={kld.min().item():.4f}")
            print(f"per_dim_kl: min={kld.min().item():.4f}, max={kld.max().item():.4f}")
        kld = torch.clamp(kld, min=0.0)
        kld_mean = kld.mean()

        return LossOutput(
            total_loss=self.weight * kld_mean,
            component_losses={MetricKey.KL_DIVERGENCE.value: kld_mean},
        )


class BinaryKLDivergenceLoss(BaseLoss):
    """KL divergence loss for Free Transformer binary latent distributions.

    Computes KL divergence between learned binary distributions and uniform prior.
    Used with Free Transformer's binary mapper output.

    Based on "The Free Transformer" (Fleuret, 2025) - arXiv:2510.17558
    """

    def __init__(self,
                 weight: float = 5.0,
                 entropy_weight: float = 0.01,
                 latent_bits: float = 64,
                 free_bits: float = 2 * math.log(2)):
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
            code_indices = torch.argmax(latent_codes, dim=-1).flatten()  # (B*token_len,)
            unique_codes = torch.unique(code_indices).numel()
            total_codes = 2 ** self.latent_bits
            usage_pct = (unique_codes / total_codes)
            all_component_losses[MetricKey.LATENT_CODE_USAGE.value] = usage_pct

        logits = predictions[BINARY_LOGITS_KEY]  # (B, T, H) or (B, H)
        if logits is None: # Inference, zero loss
            return LossOutput(
                total_loss=torch.tensor(0.0, device=next(iter(predictions.values())).device),
                component_losses=all_component_losses,
            )

        # P(B_h=1) = sigmoid(L_h) for each bit
        probs = torch.sigmoid(logits)  # (B, T, H) or (B, H)
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
            clamped_kl_per_token = torch.clamp(kl_per_token - self.free_bits, min=0.0)  # (B, T)
            clamped_kl_mean = clamped_kl_per_token.mean()  # Scalar
        else:
            clamped_kl_mean = raw_kl_mean

        all_component_losses[MetricKey.CLAMPED_KL_DIVERGENCE.value] = clamped_kl_mean
        entropy = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))  # (B,token_len,H)
        regularized_kl = clamped_kl_mean -self.entropy_weight * entropy.mean() # Scalar (avg over B,T,H)
        all_component_losses[MetricKey.POSTERIOR_ENTROPY.value] = entropy.mean()
        all_component_losses[MetricKey.KL_DIVERGENCE.value] = regularized_kl

        return LossOutput(
            total_loss=self.weight * regularized_kl,
            component_losses=all_component_losses,
        )


class TrajectoryLengthLoss(BaseLoss):
    """Loss for trajectory length consistency.

    Penalizes differences between predicted and ground truth trajectory lengths.
    """

    def __init__(self, weight: float = 0.001, action_key: str = POSITION_ACTION_KEY):
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
            raise ValueError(f"Predictions and targets must contain key '{self.action_key}' for TrajectoryLengthLoss.")
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

    def __init__(self, weight: float = 0.001, action_key: str = POSITION_ACTION_KEY):
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
            raise ValueError(f"Predictions must contain key '{self.action_key}' for TrajectorySmoothness loss.")
        pred = predictions[self.action_key]
        if pred.shape[1] < 3: # If trajectory too short, no acceleration can be computed
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
        cross_entropy_weight: float = 0.1,
        entropy_weight: float = 0.01,
        label_smoothing: float = 0.0,
    ):
        """Initialize phase classification loss.

        Args:
            cross_entropy_weight: Weight for cross-entropy loss
            entropy_weight: Weight for entropy regularization (negative encourages sparsity)
            label_smoothing: Label smoothing factor for cross-entropy
        """
        super().__init__()
        self.cross_entropy_weight = cross_entropy_weight
        self.entropy_weight = entropy_weight
        self.label_smoothing = label_smoothing

    def get_required_keys(self) -> set[str]:
        """Get required target keys for phase classification loss.

        Returns:
            Set containing the phase label key
        """
        return {PHASE_LABEL_KEY}

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
        if PHASE_LABEL_KEY not in predictions or PHASE_LABEL_KEY not in targets:
            raise ValueError(f"Predictions and targets must contain key '{PHASE_LABEL_KEY}' for PhaseClassificationLoss.")

        pred_logits = predictions[PHASE_LABEL_KEY]
        target_labels = targets[PHASE_LABEL_KEY]

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
            total_loss = total_loss + self.entropy_weight * entropy_reduced

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
        label_smoothing: float = 0.0,
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
        pred_logits = predictions[ACTION_LOGITS_KEY] # (B, num_tokens, vocab_size)
        target_tokens = targets[TOKENIZED_ACTIONS_KEY] # (B, num_tokens)
        vocab_size = pred_logits.shape[-1]
        num_tokens = pred_logits.shape[1]
        logits = pred_logits.view(-1, vocab_size, num_tokens)  # (B, vocab_size, num_tokens)
        ce_loss = F.cross_entropy(
            logits,
            target_tokens,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )
        ce_loss = reduce_loss_with_padding(ce_loss, is_pad, reduction="mean")
        predicted_tokens = torch.argmax(pred_logits, dim=-1)  # (B, seq) over C=dim=-1 (no view needed)
        correct = (predicted_tokens == target_tokens).float()  # (B, seq)
        accuracy = reduce_loss_with_padding(correct, is_pad, reduction="mean")  # Scalar %
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

        # Compute MSE loss between prediction and target
        prior_loss = F.mse_loss(
            predictions[PRIOR_PREDICTION_KEY],
            predictions[PRIOR_TARGET_KEY],
        )

        return LossOutput(
            total_loss=self.weight * prior_loss,
            component_losses={MetricKey.PRIOR_DENOISING_LOSS.value: prior_loss},
        )



class MoELoss(BaseLoss):
    """Wrapper for any BaseLoss to add MoE expert usage metric from routing weights."""

    def __init__(self, base_loss: BaseLoss):
        """Initialize MoE wrapper.

        Args:
            base_loss: Any BaseLoss instance to wrap (e.g., RegressionLoss(...))
        """
        super().__init__()
        self.base_loss = base_loss

    def get_required_keys(self) -> set[str]:
        """Union of base loss keys plus routing weight."""
        return self.base_loss.get_required_keys()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Passthrough base loss, then add expert_usage from routing weights."""
        base_output: LossOutput = self.base_loss(predictions, targets, is_pad)
        metadata = base_output.metadata if base_output.metadata is not None else {}
        required_keys = self.base_loss.get_required_keys()
        for action_key in required_keys:
            routing_key = f'{action_key}_{ROUTING_WEIGHT}'
            if routing_key in predictions:
                expert_usage = predictions[routing_key]
                if expert_usage.dim() > 0:
                    expert_usage = expert_usage.mean(dim=list(range(expert_usage.ndim - 1))) # Mean over all but last dim, which is num_experts
                usage_key = f'{action_key}_{MetadataKey.EXPERT_USAGE.value}'
                metadata[usage_key] = expert_usage

        return LossOutput(
            total_loss=base_output.total_loss,
            component_losses=base_output.component_losses,
            metadata=metadata,
        )
