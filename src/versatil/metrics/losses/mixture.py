"""Negative log-likelihood losses for mixture distributions."""

import math

import torch
import torch.nn.functional as F

from versatil.data.constants import BinaryGripperRange, GripperType
from versatil.data.metadata import ActionMetadata
from versatil.metrics.base import LossOutput, ScalarWeightedLoss
from versatil.metrics.constants import MetricKey
from versatil.metrics.losses.gripper import resolve_gripper_metadata
from versatil.models.decoding.constants import DecoderOutputKey


def _aggregate_mixture_nll(
    log_component: torch.Tensor,
    mixing_probs: torch.Tensor,
    is_pad: torch.Tensor | None,
) -> torch.Tensor:
    """Combine per-step per-component log densities into per-batch NLL.

    Both regimes return a per-step-averaged NLL so the loss magnitude is
    independent of the prediction horizon. This keeps gradients balanced
    against other terms (e.g. routing entropy) and prevents the trajectory
    log-likelihood from drowning the gating signal at long horizons.

    Dispatch on routing shape:
        (B, K)    → trajectory-level mixture: sum_t log p_k inside logsumexp_k,
                    then divide by the number of valid timesteps.
                    Components are full trajectories selected once per batch.
        (B, T, K) → per-step mixture: logsumexp_k inside, then average over
                    valid timesteps.
                    Components are selected independently at each timestep.

    Args:
        log_component: (B, T, K) per-component per-timestep log-pdf.
        mixing_probs: (B, K) or (B, T, K) mixture weights summing to 1 along K.
        is_pad: (B, T) boolean mask, True for padded positions (excluded).

    Returns:
        (B,) per-batch NLL averaged over valid timesteps.
    """
    horizon = log_component.shape[1]
    if mixing_probs.dim() == 2:
        if is_pad is not None:
            log_component = log_component.masked_fill(is_pad.unsqueeze(-1), 0.0)
            valid_count = (~is_pad).float().sum(dim=-1).clamp(min=1.0)  # (B,)
        else:
            valid_count = torch.full(
                (log_component.shape[0],),
                float(horizon),
                device=log_component.device,
                dtype=log_component.dtype,
            )
        log_traj_per_component = log_component.sum(dim=1)  # (B, K)
        log_pi = torch.log(mixing_probs + 1e-8)  # (B, K)
        joint_nll = -torch.logsumexp(log_pi + log_traj_per_component, dim=-1)  # (B,)
        return joint_nll / valid_count
    log_pi = torch.log(mixing_probs + 1e-8)  # (B, T, K)
    log_step_mix = torch.logsumexp(log_pi + log_component, dim=-1)  # (B, T)
    if is_pad is not None:
        log_step_mix = log_step_mix.masked_fill(is_pad, 0.0)
        valid_count = (~is_pad).float().sum(dim=-1).clamp(min=1.0)
        return -log_step_mix.sum(dim=-1) / valid_count  # (B,)
    return -log_step_mix.mean(dim=-1)  # (B,)


class GaussianMixtureNLLoss(ScalarWeightedLoss):
    """Negative Log-Likelihood loss for Gaussian Mixture Model.

    Supports both learned variance (from logvar predictions) and fixed variance (sigma parameter).

    Two regimes are supported, dispatched on the shape of routing_weights:

    Per-trajectory routing (B, K) — one mixture component is selected for the entire
    chunk (e.g., MoDEACT). The joint trajectory likelihood is:
        log p(a_{1:T}|s) = logsumexp_k [log π_k + Σ_t log N(a_t | μ_k(t), σ_k(t)²)]
    Components are forced to model coherent trajectories; without this formulation the
    gating collapses to one component that averages distinct trajectory modes.

    Per-timestep routing (B, T, K) — a component is selected per timestep (e.g.,
    PhaseACT). The per-step mixture likelihood applies:
        log p(a_t|s, t) = logsumexp_k [log π_kt + log N(a_t | μ_kt, σ_kt²)]
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
                - routing_weights: (B, K) for per-trajectory or (B, T, K) for per-timestep.
                - If learned_variance: {action_key}_mean (B, T, K, D), {action_key}_logvar (B, T, K, D)
                - If fixed variance: {action_key} (B, T, K, D) stacked expert means
            targets: Dictionary with action_key targets (B, T, D).
            is_pad: Optional padding mask (B, T).

        Returns:
            LossOutput with Gaussian mixture NLL.
        """
        component_losses: dict[str, torch.Tensor] = {}
        log_components_by_key: dict[str, torch.Tensor] = {}
        mixing_probs = predictions[DecoderOutputKey.ROUTING_WEIGHTS.value]
        for action_key in self.action_keys:
            target = targets[action_key]  # (B, T, D)
            mean_key = f"{action_key}_{DecoderOutputKey.MEAN.value}"
            means = predictions.get(
                mean_key, predictions.get(action_key)
            )  # (B, T, K, D)
            if self.learned_variance:
                logvar_key = f"{action_key}_{DecoderOutputKey.LOGVAR.value}"
                logvars = predictions[logvar_key]  # (B, T, K, D)
                log_component = self._compute_learned_variance_log_pdf(
                    target, means, logvars
                )
            else:
                sigma = self.sigmas.get(action_key, 0.5)
                log_component = self._compute_fixed_variance_log_pdf(
                    target, means, sigma
                )
            log_components_by_key[action_key] = log_component
            per_key_nll = _aggregate_mixture_nll(
                log_component=log_component,
                mixing_probs=mixing_probs,
                is_pad=is_pad,
            )  # (B,)
            per_key_nll_reduced = per_key_nll.mean()
            component_losses[f"{action_key}_{MetricKey.GAUSSIAN_MIXTURE_NLL.value}"] = (
                per_key_nll_reduced
            )

        if len(self.action_keys) == 1:
            action_key = self.action_keys[0]
            joint_nll_reduced = component_losses[
                f"{action_key}_{MetricKey.GAUSSIAN_MIXTURE_NLL.value}"
            ]
            total_loss = self.per_key_weights.get(action_key, 1.0) * joint_nll_reduced
        else:
            weighted_log_components = [
                self.per_key_weights.get(action_key, 1.0)
                * log_components_by_key[action_key]
                for action_key in self.action_keys
            ]
            # A shared mixture component represents the full action tuple.
            joint_log_component = torch.stack(weighted_log_components, dim=0).sum(dim=0)
            joint_nll = _aggregate_mixture_nll(
                log_component=joint_log_component,
                mixing_probs=mixing_probs,
                is_pad=is_pad,
            )
            joint_nll_reduced = joint_nll.mean()
            total_loss = joint_nll_reduced

        component_losses[MetricKey.GAUSSIAN_MIXTURE_NLL.value] = joint_nll_reduced
        return LossOutput(
            total_loss=self.weight * total_loss, component_losses=component_losses
        )

    def _compute_learned_variance_log_pdf(
        self,
        target: torch.Tensor,
        means: torch.Tensor,
        logvars: torch.Tensor,
    ) -> torch.Tensor:
        """Per-component Gaussian log-pdf with learned variance.

        Returns:
            (B, T, K) tensor of log N(a_t | μ_kt, σ_kt²) per component per timestep.
        """
        action_dimension = target.shape[-1]
        logvars = logvars.clamp(min=math.log(self.min_variance))
        target = target.unsqueeze(2)  # (B, T, 1, D)
        difference = target - means  # (B, T, K, D)
        scaled_squared_error = (difference**2) * torch.exp(-logvars)  # (B, T, K, D)
        log_normalization = -0.5 * action_dimension * math.log(2 * math.pi)
        return log_normalization - 0.5 * (logvars + scaled_squared_error).sum(dim=-1)

    @staticmethod
    def _compute_fixed_variance_log_pdf(
        target: torch.Tensor,
        means: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Per-component Gaussian log-pdf with fixed variance.

        The constant log normalization is omitted (it cancels in logsumexp_k).

        Returns:
            (B, T, K) tensor of log-pdf up to a per-component-shared constant.
        """
        target = target.unsqueeze(2)  # (B, T, 1, D)
        difference = target - means  # (B, T, K, D)
        return -0.5 * (difference**2).sum(-1) / (sigma**2)


class GripperMixtureNLLoss(ScalarWeightedLoss):
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
        self.gripper_type, self.binary_gripper_range = resolve_gripper_metadata(
            key=key, actions_metadata=actions_metadata
        )

    def get_required_keys(self) -> set[str]:
        """Return the prediction key this loss consumes."""
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
        if self.gripper_type == GripperType.BINARY.value:
            log_component = self._compute_binary_log_component(predictions, target)
        else:
            log_component = self._compute_continuous_log_component(predictions, target)
        nll_per_batch = _aggregate_mixture_nll(
            log_component=log_component,
            mixing_probs=mixing_probs,
            is_pad=is_pad,
        )
        nll_reduced = nll_per_batch.mean()
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
