"""Mixture-of-experts loss wrapper with routing regularization."""

import torch

from versatil.configs.experiment import ExperimentConfig
from versatil.metrics.base import (
    BaseLoss,
    LossOutput,
    WeightsDictionary,
    reduce_loss_with_padding,
)
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.training.callbacks.expert_usage import ExpertUsageCallback


class MoELoss(BaseLoss):
    """Wrapper for any BaseLoss to add MoE expert usage metric from routing weights."""

    def __init__(
        self,
        base_loss: BaseLoss,
        entropy_weight: float = 0.0,
        load_balance_weight: float = 0.0,
    ):
        """Initialize MoE wrapper.

        Args:
            base_loss: Any BaseLoss instance to wrap (e.g., RegressionLoss(...))
            entropy_weight: Weight for per-example routing entropy.
                Penalizes peaky-per-example routing. Pushes each example's routing
                distribution toward uniform, which prevents one example from being
                routed to a single expert with probability 1.
            load_balance_weight: Weight for Switch-Transformer-style load-balancing
                term. Penalizes batch-level imbalance in expert usage. The term is
                ``K * sum_k f_k * P_k`` where ``f_k`` is the fraction of examples
                whose argmax routes to expert k and ``P_k`` is the mean routing
                weight for expert k across the batch. Minimum value 1.0 is reached
                when usage is uniform across the batch. Crucially, this allows
                per-example routing to be peaky (so experts can specialize) while
                still forcing every expert to be used by some examples (so no
                expert dies). Use this when entropy alone produces dead experts.
        """
        super().__init__()
        self.base_loss = base_loss
        self.entropy_weight = entropy_weight
        self.load_balance_weight = load_balance_weight

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients,
        plus the wrapped ``base_loss`` weight structure nested under ``base_loss``."""
        return {
            "entropy_weight": self.entropy_weight,
            "load_balance_weight": self.load_balance_weight,
            "base_loss": self.base_loss.weights,
        }

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients and delegates
        ``base_loss`` to the wrapped loss."""
        self._validate_weights(new_weights)
        self.entropy_weight = new_weights["entropy_weight"]
        self.load_balance_weight = new_weights["load_balance_weight"]
        self.base_loss.set_weights(new_weights["base_loss"])

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
        """Passthrough base loss, then add expert_usage and optional entropy/load-balance terms."""
        predictions = self._add_weighted_mean_predictions(predictions)
        base_output: LossOutput = self.base_loss(predictions, targets, is_pad)
        metadata = base_output.metadata if base_output.metadata is not None else {}
        component_losses = dict(base_output.component_losses)
        pi = predictions[DecoderOutputKey.ROUTING_WEIGHTS.value]  # (B, K) or (B, T, K)
        total_loss = base_output.total_loss
        if self.entropy_weight != 0.0:
            entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=-1)  # (B,) or (B, T)
            if entropy.dim() == 2:
                entropy_mean = reduce_loss_with_padding(
                    entropy, is_pad, reduction="mean"
                )
            else:
                entropy_mean = entropy.mean()
            component_losses[f"{MetricKey.EXPERTS_ENTROPY.value}"] = entropy_mean
            total_loss = total_loss - self.entropy_weight * entropy_mean
        if self.load_balance_weight != 0.0:
            load_balance = self._compute_load_balance(pi=pi, is_pad=is_pad)
            component_losses[f"{MetricKey.EXPERTS_LOAD_BALANCE.value}"] = load_balance
            total_loss = total_loss + self.load_balance_weight * load_balance
        expert_usage = pi.mean(
            dim=list(range(pi.ndim - 1))
        )  # Mean over all but last dim, which is num_experts
        metadata[MetadataKey.EXPERT_USAGE.value] = expert_usage
        return LossOutput(
            total_loss=total_loss,
            component_losses=component_losses,
            metadata=metadata,
        )

    @staticmethod
    def _compute_load_balance(
        pi: torch.Tensor,
        is_pad: torch.Tensor | None,
    ) -> torch.Tensor:
        """Switch-Transformer load-balancing loss on routing weights.

        L = K * sum_k f_k * P_k
            f_k: fraction of examples whose top-1 route is k (no gradient).
            P_k: mean routing weight for k across batch (carries gradient).

        Reaches its minimum of 1.0 when usage is uniform across the batch.
        With ``pi`` of shape (B, K), the average is over B; with shape (B, T, K),
        the average is over (B, T) excluding padded timesteps.
        """
        num_experts = pi.shape[-1]
        if pi.dim() == 2:
            # (B, K) — per-trajectory routing.
            argmax_indices = pi.argmax(dim=-1)
            f = (
                torch.nn.functional.one_hot(argmax_indices, num_classes=num_experts)
                .to(pi.dtype)
                .mean(dim=0)
            )  # (K,)
            mean_routing = pi.mean(dim=0)  # (K,)
        else:
            # (B, T, K) — per-step routing; respect padding.
            argmax_indices = pi.argmax(dim=-1)
            one_hot = torch.nn.functional.one_hot(
                argmax_indices, num_classes=num_experts
            ).to(pi.dtype)  # (B, T, K)
            if is_pad is not None:
                valid = (~is_pad).to(pi.dtype).unsqueeze(-1)  # (B, T, 1)
                valid_count = valid.sum().clamp(min=1.0)
                f = (one_hot * valid).sum(dim=(0, 1)) / valid_count
                mean_routing = (pi * valid).sum(dim=(0, 1)) / valid_count
            else:
                f = one_hot.mean(dim=(0, 1))
                mean_routing = pi.mean(dim=(0, 1))
        return num_experts * (f * mean_routing).sum()
