"""Cross-entropy losses for phase and action-token classification."""

import torch
import torch.nn.functional as F

from versatil.data.constants import SampleKey
from versatil.metrics.base import (
    BaseLoss,
    LossOutput,
    ScalarWeightedLoss,
    WeightsDictionary,
    reduce_loss_with_padding,
)
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.models.decoding.constants import DecoderOutputKey


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

    @property
    def weights(self) -> WeightsDictionary:
        """Getter that returns dictionary with weight keys and scalar coefficients."""
        return {
            "cross_entropy_weight": self.cross_entropy_weight,
            "entropy_weight": self.entropy_weight,
        }

    def set_weights(self, new_weights: WeightsDictionary) -> None:
        """Setter that updates the weight scalar coefficients."""
        self._validate_weights(new_weights)
        self.cross_entropy_weight = new_weights["cross_entropy_weight"]
        self.entropy_weight = new_weights["entropy_weight"]

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

        # Store only unpadded steps: edge-padded episodes repeat their final
        # phase, which would inflate that phase in accuracy and the
        # confusion matrix computed from this metadata.
        metadata = {
            MetadataKey.PHASE_LOGITS.value: pred_flat.detach(),
            MetadataKey.PHASE_LABEL.value: target_flat.detach(),
        }

        return LossOutput(
            total_loss=total_loss,
            component_losses=component_losses,
            metadata=metadata,
        )


class ActionTokenLoss(ScalarWeightedLoss):
    """Cross-entropy loss for tokenized actions."""

    def __init__(
        self,
        weight: float = 1.0,
        label_smoothing: float = 0.2,
    ):
        """Initialize action token loss.

        Args:
            weight: Scalar multiplier applied to the cross-entropy term.
            label_smoothing: Label smoothing factor [0, 1]
        """
        super().__init__()
        self.weight = weight
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
        perplexity = torch.exp(ce_loss)
        weighted_loss = ce_loss * self.weight
        return LossOutput(
            total_loss=weighted_loss,
            component_losses={
                MetricKey.ACTION_TOKEN_CROSS_ENTROPY.value: ce_loss,
                MetricKey.PERPLEXITY.value: perplexity,
                MetricKey.TOKEN_ACCURACY.value: accuracy,
            },
        )
