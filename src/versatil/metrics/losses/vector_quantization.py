"""Losses for vector-quantized latent variable models."""

import torch
import torch.nn.functional as F

from versatil.metrics.base import LossOutput, ScalarWeightedLoss
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.models.decoding.constants import LatentKey


class VQCommitmentLoss(ScalarWeightedLoss):
    """Commitment loss for vector-quantized latent variable models.

    Penalizes the distance between the continuous encoder output and
    the quantized codebook vectors, preventing encoder outputs from
    drifting away from codebook entries. Reads pre-computed tensors
    from the predictions dict (put there by VQPosteriorEncoder).

    Ref: van den Oord et al., "Neural Discrete Representation Learning" (2017)
    """

    def __init__(
        self,
        num_codes: int,
        num_residual_layers: int,
        weight: float = 1.0,
    ):
        """Initialize VQ commitment loss.

        Args:
            num_codes: Number of codebook entries per residual layer (K).
                Must match the VQ posterior's ResidualVQ configuration.
            num_residual_layers: Number of residual VQ layers.
                Must match the VQ posterior's ResidualVQ configuration.
            weight: Loss weight for the commitment term
                ||z_continuous - sg(z_quantized)||^2.
        """
        super().__init__()
        if num_codes <= 0:
            raise ValueError(f"num_codes must be positive, got {num_codes}.")
        if num_residual_layers <= 0:
            raise ValueError(
                f"num_residual_layers must be positive, got {num_residual_layers}."
            )
        self.num_codes = num_codes
        self.num_residual_layers = num_residual_layers
        self.weight = weight

    def get_required_keys(self) -> set[str]:
        """Get required prediction keys."""
        return {
            LatentKey.VQ_Z_CONTINUOUS.value,
            LatentKey.VQ_QUANTIZED.value,
        }

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute per-layer commitment loss between encoder outputs and
        quantized codebook vectors.

        Args:
            predictions: Must contain:
                - LatentKey.VQ_Z_CONTINUOUS (L, B, code_dim): per-layer
                    pre-quantization encoder outputs in code space.
                - LatentKey.VQ_QUANTIZED (L, B, code_dim, detached):
                    per-layer hard-quantized codebook vectors.
                May also contain LatentKey.VQ_INDICES (list of L tensors
                shape (B,) with posterior choices) for codebook usage
                logging.
            targets: Unused.
            is_pad: Unused.

        Returns:
            LossOutput with weighted commitment loss averaged across layers
            (VQ-BeT style). Codebook usage metadata reports the
            fraction of total codebook capacity (num_codes * num_residual_layers)
            exercised across all layers in the batch.
        """
        required_keys = self.get_required_keys()
        if not all(k in predictions for k in required_keys):
            raise ValueError(
                f"Predictions must contain {required_keys} for VQCommitmentLoss."
            )

        z_continuous = predictions[
            LatentKey.VQ_Z_CONTINUOUS.value
        ].float()  # (L, B, code_dim)
        z_quantized = predictions[
            LatentKey.VQ_QUANTIZED.value
        ].float()  # (L, B, code_dim) — already detached

        commitment_loss = F.mse_loss(z_continuous, z_quantized)  # scalar

        posterior_indices = predictions.get(LatentKey.VQ_INDICES.value)
        metadata = {}
        if posterior_indices is not None:
            # Per-layer distinct counts summed, divided by total capacity K*L.
            # Counting distinct indices per layer separately (not over cat)
            # because layer L's code 0 is a different codebook entry than
            # layer L+1's code 0 — they index separate codebooks.
            unique_codes_total = sum(
                layer_indices.unique().numel() for layer_indices in posterior_indices
            )
            codebook_capacity = self.num_codes * self.num_residual_layers
            metadata[MetricKey.VQ_CODEBOOK_USAGE.value] = (
                unique_codes_total / codebook_capacity
            )
            metadata[MetadataKey.VQ_CODE_INDICES.value] = torch.stack(
                [
                    layer_indices.long() for layer_indices in posterior_indices
                ],  # list[L] of (B,)
                dim=0,
            )  # (L, B)
            metadata[MetadataKey.VQ_NUM_CODES.value] = torch.tensor(
                self.num_codes,
                device=z_continuous.device,
            )  # ()

        return LossOutput(
            total_loss=self.weight * commitment_loss,
            component_losses={
                MetricKey.VQ_COMMITMENT_LOSS.value: commitment_loss,
            },
            metadata=metadata,
        )


class VQPriorCrossEntropyLoss(ScalarWeightedLoss):
    """Cross-entropy loss training a learned categorical prior to predict
    the posterior's codebook index choices.

    Computes -log p(k*|s) summed over residual VQ layers, where k* is
    the index the posterior chose and p(k|s) is the prior's predicted
    categorical. This is the KL term in the ELBO for a delta posterior
    against a categorical prior.

    Ref: van den Oord et al. (2017) train the prior in a second stage;
    this loss enables end-to-end joint training as a tighter ELBO.
    """

    def __init__(
        self,
        weight: float = 1.0,
    ):
        """Initialize VQ prior cross-entropy loss.

        Args:
            weight: Loss weight for the cross-entropy term.
        """
        super().__init__()
        self.weight = weight

    def get_required_keys(self) -> set[str]:
        """Get required prediction keys."""
        return {
            LatentKey.VQ_INDICES.value,
            LatentKey.PRIOR_CODE_LOGITS.value,
        }

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        """Compute cross-entropy between prior logits and posterior indices.

        Args:
            predictions: Must contain:
                - LatentKey.VQ_INDICES: list of (B,) long tensors (posterior choices)
                - LatentKey.PRIOR_CODE_LOGITS: list of (B, K) float tensors (prior logits)
            targets: Unused.
            is_pad: Unused.

        Returns:
            LossOutput with weighted cross-entropy loss.
        """
        required_keys = self.get_required_keys()
        if not all(k in predictions for k in required_keys):
            raise ValueError(
                f"Predictions must contain {required_keys} for VQPriorCrossEntropyLoss."
            )

        posterior_indices = predictions[LatentKey.VQ_INDICES.value]
        prior_logits = predictions[LatentKey.PRIOR_CODE_LOGITS.value]
        if len(prior_logits) == 0:
            raise ValueError("VQPriorCrossEntropyLoss received no prior logits.")
        if len(prior_logits) != len(posterior_indices):
            raise ValueError(
                f"VQPriorCrossEntropyLoss expected the same number of prior logit "
                f"layers and posterior index layers, got {len(prior_logits)} "
                f"and {len(posterior_indices)}."
            )

        total_ce = torch.tensor(0.0, device=prior_logits[0].device)
        for layer_index, (layer_logits, layer_indices) in enumerate(
            zip(prior_logits, posterior_indices, strict=True)
        ):
            # layer_logits: (B, K), layer_indices: (B,)
            if layer_logits.ndim != 2:
                raise ValueError(
                    f"Prior logits for VQ layer {layer_index} must have shape "
                    f"(B, K), got {tuple(layer_logits.shape)}."
                )
            if layer_indices.ndim != 1:
                raise ValueError(
                    f"Posterior indices for VQ layer {layer_index} must have shape "
                    f"(B,), got {tuple(layer_indices.shape)}."
                )
            if layer_logits.shape[0] != layer_indices.shape[0]:
                raise ValueError(
                    f"Prior logits and posterior indices for VQ layer {layer_index} "
                    f"must have the same batch size, got {layer_logits.shape[0]} "
                    f"and {layer_indices.shape[0]}."
                )
            total_ce = total_ce + F.cross_entropy(
                layer_logits, layer_indices.long()
            )  # scalar, averaged over batch

        return LossOutput(
            total_loss=self.weight * total_ce,
            component_losses={
                MetricKey.VQ_PRIOR_CROSS_ENTROPY.value: total_ce,
            },
        )
