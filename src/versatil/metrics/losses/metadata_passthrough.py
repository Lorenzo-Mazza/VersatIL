"""Passthrough loss that forwards target keys into metadata."""

import torch

from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.metrics.base import BaseLoss, LossOutput


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
