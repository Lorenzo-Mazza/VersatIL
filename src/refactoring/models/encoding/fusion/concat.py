import torch

from refactoring.models.encoding.fusion.base import FusionOutput
from refactoring.models.encoding.fusion.sequential import SequentialFusion


class ConcatFusion(SequentialFusion):
    """Combines sequence features by projecting them into a shared embedding space and then concatenating them."""

    def __init__(
        self,
        input_features: list[str],
        output_name: str,
        hidden_dim: int,
    ):
        super().__init__(
            input_features=input_features,
            output_name=output_name,
            hidden_dim=hidden_dim,
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of sequence features [B, T, D_i] or [B, D_i]

        Returns:
            Fused features [B, T, hidden_dim] or [B, hidden_dim]
        """
        if self.projections is None:
            raise RuntimeError("Projections must be set up before forward pass")
        projected = []
        for feat, proj in zip(features, self.projections):
            projected.append(proj(feat))
        return torch.cat(projected, dim=-1)

    def get_output_specification(self) -> FusionOutput:
        """Get output specification."""
        output_dim = self.hidden_dim * len(self.input_features)
        return FusionOutput(
            output_name=self.output_name,
            output_dim=output_dim,
        )
