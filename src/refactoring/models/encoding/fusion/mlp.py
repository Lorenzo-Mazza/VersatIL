
import torch

from refactoring.models.encoding.fusion.base import FusionOutput
from refactoring.models.encoding.fusion.sequential import SequentialFusion
from refactoring.models.layers import MLP
from refactoring.models.layers.activation import ActivationFunction


class MLPFusion(SequentialFusion):
    """Combines sequence features by projecting them into a shared embedding space, concatenating, and then applying an MLP."""
    def __init__(
            self,
            input_features: list[str],
            output_name: str,
            hidden_dim: int,
            mlp_hidden_dims : list[int],
            activation_name: str = ActivationFunction.GELU.value,
            dropout: float = 0.1,
    ):
        """
        Args:
            input_features: List of feature names to fuse.
            output_name: Name of the output fused feature.
            hidden_dim: Dimension to project each input feature to before fusion.
            mlp_hidden_dims: List of hidden layer dimensions for the MLP.
            activation_name: Name of the activation function to use in the MLP.
            dropout: Dropout rate for the MLP.
        """
        super().__init__(input_features=input_features, output_name=output_name, hidden_dim=hidden_dim,)
        self.mlp = MLP(
            input_dim=hidden_dim * len(input_features),
            hidden_dims=mlp_hidden_dims,
            activation_function=ActivationFunction(activation_name).to_torch_activation(),
            dropout=dropout,
        )
        self.output_dim = mlp_hidden_dims[-1]

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
        concat = torch.cat(projected, dim=-1)
        result: torch.Tensor = self.mlp(concat)
        return result


    def get_output_specification(self) -> FusionOutput:
        """Get output specification."""
        return FusionOutput(
            output_name=self.output_name,
            output_dim=self.output_dim,
        )
