import torch

from versatil.models.encoding.fusion.base import SequentialFusion
from versatil.models.feature_meta import FeatureMetadata, FeatureType
from versatil.models.layers import MLP
from versatil.models.layers.activation import ActivationFunction


class MLPFusion(SequentialFusion):
    """Combines sequence features by projecting them into a shared embedding space, concatenating, and then applying an MLP."""

    def __init__(
        self,
        input_features: list[str],
        output_name: str,
        hidden_dimension: int,
        mlp_hidden_dims: list[int],
        activation_name: str = ActivationFunction.GELU.value,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_features: List of feature names to fuse.
            output_name: Name of the output fused feature.
            hidden_dimension: Dimension to project each input feature to before fusion.
            mlp_hidden_dims: List of hidden layer dimensions for the MLP.
            activation_name: Name of the activation function to use in the MLP.
            dropout: Dropout rate for the MLP.
        """
        super().__init__(
            input_features=input_features,
            output_name=output_name,
            hidden_dimension=hidden_dimension,
        )
        self.mlp = MLP(
            input_dimension=hidden_dimension * len(input_features),
            hidden_dimensions=mlp_hidden_dims,
            activation_function=ActivationFunction(
                activation_name
            ).to_torch_activation(),
            dropout=dropout,
        )
        self.output_dim = mlp_hidden_dims[-1]

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of sequence or flat features [B, Seq, D_i], [B, D_i]. Or if observation horizon spans
                multiple timesteps, [B, T, Seq, D_i] or [B, T, D_i].

        Returns:
            Fused features of shape [B, Seq, output_dim] or [B, output_dim]. If observation horizon spans
            multiple timesteps, returns [B, T, Seq, output_dim] or [B, T, output_dim].
        """
        if self.projections is None:
            raise RuntimeError("Projections must be set up before forward pass")
        projected = []
        for feat, proj in zip(features, self.projections):
            projected.append(proj(feat))
        concat = torch.cat(projected, dim=-1)
        result: torch.Tensor = self.mlp(concat)
        return result

    def get_output_specification(self) -> FeatureMetadata:
        """Get output specification."""
        dimension: tuple[int, ...] = (self.output_dim,)
        if self._output_feature_type == FeatureType.SEQUENTIAL.value:
            dimension = (self._output_sequence_length, self.output_dim)
        return FeatureMetadata(
            key=self.output_name,
            feature_type=self._output_feature_type,
            dimension=dimension,
        )
