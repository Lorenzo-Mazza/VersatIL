import abc

from torch import nn

from refactoring.models.encoding.fusion.base import FusionInput, FusionModule
from refactoring.models.encoding.fusion.constants import FeatureType


class SequentialFusion(FusionModule, abc.ABC):
    """Combines sequence features (e.g., for temporal or token sequences)."""
    def __init__(
            self,
            input_features: list[str],
            output_name: str,
            hidden_dim: int,
    ):
        """
        Args:
            input_features: List of feature names to fuse.
            output_name: Name of the output fused feature.
            hidden_dim: Dimension to project each input feature to before fusion.
        """
        input_specification = FusionInput(input_features=input_features, feature_type=FeatureType.ANY.value)
        super().__init__(input_specification=input_specification, output_name=output_name)
        self.projections: nn.ModuleList | None = None
        self.hidden_dim = hidden_dim


    def _setup_layers(self, feature_keys_to_dims: dict[str, int | tuple]):
        """Build projection layers..."""
        input_dims_raw = [feature_keys_to_dims[feat] for feat in self.input_features]
        input_dims: list[int] = []
        for feat_name, dim in zip(self.input_features, input_dims_raw):
            if isinstance(dim, tuple):
                if len(dim)>2:
                    raise ValueError(f"SequentialFusion requires flat or sequential dimensions, but '{feat_name}' has dimension {dim}. "
                        f"Use SpatialFusion for spatial features.")
                proj_dim = dim[-1]
            else:
                proj_dim = dim
            input_dims.append(proj_dim)
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.hidden_dim)
            for dim in input_dims
        ])
