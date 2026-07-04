import torch
from torch import nn

from versatil.models.encoding.fusion.base import SequentialFusion
from versatil.models.feature_meta import FeatureMetadata, FeatureType


class AttentionFusion(SequentialFusion):
    """Combines sequence features by projecting them into a shared embedding space and then applying cross-attention to them.
    If only one feature is provided, it is returned as-is after projection."""

    def __init__(
        self,
        input_features: list[str],
        output_name: str,
        hidden_dimension: int,
        input_feature_query: str | None = None,
        number_of_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_norm: bool = True,
    ):
        """
        Args:
            input_features: List of feature names to fuse.
            output_name: Name of the output fused feature.
            hidden_dimension: Dimension to project each input feature to before fusion.
            input_feature_query: Name of the feature to use as query in cross-attention. If None, uses the first feature.
            number_of_heads: Number of attention heads.
            dropout: Dropout rate for attention weights.
            use_residual: Whether to add a residual connection from the input to the output.
            use_norm: Whether to apply layer normalization after projection and before fusion.
        """
        super().__init__(
            input_features=input_features,
            output_name=output_name,
            hidden_dimension=hidden_dimension,
        )
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.attention = nn.MultiheadAttention(
            hidden_dimension,
            num_heads=number_of_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.input_feature_query = input_feature_query
        self.norms: nn.ModuleList | None = None
        if self.use_norm:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(self.hidden_dimension) for _ in input_features]
            )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of features shaped [B, T, S, D_i] (sequential),
                [B, T, D_i] (flat with time), or [B, D_i] (flat).

        Returns:
            Fused features [B, T, hidden_dimension] or [B, hidden_dimension]
        """
        if self.projections is None:
            raise RuntimeError("Projections must be set up before forward pass")
        projected = [
            proj(feat) for feat, proj in zip(features, self.projections, strict=True)
        ]
        if self.use_norm:
            if self.norms is None:
                raise RuntimeError("Norms should be initialized when use_norm is True")
            projected = [norm(p) for p, norm in zip(projected, self.norms, strict=True)]
        # Sequential features arrive as (B, T, S, D) — fusion runs before the
        # pipeline's T=1 squeeze — so batch and time are merged for attention
        # over tokens and restored afterwards.
        batch_time_shape: torch.Size | None = None
        if projected[0].dim() == 4:
            batch_time_shape = projected[0].shape[:2]
            projected = [
                p.reshape(-1, p.shape[2], p.shape[3]) for p in projected
            ]  # [B*T, S, D]
        has_sequence = projected[0].dim() == 3
        if not has_sequence:
            projected = [p.unsqueeze(1) for p in projected]  # [B, 1, D]

        feature_to_use_as_query = self.input_feature_query or self.input_features[0]
        query_idx = self.input_features.index(feature_to_use_as_query)
        query = projected[query_idx]
        other_features = [p for i, p in enumerate(projected) if i != query_idx]

        if len(other_features) > 0:
            key_value = torch.cat(other_features, dim=1)  # [B, sum(S_i), D]
            attention_map, _ = self.attention(query, key_value, key_value)
            fused = query + attention_map if self.use_residual else attention_map
        else:
            fused = query

        if not has_sequence:
            fused = fused.squeeze(1)
        if batch_time_shape is not None:
            fused = fused.reshape(
                batch_time_shape[0], batch_time_shape[1], fused.shape[1], fused.shape[2]
            )

        result: torch.Tensor = fused
        return result

    def get_output_specification(self) -> FeatureMetadata:
        """Get output specification."""
        dimension: tuple[int, ...] = (self.hidden_dimension,)
        if self._output_feature_type == FeatureType.SEQUENTIAL.value:
            dimension = (self._output_sequence_length, self.hidden_dimension)
        return FeatureMetadata(
            key=self.output_name,
            feature_type=self._output_feature_type,
            dimension=dimension,
        )
