"""torch.nn.Module to construct an input feature vector to use as conditioner for a U-Net."""

import torch
from torch import nn

from versatil.data.constants import SampleKey
from versatil.models.decoding.constants import AlgorithmContextKey
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.layers.feature_projection import FeatureProjection


class UNetInputBuilder(nn.Module):
    """Builds a flattened conditioning vector for U-Net decoders from multi-modal features.

    This module takes a dictionary of encoded features (from various encoders like
    RGB, depth, proprioceptive, etc.), projects them to a common embedding dimension,
    and concatenates them into a single conditioning vector suitable for U-Net input.

    Features are processed based on their dimensionality:
        - 2D (B, Emb): Used directly (pooled/single token features)
        - 3D (B, Seq, Emb): Flattened to (B, Seq*Emb)
        - 4D with time (B, T, Seq, Emb): Flattened to (B, T*Seq*Emb)
        - 4D spatial or 5D: Not supported (must be pooled first)

    Class tokens (if present) are appended at the end of the feature vector.

    Args:
        embedding_dimension: Target dimension for projecting all features.

    Example:
        >>> builder = UNetInputBuilder(embedding_dimension=256)
        >>> features = {"rgb_pooled": torch.randn(4, 512), "proprio": torch.randn(4, 64)}
        >>> conditioning = builder(features)  # Shape: (4, 256 + 256)
    """

    def __init__(
        self,
        embedding_dimension: int,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.projection = FeatureProjection(embedding_dimension)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Project and concatenate features into a single conditioning vector.

        Args:
            features: Dictionary mapping feature names to tensors. Padding masks
                and pad action keys are automatically filtered out.

        Returns:
            Concatenated feature tensor of shape (B, total_features * embedding_dimension),
            or None if no valid features are provided.

        Raises:
            ValueError: If a feature has an unsupported shape (4D spatial or 5D).
        """
        clean_features = {
            k: v
            for k, v in features.items()
            if EncoderOutputKeys.PADDING_MASK.value not in k
            and k != SampleKey.IS_PAD_ACTION.value
        }
        projected = self.projection(
            clean_features
        )  # Project all features to common embedding dim
        flat_features_list = []
        cls_token = None
        for name in sorted(projected.keys()):
            x = projected[name]
            if x.ndim == 2:  # algorithm context (B, Emb)
                feature_embedding = x  # (B, Emb)
            elif x.ndim == 3:  # temporal vector (B, T, Emb)
                B, _, _ = x.shape
                feature_embedding = x.reshape(B, -1)  # (B, T*Emb)
            elif x.ndim == 4:  # temporal token sequence (B, T, Seq, Emb)
                B, _, _, _ = x.shape
                feature_embedding = x.reshape(B, -1)  # (B, T*Seq*Emb)
            elif x.ndim == 5:  # temporal spatial (B, T, Emb, H, W)
                raise ValueError(
                    f"5D feature '{name}' is not supported as input to U-Net Decoder. "
                    "Please pool your features accordingly using the encoding pipeline."
                )
            else:
                raise ValueError(f"Feature '{name}' has unsupported shape {x.shape}")

            if AlgorithmContextKey.CLASS_TOKEN.value in name:
                cls_token = feature_embedding
            else:
                flat_features_list.append(feature_embedding)

        # Append cls token at the end, if present
        if cls_token is not None:
            flat_features_list.append(cls_token)

        return (
            torch.cat(flat_features_list, dim=-1) if flat_features_list else None
        )  # (B, Total_len*Emb)
