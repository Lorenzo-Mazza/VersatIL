"""torch.nn.Module to construct transformer input token sequences from a group of diverse input features.

Across the module, the following abbreviations are used:
- B: Batch size
- T: Temporal length (if applicable)
- C: Channels/ Original Feature dimension
- H: Height (for spatial features)
- W: Width (for spatial features)
- Emb: Embedding dimension
- PE: Positional Encoding
"""

import torch
from torch import nn as nn

from versatil.data.constants import SampleKey
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.layers.dynamic_feature_embedding import DynamicFeatureEmbedding
from versatil.models.layers.feature_projection import FeatureProjection
from versatil.models.layers.positional_encoding.base import PositionalEncoding2D, PositionalEncoding1D


class TransformerInputBuilder(nn.Module):
    """Transforms input features into a sequence of token embeddings to feed into a transformer.

    Note:

        This module:
        i). projects multiple features into a common embedding dimension.
          - Spatial feature maps (B, C, H, W) or temporal-spatial (B, T, C, H, W) are projected into the common channel
          embedding dimension Emb using 1x1 convolutions, with output (B, Optional[T], Emb, H, W).
          - Flat features (B, D) or sequential features (B, T, D) are projected using linear layer, with output
           (B, Emb) or (B, T, Emb).
        ii). Each spatial feature is flattened into token sequences of size (B, H*W, Emb) or (B, T*H*W, Emb)
        iii). The feature tokens are concatenated together along the sequence dimension to produce a unified
          token sequence (B, Total_Seq, Emb).

        Optionally, accepts a spatial positional encoding layer (1D or 2D, sinusoidal or learned), a temporal one (1D)
         and a flat feature positional encoding (1D)cto compute and return the final matching positional encodings
         with shape (B, Total_Seq, Emb).
        For spatial features
        - if `spatial_positional_encoding_layer` (2D) is provided and no temporal layer →  identical 2D PE
          is repeated for every frame
        - If both `spatial` (2D) and `temporal` (1D) PE layers are provided → repeated 2D spatial PE
          + 1D temporal PE broadcast over H×W tokens and added

        - If spatial_positional_encoding_layer is not provided → all features (visual + flat) receive
         one global 1D positional encoding from `flat_positional_encoding_layer`.

        Features may have different spatial sizes (H, W) or temporal lengths (T),
         as long as batch dims are consistent within each feature.

    Example:
        >>> pos_enc = SinusoidalPositionalEncoding2D(embedding_dimension=256)
        >>> input_builder = TransformerInputBuilder(embedding_dim=256, spatial_positional_encoding_layer=pos_enc)
        >>> features = {
        ...     "rgb": torch.randn(8, 3, 16, 16),         # (B, C, H, W)
        ...     "depth": torch.randn(8, 5, 1, 32, 32),    # (B, T, C, H, W)
        ... }
        >>> tokens, pos = input_builder(features)
        >>> tokens.shape  # (8, (16*16 + 5*32*32), 256)
        >>> pos.shape     # (8, (16*16 + 5*32*32), 256)
    """
    def __init__(
            self,
            embedding_dim: int,
            has_time_dim: bool = False,
            spatial_positional_encoding_layer: PositionalEncoding2D | None = None,
            flat_positional_encoding_layer: PositionalEncoding1D | None = None,
            temporal_positional_encoding_layer: PositionalEncoding1D | None = None,
            use_camera_embeddings: bool = True,
            exclude_keys: list[str] | None = None,
    ):
        """Initialize TransformerInputBuilder.

        Args:
            embedding_dim: Common embedding dimension for all features.
            has_time_dim: Whether input features include a time dimension.
            spatial_positional_encoding_layer: Optional 2D positional encoding layer for spatial features.
            flat_positional_encoding_layer: Optional 1D positional encoding layer for flat/sequential features.
            temporal_positional_encoding_layer: Optional 1D positional encoding layer for temporal dimension.
            use_camera_embeddings: Whether to use camera embeddings for multi-camera 2D PE, so that each camera
                view can be distinguished in the transformer input.
            exclude_keys: Optional list of feature keys to exclude from the input sequence.
        Raises:
            ValueError: If provided positional encoding layers do not match expected types or dimensions.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.exclude_keys = set(exclude_keys) if exclude_keys else set()
        self.projection = FeatureProjection(embedding_dim, has_time_dim=has_time_dim)
        if spatial_positional_encoding_layer is not None:
            if not isinstance(spatial_positional_encoding_layer, PositionalEncoding2D):
                raise ValueError("spatial_positional_encoding_layer must be PositionalEncoding2D.")
            if spatial_positional_encoding_layer.embedding_dimension != embedding_dim:
                raise ValueError("spatial_positional_encoding_layer embedding dimension does not match.")
        if temporal_positional_encoding_layer is not None:
            if not isinstance(temporal_positional_encoding_layer, PositionalEncoding1D):
                raise ValueError("temporal_positional_encoding_layer must be PositionalEncoding1D.")
            if temporal_positional_encoding_layer.embedding_dimension != embedding_dim:
                raise ValueError("temporal_positional_encoding_layer embedding dimension does not match.")
        if flat_positional_encoding_layer is not None:
            if not isinstance(flat_positional_encoding_layer, PositionalEncoding1D):
                raise ValueError("flat_positional_encoding_layer must be PositionalEncoding1D.")
            if flat_positional_encoding_layer.embedding_dimension != embedding_dim:
                raise ValueError("flat_positional_encoding_layer embedding dimension does not match.")
        self.spatial_positional_encoding_layer = spatial_positional_encoding_layer
        self.temporal_positional_encoding_layer = temporal_positional_encoding_layer
        self.has_time_dim = has_time_dim
        self.flat_positional_encoding_layer = flat_positional_encoding_layer
        self.camera_embeddings = DynamicFeatureEmbedding(embedding_dim) if use_camera_embeddings else None


    def forward(self, features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None,
    torch.Tensor | None]:
        """Projects and concatenate features into sequences of token embeddings, with optional positional encodings.
        Args:
            features: Dict of features with several possible shapes:
                - spatial features (B, C, H, W)
                - temporal-spatial features(B, T, C, H, W)
                - flat features(B, D)
                - sequential features (B, T, D)
                - temporal-sequential features (B, T, Seq, D)
                - padding mask of shape (B, Seq) or (B, T) or (B, T, Seq) with boolean values

        Note: If the CLS token is included, it is always appended at the end of the sequence.

        Returns:
            Tuple of:
            - concatenated token sequences (B, Total_Seq, Emb)
            - optional positional encodings (B, Total_Seq, Emb)
            - optional is-padding mask (B, Total_Seq), where True indicates padded tokens
        """
        action_padding_mask = features.get(SampleKey.IS_PAD_ACTION.value, None)
        clean_features = {
            k: v for k, v in features.items()
            if not EncoderOutputKeys.PADDING_MASK.value in k
            and k != SampleKey.IS_PAD_ACTION.value
            and k not in self.exclude_keys
        }
        projected = self.projection(clean_features) # Project all features to common embedding dim
        spatial_tokens_list = []
        spatial_positional_encodings = []
        spatial_mask_list = []
        flat_tokens_list = []
        flat_mask_list = []
        cls_token, cls_token_positional_encoding, cls_token_padding_mask = None, None, None
        for name in sorted(projected.keys()):
            x = projected[name]
            B, T, Emb, H, W = None, None, None, None, None
            if x.ndim == 2:  # pooled / single token (always T=1)
                token_embeddings = x.unsqueeze(1)  # (B, 1, Emb)
                is_spatial = False
                B, tokens_per_frame, Emb = token_embeddings.shape
                T = 1
            elif x.ndim == 3:
                token_embeddings = x
                is_spatial = False
                if self.has_time_dim: # (B, T, Emb)
                    B, T, Emb = token_embeddings.shape
                    tokens_per_frame = 1
                else: # (B, seq_len, Emb)
                    B, tokens_per_frame, Emb = token_embeddings.shape
                    T = 1
            elif x.ndim == 4:
                if self.has_time_dim:  # (B, T, Seq, Emb)
                    B, T, tokens_per_frame, Emb = x.shape
                    is_spatial = False
                    token_embeddings = x.reshape(B, -1, Emb) # (B, T*Seq, Emb)
                else:  # spatial (B, Emb, H, W)
                    B, Emb, H, W = x.shape
                    T = 1
                    tokens_per_frame = H * W
                    is_spatial = True
                    token_embeddings = x.flatten(2).transpose(1, 2)  # (B, HW, Emb)
            elif x.ndim == 5:  # temporal spatial (B, T, Emb, H, W)
                B, T, Emb, H, W = x.shape
                tokens_per_frame = H * W
                is_spatial = True
                token_embeddings = x.flatten(3).transpose(2,3).reshape(B, T * H * W, Emb)  # (B, T*H*W, Emb)
            else:
                raise ValueError(f"Feature '{name}' has unsupported shape {x.shape}")

            padding_mask = features.get(f"{name}_padding_mask", None)
            if padding_mask is None and SampleKey.ACTION.value in name and action_padding_mask is not None:
                padding_mask = action_padding_mask.clone()
            if padding_mask is not None:
                padding_mask = padding_mask.to(torch.bool)
                match padding_mask.ndim:
                    case 1: # (B,), comes from a pooled feature
                        reshaped_mask = padding_mask.unsqueeze(1) # (B, 1)
                    case 2: # (B, Seq) or (B, T)
                        reshaped_mask = padding_mask # (B, T) or (B, Seq)
                    case 3: # (B, T, Seq)
                        reshaped_mask = padding_mask.reshape(B, -1) # (B, T*Seq)
                    case _:
                        raise ValueError(f"Padding masks not supported for spatial features, "
                                         f"got {padding_mask.ndim} for {name}")
            else:
                reshaped_mask = torch.zeros(B, T*tokens_per_frame, dtype=torch.bool, device=x.device) # B , T*Seq

            if is_spatial and self.spatial_positional_encoding_layer is not None:
                pe_2d = self.spatial_positional_encoding_layer(torch.zeros(1, 1, H, W, device=x.device))  # (1, Emb, H, W)
                pe_flat = pe_2d.flatten(2).transpose(1, 2)  # (1, HW, Emb)
                if self.temporal_positional_encoding_layer is not None and self.has_time_dim:
                    pe_spatial = pe_flat.repeat(1, T, 1) # (1, T*H*W, Emb)
                    pe_time = self.temporal_positional_encoding_layer(torch.zeros(1, T, 1, device=x.device)) # (1, T, Emb)
                    pe_time = pe_time.repeat_interleave(H * W, dim=1)  # (1, T*H*W, Emb)
                    pe = pe_spatial + pe_time  # (1,T*H*W, Emb)
                else:
                    pe = pe_flat.repeat(1, T, 1) if self.has_time_dim else pe_flat  # (1, T*H*W, Emb) or (1, H*W, Emb)

                pe = pe.repeat(B, 1, 1) # (B, seq_len, Emb)
                if self.camera_embeddings is not None:
                    pe = pe + self.camera_embeddings(name, x.device)  # extract camera embeddings with correct key and add to PE
                spatial_positional_encodings.append(pe)


            if is_spatial:
                spatial_tokens_list.append(token_embeddings)
                spatial_mask_list.append(reshaped_mask)
            else:
                if DecoderOutputKey.CLASS_TOKEN.value in name:
                    cls_token = token_embeddings
                    cls_token_padding_mask = reshaped_mask
                else:
                    flat_tokens_list.append(token_embeddings)
                    flat_mask_list.append(reshaped_mask)

        # Append cls token at the end, if present
        if cls_token is not None and cls_token_padding_mask is not None:
            flat_tokens_list.append(cls_token)
            flat_mask_list.append(cls_token_padding_mask)

        spatial_tokens = torch.cat(spatial_tokens_list, dim=1) if spatial_tokens_list else None # (B, L_spatial, Emb)
        flat_tokens = torch.cat(flat_tokens_list, dim=1) if flat_tokens_list else None  # (B, L_flat, Emb)
        tokens = torch.cat([t for t in [spatial_tokens, flat_tokens] if t is not None], dim=1) # (B, Total_Seq, Emb)

        # Compute positional encodings based on configuration
        B = tokens.shape[0]
        if self.spatial_positional_encoding_layer is None:
            # No spatial PE: apply flat PE to ALL tokens (spatial + flat)
            if self.flat_positional_encoding_layer is not None:
                positional_encodings = self.flat_positional_encoding_layer(
                    torch.zeros(1, tokens.shape[1], device=tokens.device)
                )  # (1, Total_Seq, Emb)
                positional_encodings = positional_encodings.expand(B, -1, -1)  # (B, Total_Seq, Emb)
            else:
                positional_encodings = None
        else:
            # Spatial PE exists: use separate spatial PE + flat PE
            spatial_positional_encodings = torch.cat(spatial_positional_encodings, dim=1) if spatial_positional_encodings else None
            flat_positional_encodings = None
            if flat_tokens is not None:
                if self.flat_positional_encoding_layer is not None:
                    flat_positional_encodings = self.flat_positional_encoding_layer(
                        torch.zeros(1, flat_tokens.shape[1], device=flat_tokens.device)
                    )  # (1, L_flat, Emb)
                    flat_positional_encodings = flat_positional_encodings.expand(B, -1, -1)  # (B, L_flat, Emb)
                else:
                    flat_positional_encodings = torch.zeros(
                        B, flat_tokens.shape[1], self.embedding_dim, device=flat_tokens.device
                    )  # (B, L_flat, Emb)
            pe_list = [p for p in [spatial_positional_encodings, flat_positional_encodings] if p is not None]
            positional_encodings = torch.cat(pe_list, dim=1) if pe_list else None  # (B, Total_Seq, Emb)

        padding_mask = torch.cat(spatial_mask_list + flat_mask_list, dim=1) # (B, Total_Seq)
        if not torch.compiler.is_compiling() and not padding_mask.any():
            padding_mask = None
        return tokens, positional_encodings, padding_mask











