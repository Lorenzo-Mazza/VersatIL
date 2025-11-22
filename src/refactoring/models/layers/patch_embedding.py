import enum

import torch
import torch.nn as nn


class PatchEmbedType(str, enum.Enum):
    """Patch embedding implementation types."""
    STANDARD = "standard"  # Single conv, standard ViT
    PROGRESSIVE = "progressive"  # Multi-stage like DFormer/Swin
    OVERLAPPING = "overlapping"  # Overlapping patches


class PatchEmbedding(nn.Module):
    """Flexible patch embedding supporting multiple strategies."""
    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            embed_type: str = PatchEmbedType.STANDARD.value,
            norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.embed_type = embed_type

        if embed_type == PatchEmbedType.STANDARD.value:
            self.projection = self._build_standard_projection()
        elif embed_type == PatchEmbedType.PROGRESSIVE.value:
            # Use LayerNorm as default if norm_layer is None
            norm = norm_layer if norm_layer is not None else nn.LayerNorm
            self.projection = self._build_progressive_projection(norm_layer=norm)
        elif embed_type == PatchEmbedType.OVERLAPPING.value:
            self.projection = self._build_overlapping_projection()
        else:
            raise ValueError(f"Unknown embed_type: {embed_type}")

        self.norm = nn.LayerNorm(self.embed_dim) if norm_layer else nn.Identity()


    def _build_standard_projection(self) -> nn.Module:
        """Standard ViT: single large-stride convolution."""
        return nn.Conv2d(
            self.in_chans,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )


    def _build_progressive_projection(self, norm_layer: type[nn.Module]) -> nn.Module:
        """Progressive downsampling like DFormer."""
        # Progressive: downsample by 2 at each stage
        stages = []
        current_dim = self.in_chans
        # Stage 1: in_chans -> embedding_dimension // 2, stride 2
        stages.extend([
            nn.Conv2d(current_dim, self.embed_dim // 2, 3, 2, 1),
            norm_layer(self.embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim // 2, 3, 1, 1),
            norm_layer(self.embed_dim // 2),
            nn.GELU(),
        ])
        # Stage 2: embedding_dimension // 2 -> embedding_dimension, stride 2
        stages.extend([
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, 3, 2, 1),
            norm_layer(self.embed_dim),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1),
            norm_layer(self.embed_dim),
        ])
        # Additional stages if patch_size requires more downsampling
        total_stride = 4  # 2 * 2 from above
        while total_stride < self.patch_size:
            stages.extend([
                nn.Conv2d(self.embed_dim, self.embed_dim, 3, 2, 1),
                norm_layer(self.embed_dim),
                nn.GELU(),
            ])
            total_stride *= 2
        return nn.Sequential(*stages)


    def _build_overlapping_projection(self) -> nn.Module:
        """Overlapping patches with smaller stride."""
        stride = self.patch_size // 2
        padding = self.patch_size // 4
        return nn.Conv2d(
            self.in_chans,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=stride,
            padding=padding
        )


    def forward(self, x: torch.Tensor, return_patch_size: bool = False) -> torch.Tensor | tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Tensor of images with shape (batch size, channels, height, width)
            return_patch_size: If True, also return the effective patch size after embedding.

        Returns:
            For PROGRESSIVE: Tensor of shape (batch size, H', W', embedding_dim)
            For STANDARD/OVERLAPPING: Tensor of shape (batch size, N, embedding_dim) where N = num_patches
        """
        x = self.projection(x)  # (B, embedding_dimension, H', W')
        _, _, H, W = x.shape
        if self.embed_type == PatchEmbedType.PROGRESSIVE.value:
            x = x.permute(0, 2, 3, 1)  # (B, H', W', embedding_dimension)
            if return_patch_size:
                return x, H, W
            else:
                return x
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if return_patch_size:
            return x, H, W
        else:
            return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer: Downsamples spatial dims by 2x using strided conv, changes channel dim.
    Input: [B, H, W, C] (H/W should be even for integer downsampling).
    Output: [B, H//2, W//2, out_dim].
    """
    def __init__(self, dim: int, out_dim: int, norm_layer = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.reduction = nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_layer = norm_layer
        # Instantiate norm based on type
        if norm_layer == nn.LayerNorm:
            # Create LayerNorm once during init, not during forward!
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = norm_layer(out_dim)  # e.g., nn.SyncBatchNorm(out_dim)


    def forward(self, x):
        """
        Forward pass.
        Args:
            x: [B, H, W, C] input tokens.
        Returns:
            [B, H//2, W//2, out_dim] merged tokens (approximate for odd dimensions).

        Note:
            The stride-2 convolution handles odd spatial dimensions naturally via rounding.
            This matches the original DFormerv2 implementation behavior.
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x = self.reduction(x)  # [B, out_dim, H//2, W//2]
        if self.norm_layer != nn.LayerNorm:
            # Apply norm in image format if BatchNorm-like
            x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H//2, W//2, out_dim]
        if self.norm_layer == nn.LayerNorm:
            # Apply LayerNorm in token format (reuse instance created in __init__)
            x = self.norm(x)
        return x
