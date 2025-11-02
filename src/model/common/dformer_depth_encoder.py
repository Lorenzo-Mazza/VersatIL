from typing import Optional

import torch
from torch import nn

from model.common.dformerv2 import DFormerv2_S, DFormerv2_B, DFormerv2_L
from model.common.spatial_softmax import SpatialSoftmax2d
from model.diffusion.vision_encoder.encoder_utils import replace_bn_with_gn


class DFormerDepthEncoder(nn.Module):
    """
    Fused RGB+Depth encoder using DFormerv2 (GroupNorm-converted).
    Expects: rgb=(B,3,H,W), depth=(B,1,H,W); returns (B, out_dim).
    """


    def __init__(self, variant: str = 'S', out_dim: int = 512,
                 out_indices = (3,), drop_path_rate: float = 0.0, pretrained: Optional[str] = None, **dformer_kwargs):
        super().__init__()
        # Build DFormer
        if variant.upper() == 'S':
            self.backbone = DFormerv2_S(out_indices=out_indices, drop_path_rate=drop_path_rate, **dformer_kwargs)
        elif variant.upper() == 'B':
            self.backbone = DFormerv2_B(out_indices=out_indices, drop_path_rate=drop_path_rate, **dformer_kwargs)
        elif variant.upper() == 'L':
            self.backbone = DFormerv2_L(out_indices=out_indices, drop_path_rate=drop_path_rate, **dformer_kwargs)
        else:
            raise ValueError(f"Unknown DFormer variant: {variant}")

        # Load pretrained if provided
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)

        # Convert BN -> GN
        self.backbone = replace_bn_with_gn(self.backbone)
        last_stage_c = self.backbone.layers[out_indices[-1]].embed_dim
        self.pool = SpatialSoftmax2d(normalize=True, temperature=1.0, learnable_temperature=False)
        self.head = nn.Linear(2 * last_stage_c, out_dim)

    @torch.no_grad()
    def get_output_dim(self) -> int:
        return self.head.out_features

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, return_feature_map: bool = False) -> torch.Tensor:
        """
        rgb:   (B,3,H,W)
        depth: (B,1,H,W)   (only the first channel is used internally)
        """
        feats = self.backbone(rgb, depth)  # tuple of feature maps at out_indices
        x = feats[-1]                    # (B, C, Hs, Ws)
        if return_feature_map:
            return x  # Return last feature map
        pooled = self.pool(x)            # (B, 2C)
        return self.head(pooled)         # (B, out_dim)



