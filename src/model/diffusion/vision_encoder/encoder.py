from typing import Union, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from legacy_constants import Cameras, DepthFusionStrategy
from model.common.dformer_depth_encoder import DFormerDepthEncoder
from model.common.spatial_softmax import SpatialSoftmax2d
from model.diffusion.vision_encoder.crop_randomizer import CropRandomizer
from model.diffusion.vision_encoder.encoder_utils import replace_bn_with_gn, get_resnet


class DepthResNet(nn.Module):
    def __init__(
        self, backbone="resnet18", num_classes=512, use_group_norm=True, pretrained=True
    ):
        super().__init__()

        # Load pretrained RGB model
        base_model = get_resnet(
            backbone, weights="IMAGENET1K_V1" if pretrained else None
        )

        # Adapted conv1 for 1 channel
        self.conv1 = nn.Conv2d(
            1,
            base_model.conv1.out_channels,
            kernel_size=base_model.conv1.kernel_size,
            stride=base_model.conv1.stride,
            padding=base_model.conv1.padding,
            bias=False,
        )

        # Adapt weights: average RGB channels
        if pretrained:
            original_weight = base_model.conv1.weight.data
            self.conv1.weight.data = original_weight.sum(dim=1, keepdim=True) / 3.0

        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Spatial softmax pooling
        self.spatial_softmax = SpatialSoftmax2d(
            normalize=True, temperature=1.0, learnable_temperature=False
        )

        # Linear layer after pooling
        in_features = (
            base_model.layer4[-1].conv2.out_channels * 2
        )  # for action_embedding,y per channel
        self.fc = nn.Linear(in_features, num_classes)

        if use_group_norm:
            self = replace_bn_with_gn(self)

    def forward(self, x, return_feature_map=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_feature_map:
            return x  # (B, C, H', W')
        pooled = self.spatial_softmax(x)
        return self.fc(pooled)


# Modified RGB ResNet to optionally return feature map
class RGBResNet(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        num_classes=512,
        input_channels=3,
        use_group_norm=True,
        pretrained=False,
    ):
        super().__init__()

        base_model = get_resnet(
            backbone, weights="IMAGENET1K_V1" if pretrained else None
        )

        self.conv1 = nn.Conv2d(
            input_channels,
            base_model.conv1.out_channels,
            kernel_size=base_model.conv1.kernel_size,
            stride=base_model.conv1.stride,
            padding=base_model.conv1.padding,
            bias=False,
        )
        if pretrained:
            original_weight = base_model.conv1.weight.data
            if input_channels == 3:
                self.conv1.weight.data = original_weight
            elif input_channels < 3:
                self.conv1.weight.data = original_weight[:, :input_channels]
            else:  # >3, repeat channels
                self.conv1.weight.data[:, :3] = original_weight
                remaining = input_channels - 3
                self.conv1.weight.data[:, 3:] = original_weight[
                    :, : remaining % 3
                ].repeat(1, remaining // 3 + 1, 1, 1)[:, :remaining]

        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Spatial softmax pooling
        self.spatial_softmax = SpatialSoftmax2d(
            normalize=True, temperature=1.0, learnable_temperature=False
        )

        # Linear layer after pooling
        in_features = (
            base_model.layer4[-1].conv2.out_channels * 2
        )  # for action_embedding,y per channel
        self.fc = nn.Linear(in_features, num_classes)

        if use_group_norm:
            self = replace_bn_with_gn(self)

    def forward(self, x, return_feature_map=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_feature_map:
            return x
        pooled = self.spatial_softmax(x)
        return self.fc(pooled)


class MultiImageObsEncoder(nn.Module):
    def __init__(
        self,
        obs_shapes: dict,
        backbone: str = "resnet18",
        resize_shape: Union[tuple, dict, None] = None,
        crop_shape: Union[tuple, dict, None] = None,
        random_crop: bool = False,
        use_group_norm: bool = True,
        imagenet_norm: bool = False,
        depth_fusion_strategy: str = DepthFusionStrategy.SEPARATE.value,  # 'separate', 'width', 'left_channel_wise', 'attention'
        pretrained: bool = True,
        freeze_dformer: bool = False,
        dformer_checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.image_keys = []
        self.state_keys = []
        self.key_transforms = nn.ModuleDict()
        self.key_encoders = nn.ModuleDict()
        self.key_shapes = dict(obs_shapes)

        has_depth = Cameras.DEPTH.value in obs_shapes
        self.depth_fusion_strategy = None if not has_depth else depth_fusion_strategy

        for key, shape in obs_shapes.items():
            if len(shape) == 3:  # Image
                self.image_keys.append(key)
                # Transforms (resize, crop, norm)
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape:
                    h, w = (
                        resize_shape.get(key, resize_shape)
                        if isinstance(resize_shape, dict)
                        else resize_shape
                    )
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                this_randomizer = nn.Identity()
                if crop_shape:
                    h, w = (
                        crop_shape.get(key, crop_shape)
                        if isinstance(crop_shape, dict)
                        else crop_shape
                    )
                    this_randomizer = (
                        CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                        if random_crop
                        else torchvision.transforms.CenterCrop(size=(h, w))
                    )
                if key != Cameras.DEPTH.value:
                    this_normalizer = (
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        )
                        if imagenet_norm
                        else nn.Identity()
                    )
                else:
                    this_normalizer = nn.Identity()  # No norm for depth (0-1)

                self.key_transforms[key] = nn.Sequential(
                    this_resizer, this_randomizer, this_normalizer
                )
                # Skip encoder creation for fused keys in 'dformer'
                if (
                    self.depth_fusion_strategy
                    == DepthFusionStrategy.GEOMETRIC_ATTENTION.value
                    and key in [Cameras.LEFT.value, Cameras.DEPTH.value]
                ):
                    continue

                if key == Cameras.DEPTH.value:
                    self.key_encoders[key] = DepthResNet(
                        backbone=backbone,
                        use_group_norm=use_group_norm,
                        pretrained=pretrained,
                    )
                else:
                    self.key_encoders[key] = RGBResNet(
                        backbone=backbone,
                        input_channels=shape[0],
                        use_group_norm=use_group_norm,
                        pretrained=pretrained,
                    )

            else:  # State
                self.state_keys.append(key)

        self.image_keys = sorted(self.image_keys)
        self.state_keys = sorted(self.state_keys)
        if self.key_encoders:
            first_encoder = self.key_encoders[list(self.key_encoders.keys())[0]]
            self.map_dim = first_encoder.layer4[-1].conv2.out_channels
            self.feature_dim = first_encoder.fc.out_features
        else:
            first_encoder = self.dformer_encoder
            self.map_dim = first_encoder.backbone.layers[
                -1
            ].embed_dim  # Last stage channels in DFormerv2
            self.feature_dim = (
                first_encoder.get_output_dim()
            )  # Custom method in DFormerDepthEncoder

        if depth_fusion_strategy == DepthFusionStrategy.LEFT_CHANNEL_WISE.value:
            if Cameras.LEFT.value not in self.image_keys:
                raise ValueError(
                    "Left-channel depth fusion requires depth and left rgb key."
                )
            # Fusion proj for channel cat on features
            self.fusion_proj = nn.Conv2d(self.map_dim * 2, self.map_dim, kernel_size=1)

        if depth_fusion_strategy == DepthFusionStrategy.ATTENTION.value:
            self.attention_fusion = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.feature_dim, nhead=8, batch_first=True
                ),
                num_layers=1,
            )
        if self.depth_fusion_strategy == DepthFusionStrategy.GEOMETRIC_ATTENTION.value:
            if Cameras.LEFT.value not in self.image_keys:
                raise ValueError(
                    "Geometric attention depth fusion requires depth and left rgb key."
                )

            self.dformer_encoder = DFormerDepthEncoder(
                variant="S",
                out_dim=512,  # Default
                pretrained=dformer_checkpoint_path,
            )
            self.dformer_encoder.requires_grad_(False)
            if not freeze_dformer:
                for param in self.dformer_encoder.backbone.layers[-1].parameters():
                    param.requires_grad = True  # Unfreeze last layer

    def forward(self, obs_dict, is_train: Optional[bool] = None):
        features = []
        is_sequence = any(obs_dict[key].dim() == 5 for key in self.image_keys)
        B, T = (
            obs_dict[self.image_keys[0]].shape[:2]
            if is_sequence
            else (obs_dict[self.image_keys[0]].shape[0], None)
        )
        flat_dim = (B * T) if is_sequence else B
        cat_dim_spatial = (
            4 if is_sequence else 3
        )  # For feat_maps: seq (B,T,C,H,W), cat width=4; non-seq (B,C,H,W), cat=3
        cat_dim_channel = 2 if is_sequence else 1

        if self.depth_fusion_strategy == DepthFusionStrategy.GEOMETRIC_ATTENTION.value:
            # Fuse left RGB + Depth using DFormer's geometric attention encoding
            if Cameras.LEFT.value not in self.image_keys:
                raise ValueError(
                    "Geometric attention fusion requires depth and left rgb key."
                )
            rgb = obs_dict[Cameras.LEFT.value]
            depth = obs_dict[Cameras.DEPTH.value]
            if is_sequence:
                rgb = rgb.view(flat_dim, *rgb.shape[2:])
                depth = depth.view(flat_dim, *depth.shape[2:])
            rgb = self.key_transforms[Cameras.LEFT.value](rgb) if is_train else rgb
            depth = (
                self.key_transforms[Cameras.DEPTH.value](depth) if is_train else depth
            )
            fused_feat = self.dformer_encoder(rgb, depth)
            if is_sequence:
                fused_feat = fused_feat.view(B, T, -1)
            features.append(fused_feat)

            # Encode remaining RGB views separately
            other_keys = [
                k
                for k in self.image_keys
                if k not in [Cameras.LEFT.value, Cameras.DEPTH.value]
            ]
            for key in other_keys:
                img = obs_dict[key]
                if is_sequence:
                    img = img.view(flat_dim, *img.shape[2:])
                img = self.key_transforms[key](img) if is_train else img
                feat = self.key_encoders[key](img)
                if is_sequence:
                    feat = feat.view(B, T, -1)
                features.append(feat)

        elif (
            self.depth_fusion_strategy == DepthFusionStrategy.SEPARATE.value
            or self.depth_fusion_strategy is None
        ):
            # Encode each separately, cat flats
            for key in self.image_keys:
                img = obs_dict[key]
                if is_sequence:  # B,T,C,H,W
                    img = img.flatten(0, 1)
                img = self.key_transforms[key](img) if is_train else img
                feat = self.key_encoders[key](img)
                if is_sequence:
                    feat = feat.reshape(B, T, -1)
                features.append(feat)

        elif self.depth_fusion_strategy == DepthFusionStrategy.WIDTH.value:
            # Encode to feature maps, cat along width, then pool and fc
            feat_maps = {}
            for key in self.image_keys:
                img = obs_dict[key]
                if is_sequence:
                    img = img.view(flat_dim, *img.shape[2:])
                img = self.key_transforms[key](img) if is_train else img
                feat_map = self.key_encoders[key](
                    img, return_feature_map=True
                )  # (flat_dim,512,H',W')
                if is_sequence:
                    feat_map = feat_map.view(B, T, *feat_map.shape[1:])
                feat_maps[key] = feat_map

            all_maps = [feat_maps[k] for k in self.image_keys]
            fused_map = torch.cat(all_maps, dim=cat_dim_spatial)  # Cat along width
            if is_sequence:
                fused_map = fused_map.view(flat_dim, *fused_map.shape[2:])
            pooled = self.key_encoders[self.image_keys[0]].spatial_softmax(fused_map)
            fused_feat = self.key_encoders[self.image_keys[0]].fc(pooled)
            if is_sequence:
                fused_feat = fused_feat.view(B, T, -1)
            features.append(fused_feat)

        elif self.depth_fusion_strategy == DepthFusionStrategy.LEFT_CHANNEL_WISE.value:
            # Mid-fusion: channel cat left rgb features + depth features, project, cat with others along width
            feat_maps = {}
            for key in self.image_keys:
                img = obs_dict[key]
                if is_sequence:
                    img = img.view(flat_dim, *img.shape[2:])
                img = self.key_transforms[key](img) if is_train else img
                feat_map = self.key_encoders[key](img, return_feature_map=True)
                if is_sequence:
                    feat_map = feat_map.view(B, T, *feat_map.shape[1:])
                feat_maps[key] = feat_map

            # Fuse primary + depth
            fused_primary = torch.cat(
                [feat_maps[Cameras.LEFT.value], feat_maps[Cameras.DEPTH.value]],
                dim=cat_dim_channel,
            )
            if is_sequence:
                fused_primary = fused_primary.view(flat_dim, *fused_primary.shape[2:])
            fused_primary = self.fusion_proj(fused_primary)
            if is_sequence:
                fused_primary = fused_primary.view(B, T, *fused_primary.shape[1:])

            # Cat with remaining (flexible count)
            remaining_keys = [
                k
                for k in self.image_keys
                if k not in [Cameras.LEFT.value, Cameras.DEPTH.value]
            ]
            fused_map = torch.cat(
                [fused_primary] + [feat_maps[k] for k in remaining_keys],
                dim=cat_dim_spatial,
            )
            if is_sequence:
                fused_map = fused_map.view(flat_dim, *fused_map.shape[2:])
            pooled = self.key_encoders[Cameras.LEFT.value].spatial_softmax(fused_map)
            fused_feat = self.key_encoders[Cameras.LEFT.value].fc(pooled)
            if is_sequence:
                fused_feat = fused_feat.view(B, T, -1)
            features.append(fused_feat)

        elif self.depth_fusion_strategy == DepthFusionStrategy.ATTENTION.value:
            # Encode each to flat, stack as tokens, attention fuse, mean pool
            view_feats = []
            for key in self.image_keys:
                img = obs_dict[key]
                if is_sequence:
                    img = img.view(flat_dim, *img.shape[2:])
                img = self.key_transforms[key](img) if is_train else img
                feat = self.key_encoders[key](img)  # flat feat (flat_dim, 512)
                if is_sequence:
                    feat = feat.view(B, T, -1)
                view_feats.append(feat.unsqueeze(-2))  # (B,T,1,512) or (B,1,512)

            stacked = torch.cat(
                view_feats, dim=-2
            )  # (B,T,num_views,512) or (B,num_views,512)
            if is_sequence:
                stacked = stacked.view(flat_dim, stacked.shape[-2], stacked.shape[-1])
            fused = self.attention_fusion(stacked)  # (flat_dim, num_views, 512)
            fused_feat = fused.mean(dim=1)  # (flat_dim, 512)
            if is_sequence:
                fused_feat = fused_feat.view(B, T, -1)
            features.append(fused_feat)

        # Add state features
        for key in self.state_keys:
            features.append(obs_dict[key])

        return torch.cat(features, dim=-1)

    def get_output_dim(self):
        rgb_count = len([k for k in self.image_keys if k != Cameras.DEPTH.value])
        has_depth = Cameras.DEPTH.value in self.image_keys
        state_dim = sum(self.key_shapes[k][0] for k in self.state_keys)
        if self.depth_fusion_strategy in [
            DepthFusionStrategy.LEFT_CHANNEL_WISE.value,
            DepthFusionStrategy.ATTENTION.value,
            DepthFusionStrategy.WIDTH.value,
        ]:
            return self.feature_dim + state_dim
        elif (
            self.depth_fusion_strategy == DepthFusionStrategy.GEOMETRIC_ATTENTION.value
        ):
            return (self.feature_dim * rgb_count) + state_dim
        else:  # Separate encoder for depth
            return (
                self.feature_dim * (rgb_count + (1 if has_depth else 0))
            ) + state_dim
