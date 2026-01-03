# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from legacy_constants import Cameras
from model.detr.utils import NestedTensor, is_main_process
from model.detr.position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


WEIGHTS_MAP = {
    "resnet18": ResNet18_Weights.DEFAULT,
    "resnet34": ResNet34_Weights.DEFAULT,
    "resnet50": ResNet50_Weights.DEFAULT,
    "resnet101": ResNet101_Weights.DEFAULT,
    "resnet152": ResNet152_Weights.DEFAULT,
}


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        weights = WEIGHTS_MAP.get(name) if is_main_process() else None
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class DepthBackbone(BackboneBase):
    """Specialized ResNet backbone for depth (grayscale) images."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        weights = WEIGHTS_MAP.get(name) if is_main_process() else None
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )

        # Replace the first conv layer to accept single-channel input
        # Save the weights for the RGB model's first layer to adapt to grayscale
        original_weight = backbone.conv1.weight.clone()

        backbone.conv1 = nn.Conv2d(
            1,
            backbone.conv1.out_channels,
            kernel_size=backbone.conv1.kernel_size,
            stride=backbone.conv1.stride,
            padding=backbone.conv1.padding,
            bias=False if backbone.conv1.bias is None else True,
        )

        # Initialize the new layer with the average of RGB weights
        # This provides a better starting point than random initialization
        backbone.conv1.weight.data = original_weight.sum(dim=1, keepdim=True)

        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(
    backbone: str,
    hidden_dim: int,
    position_embedding: str,
    lr_backbone: float,
    dilation: bool,
    masks: bool = False,
    is_depth: bool = False,
) -> nn.Module:
    """Build a backbone for image processing.

    Args:
        backbone: Name of the backbone model (e.g., 'resnet18', 'resnet50')
        hidden_dim: Dimension of the hidden layers
        position_embedding: Type of position encoding to use (e.g., 'sine', 'learned')
        lr_backbone: Learning rate for the backbone
        dilation: Whether to use dilation in the backbone
        masks: Whether to return intermediate layers for segmentation masks
        is_depth: Whether this backbone is for a depth image (grayscale)
    """
    position_embedding = build_position_encoding(
        hidden_dim=hidden_dim, position_embedding=position_embedding
    )
    train_backbone = lr_backbone > 0
    return_interm_layers = masks
    if is_depth:
        # Create a grayscale-specific backbone for depth images
        backbone = DepthBackbone(
            backbone, train_backbone, return_interm_layers, dilation
        )
    else:
        # Regular RGB backbone
        backbone = Backbone(backbone, train_backbone, return_interm_layers, dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_backbones(
    camera_names: List[str],
    backbone: str,
    hidden_dim: int,
    position_embedding: str,
    lr_backbone: float,
    dilation: bool,
    masks: bool = False,
) -> Dict[str, nn.Module]:
    """Build backbones for all cameras specified in the config.

    Args:
        camera_names: List of camera names (e.g., ['left', 'right', 'depth'])
        backbone: Name of the backbone model to use
        hidden_dim: Dimension of the hidden layers
        position_embedding: Type of position encoding to use
        lr_backbone: Learning rate for the backbone
        dilation: Whether to use dilation in the backbone
        masks: Whether to return intermediate layers for segmentation masks
    """
    backbones = OrderedDict()
    for cam_name in camera_names:
        is_depth = cam_name == Cameras.DEPTH.value
        backbones[cam_name] = build_backbone(
            backbone=backbone,
            hidden_dim=hidden_dim,
            position_embedding=position_embedding,
            lr_backbone=lr_backbone,
            dilation=dilation,
            masks=masks,
            is_depth=is_depth,
        )
    return backbones
