"""Conv2d + BatchNorm2d fusion for quantization preparation."""

import torch
from torch import nn

from versatil.post_training_compression.preparation.batchnorm import (
    extract_activation,
    extract_batchnorm_parameters,
    has_batchnorm_buffers,
)


def fuse_conv_batchnorm(
    conv: nn.Conv2d,
    batchnorm: nn.Module,
) -> nn.Conv2d:
    """Create new Conv2d with BN folded into weights and bias."""
    parameters = extract_batchnorm_parameters(batchnorm)
    if parameters is None:
        raise ValueError(
            f"Module {type(batchnorm).__name__} does not have "
            f"the required BatchNorm buffers"
        )
    running_mean, running_var, weight, bias, eps = parameters
    device = conv.weight.device
    scale = weight.to(device) / torch.sqrt(running_var.to(device) + eps)
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
        device=device,
    )

    running_mean = running_mean.to(device)
    bias = bias.to(device)
    reshape_dims = (-1,) + (1,) * (conv.weight.dim() - 1)
    fused_conv.weight.data = conv.weight.data * scale.reshape(reshape_dims)
    if conv.bias is not None:
        fused_conv.bias.data = bias + (conv.bias.data - running_mean) * scale
    else:
        fused_conv.bias.data = bias - running_mean * scale
    return fused_conv


def _unwrap_batchnorm_container(
    module: nn.Module,
) -> tuple[nn.Module | None, nn.Module | None]:
    """Return the BN module and trailing activation held by a child module.

    ``replace_frozen_batchnorm`` rewrites BN+activation modules into
    ``nn.Sequential(batchnorm, activation)``; fusion must look through that
    wrapper or it silently skips exactly the modules the preparation targeted.
    """
    if has_batchnorm_buffers(module):
        return module, extract_activation(module)
    if (
        isinstance(module, nn.Sequential)
        and len(module) > 0
        and has_batchnorm_buffers(module[0])
    ):
        activation = module[1] if len(module) > 1 else None
        return module[0], activation
    return None, None


def _find_conv_batchnorm_pairs(
    module: nn.Module,
) -> list[tuple[nn.Module, str, str]]:
    """Find consecutive Conv2d + BN child pairs in a module's children.

    Pairing assumes children are declared in forward-execution order (the
    torchvision/timm convention). Pairs whose channel counts disagree are
    skipped: the BN cannot belong to the preceding convolution.
    """
    pairs = []
    children = list(module.named_children())
    for index in range(len(children) - 1):
        current_name, current_child = children[index]
        next_name, next_child = children[index + 1]
        if not isinstance(current_child, nn.Conv2d):
            continue
        batchnorm, _ = _unwrap_batchnorm_container(next_child)
        if batchnorm is None:
            continue
        if batchnorm.running_mean.shape[0] != current_child.out_channels:
            continue
        pairs.append((module, current_name, next_name))
    return pairs


def fuse_all_conv_batchnorm_pairs(model: nn.Module) -> int:
    """Find and fuse all consecutive Conv2d+BN children in all submodules.

    Returns the number of pairs fused.
    """
    fusion_count = 0
    for parent in list(model.modules()):
        pairs = _find_conv_batchnorm_pairs(parent)
        for container, conv_name, batchnorm_name in pairs:
            conv = getattr(container, conv_name)
            batchnorm, activation = _unwrap_batchnorm_container(
                getattr(container, batchnorm_name)
            )
            fused = fuse_conv_batchnorm(conv=conv, batchnorm=batchnorm)
            setattr(container, conv_name, fused)
            if activation is not None:
                setattr(container, batchnorm_name, activation)
            else:
                setattr(container, batchnorm_name, nn.Identity())
            fusion_count += 1
    return fusion_count
