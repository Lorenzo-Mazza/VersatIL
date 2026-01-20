import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def convert_layers(
    model: nn.Module,
    layer_type_old: type[nn.Module],
    layer_type_new: type[nn.Module],
    convert_weights: bool = False,
) -> nn.Module:
    """
    Recursively convert layers of a specific type in the model to a new type, in-place.

    For GroupNorm conversion, num_groups is automatically computed as num_channels // 16
    with fallback to ensure divisibility.
    Args:
        model: The model to modify in-place.
        layer_type_old: The type of layer to replace (e.g., nn.BatchNorm2d).
        layer_type_new: The type of layer to replace with (e.g., nn.GroupNorm).
        convert_weights: If True, copy weights and biases from old to new layers.

    Returns:
        The modified model with layers converted.
    """
    for name, module in reversed(model._modules.items()):
        if module is None:
            continue
        if len(list(module.children())) > 0:
            model._modules[name] = convert_layers(
                module, layer_type_old, layer_type_new, convert_weights
            )

        if isinstance(module, layer_type_old):
            layer_old = module
            num_channels = module.num_features

            # Compute num_groups automatically for GroupNorm
            if layer_type_new == nn.GroupNorm:
                computed_num_groups = _compute_num_groups(num_channels)
            else:
                computed_num_groups = num_channels

            layer_new = layer_type_new(
                num_groups=computed_num_groups,
                num_channels=num_channels,
                eps=module.eps,
                affine=module.affine,
            )

            if convert_weights and module.affine:
                layer_new.weight.data.copy_(layer_old.weight.data)
                layer_new.bias.data.copy_(layer_old.bias.data)

            model._modules[name] = layer_new

    return model


def _compute_num_groups(num_channels: int) -> int:
    """Compute number of groups for GroupNorm based on num_channels.

    Strategy: num_channels // 16, with fallback to largest divisor.

    Args:
        num_channels: Number of channels in the layer.

    Returns:
        Number of groups that divides num_channels evenly.
    """
    # Target: num_channels // 16
    target_groups = max(1, num_channels // 16)

    # Find largest divisor <= target_groups
    for num_groups in range(target_groups, 0, -1):
        if num_channels % num_groups == 0:
            return num_groups

    # Fallback: use 1 (equivalent to LayerNorm)
    return 1


def replace_batchnorm_with_groupnorm(model: nn.Module) -> nn.Module:
    """Replace all kinds of `_BatchNorm` layers (including SyncBatchNorm) in the model with `GroupNorm` layers.

    Note: The number of groups for `GroupNorm` will be computed automatically as `num_channels // 16`.

    Args:
        model: The model to modify

    Returns:
        The modified model with `BatchNorm` replaced
    """
    return convert_layers(
        model,
        layer_type_old=_BatchNorm,
        layer_type_new=nn.GroupNorm,
        convert_weights=False,
    )
