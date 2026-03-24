"""BatchNorm detection and replacement for quantization preparation."""

import torch
from torch import nn

BATCHNORM_ATTRIBUTE_NAMES = ("running_mean", "running_var", "weight", "bias")

STANDARD_BATCHNORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


def has_batchnorm_buffers(module: nn.Module) -> bool:
    """Check if module has the four canonical BN attributes.

    Checks for running_mean, running_var, weight, and bias as either
    buffers or parameters, since standard BN stores weight/bias as
    parameters while frozen variants store them as buffers.
    """
    for name in BATCHNORM_ATTRIBUTE_NAMES:
        attribute = getattr(module, name, None)
        if not isinstance(attribute, torch.Tensor):
            return False
    return True


def _is_standard_batchnorm_already_prepared(module: nn.Module) -> bool:
    """Check if module is a standard BN already in non-tracking eval state."""
    if not isinstance(module, STANDARD_BATCHNORM_TYPES):
        return False
    return not module.training and not module.track_running_stats


def is_frozen_batchnorm(module: nn.Module) -> bool:
    """Check if module is a frozen or non-standard BN that needs replacement."""
    if not has_batchnorm_buffers(module):
        return False
    if _is_standard_batchnorm_already_prepared(module):
        return False
    return not isinstance(module, STANDARD_BATCHNORM_TYPES)


def extract_batchnorm_parameters(
    batchnorm: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float] | None:
    """Extract (running_mean, running_var, weight, bias, eps) from any BN variant."""
    if not has_batchnorm_buffers(batchnorm):
        return None
    running_mean = batchnorm.running_mean
    running_var = batchnorm.running_var
    weight = batchnorm.weight
    bias = batchnorm.bias
    eps = getattr(batchnorm, "eps", 1e-5)
    return running_mean, running_var, weight, bias, eps


def extract_activation(batchnorm: nn.Module) -> nn.Module | None:
    """Extract fused activation from BN modules like FrozenBatchNormAct2d."""
    activation = getattr(batchnorm, "act", None)
    if activation is None:
        return None
    if isinstance(activation, nn.Identity):
        return None
    return activation


def _create_replacement_batchnorm(
    batchnorm: nn.Module,
    num_features: int,
) -> nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d:
    """Create a standard BatchNorm with parameters copied from a frozen BN.

    Detects the original batch norm dimension from the module type.
    Falls back to BatchNorm2d for frozen or non-standard variants.
    """
    class_name = type(batchnorm).__name__.lower()
    # This string detection is hacky but allows handling arbitrary BN variants
    if "1d" in class_name:
        replacement_class = nn.BatchNorm1d
    elif "3d" in class_name:
        replacement_class = nn.BatchNorm3d
    else:
        replacement_class = nn.BatchNorm2d

    parameters = extract_batchnorm_parameters(batchnorm)
    if parameters is None:
        raise ValueError(
            f"Module {type(batchnorm).__name__} does not have "
            f"the required BatchNorm buffers"
        )
    running_mean, running_var, weight, bias, eps = parameters
    device = running_mean.device
    replacement = replacement_class(num_features=num_features, eps=eps, device=device)
    replacement.running_mean.data.copy_(running_mean.data)
    replacement.running_var.data.copy_(running_var.data)
    replacement.weight.data.copy_(weight.data)
    replacement.bias.data.copy_(bias.data)
    replacement.eval()
    replacement.track_running_stats = False
    return replacement


def replace_frozen_batchnorm(model: nn.Module) -> int:
    """Recursively replace all frozen BN with standard BatchNorm variants.

    Returns the number of modules replaced.
    """
    replacement_count = 0
    for name, child in list(model.named_children()):
        if is_frozen_batchnorm(child):
            num_features = child.running_mean.shape[0]
            replacement = _create_replacement_batchnorm(
                batchnorm=child,
                num_features=num_features,
            )
            activation = extract_activation(child)
            if activation is not None:
                combined = nn.Sequential(replacement, activation)
                setattr(model, name, combined)
            else:
                setattr(model, name, replacement)
            replacement_count += 1
        else:
            replacement_count += replace_frozen_batchnorm(child)
    return replacement_count


def prepare_batchnorms_for_quantization(model: nn.Module) -> int:
    """Replace frozen BN and set all BN to eval mode with tracking disabled.

    Returns the total number of frozen BN modules replaced.
    """
    replacement_count = replace_frozen_batchnorm(model)
    for module in model.modules():
        if isinstance(module, STANDARD_BATCHNORM_TYPES):
            module.eval()
            module.track_running_stats = False
    return replacement_count
