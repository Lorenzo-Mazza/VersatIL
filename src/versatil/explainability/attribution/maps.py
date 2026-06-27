"""Feature-grid and image-grid attribution map helpers."""

import math

import torch
import torch.nn.functional as F

from versatil.explainability.constants import ExplanationType
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
    VisionExplanationTarget,
)


def get_batch_temporal_shape(camera_tensor: torch.Tensor) -> tuple[int, int]:
    """Return batch and temporal dimensions from a camera tensor.

    Args:
        camera_tensor: Camera tensor with shape ``(B, T, C, H, W)`` or
            ``(B, C, H, W)``.

    Returns:
        ``(batch_size, temporal_length)``. Non-temporal inputs use
        ``temporal_length=1``.

    Raises:
        ValueError: If the tensor is neither 4D nor 5D.
    """
    if camera_tensor.dim() == 5:
        return camera_tensor.shape[0], camera_tensor.shape[1]
    if camera_tensor.dim() == 4:
        return camera_tensor.shape[0], 1
    raise ValueError(
        f"Camera tensor must have shape (B, T, C, H, W) or (B, C, H, W). "
        f"Got: {tuple(camera_tensor.shape)}"
    )


def get_image_size(camera_tensor: torch.Tensor) -> tuple[int, int]:
    """Return image height and width from a camera tensor.

    Args:
        camera_tensor: Camera tensor with shape ``(B, T, C, H, W)`` or
            ``(B, C, H, W)``.

    Returns:
        ``(image_height, image_width)``.

    Raises:
        ValueError: If the tensor is neither 4D nor 5D.
    """
    if camera_tensor.dim() not in {4, 5}:
        raise ValueError(
            f"Camera tensor must have shape (B, T, C, H, W) or (B, C, H, W). "
            f"Got: {tuple(camera_tensor.shape)}"
        )
    return camera_tensor.shape[-2], camera_tensor.shape[-1]


def normalize_map(heatmap: torch.Tensor) -> torch.Tensor:
    """Normalize each heatmap independently to ``[0, 1]``.

    Args:
        heatmap: Heatmap tensor with spatial dimensions in the final two axes.

    Returns:
        Tensor with the same shape as ``heatmap`` and per-map min/max scaling.
    """
    flattened = heatmap.flatten(start_dim=-2)
    minimum = flattened.min(dim=-1, keepdim=True).values.unsqueeze(-1)
    maximum = flattened.max(dim=-1, keepdim=True).values.unsqueeze(-1)
    return (heatmap - minimum) / (maximum - minimum + 1e-8)


def activation_to_nchw(
    tensor: torch.Tensor,
    activation_layout: str,
) -> torch.Tensor:
    """Convert captured feature-map activations to NCHW.

    Args:
        tensor: Captured activation or gradient tensor.
        activation_layout: Layout from :class:`ActivationLayout`.

    Returns:
        Tensor with shape ``(N, C, H, W)``.

    Raises:
        ValueError: If ``activation_layout`` is not a spatial layout.
    """
    if activation_layout == ActivationLayout.NCHW.value:
        return tensor
    if activation_layout == ActivationLayout.NHWC.value:
        return tensor.permute(0, 3, 1, 2).contiguous()
    raise ValueError(
        f"Activation layout '{activation_layout}' is not a spatial feature-map layout."
    )


def nchw_to_activation_layout(
    tensor: torch.Tensor,
    activation_layout: str,
) -> torch.Tensor:
    """Convert NCHW activations back to their captured layout.

    Args:
        tensor: Tensor with shape ``(N, C, H, W)``.
        activation_layout: Target layout from :class:`ActivationLayout`.

    Returns:
        Tensor in ``activation_layout``.

    Raises:
        ValueError: If ``activation_layout`` is not a spatial layout.
    """
    if activation_layout == ActivationLayout.NCHW.value:
        return tensor
    if activation_layout == ActivationLayout.NHWC.value:
        return tensor.permute(0, 2, 3, 1).contiguous()
    raise ValueError(
        f"Activation layout '{activation_layout}' is not a spatial feature-map layout."
    )


def token_sequence_to_nchw(
    tensor: torch.Tensor,
    target: VisionExplanationTarget,
    tensor_name: str,
) -> torch.Tensor:
    """Convert ViT patch-token tensors to NCHW.

    Args:
        tensor: Token tensor with shape ``(N, L, C)``.
        target: Token target metadata containing prefix-token and patch-grid
            information.
        tensor_name: Name used in validation errors.

    Returns:
        Tensor with shape ``(N, C, H_patch, W_patch)`` after dropping prefix
        tokens.

    Raises:
        ValueError: If the target layout is not ``NLC``.
        ValueError: If ``tensor`` is not 3D.
        RuntimeError: If patch-grid resolution fails.
    """
    if target.activation_layout != ActivationLayout.NLC.value:
        raise ValueError(
            f"ViT token targets require NLC activations. "
            f"Got: {target.activation_layout}"
        )
    if tensor.dim() != 3:
        raise ValueError(
            f"ViT token targets expect (N, L, C) {tensor_name}. "
            f"Got: {tuple(tensor.shape)}"
        )

    patch_tokens = tensor[:, target.prefix_token_count :]
    patch_height, patch_width = resolve_patch_grid(
        target=target,
        token_count=patch_tokens.shape[1],
    )
    return patch_tokens.reshape(
        patch_tokens.shape[0],
        patch_height,
        patch_width,
        patch_tokens.shape[2],
    ).permute(0, 3, 1, 2)


def target_tensor_to_nchw(
    tensor: torch.Tensor,
    target: VisionExplanationTarget,
    tensor_name: str,
) -> torch.Tensor:
    """Convert a captured target tensor to NCHW for feature-grid attribution.

    Args:
        tensor: Captured activation or gradient tensor.
        target: Target metadata describing tensor kind and layout.
        tensor_name: Name used in validation errors.

    Returns:
        Tensor with shape ``(N, C, H, W)``.

    Raises:
        ValueError: If the target kind or activation layout is unsupported.
    """
    if target.target_kind == ExplanationTargetKind.TOKEN_SEQUENCE.value:
        return token_sequence_to_nchw(
            tensor=tensor,
            target=target,
            tensor_name=tensor_name,
        )
    if target.target_kind == ExplanationTargetKind.SPATIAL_FEATURE_MAP.value:
        return activation_to_nchw(
            tensor=tensor,
            activation_layout=target.activation_layout,
        )
    raise ValueError(f"Unsupported target kind: {target.target_kind}")


def nchw_to_target_tensor(
    tensor: torch.Tensor,
    target: VisionExplanationTarget,
    original_tensor: torch.Tensor,
) -> torch.Tensor:
    """Convert ablated NCHW activations back to the target-layer output layout.

    Args:
        tensor: Ablated activation tensor with shape ``(N, C, H, W)``.
        target: Target metadata describing whether the activation came from a
            spatial feature map or a ViT token sequence.
        original_tensor: Original selected target-layer output. Token targets
            keep prefix tokens from this tensor because they are not part of the
            patch heatmap grid.

    Returns:
        Replacement tensor matching ``original_tensor`` layout.

    Raises:
        ValueError: If the target kind or activation layout is unsupported.
        ValueError: If a token target does not have a 3D original tensor.
        RuntimeError: If token patch-grid metadata disagrees with ``tensor``.
    """
    if target.target_kind == ExplanationTargetKind.SPATIAL_FEATURE_MAP.value:
        return nchw_to_activation_layout(
            tensor=tensor,
            activation_layout=target.activation_layout,
        )

    if target.target_kind != ExplanationTargetKind.TOKEN_SEQUENCE.value:
        raise ValueError(f"Unsupported target kind: {target.target_kind}")

    if target.activation_layout != ActivationLayout.NLC.value:
        raise ValueError(
            f"ViT target replacement requires NLC activations. "
            f"Got: {target.activation_layout}"
        )
    if original_tensor.dim() != 3:
        raise ValueError(
            "ViT target replacement expects (N, L, C) original activations. "
            f"Got: {tuple(original_tensor.shape)}"
        )

    original_patch_count = original_tensor.shape[1] - target.prefix_token_count
    if original_patch_count < 0:
        raise ValueError(
            f"prefix_token_count {target.prefix_token_count} exceeds original "
            f"token count {original_tensor.shape[1]}."
        )
    patch_height, patch_width = resolve_patch_grid(
        target=target,
        token_count=original_patch_count,
    )
    if (patch_height, patch_width) != (tensor.shape[2], tensor.shape[3]):
        raise RuntimeError(
            f"Ablated token grid {(tensor.shape[2], tensor.shape[3])} does not "
            f"match target patch grid {(patch_height, patch_width)}."
        )
    if original_tensor.shape[0] != tensor.shape[0]:
        raise RuntimeError(
            f"Ablated token batch size {tensor.shape[0]} does not match original "
            f"batch size {original_tensor.shape[0]}."
        )
    if original_tensor.shape[2] != tensor.shape[1]:
        raise RuntimeError(
            f"Ablated token channel count {tensor.shape[1]} does not match "
            f"original channel count {original_tensor.shape[2]}."
        )

    patch_token_count = patch_height * patch_width
    patch_tokens = tensor.permute(0, 2, 3, 1).reshape(
        tensor.shape[0],
        patch_token_count,
        tensor.shape[1],
    )
    prefix_tokens = original_tensor[:, : target.prefix_token_count]
    return torch.cat([prefix_tokens, patch_tokens], dim=1)


def compute_gradcam(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    target: VisionExplanationTarget,
    explanation_type: str,
) -> torch.Tensor:
    """Compute GradCAM or GradCAM++ from captured target activations.

    Args:
        activation: Captured target activation.
        gradient: Gradient of the selected prediction with respect to
            ``activation``.
        target: Target metadata describing activation layout.
        explanation_type: GradCAM variant from :class:`ExplanationType`.

    Returns:
        Unnormalized heatmaps with shape ``(N, H_feature, W_feature)``.

    Raises:
        ValueError: If the target layout is unsupported.
        RuntimeError: If token patch-grid resolution fails.
    """
    activation = target_tensor_to_nchw(
        tensor=activation,
        target=target,
        tensor_name="activations",
    )
    gradient = target_tensor_to_nchw(
        tensor=gradient,
        target=target,
        tensor_name="gradients",
    )
    if explanation_type == ExplanationType.GRADCAM_PLUS_PLUS.value:
        gradient_squared = gradient.square()
        gradient_cubed = gradient_squared * gradient
        activation_sum = activation.sum(dim=(2, 3), keepdim=True)
        alpha = gradient_squared / (
            2 * gradient_squared + activation_sum * gradient_cubed + 1e-8
        )
        weights = (alpha * F.relu(gradient)).sum(dim=(2, 3))
    else:
        weights = gradient.mean(dim=(2, 3))
    return F.relu((weights[:, :, None, None] * activation).sum(dim=1))


def resolve_patch_grid(
    target: VisionExplanationTarget,
    token_count: int,
) -> tuple[int, int]:
    """Resolve patch-token count to a 2D grid.

    Args:
        target: Token target metadata. ``patch_grid`` is used when present.
        token_count: Number of patch tokens after dropping prefix tokens.

    Returns:
        ``(patch_grid_height, patch_grid_width)``.

    Raises:
        RuntimeError: If an explicit grid does not match ``token_count``.
        RuntimeError: If no grid is configured and ``token_count`` is not a
            perfect square.
    """
    if target.patch_grid is not None:
        patch_height, patch_width = target.patch_grid
        if patch_height * patch_width != token_count:
            raise RuntimeError(
                f"Target patch grid {target.patch_grid} does not match "
                f"{token_count} patch tokens."
            )
        return patch_height, patch_width

    grid_size = math.isqrt(token_count)
    if grid_size * grid_size != token_count:
        raise RuntimeError(
            f"Cannot infer a square patch grid from {token_count} tokens. "
            "Set patch_grid on the encoder explainability target."
        )
    return grid_size, grid_size


def compute_target_map(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    target: VisionExplanationTarget,
    explanation_type: str,
) -> torch.Tensor:
    """Compute a feature-grid heatmap from a captured target.

    Args:
        activation: Captured target activation.
        gradient: Captured gradient for ``activation``.
        target: Target metadata describing target kind and layout.
        explanation_type: Requested gradient explanation type.

    Returns:
        Unnormalized heatmaps on the target feature grid.
    """
    return compute_gradcam(
        activation=activation,
        gradient=gradient,
        target=target,
        explanation_type=explanation_type,
    )


def resize_feature_heatmap_to_camera(
    feature_heatmap: torch.Tensor,
    camera_tensor: torch.Tensor,
) -> torch.Tensor:
    """Resize flattened feature heatmaps back to camera image windows.

    Args:
        feature_heatmap: Feature-grid heatmap with shape ``(B*T, H, W)``.
        camera_tensor: Source camera tensor with shape ``(B, T, C, H, W)`` or
            ``(B, C, H, W)``.

    Returns:
        Normalized image-space heatmap with shape ``(B, T, H_image, W_image)``.

    Raises:
        ValueError: If ``camera_tensor`` rank is unsupported.
    """
    image_height, image_width = get_image_size(camera_tensor=camera_tensor)
    image_heatmap = F.interpolate(
        feature_heatmap[:, None],
        size=(image_height, image_width),
        mode="bicubic",
        align_corners=False,
    )[:, 0]
    batch_size, temporal_length = get_batch_temporal_shape(camera_tensor=camera_tensor)
    image_heatmap = image_heatmap.reshape(
        batch_size,
        temporal_length,
        image_height,
        image_width,
    )
    return normalize_map(image_heatmap)
