"""Explainability contracts exposed by encoding modules."""

import enum
from dataclasses import dataclass

import torch.nn as nn


class ExplanationTargetKind(enum.StrEnum):
    """Kinds of encoder activations that can be converted into visual maps."""

    SPATIAL_FEATURE_MAP = "spatial_feature_map"
    TOKEN_SEQUENCE = "token_sequence"


class ActivationLayout(enum.StrEnum):
    """Tensor layouts produced by explainability target layers."""

    NCHW = "nchw"
    NHWC = "nhwc"
    NLC = "nlc"


@dataclass(frozen=True)
class VisionExplanationTarget:
    """Target-layer metadata needed to convert activations into image maps.

    Attributes:
        layer: Module whose forward activation should be captured.
        target_kind: Target category from :class:`ExplanationTargetKind`.
        activation_layout: Layout of the captured activation from
            :class:`ActivationLayout`.
        output_index: Optional index when ``layer`` returns a tuple. ``None``
            selects the first tensor output.
        prefix_token_count: Number of prefix tokens to discard before reshaping
            ViT patch-token attributions.
        patch_grid: Optional ``(height, width)`` patch grid for token targets.
            If omitted, token maps can only be inferred when the remaining token
            count is a perfect square.
    """

    layer: nn.Module
    target_kind: str
    activation_layout: str
    output_index: int | None = None
    prefix_token_count: int = 0
    patch_grid: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Validate target metadata.

        Raises:
            ValueError: If ``target_kind`` or ``activation_layout`` is not a
                supported enum value.
            ValueError: If ``prefix_token_count`` is negative.
        """
        valid_kinds = [kind.value for kind in ExplanationTargetKind]
        if self.target_kind not in valid_kinds:
            raise ValueError(
                f"Invalid target_kind '{self.target_kind}'. "
                f"Must be one of: {valid_kinds}"
            )

        valid_layouts = [layout.value for layout in ActivationLayout]
        if self.activation_layout not in valid_layouts:
            raise ValueError(
                f"Invalid activation_layout '{self.activation_layout}'. "
                f"Must be one of: {valid_layouts}"
            )

        if self.prefix_token_count < 0:
            raise ValueError(
                f"prefix_token_count must be non-negative. "
                f"Got: {self.prefix_token_count}"
            )


def resolve_timm_feature_info_layer(
    backbone: nn.Module,
    layer_index: int,
) -> nn.Module | None:
    """Resolve a timm feature extractor module from ``feature_info`` metadata.

    Args:
        backbone: timm feature extractor module.
        layer_index: Feature-info output index selected by the encoder.

    Returns:
        Module that produces the selected feature output, or ``None`` when the
        backbone has no resolvable feature-info module for that index.
    """
    feature_info = getattr(backbone, "feature_info", None)
    if feature_info is None:
        return None
    module_name = feature_info.module_name(layer_index)
    named_modules = dict(backbone.named_modules())
    if module_name in named_modules:
        return named_modules[module_name]
    flattened_module_name = module_name.replace(".", "_")
    return named_modules.get(flattened_module_name)
