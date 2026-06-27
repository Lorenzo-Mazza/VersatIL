"""Heatmap function registry for visual explanation methods."""

from functools import partial

from versatil.explainability.attribution.ablation import (
    compute_ablation_maps_for_policy,
)
from versatil.explainability.attribution.gradients import (
    compute_gradient_maps_for_policy,
)
from versatil.explainability.constants import ExplanationType
from versatil.explainability.typedefs import ExplanationHeatmapFunction


def to_explanation_heatmaps(
    channel_batch_size: int,
) -> dict[str, ExplanationHeatmapFunction]:
    """Build the registry of supported policy heatmap computations.

    Args:
        channel_batch_size: Number of feature channels ablated per policy
            forward pass for Ablation-CAM.

    Returns:
        Mapping from ``ExplanationType`` values to callables that compute
        camera heatmaps for a policy and observation batch.
    """
    return {
        ExplanationType.GRADCAM.value: partial(
            compute_gradient_maps_for_policy,
            explanation_type=ExplanationType.GRADCAM.value,
        ),
        ExplanationType.GRADCAM_PLUS_PLUS.value: partial(
            compute_gradient_maps_for_policy,
            explanation_type=ExplanationType.GRADCAM_PLUS_PLUS.value,
        ),
        ExplanationType.ABLATION_CAM.value: partial(
            compute_ablation_maps_for_policy,
            channel_batch_size=channel_batch_size,
        ),
    }
