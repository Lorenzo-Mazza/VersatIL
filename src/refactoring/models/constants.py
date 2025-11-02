"""Constants for model/policy components."""

import enum


class ExplanationType(str, enum.Enum):
    """Enum for model explanation/interpretability types."""
    GRADCAM = "gradcam"
    GRADCAM_PLUS_PLUS = "gradcam++"
    ABLATION_CAM = "ablation_cam"
    SALIENCY_MAP = "saliency_map"
    INTEGRATED_GRADIENT = "integrated_gradient"
