"""Constants for explanation types used for policy interpretability."""
import enum


class ExplanationType(enum.Enum):
    GRADCAM = "gradcam"
    GRADCAM_PLUS_PLUS = "gradcam++"
    ABLATION_CAM = "ablation_cam"
    SALIENCY_MAP = "saliency_map"
    INTEGRATED_GRADIENT = "integrated_gradient"
