"""Constants for explanation types used for policy interpretability."""

import enum


class ExplanationType(enum.StrEnum):
    """Supported visual-explanation methods."""

    GRADCAM = "gradcam"
    GRADCAM_PLUS_PLUS = "gradcam++"
    ABLATION_CAM = "ablation_cam"


VALID_EXPLANATION_TYPES: tuple[str, ...] = tuple(
    member.value for member in ExplanationType
)


class ExplanationSourceType(enum.StrEnum):
    """Supported sources for explanation batches."""

    DATASET = "dataset"
    ONLINE_INFERENCE = "online_inference"


class ExplanationDatasetSplit(enum.StrEnum):
    """Dataset splits available to offline explanation runs."""

    TRAIN = "train"
    VAL = "val"
    ALL = "all"


VALID_EXPLANATION_SOURCE_TYPES: tuple[str, ...] = tuple(
    member.value for member in ExplanationSourceType
)


class VisionCaptureMode(enum.StrEnum):
    """Forward-hook routing used by visual attribution targets."""

    SINGLE_CALL = "single_call"
    PER_CAMERA_CALL = "per_camera_call"
    STACKED_CAMERA_BATCH = "stacked_camera_batch"
