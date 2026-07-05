"""Shared explainability type aliases."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from versatil.models.encoding.explainability import VisionExplanationTarget
from versatil.models.policy import Policy

type ObservationValue = torch.Tensor | str | list[str] | list[list[str]]
type ObservationBatch = dict[str, ObservationValue]
type ActionBatch = dict[str, torch.Tensor]
type ExplanationMetadataValue = str | int | float | bool | None | list[int] | list[str]
type PolicyPredictionSelector = Callable[[dict[str, torch.Tensor]], torch.Tensor]
type TensorModuleOutput = torch.Tensor | tuple[torch.Tensor | None, ...]
type PolicyExplanationObjective = Callable[
    [Policy, ObservationBatch, ActionBatch | None, bool],
    torch.Tensor,
]


@dataclass(frozen=True)
class VisionExplainableModule:
    """Camera-addressable visual module exposed to the explainer.

    Attributes:
        name: Stable module path used by runner filters and output metadata.
        module: Module that owns ``target`` and participates in policy forward
            passes.
        target: Target layer metadata used to capture activations and reshape
            feature maps.
        camera_keys: Observation camera keys that can be attributed through
            this module.
        capture_mode: Hook routing mode from ``VisionCaptureMode``. It tells
            attribution code whether the target layer runs once, once per
            camera, or once on a stacked camera batch.
    """

    name: str
    module: torch.nn.Module
    target: VisionExplanationTarget
    camera_keys: tuple[str, ...]
    capture_mode: str


@dataclass(frozen=True)
class CameraExplanationTarget:
    """Concrete visual target for one output camera heatmap.

    Attributes:
        camera_key: Observation camera key whose overlay should be generated.
        vision_module_name: Name of the module that produced the target
            activation.
        target: Target layer metadata used by Grad-CAM and Ablation-CAM.
        capture_mode: Hook routing mode from ``VisionCaptureMode``.
        invocation_index: Selected forward-hook call for modules invoked once
            per camera. ``None`` means the first captured call is used.
        stacked_camera_index: Camera index inside a stacked camera batch when
            ``capture_mode`` is ``stacked_camera_batch``.
        stacked_camera_count: Number of cameras inside the stacked batch.
    """

    camera_key: str
    vision_module_name: str
    target: VisionExplanationTarget
    capture_mode: str
    invocation_index: int | None = None
    stacked_camera_index: int | None = None
    stacked_camera_count: int | None = None


class ExplanationHeatmapFunction(Protocol):
    """Callable contract used by the explanation method registry."""

    def __call__(
        self,
        policy: Policy,
        observation: ObservationBatch,
        actions: ActionBatch | None,
        target_camera: str | None,
        target_vision_module_names: list[str] | None,
        preprocess_observation: bool,
    ) -> dict[str, torch.Tensor]:
        """Compute heatmaps for one explanation method.

        Args:
            policy: Policy whose prediction is being explained.
            observation: Observation tensors keyed by observation-space names.
            actions: Optional action tensors used by discrete predictors to
                score teacher-forced action-token likelihoods.
            target_camera: Optional camera key selected by runner filtering.
            target_vision_module_names: Optional visual module allowlist.
            preprocess_observation: Whether to run policy preprocessing before
                attribution.

        Returns:
            Heatmaps keyed by camera name.
        """
