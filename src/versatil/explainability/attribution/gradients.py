"""Gradient-based visual attribution methods."""

import torch

from versatil.explainability.attribution.capture import GradientCapture
from versatil.explainability.attribution.maps import (
    compute_target_map,
    resize_feature_heatmap_to_camera,
)
from versatil.explainability.attribution.objectives import (
    compute_policy_explanation_objective,
    resolve_actions_for_explanation,
)
from versatil.explainability.constants import ExplanationType
from versatil.explainability.typedefs import (
    ActionBatch,
    ObservationBatch,
    PolicyPredictionSelector,
)
from versatil.explainability.vision_modules import resolve_camera_explanation_targets
from versatil.models.policy import Policy


def compute_gradient_maps_for_policy(
    policy: Policy,
    observation: ObservationBatch,
    actions: ActionBatch | None = None,
    explanation_type: str = ExplanationType.GRADCAM.value,
    output_selector: PolicyPredictionSelector | None = None,
    target_camera: str | None = None,
    target_vision_module_names: list[str] | None = None,
    preprocess_observation: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute gradient visual maps for policy visual modules.

    This is the policy-level entry point for gradient explanations. CNN
    feature maps and ViT patch-token activations use the same public method;
    token activations are reshaped to a patch grid before Grad-CAM weighting.

    Args:
        policy: Policy instance to explain.
        observation: Raw observation tensors keyed by camera names. Camera
            tensors must have shape ``(B, T, C, H, W)`` or ``(B, C, H, W)``.
        actions: Optional action tensors. Discrete predictors use tokenized
            actions when available; otherwise pseudo-target tokens are generated
            before attribution hooks are registered.
        explanation_type: Either ``gradcam`` or ``gradcam++``.
        output_selector: Optional function selecting the prediction tensor to
            explain for continuous predictors.
        target_camera: Optional camera key to explain. If omitted, every camera
            exposed by a visual module is explained.
        target_vision_module_names: Optional visual module allowlist.
        preprocess_observation: Whether to normalize/tokenize ``observation``
            before attribution. Online inference windows should use ``True``;
            dataset batches that already passed through ``EpisodicDataset``
            should use ``False``.

    Returns:
        Heatmaps keyed by camera with shape ``(B, T, H, W)``.

    Raises:
        ValueError: If ``explanation_type`` is not a supported gradient method.
        ValueError: If ``target_camera`` is not exposed by a visual module.
        ValueError: If a camera tensor has an unsupported rank.
        RuntimeError: If the selected visual module does not expose a compatible
            explainability target.
        RuntimeError: If target-layer hooks do not capture activation or
            gradient tensors.
    """
    valid_types = [
        ExplanationType.GRADCAM.value,
        ExplanationType.GRADCAM_PLUS_PLUS.value,
    ]
    if explanation_type not in valid_types:
        raise ValueError(
            f"Unsupported gradient explanation_type '{explanation_type}'. "
            f"Use one of: {valid_types}"
        )

    resolved_actions = resolve_actions_for_explanation(
        policy=policy,
        observation=observation,
        actions=actions,
        preprocess_observation=preprocess_observation,
    )
    # Frozen vision towers produce requires_grad=False activations unless the
    # inputs carry gradients.
    observation = {
        key: value.clone().requires_grad_(True)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for key, value in observation.items()
    }
    camera_targets = resolve_camera_explanation_targets(
        policy=policy,
        target_camera=target_camera,
        target_vision_module_names=target_vision_module_names,
    )
    heatmap_accumulator: dict[str, list[torch.Tensor]] = {}

    for camera_target in camera_targets:
        camera = camera_target.camera_key
        if camera not in observation:
            raise ValueError(
                f"Camera '{camera}' resolved as an explanation target but is "
                "missing from the observation; the checkpoint's observation "
                "space and the provided observation disagree."
            )

        capture = GradientCapture(target=camera_target)
        forward_handle = camera_target.target.layer.register_forward_hook(
            capture.forward_hook
        )
        try:
            policy.zero_grad()
            objective = compute_policy_explanation_objective(
                policy=policy,
                observation=observation,
                actions=resolved_actions,
                preprocess_observation=preprocess_observation,
                output_selector=output_selector,
            )
            objective.mean().backward()
            activation, gradient = capture.require_tensors()
        finally:
            forward_handle.remove()

        feature_heatmap = compute_target_map(
            activation=activation,
            gradient=gradient,
            target=camera_target.target,
            explanation_type=explanation_type,
        )
        camera_tensor = observation[camera]
        if not isinstance(camera_tensor, torch.Tensor):
            raise ValueError(f"Camera '{camera}' is not a tensor observation.")
        heatmap_accumulator.setdefault(camera, []).append(
            resize_feature_heatmap_to_camera(
                feature_heatmap=feature_heatmap,
                camera_tensor=camera_tensor,
            )
        )

    return {
        camera: torch.stack(camera_heatmaps, dim=0).mean(dim=0)
        for camera, camera_heatmaps in heatmap_accumulator.items()
    }
