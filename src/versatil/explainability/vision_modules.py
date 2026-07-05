"""Discovery of camera-addressable visual targets for policy explanations."""

import torch

from versatil.explainability.constants import VisionCaptureMode
from versatil.explainability.typedefs import (
    CameraExplanationTarget,
    VisionExplainableModule,
)
from versatil.models.encoding.encoders.base import EncodingMixin
from versatil.models.encoding.explainability import (
    ExplanationTargetKind,
    VisionExplanationTarget,
)
from versatil.models.policy import Policy

COMPATIBLE_EXPLANATION_TARGET_KINDS = (
    ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
    ExplanationTargetKind.TOKEN_SEQUENCE.value,
)


def get_policy_encoders(policy: Policy) -> dict[str, EncodingMixin]:
    """Return encoders registered in the policy encoding pipeline."""
    return {
        **policy.encoding_pipeline.encoders,
        **policy.encoding_pipeline.conditional_encoders,
    }


def get_vision_explainable_modules(policy: Policy) -> list[VisionExplainableModule]:
    """Return visual modules that can produce camera heatmaps.

    Args:
        policy: Policy whose encoding pipeline and decoder should be inspected.

    Returns:
        Visual modules from the encoding pipeline and decoder-owned VLM vision
        towers. Each entry includes the camera keys that can be attributed
        through that module and the hook routing mode needed to isolate a
        camera when the same module is reused.

    Raises:
        RuntimeError: If no camera-addressable visual target is exposed.
    """
    modules = [
        *get_encoding_pipeline_vision_modules(policy=policy),
        *get_decoder_vision_modules(policy=policy),
    ]
    if modules:
        return modules
    raise RuntimeError(
        "No compatible vision explainability modules found. "
        "Explainability requires visual modules that expose target metadata "
        "through get_explainability_targets()."
    )


def get_encoding_pipeline_vision_modules(
    policy: Policy,
) -> list[VisionExplainableModule]:
    """Return explainable visual modules from the encoding pipeline."""
    modules = []
    for encoder_name, encoder in get_policy_encoders(policy=policy).items():
        camera_keys = _camera_keys_for_module(policy=policy, module=encoder)
        if not camera_keys:
            continue
        target = select_explainability_target(
            targets=_get_explainability_targets(module=encoder),
            module_name=encoder_name,
        )
        if target is None:
            continue
        modules.append(
            VisionExplainableModule(
                name=encoder_name,
                module=encoder,
                target=target,
                camera_keys=camera_keys,
                capture_mode=_capture_mode_for_module(
                    module=encoder,
                    camera_keys=camera_keys,
                ),
            )
        )
    return modules


def get_decoder_vision_modules(policy: Policy) -> list[VisionExplainableModule]:
    """Return explainable visual modules owned by decoder VLM backbones."""
    vlm_backbone = _decoder_vlm_backbone(policy=policy)
    if vlm_backbone is None:
        return []

    modules: list[VisionExplainableModule] = []
    camera_keys = _camera_keys_for_module(policy=policy, module=vlm_backbone)
    direct_target = select_explainability_target(
        targets=_get_explainability_targets(module=vlm_backbone),
        module_name="decoder.vlm_backbone",
    )
    if direct_target is not None and camera_keys:
        modules.append(
            VisionExplainableModule(
                name="decoder.vlm_backbone",
                module=vlm_backbone,
                target=direct_target,
                camera_keys=camera_keys,
                capture_mode=_capture_mode_for_module(
                    module=vlm_backbone,
                    camera_keys=camera_keys,
                ),
            )
        )

    vision_encoders = getattr(vlm_backbone, "vision_encoders", None)
    if not isinstance(vision_encoders, torch.nn.ModuleList):
        return modules

    for index, vision_encoder in enumerate(vision_encoders):
        target = select_explainability_target(
            targets=_get_explainability_targets(module=vision_encoder),
            module_name=f"decoder.vlm_backbone.vision_encoders.{index}",
        )
        if target is None or not camera_keys:
            continue
        modules.append(
            VisionExplainableModule(
                name=f"decoder.vlm_backbone.vision_encoders.{index}",
                module=vision_encoder,
                target=target,
                camera_keys=camera_keys,
                capture_mode=_capture_mode_for_decoder_vision_tower(
                    camera_keys=camera_keys
                ),
            )
        )
    return modules


def resolve_camera_explanation_targets(
    policy: Policy,
    target_camera: str | None = None,
    target_vision_module_names: list[str] | None = None,
) -> list[CameraExplanationTarget]:
    """Resolve runner filters into concrete camera-level targets.

    Args:
        policy: Policy whose visual modules should be explained.
        target_camera: Optional camera key selected by the runner.
        target_vision_module_names: Optional visual module allowlist. Names are
            the values returned by ``get_vision_explainable_modules()``.

    Returns:
        Concrete camera-target bindings for attribution methods.

    Raises:
        ValueError: If a configured camera or module allowlist matches nothing.
        RuntimeError: If the policy exposes no visual explainability targets.
    """
    modules = get_vision_explainable_modules(policy=policy)
    if target_vision_module_names is not None:
        target_name_set = set(target_vision_module_names)
        modules = [module for module in modules if module.name in target_name_set]
        if not modules:
            available_names = [
                module.name for module in get_vision_explainable_modules(policy=policy)
            ]
            raise ValueError(
                f"target_vision_module_names={target_vision_module_names} did not "
                f"match visual modules: {available_names}"
            )

    targets = []
    for module in modules:
        camera_keys = (
            (target_camera,)
            if target_camera is not None and target_camera in module.camera_keys
            else module.camera_keys
        )
        if target_camera is not None and target_camera not in module.camera_keys:
            continue
        for camera_key in camera_keys:
            targets.append(
                CameraExplanationTarget(
                    camera_key=camera_key,
                    vision_module_name=module.name,
                    target=module.target,
                    capture_mode=module.capture_mode,
                    invocation_index=_invocation_index_for_camera(
                        camera_key=camera_key,
                        module=module,
                    ),
                    stacked_camera_index=_stacked_camera_index_for_camera(
                        camera_key=camera_key,
                        module=module,
                    ),
                    stacked_camera_count=_stacked_camera_count_for_module(
                        module=module
                    ),
                )
            )
    # TODO: Multiple visual modules can explain the same camera. Attribution
    #  methods aggregate those maps today; expose per-module output names later.
    if targets:
        return targets

    available_cameras = sorted(
        {camera_key for module in modules for camera_key in module.camera_keys}
    )
    raise ValueError(
        f"target_camera={target_camera!r} did not match visual module cameras: "
        f"{available_cameras}"
    )


def select_explainability_target(
    targets: list[VisionExplanationTarget],
    module_name: str,
) -> VisionExplanationTarget | None:
    """Select the single image-map target exposed by a visual module.

    Args:
        targets: Target metadata exposed by a visual module.
        module_name: Name used in error messages when target metadata is
            malformed.

    Returns:
        The compatible target, or ``None`` when the module exposes no targets.

    Raises:
        RuntimeError: If targets are exposed but none are compatible with visual
            heatmap computation.
        RuntimeError: If more than one compatible target is exposed. The runner
            currently has no config field to disambiguate multiple target layers
            inside the same visual module.
    """
    if not targets:
        return None
    compatible_targets = [
        target
        for target in targets
        if target.target_kind in COMPATIBLE_EXPLANATION_TARGET_KINDS
    ]
    if len(compatible_targets) == 1:
        return compatible_targets[0]
    if len(compatible_targets) > 1:
        target_kinds = [target.target_kind for target in compatible_targets]
        raise RuntimeError(
            f"Visual module '{module_name}' exposes multiple compatible "
            f"explainability targets {target_kinds}. Configure the module to "
            "expose exactly one target until per-target selection is supported."
        )
    raise RuntimeError(
        f"Visual module '{module_name}' does not expose a compatible "
        "explainability target."
    )


def _decoder_vlm_backbone(policy: Policy) -> torch.nn.Module | None:
    decoder = policy.decoder
    module_children = getattr(decoder, "_modules", {})
    if "vlm_backbone" in module_children:
        return module_children["vlm_backbone"]
    return None


def _camera_keys_for_module(
    policy: Policy,
    module: torch.nn.Module,
) -> tuple[str, ...]:
    camera_keys = getattr(module, "camera_keys", None)
    if isinstance(camera_keys, list | tuple):
        return tuple(camera_keys)

    policy_camera_keys = _policy_camera_keys(policy=policy)
    input_specification = getattr(module, "input_specification", None)
    input_keys = getattr(input_specification, "keys", None)
    if not isinstance(input_keys, list | tuple):
        return ()
    return tuple(key for key in input_keys if key in policy_camera_keys)


def _policy_camera_keys(policy: Policy) -> set[str]:
    cameras = policy.observation_space.cameras
    if isinstance(cameras, dict):
        return set(cameras)
    return set(cameras)


def _get_explainability_targets(
    module: torch.nn.Module,
) -> list[VisionExplanationTarget]:
    target_getter = getattr(module, "get_explainability_targets", None)
    if target_getter is None:
        return []
    targets = target_getter()
    if not isinstance(targets, list):
        raise RuntimeError(
            f"{type(module).__name__}.get_explainability_targets() must return a list."
        )
    return targets


def _capture_mode_for_module(
    module: torch.nn.Module,
    camera_keys: tuple[str, ...],
) -> str:
    if len(camera_keys) <= 1:
        return VisionCaptureMode.SINGLE_CALL.value
    if getattr(module, "is_stacked_camera_batch", False):
        return VisionCaptureMode.STACKED_CAMERA_BATCH.value
    if getattr(module, "is_multi_camera", False):
        return VisionCaptureMode.PER_CAMERA_CALL.value
    return VisionCaptureMode.SINGLE_CALL.value


def _capture_mode_for_decoder_vision_tower(camera_keys: tuple[str, ...]) -> str:
    if len(camera_keys) > 1:
        return VisionCaptureMode.PER_CAMERA_CALL.value
    return VisionCaptureMode.SINGLE_CALL.value


def _invocation_index_for_camera(
    camera_key: str,
    module: VisionExplainableModule,
) -> int | None:
    if module.capture_mode != VisionCaptureMode.PER_CAMERA_CALL.value:
        return None
    return module.camera_keys.index(camera_key)


def _stacked_camera_index_for_camera(
    camera_key: str,
    module: VisionExplainableModule,
) -> int | None:
    if module.capture_mode != VisionCaptureMode.STACKED_CAMERA_BATCH.value:
        return None
    return module.camera_keys.index(camera_key)


def _stacked_camera_count_for_module(
    module: VisionExplainableModule,
) -> int | None:
    if module.capture_mode != VisionCaptureMode.STACKED_CAMERA_BATCH.value:
        return None
    return len(module.camera_keys)
