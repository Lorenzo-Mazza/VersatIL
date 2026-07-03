"""Ablation-CAM visual attribution."""

import torch
import torch.nn.functional as F

from versatil.explainability.attribution.capture import (
    ActivationCapture,
    replace_camera_tensor,
    select_camera_tensor,
    select_tensor_output,
    should_capture_invocation,
)
from versatil.explainability.attribution.maps import (
    nchw_to_target_tensor,
    resize_feature_heatmap_to_camera,
    target_tensor_to_nchw,
)
from versatil.explainability.attribution.objectives import (
    compute_policy_explanation_objective,
    repeat_action_batch,
    resolve_actions_for_explanation,
)
from versatil.explainability.typedefs import (
    ActionBatch,
    CameraExplanationTarget,
    ObservationBatch,
    PolicyPredictionSelector,
    TensorModuleOutput,
)
from versatil.explainability.vision_modules import resolve_camera_explanation_targets
from versatil.models.policy import Policy


def repeat_observation_batch(
    observation: ObservationBatch,
    repeat_count: int,
) -> ObservationBatch:
    """Repeat observation batch rows for channel-wise ablations.

    Args:
        observation: Observation values keyed by observation name.
        repeat_count: Number of repeated copies to concatenate along batch.

    Returns:
        Observation dictionary with tensor batch dimensions multiplied by
        ``repeat_count``. Batched list values, such as language prompts, are
        repeated along the same batch axis so policy preprocessing sees a
        consistent batch size.
    """
    repeated_observation: ObservationBatch = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            repeated_observation[key] = torch.cat(
                [value.clone() for _ in range(repeat_count)],
                dim=0,
            )
        elif isinstance(value, list):
            repeated_observation[key] = list(value) * repeat_count
        else:
            repeated_observation[key] = value
    return repeated_observation


def replace_target_output(
    module_output: TensorModuleOutput,
    replacement: torch.Tensor,
    output_index: int | None,
) -> TensorModuleOutput:
    """Replace the selected target-layer output with an ablated tensor.

    Args:
        module_output: Original module output from the forward hook.
        replacement: Ablated tensor in the same layout as the selected output.
        output_index: Optional tuple output index. ``None`` replaces tensor
            outputs directly or the first non-None tensor in tuple outputs.

    Returns:
        Module output with the selected tensor replaced.

    Raises:
        RuntimeError: If no replaceable tensor is present.
    """
    if isinstance(module_output, torch.Tensor):
        return replacement

    output_list = list(module_output)
    if output_index is not None:
        if output_list[output_index] is None:
            raise RuntimeError(
                f"Target layer output_index={output_index} has no tensor."
            )
        output_list[output_index] = replacement
        return tuple(output_list)

    for index, value in enumerate(output_list):
        if value is not None:
            output_list[index] = replacement
            return tuple(output_list)
    raise RuntimeError("Target layer did not produce a tensor output.")


def ablate_target_channels(
    module_output: TensorModuleOutput,
    camera_target: CameraExplanationTarget,
    channel_start: int,
    channel_count: int,
    sample_count: int,
) -> TensorModuleOutput:
    """Zero one target activation channel per repeated batch group.

    Args:
        module_output: Target module output from the forward hook.
        camera_target: Concrete camera target whose activation should be
            perturbed.
        channel_start: First channel index ablated in this chunk.
        channel_count: Number of channels ablated in this chunk.
        sample_count: Number of original batch rows before repetition.

    Returns:
        Module output with one channel zeroed in each repeated batch group.

    Raises:
        ValueError: If the target activation cannot be converted to NCHW.
        RuntimeError: If the selected module output cannot be replaced.
    """
    activation = select_tensor_output(
        value=module_output,
        output_index=camera_target.target.output_index,
    )
    camera_activation = select_camera_tensor(
        tensor=activation,
        target=camera_target,
    )
    nchw_activation = target_tensor_to_nchw(
        tensor=camera_activation,
        target=camera_target.target,
        tensor_name="activations",
    ).clone()
    for channel_offset in range(channel_count):
        start = channel_offset * sample_count
        end = (channel_offset + 1) * sample_count
        nchw_activation[start:end, channel_start + channel_offset] = 0
    camera_replacement = nchw_to_target_tensor(
        tensor=nchw_activation,
        target=camera_target.target,
        original_tensor=camera_activation,
    )
    replacement = replace_camera_tensor(
        tensor=activation,
        replacement=camera_replacement,
        target=camera_target,
    )
    return replace_target_output(
        module_output=module_output,
        replacement=replacement,
        output_index=camera_target.target.output_index,
    )


def compute_ablation_maps_for_policy(
    policy: Policy,
    observation: ObservationBatch,
    actions: ActionBatch | None = None,
    output_selector: PolicyPredictionSelector | None = None,
    target_camera: str | None = None,
    target_vision_module_names: list[str] | None = None,
    channel_batch_size: int = 32,
    preprocess_observation: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute Ablation-CAM maps for policy visual modules.

    Note:
        Ablation-CAM is perturbation-based: each channel in the selected visual
        target activation is zeroed and the channel weight is the resulting
        drop in the selected prediction score. It is kept separate from
        gradient methods because it does not use backpropagation.

    Args:
        policy: Policy instance to explain.
        observation: Raw observation tensors keyed by camera names. Camera
            tensors must have shape ``(B, T, C, H, W)`` or ``(B, C, H, W)``.
        actions: Optional action tensors. Discrete predictors use tokenized
            actions when available; otherwise pseudo-target tokens are generated
            before attribution hooks are registered.
        output_selector: Optional function selecting the prediction tensor to
            score for continuous predictors.
        target_camera: Optional camera key to explain. If omitted, every camera
            exposed by a visual module is explained.
        target_vision_module_names: Optional visual module allowlist.
        channel_batch_size: Number of feature channels ablated per forward pass.
        preprocess_observation: Whether to normalize/tokenize ``observation``
            before attribution. Online inference windows should use ``True``;
            dataset batches that already passed through ``EpisodicDataset``
            should use ``False``.

    Returns:
        Heatmaps keyed by camera with shape ``(B, T, H, W)``.

    Raises:
        ValueError: If ``channel_batch_size`` is not positive.
        ValueError: If ``target_camera`` is not exposed by a visual module.
        ValueError: If a camera tensor has an unsupported rank.
        RuntimeError: If the selected visual module does not expose a compatible
            explainability target.
        RuntimeError: If target-layer activation capture fails.
    """
    if channel_batch_size <= 0:
        raise ValueError(
            f"channel_batch_size must be positive. Got: {channel_batch_size}"
        )

    resolved_actions = resolve_actions_for_explanation(
        policy=policy,
        observation=observation,
        actions=actions,
        preprocess_observation=preprocess_observation,
    )
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

        capture = ActivationCapture(target=camera_target)
        capture_handle = camera_target.target.layer.register_forward_hook(
            capture.forward_hook
        )
        try:
            with torch.no_grad():
                baseline_score = compute_policy_explanation_objective(
                    policy=policy,
                    observation=observation,
                    actions=resolved_actions,
                    preprocess_observation=preprocess_observation,
                    output_selector=output_selector,
                ).mean()
        finally:
            capture_handle.remove()

        activation = capture.require_activation()
        nchw_activation = target_tensor_to_nchw(
            tensor=activation,
            target=camera_target.target,
            tensor_name="activations",
        )
        sample_count, channel_count, feature_height, feature_width = (
            nchw_activation.shape
        )
        drops = torch.zeros(
            channel_count,
            dtype=nchw_activation.dtype,
            device=nchw_activation.device,
        )

        for channel_start in range(0, channel_count, channel_batch_size):
            current_channel_count = min(
                channel_batch_size,
                channel_count - channel_start,
            )
            repeated_observation = repeat_observation_batch(
                observation=observation,
                repeat_count=current_channel_count,
            )
            repeated_actions = repeat_action_batch(
                actions=resolved_actions,
                repeat_count=current_channel_count,
            )
            hook_call_index = 0

            def ablation_hook(
                module: torch.nn.Module,
                module_input: tuple[torch.Tensor, ...],
                module_output: TensorModuleOutput,
            ) -> TensorModuleOutput:
                """Apply current channel ablations to the selected camera call.

                Args:
                    module: Module registered as the target layer.
                    module_input: Positional module inputs from PyTorch's hook
                        API.
                    module_output: Target-layer output to perturb.

                Returns:
                    Module output with one channel zeroed per repeated batch
                    group.
                """
                nonlocal hook_call_index
                current_call_index = hook_call_index
                hook_call_index += 1
                if not should_capture_invocation(
                    call_index=current_call_index,
                    target=camera_target,
                ):
                    return module_output
                return ablate_target_channels(
                    module_output=module_output,
                    camera_target=camera_target,
                    channel_start=channel_start,
                    channel_count=current_channel_count,
                    sample_count=sample_count,
                )

            ablation_handle = camera_target.target.layer.register_forward_hook(
                ablation_hook
            )
            try:
                with torch.no_grad():
                    ablated_scores = compute_policy_explanation_objective(
                        policy=policy,
                        observation=repeated_observation,
                        actions=repeated_actions,
                        preprocess_observation=preprocess_observation,
                        output_selector=output_selector,
                    ).reshape(current_channel_count, -1)
            finally:
                ablation_handle.remove()

            drops[channel_start : channel_start + current_channel_count] = (
                baseline_score - ablated_scores.mean(dim=1)
            )

        weights = F.relu(drops)
        feature_heatmap = F.relu(
            (weights[None, :, None, None] * nchw_activation).sum(dim=1)
        )
        feature_heatmap = feature_heatmap.reshape(
            sample_count,
            feature_height,
            feature_width,
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
