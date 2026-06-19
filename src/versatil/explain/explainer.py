"""Model explainability utilities for visual attention and gradient-based explanations.

This module provides functions for explaining model predictions through various
interpretability techniques including GradCAM, saliency maps, and integrated gradients.
"""

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from torch import nn

from versatil.data.constants import ProprioKey
from versatil.data.processing.transform import normalize_observation
from versatil.explain.constants import ExplanationType
from versatil.models.policy import Policy


class PolicyExplainerWrapper(nn.Module):
    """Wrapper to adapt Policy for explainability methods.

    This wrapper makes a Policy compatible with gradient-based explanation
    methods by providing a __call__ method that matches the expected signature.
    """

    def __init__(self, policy: Policy):
        """Initialize wrapper.

        Args:
            policy: Policy instance to wrap
        """
        super().__init__()
        self.policy = policy

    def forward(
        self, observation: dict[str, torch.Tensor], **kwargs: Any
    ) -> tuple[dict[str, torch.Tensor], ...]:
        """Forward pass compatible with explainer signature.

        Args:
            observation: Dictionary of observation tensors
            **kwargs: Additional arguments (ignored)

        Returns:
            Tuple with predictions as first element (for compatibility with explainer's unpacking)
        """
        normalized_obs = normalize_observation(
            observation=observation,
            normalizer=self.policy.normalizer,
            observation_space=self.policy.observation_space,
        )
        features = self.policy.encoding_pipeline(normalized_obs)
        predictions = self.policy.decoder(features, actions=None)
        return (predictions,)


def create_target_layers_getter_from_policy(
    policy: Policy,
) -> Callable[[str], list[nn.Module]]:
    """Create a target_layers_getter function from a Policy instance.

    This helper creates a closure that uses the Policy's standardized methods
    to get vision encoder target layers for GradCAM.

    Args:
        policy: Policy instance with encoding pipeline

    Returns:
        Function that takes a camera name and returns list of target layers

    """
    camera_to_encoder = policy.get_camera_to_encoder_mapping()

    def get_target_layers(camera_name: str) -> list[nn.Module]:
        """Get target layers for a specific camera.

        Args:
            camera_name: Name of the camera/observation key

        Returns:
            List of target layers for GradCAM

        Raises:
            ValueError: If camera has no corresponding encoder
        """
        if camera_name not in camera_to_encoder:
            available = list(camera_to_encoder.keys())
            raise ValueError(
                f"Camera '{camera_name}' has no corresponding vision encoder. "
                f"Available cameras: {available}"
            )

        encoder_name = camera_to_encoder[camera_name]
        target_layers: list[nn.Module] = policy.get_gradcam_target_layers(encoder_name)
        return target_layers

    return get_target_layers


def compute_gradcam_custom(
    model: nn.Module,
    explanation_type: str,
    observation: dict[str, torch.Tensor],
    camera_names: list[str],
    input_channels: dict[str, int],
    target_layers_getter: Callable[[str], list[nn.Module]],
    output_selector: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    target_camera: str | None = None,
    eigen_smooth: bool = False,
    **forward_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Compute GradCAM-based explanations for model predictions.

    Args:
        model: PyTorch model to explain
        explanation_type: Type of explanation (gradcam, gradcam++, ablation_cam)
        observation: Dictionary of observation tensors
        camera_names: List of camera keys to explain
        input_channels: Dictionary mapping camera names to input channel counts
        target_layers_getter: Function that returns target layers for each camera
        output_selector: Function that selects the target output from predictions
        target_camera: Optional specific camera to explain (None for all)
        eigen_smooth: Whether to apply eigen-smoothing
        **forward_kwargs: Additional arguments passed to model forward

    Returns:
        Dictionary mapping camera names to heatmap tensors (T, H, W)
    """
    model.eval()
    cameras = [target_camera] if target_camera else camera_names
    has_seq = any(
        len(observation.get(cam, torch.empty(0)).shape) == 5 for cam in cameras
    )
    example_cam = cameras[0]
    B = observation[example_cam].shape[0]
    T = observation[example_cam].shape[1] if has_seq else 1

    channel_dim = 1
    heatmaps = {}
    eps = 1e-8
    for camera in cameras:
        if camera not in observation:
            continue
        target_layer = target_layers_getter(camera)[0]
        activations = []

        def forward_hook(module, inp, out):
            # Handle modules that return tuples (e.g., DFormerStage)
            """Record module activations, unwrapping tuple outputs."""
            if isinstance(out, tuple):
                activations.append(out[0].detach())
            else:
                activations.append(out.detach())

        gradients = []

        def backward_hook(module, grad_inp, grad_out):
            # Handle None gradients or tuple gradient outputs
            # For modules that return tuples (e.g., DFormerStage), grad_out may have None elements
            """Record output gradients, skipping None entries in tuples."""
            if grad_out[0] is not None:
                gradients.append(grad_out[0].detach())
            elif isinstance(grad_out, tuple) and len(grad_out) > 1:
                # Find first non-None gradient in tuple
                for g in grad_out:
                    if g is not None:
                        gradients.append(g.detach())
                        break
                else:
                    # All gradients are None - this shouldn't happen but handle gracefully
                    raise RuntimeError(
                        "All gradients in grad_out are None. Cannot compute GradCAM."
                    )

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        predicted_actions, *_ = model(observation=observation, **forward_kwargs)
        model.zero_grad()
        target_output = output_selector(predicted_actions).mean()
        target_output.backward()
        if explanation_type in [
            ExplanationType.GRADCAM.value,
            ExplanationType.GRADCAM_PLUS_PLUS.value,
        ]:
            activation = activations[0]
            gradient = gradients[0]

            C_f = activation.shape[channel_dim]
            H_f, W_f = activation.shape[-2], activation.shape[-1]
            if has_seq:
                activation = activation.view(B, T, C_f, H_f, W_f)
                gradient = gradient.view(B, T, C_f, H_f, W_f)
            else:
                activation = activation.view(B, C_f, H_f, W_f).unsqueeze(1)
                gradient = gradient.view(B, C_f, H_f, W_f).unsqueeze(1)

            if explanation_type == ExplanationType.GRADCAM.value:
                weights = gradient.mean(dim=(3, 4))
            elif explanation_type == ExplanationType.GRADCAM_PLUS_PLUS.value:
                grad2 = gradient**2
                grad3 = gradient**3
                sum_act = activation.sum(dim=(3, 4)).unsqueeze(3).unsqueeze(4)
                alpha = grad2 / (2 * grad2 + sum_act * grad3 + eps)
                aij = alpha * F.relu(gradient)
                weights = aij.sum(dim=(3, 4))
            cam = F.relu((weights.unsqueeze(3).unsqueeze(4) * activation).sum(dim=2))

        elif explanation_type == ExplanationType.ABLATION_CAM.value:
            activation = activations[0]
            C_f = activation.shape[channel_dim]
            H_f, W_f = activation.shape[-2], activation.shape[-1]
            if has_seq:
                activation = activation.view(B, T, C_f, H_f, W_f)
                full_batch = B * T
            else:
                activation = activation.view(B, C_f, H_f, W_f).unsqueeze(1)
                full_batch = B

            batch_size_ablate = 32
            drops = torch.zeros(C_f, device=activation.device)

            for i in range(0, C_f, batch_size_ablate):
                num_channels = min(batch_size_ablate, C_f - i)
                obs_repeated_chunk = {
                    k: v.clone().repeat(num_channels, *([1] * (len(v.shape) - 1)))
                    for k, v in observation.items()
                }

                def ablation_hook_chunk(module, inp, out):
                    """Zero one activation channel per batch replica."""
                    for j in range(num_channels):
                        start = j * full_batch
                        end = (j + 1) * full_batch
                        out[start:end, i + j] = 0
                    return out

                handle_ablate = target_layer.register_forward_hook(ablation_hook_chunk)
                predicted_actions_chunk, *_ = model(
                    observation=obs_repeated_chunk, **forward_kwargs
                )
                handle_ablate.remove()

                target_outputs_chunk = output_selector(predicted_actions_chunk)
                drops_chunk = (
                    target_output.detach().repeat(num_channels)
                    - target_outputs_chunk.detach()
                )
                drops[i : i + num_channels] = drops_chunk

            weights = F.relu(drops).view(B, C_f)
            cam = F.relu(
                (weights.unsqueeze(1).unsqueeze(3).unsqueeze(4) * activation).sum(dim=2)
            )

        handle_fwd.remove()
        handle_bwd.remove()

        input_size = (
            observation[camera].shape[-2:]
            if not has_seq
            else observation[camera].shape[3:5]
        )
        cam = cam.view(B * T, 1, H_f, W_f)
        cam = F.interpolate(
            cam, size=input_size, mode="bicubic", align_corners=False
        ).view(B, T, *input_size)

        cam_min = cam.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        cam_max = cam.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + eps)

        heatmaps[camera] = cam.squeeze(0)
    return heatmaps


def compute_saliency_maps(
    model: nn.Module,
    observation: dict[str, torch.Tensor],
    camera_names: list[str],
    output_selector: Callable[[torch.Tensor], torch.Tensor],
    target_camera: str | None = None,
    smooth: bool = False,
    **forward_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Compute vanilla saliency maps (input gradients) for input images.

    Args:
        model: PyTorch model to explain
        observation: Dictionary of observation tensors
        camera_names: List of camera keys
        output_selector: Function that selects the target output from predictions
        target_camera: Optional specific camera to explain (None for all)
        smooth: Whether to apply Gaussian smoothing
        **forward_kwargs: Additional arguments passed to model forward

    Returns:
        Dictionary mapping camera names to saliency maps (H, W)
    """
    model.eval()
    cameras = [target_camera] if target_camera else camera_names
    obs_clone = {k: v.clone() for k, v in observation.items()}
    saliency_maps = {}

    for camera in cameras:
        if camera in obs_clone:
            obs_clone[camera].requires_grad_(True)

        predicted_actions, *_ = model.forward(observation=obs_clone, **forward_kwargs)
        target_output = output_selector(predicted_actions)

        model.zero_grad()
        target_output.backward()

        grad_tensor = obs_clone[camera].grad
        if grad_tensor is None:
            raise RuntimeError("Gradient should be computed after backward()")
        grad = grad_tensor.abs()
        saliency = grad.max(dim=1)[0].squeeze(0)

        if smooth:
            saliency = tvf.gaussian_blur(
                saliency.unsqueeze(0).unsqueeze(0), kernel_size=(5, 5), sigma=(1.5, 1.5)
            ).squeeze()

        saliency = (saliency - saliency.min()) / (
            saliency.max() - saliency.min() + 1e-8
        )
        saliency_maps[camera] = (
            saliency if saliency.dim() == 3 else saliency.unsqueeze(0)
        )

        obs_clone[camera].requires_grad_(False)
        obs_clone[camera].grad = None

    return saliency_maps


def compute_integrated_grad_maps(
    model: nn.Module,
    observation: dict[str, torch.Tensor],
    camera_names: list[str],
    output_selector: Callable[[torch.Tensor], torch.Tensor],
    target_camera: str | None = None,
    num_steps: int = 50,
    baseline: torch.Tensor | None = None,
    smooth: bool = False,
    **forward_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Compute Integrated Gradients attributions for input images.

    Args:
        model: PyTorch model to explain
        observation: Dictionary of observation tensors
        camera_names: List of camera keys
        output_selector: Function that selects the target output from predictions
        target_camera: Optional specific camera to explain (None for all)
        num_steps: Number of interpolation steps
        baseline: Optional baseline input (default: zeros)
        smooth: Whether to apply Gaussian smoothing
        **forward_kwargs: Additional arguments passed to model forward

    Returns:
        Dictionary mapping camera names to attribution maps (H, W)
    """
    model.eval()
    cameras = [target_camera] if target_camera else camera_names
    heatmaps = {}

    for camera in cameras:
        if camera not in observation:
            continue

        obs_clone = {k: v.clone() for k, v in observation.items()}
        input_img = obs_clone[camera]
        if baseline is None:
            baseline = torch.zeros_like(input_img)

        attributions = torch.zeros_like(input_img)
        diff = input_img - baseline

        for i in range(1, num_steps + 1):
            alpha = i / float(num_steps)
            interp = baseline + alpha * diff
            interp.requires_grad_(True)

            obs_interp = obs_clone.copy()
            obs_interp[camera] = interp
            predicted_actions, *_ = model.forward(obs_interp, **forward_kwargs)

            target_output = output_selector(predicted_actions)

            model.zero_grad()
            target_output.backward()

            if interp.grad is None:
                raise RuntimeError("Gradient should be computed after backward()")
            attributions += interp.grad / num_steps

        attributions *= diff
        heatmap = attributions.abs().sum(dim=1).squeeze(0)

        if smooth:
            heatmap = tvf.gaussian_blur(
                heatmap.unsqueeze(0).unsqueeze(0), kernel_size=(5, 5), sigma=(1.5, 1.5)
            ).squeeze()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmaps[camera] = heatmap if heatmap.dim() == 3 else heatmap.unsqueeze(0)

    return heatmaps


def compute_gradcam_for_policy(
    policy: Policy,
    observation: dict[str, torch.Tensor],
    explanation_type: str = ExplanationType.GRADCAM.value,
    output_selector: Callable[[dict[str, torch.Tensor]], torch.Tensor] | None = None,
    target_camera: str | None = None,
    eigen_smooth: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute GradCAM explanations for a Policy instance.

    This is a convenience wrapper that automatically extracts vision encoders
    and target layers from a Policy's encoding pipeline.

    Args:
        policy: Policy instance to explain
        observation: Dictionary of observation tensors
        explanation_type: Type of explanation (gradcam, gradcam++, ablation_cam)
        output_selector: Optional function to select target output. If None, uses mean of all actions
        target_camera: Optional specific camera to explain (None for all)
        eigen_smooth: Whether to apply eigen-smoothing

    Returns:
        Dictionary mapping camera names to heatmap tensors (T, H, W)

    """
    # Get camera to encoder mapping
    camera_to_encoder = policy.get_camera_to_encoder_mapping()
    camera_names = list(camera_to_encoder.keys())

    # Create target layers getter
    target_layers_getter = create_target_layers_getter_from_policy(policy)

    # Get input channels from observation space
    input_channels = {}
    for cam in camera_names:
        if cam in observation:
            # Infer channels from observation shape
            obs_tensor = observation[cam]
            if len(obs_tensor.shape) == 4:  # (B, C, H, W)
                input_channels[cam] = obs_tensor.shape[1]
            elif len(obs_tensor.shape) == 5:  # (B, T, C, H, W)
                input_channels[cam] = obs_tensor.shape[2]
            else:
                input_channels[cam] = 3  # Default to RGB

    # Default output selector: mean of all position actions
    if output_selector is None:

        def default_output_selector(
            predictions: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            # Use position actions if available, otherwise use first available action type
            """Select position actions, falling back to the first action tensor."""
            if ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value in predictions:
                return predictions[ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value]
            else:
                # Use first available action
                first_key = list(predictions.keys())[0]
                return predictions[first_key]

        output_selector = default_output_selector

    # Wrap policy for explainer compatibility
    wrapped_model = PolicyExplainerWrapper(policy)

    return compute_gradcam_custom(
        model=wrapped_model,
        explanation_type=explanation_type,
        observation=observation,
        camera_names=camera_names,
        input_channels=input_channels,
        target_layers_getter=target_layers_getter,
        output_selector=output_selector,
        target_camera=target_camera,
        eigen_smooth=eigen_smooth,
    )


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.5,
) -> np.ndarray:
    """Overlay heatmap on image.

    Args:
        img: Base image in RGB or BGR format
        mask: CAM mask (heatmap)
        use_rgb: Whether to use RGB or BGR heatmap
        colormap: OpenCV colormap to use
        image_weight: Weight for the original image in the overlay

    Returns:
        Image with heatmap overlay as uint8 array
    """
    heatmap = cv2.applyColorMap((255 * mask).astype(np.uint8), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    if np.max(img) > 1:
        raise ValueError("The input image should be np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise ValueError(
            f"image_weight should be in the range [0, 1]. Got: {image_weight}"
        )

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    result: np.ndarray = (255 * cam).astype(np.uint8)
    return result
