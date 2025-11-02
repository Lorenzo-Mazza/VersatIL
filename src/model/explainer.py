from typing import Dict, Optional, Callable, List

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf

from legacy_constants import ExplanationType


def compute_gradcam_custom(
    model: nn.Module,
    explanation_type: str,
    observation: Dict[str, torch.Tensor],
    camera_names: List[str],
    input_channels: Dict[str, int],
    target_layers_getter: Callable[[str], List[nn.Module]],
    output_selector: Callable[[torch.Tensor], torch.Tensor],
    target_camera: Optional[str] = None,
    eigen_smooth: bool = False,
    **forward_kwargs
) -> Dict[str, torch.Tensor]:
    model.eval()
    cameras = [target_camera] if target_camera else camera_names
    has_seq = any(len(observation.get(cam, torch.empty(0)).shape) == 5 for cam in cameras)
    example_cam = cameras[0]
    B = observation[example_cam].shape[0]
    if has_seq:
        T = observation[example_cam].shape[1]
    else:
        T = 1

    channel_dim = 1
    heatmaps = {}
    eps = 1e-8
    for camera in cameras:
        if camera not in observation:
            continue
        target_layer = target_layers_getter(camera)[0]  # Assume single target layer
        activations = []


        def forward_hook(module, inp, out):
            activations.append(out.detach())
        gradients = []
        def backward_hook(module, grad_inp, grad_out):
            gradients.append(grad_out[0].detach())
        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        predicted_actions, *_ = model(observation=observation, **forward_kwargs)
        model.zero_grad()
        target_output = output_selector(predicted_actions).mean()
        target_output.backward()
        if explanation_type in [ExplanationType.GRADCAM.value, ExplanationType.GRADCAM_PLUS_PLUS.value]:
            activation = activations[0]  # (B*T, C_f, H_f, W_f)
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
                weights = gradient.mean(dim=(3, 4))  # (B, T, C_f)
            elif explanation_type == ExplanationType.GRADCAM_PLUS_PLUS.value:
                grad2 = gradient ** 2
                grad3 = gradient ** 3
                sum_act = activation.sum(dim=(3, 4)).unsqueeze(3).unsqueeze(4)
                alpha = grad2 / (2 * grad2 + sum_act * grad3 + eps)
                aij = alpha * F.relu(gradient)
                weights = aij.sum(dim=(3, 4))  # (B, T, C_f)
            cam = F.relu((weights.unsqueeze(3).unsqueeze(4) * activation).sum(dim=2))  # (B, T, H_f, W_f)

        elif explanation_type == ExplanationType.ABLATION_CAM.value:
            activation = activations[0]  # (B*T or B, C_f, H_f, W_f)
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
                obs_repeated_chunk = {k: v.clone().repeat(num_channels, *([1] * (len(v.shape) - 1))) for k, v in observation.items() }

                def ablation_hook_chunk(module, inp, out):
                    for j in range(num_channels):
                        start = j * full_batch
                        end = (j + 1) * full_batch
                        out[start:end, i + j] = 0
                    return out

                handle_ablate = target_layer.register_forward_hook(ablation_hook_chunk)
                predicted_actions_chunk, *_ = model(observation=obs_repeated_chunk, **forward_kwargs)
                handle_ablate.remove()

                target_outputs_chunk = output_selector(predicted_actions_chunk)  
                drops_chunk = target_output.detach().repeat(num_channels) - target_outputs_chunk.detach()
                drops[i:i + num_channels] = drops_chunk

            weights = F.relu(drops).view(B, C_f)
            cam = F.relu((weights.unsqueeze(1).unsqueeze(3).unsqueeze(4) * activation).sum(dim=2))  # (B, T, H_f, W_f)

        handle_fwd.remove()

        input_size = observation[camera].shape[-2:] if not has_seq else observation[camera].shape[3:5]
        cam = cam.view(B * T, 1, H_f, W_f)
        cam = F.interpolate(cam, size=input_size, mode='bicubic', align_corners=False).view(B, T, *input_size)

        cam_min = cam.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        cam_max = cam.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + eps)

        heatmaps[camera] = cam.squeeze(0)
    return heatmaps


def compute_saliency_maps(
    model: nn.Module,
    observation: Dict[str, torch.Tensor],
    camera_names: List[str],
    output_selector: Callable[[torch.Tensor], torch.Tensor],
    target_camera: Optional[str] = None,
    smooth: bool = False,
    **forward_kwargs
) -> Dict[str, torch.Tensor]:
    """
    Compute vanilla saliency maps (input gradients) for input images using the full observation.

    Args:
        model: The PyTorch model to explain.
        observation: Input observations.
        camera_names: List of camera keys.
        output_selector: Callable that takes predicted_actions and returns the scalar target.
        target_camera: Camera to compute for (or None for all).
        smooth: Apply Gaussian smoothing.
        **forward_kwargs: Additional arguments for model.forward.

    Returns:
        Dict[str, torch.Tensor]: Saliency maps per camera, shape (H, W) normalized to [0,1].
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

        grad = obs_clone[camera].grad.abs()
        saliency = grad.max(dim=1)[0].squeeze(0)

        if smooth:
            saliency = tvf.gaussian_blur(saliency.unsqueeze(0).unsqueeze(0), kernel_size=(5, 5), sigma=(1.5, 1.5)).squeeze()

        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        saliency_maps[camera] = saliency if saliency.dim() == 3 else saliency.unsqueeze(0)

        obs_clone[camera].requires_grad_(False)
        obs_clone[camera].grad = None

    return saliency_maps

def compute_integrated_grad_maps(
    model: nn.Module,
    observation: Dict[str, torch.Tensor],
    camera_names: List[str],
    output_selector: Callable[[torch.Tensor], torch.Tensor],
    target_camera: Optional[str] = None,
    num_steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
    smooth: bool = False,
    **forward_kwargs
) -> Dict[str, torch.Tensor]:
    """
    Compute Integrated Gradients attributions for input images.

    Args:
        model: The PyTorch model to explain.
        observation: Input observations.
        camera_names: List of camera keys.
        output_selector: Callable that takes predicted_actions and returns the scalar target.
        target_camera: Camera to compute for (or None for all).
        num_steps: Number of interpolation steps.
        baseline: Baseline input (default: zeros).
        smooth: Apply Gaussian smoothing.
        **forward_kwargs: Additional arguments for model.forward.

    Returns:
        Dict[str, torch.Tensor]: Attribution maps per camera, shape (H, W) normalized to [0,1].
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

            attributions += interp.grad / num_steps

        attributions *= diff
        heatmap = attributions.abs().sum(dim=1).squeeze(0)

        if smooth:
            heatmap = tvf.gaussian_blur(heatmap.unsqueeze(0).unsqueeze(0), kernel_size=(5, 5), sigma=(1.5, 1.5)).squeeze()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmaps[camera] = heatmap if heatmap.dim()==3 else heatmap.unsqueeze(0)

    return heatmaps


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """Taken from grad-cam library.

    This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
