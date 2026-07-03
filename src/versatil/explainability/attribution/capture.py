"""Activation and gradient capture helpers for attribution methods."""

import torch

from versatil.explainability.constants import VisionCaptureMode
from versatil.explainability.typedefs import (
    CameraExplanationTarget,
    TensorModuleOutput,
)


def select_tensor_output(
    value: TensorModuleOutput,
    output_index: int | None,
) -> torch.Tensor:
    """Select the tensor output used for attribution.

    Args:
        value: Tensor output from a module hook, or a tuple of tensor/None
            entries from modules such as DFormer stages.
        output_index: Optional index used when the target layer returns a tuple.
            ``None`` selects the first non-None tensor output.

    Returns:
        The tensor to use as the attribution target.

    Raises:
        RuntimeError: If ``output_index`` selects ``None``.
        RuntimeError: If no tensor output is available.
    """
    if isinstance(value, torch.Tensor):
        return value

    if output_index is not None:
        indexed_value = value[output_index]
        if indexed_value is None:
            raise RuntimeError(
                f"Target layer output_index={output_index} has no tensor."
            )
        return indexed_value

    for tensor in value:
        if tensor is not None:
            return tensor
    raise RuntimeError("Target layer did not produce a tensor output.")


def select_camera_tensor(
    tensor: torch.Tensor,
    target: CameraExplanationTarget,
) -> torch.Tensor:
    """Select the camera-specific slice from a captured target tensor.

    Args:
        tensor: Tensor selected from the target-layer output.
        target: Concrete camera target describing capture routing.

    Returns:
        Tensor belonging to the requested camera.

    Raises:
        RuntimeError: If a stacked camera batch cannot be split evenly.
    """
    if target.capture_mode != VisionCaptureMode.STACKED_CAMERA_BATCH.value:
        return tensor
    if target.stacked_camera_index is None or target.stacked_camera_count is None:
        raise RuntimeError(
            "Stacked camera capture requires stacked_camera_index and "
            "stacked_camera_count."
        )
    if tensor.shape[0] % target.stacked_camera_count != 0:
        raise RuntimeError(
            f"Cannot split stacked camera batch of size {tensor.shape[0]} into "
            f"{target.stacked_camera_count} cameras."
        )
    reshaped = tensor.reshape(
        tensor.shape[0] // target.stacked_camera_count,
        target.stacked_camera_count,
        *tensor.shape[1:],
    )
    return reshaped[:, target.stacked_camera_index].reshape(
        -1,
        *tensor.shape[1:],
    )


def replace_camera_tensor(
    tensor: torch.Tensor,
    replacement: torch.Tensor,
    target: CameraExplanationTarget,
) -> torch.Tensor:
    """Replace the camera-specific slice inside a target-layer tensor.

    Args:
        tensor: Original target-layer tensor.
        replacement: Replacement tensor for the selected camera.
        target: Concrete camera target describing capture routing.

    Returns:
        Tensor with the selected camera slice replaced.

    Raises:
        RuntimeError: If a stacked camera batch cannot be split evenly.
    """
    if target.capture_mode != VisionCaptureMode.STACKED_CAMERA_BATCH.value:
        return replacement
    if target.stacked_camera_index is None or target.stacked_camera_count is None:
        raise RuntimeError(
            "Stacked camera replacement requires stacked_camera_index and "
            "stacked_camera_count."
        )
    if tensor.shape[0] % target.stacked_camera_count != 0:
        raise RuntimeError(
            f"Cannot split stacked camera batch of size {tensor.shape[0]} into "
            f"{target.stacked_camera_count} cameras."
        )
    reshaped = tensor.reshape(
        tensor.shape[0] // target.stacked_camera_count,
        target.stacked_camera_count,
        *tensor.shape[1:],
    ).clone()
    reshaped[:, target.stacked_camera_index] = replacement.reshape(
        tensor.shape[0] // target.stacked_camera_count,
        *tensor.shape[1:],
    )
    return reshaped.reshape_as(tensor)


def should_capture_invocation(
    call_index: int,
    target: CameraExplanationTarget,
) -> bool:
    """Return whether a hook call belongs to the requested camera."""
    if target.capture_mode != VisionCaptureMode.PER_CAMERA_CALL.value:
        return True
    return call_index == target.invocation_index


class GradientCapture:
    """Capture activation and gradient for one camera target."""

    def __init__(self, target: CameraExplanationTarget) -> None:
        """Initialize the capture state.

        Args:
            target: Concrete camera target whose activation should be captured.
        """
        self.target = target
        self.call_index = 0
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None

    def forward_hook(
        self,
        module: torch.nn.Module,
        module_input: tuple[torch.Tensor, ...],
        module_output: TensorModuleOutput,
    ) -> None:
        """Capture the selected camera activation and attach a gradient hook.

        Args:
            module: Module that produced the activation.
            module_input: Positional module inputs from PyTorch's hook API.
            module_output: Module output to capture.

        Raises:
            RuntimeError: If the selected activation does not require gradients,
                or if the layer already produced a captured activation in this
                forward pass, which would pair the last activation with the
                gradient of an earlier one.
        """
        current_call_index = self.call_index
        self.call_index += 1
        if not should_capture_invocation(
            call_index=current_call_index,
            target=self.target,
        ):
            return
        if self.activation is not None:
            raise RuntimeError(
                f"Target layer was invoked more than once for camera "
                f"'{self.target.camera_key}' in capture mode "
                f"'{self.target.capture_mode}'; the activation/gradient "
                "pairing would be ambiguous. Use a per-camera capture mode or "
                "a more specific target layer."
            )
        activation = select_tensor_output(
            value=module_output,
            output_index=self.target.target.output_index,
        )
        camera_activation = select_camera_tensor(
            tensor=activation,
            target=self.target,
        )
        if not activation.requires_grad:
            raise RuntimeError("Target activation does not require gradients.")
        self.activation = camera_activation.detach()
        activation.register_hook(self.gradient_hook)

    def gradient_hook(self, gradient: torch.Tensor) -> None:
        """Record the gradient of the selected activation.

        Args:
            gradient: Gradient tensor produced for the captured activation.
        """
        self.gradient = select_camera_tensor(
            tensor=gradient,
            target=self.target,
        ).detach()

    def require_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return captured activation and gradient tensors.

        Returns:
            Tuple of ``(activation, gradient)`` from the most recent forward and
            backward pass.

        Raises:
            RuntimeError: If either hook did not run.
        """
        if self.activation is None:
            raise RuntimeError("Target activation was not captured.")
        if self.gradient is None:
            raise RuntimeError("Target gradient was not captured.")
        return self.activation, self.gradient


class ActivationCapture:
    """Capture activation for one perturbation target."""

    def __init__(self, target: CameraExplanationTarget) -> None:
        """Initialize the capture state.

        Args:
            target: Concrete camera target whose activation should be captured.
        """
        self.target = target
        self.call_index = 0
        self.activation: torch.Tensor | None = None

    def forward_hook(
        self,
        module: torch.nn.Module,
        module_input: tuple[torch.Tensor, ...],
        module_output: TensorModuleOutput,
    ) -> None:
        """Capture the selected camera activation.

        Args:
            module: Module that produced the activation.
            module_input: Positional module inputs from PyTorch's hook API.
            module_output: Module output to capture.
        """
        current_call_index = self.call_index
        self.call_index += 1
        if not should_capture_invocation(
            call_index=current_call_index,
            target=self.target,
        ):
            return
        activation = select_tensor_output(
            value=module_output,
            output_index=self.target.target.output_index,
        )
        self.activation = select_camera_tensor(
            tensor=activation,
            target=self.target,
        ).detach()

    def require_activation(self) -> torch.Tensor:
        """Return the captured activation tensor.

        Returns:
            Activation from the most recent forward pass.

        Raises:
            RuntimeError: If the forward hook did not run.
        """
        if self.activation is None:
            raise RuntimeError("Target activation was not captured.")
        return self.activation
