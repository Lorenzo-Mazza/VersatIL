"""Policy-level regularizers for local sensitivity control."""

import abc

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from versatil.common.tensor_ops import (
    TensorTree,
    batch_rms,
    detach_floating_tensor_dictionary,
    normalize_tensor_tuple,
    reshape_batch_scale_for_broadcast,
    slice_tensor_dictionary,
)
from versatil.data.constants import ActionComputationMethod, ProprioceptiveType
from versatil.data.metadata import OnTheFlyActionMetadata, PositionActionMetadata
from versatil.metrics.base import LossOutput
from versatil.metrics.constants import (
    FiniteDifferencePerturbationMode,
    ImageAugmentationConsistencyLossMode,
    MetricKey,
)
from versatil.metrics.regularization_context import (
    PolicyForwardContext,
    PolicyGraphInputDomain,
    PolicyRegularizationGraph,
)


class PolicyRegularizer(nn.Module, abc.ABC):
    """Base class for losses that perturb a named policy graph boundary."""

    def __init__(
        self,
        input_keys: list[str],
        input_domain: str,
        output_keys: list[str] | None = None,
        apply_during_eval: bool = False,
        max_batch_size: int | None = None,
        detach_inputs: bool = True,
        disable_decoder_stochastic: bool = True,
    ) -> None:
        """Initialize graph-boundary regularization settings.

        Args:
            input_keys: Tensor keys to perturb inside ``input_domain``.
            input_domain: Policy graph boundary containing ``input_keys``. Valid
                values are ``"observation"``, ``"encoded_features"``, and
                ``"decoder_features"``.
            output_keys: Prediction tensor keys used to measure output change. If
                omitted, the policy graph's default loss output keys are used.
            apply_during_eval: Whether to compute the regularizer when the policy
                graph was built in eval mode.
            max_batch_size: Optional leading-batch slice used only for the
                regularizer pass.
            detach_inputs: Whether to detach floating tensors at ``input_domain``
                before perturbing them. This blocks gradients into upstream graph
                segments while preserving gradients through the re-entered suffix.
            disable_decoder_stochastic: Whether the graph should run decoder
                stochastic layers in eval mode during perturbed forwards.
        """
        super().__init__()
        self.input_keys = input_keys
        self.input_domain = PolicyGraphInputDomain(input_domain)
        self.output_keys = output_keys
        self.apply_during_eval = apply_during_eval
        self.max_batch_size = max_batch_size
        self.detach_inputs = detach_inputs
        self.disable_decoder_stochastic = disable_decoder_stochastic

    @abc.abstractmethod
    def forward(
        self,
        graph: PolicyRegularizationGraph,
    ) -> LossOutput:
        """Compute a regularization loss for one policy forward graph.

        Args:
            graph: Batch-local graph object created by ``Policy``. It contains
                the original forward tensors and the callback used to re-enter
                the exact policy execution order with replacements.

        Returns:
            Loss output with the weighted regularization loss and diagnostics.
        """
        raise NotImplementedError

    def _prepare_context(self, context: PolicyForwardContext) -> PolicyForwardContext:
        """Slice the context to ``max_batch_size`` rows and apply detachment.

        Floating tensors in the configured input domain are detached when
        ``detach_inputs`` is enabled.
        """
        observation = slice_tensor_dictionary(
            values=context.observation,
            max_batch_size=self.max_batch_size,
        )
        encoded_features = slice_tensor_dictionary(
            values=context.encoded_features,
            max_batch_size=self.max_batch_size,
        )
        decoder_features = slice_tensor_dictionary(
            values=context.decoder_features,
            max_batch_size=self.max_batch_size,
        )
        predictions = slice_tensor_dictionary(
            values=context.predictions,
            max_batch_size=self.max_batch_size,
        )
        actions = slice_tensor_dictionary(
            values=context.actions,
            max_batch_size=self.max_batch_size,
        )
        if observation is None or encoded_features is None or decoder_features is None:
            raise ValueError("PolicyForwardContext dictionaries cannot be None.")
        if predictions is None:
            raise ValueError("PolicyForwardContext predictions cannot be None.")
        if self.detach_inputs:
            match self.input_domain:
                case PolicyGraphInputDomain.OBSERVATION:
                    observation = detach_floating_tensor_dictionary(values=observation)
                case PolicyGraphInputDomain.ENCODED_FEATURES:
                    encoded_features = detach_floating_tensor_dictionary(
                        values=encoded_features
                    )
                case PolicyGraphInputDomain.DECODER_FEATURES:
                    decoder_features = detach_floating_tensor_dictionary(
                        values=decoder_features
                    )
        return PolicyForwardContext(
            observation=observation,
            encoded_features=encoded_features,
            decoder_features=decoder_features,
            predictions=predictions,
            actions=actions,
        )

    def _domain_inputs(
        self,
        context: PolicyForwardContext,
    ) -> dict[str, TensorTree]:
        """Return the context dictionary for ``self.input_domain``."""
        match self.input_domain:
            case PolicyGraphInputDomain.OBSERVATION:
                return context.observation
            case PolicyGraphInputDomain.ENCODED_FEATURES:
                return context.encoded_features
            case PolicyGraphInputDomain.DECODER_FEATURES:
                return context.decoder_features

    def _validate_input_keys(self, context: PolicyForwardContext) -> None:
        """Raise if a configured input key is missing or not a floating tensor."""
        domain_inputs = self._domain_inputs(context=context)
        missing_keys = sorted(set(self.input_keys) - set(domain_inputs))
        if missing_keys:
            raise ValueError(
                f"{type(self).__name__} input keys {missing_keys} were not found "
                f"in domain '{self.input_domain.value}'. "
                f"Available keys: {sorted(domain_inputs)}."
            )
        non_tensor_keys = [
            key
            for key in self.input_keys
            if not isinstance(domain_inputs[key], torch.Tensor)
        ]
        if non_tensor_keys:
            raise ValueError(
                f"{type(self).__name__} can only perturb tensor inputs, "
                f"got non-tensor input keys: {non_tensor_keys}."
            )
        non_floating_keys = [
            key
            for key in self.input_keys
            if not torch.is_floating_point(domain_inputs[key])
        ]
        if non_floating_keys:
            raise ValueError(
                f"{type(self).__name__} can only perturb floating-point tensors, "
                f"got non-floating input keys: {non_floating_keys}."
            )

    def _resolve_output_keys(
        self,
        graph: PolicyRegularizationGraph,
        predictions: dict[str, torch.Tensor],
    ) -> list[str]:
        """Resolve prediction keys used to measure sensitivity.

        Falls back to the graph's default loss output keys, then to all
        floating prediction keys. Raises if keys are missing, non-floating,
        or cannot be inferred.
        """
        if self.output_keys is not None:
            keys = self.output_keys
        else:
            keys = graph.default_output_keys
            if not keys:
                keys = sorted(
                    key
                    for key, value in predictions.items()
                    if torch.is_floating_point(value)
                )
        missing_keys = sorted(set(keys) - set(predictions))
        if missing_keys:
            raise ValueError(
                f"{type(self).__name__} output keys {missing_keys} were not found "
                f"in predictions. Available keys: {sorted(predictions)}."
            )
        non_floating_keys = [
            key for key in keys if not torch.is_floating_point(predictions[key])
        ]
        if non_floating_keys:
            raise ValueError(
                f"{type(self).__name__} output keys must be floating-point tensors, "
                f"got non-floating keys: {non_floating_keys}."
            )
        if not keys:
            raise ValueError(
                f"{type(self).__name__} could not infer output keys. Configure "
                "output_keys explicitly."
            )
        return keys

    def _flatten_outputs(
        self,
        predictions: dict[str, torch.Tensor],
        output_keys: list[str],
    ) -> torch.Tensor:
        """Concatenate selected ``(B, ...)`` predictions into ``(B, D_out)``."""
        return torch.cat(
            [
                predictions[key].reshape(predictions[key].shape[0], -1)
                for key in output_keys
            ],
            dim=1,
        )

    @staticmethod
    def _flatten_batch_tensors(
        tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        """Flatten ``(B, ...)`` tensors to ``(B, D_i)``, validating batch sizes."""
        if not tensors:
            raise ValueError("At least one tensor is required.")
        batch_size: int | None = None
        flattened_tensors: list[torch.Tensor] = []
        for tensor in tensors:
            if tensor.ndim == 0:
                raise ValueError("Batched tensors must have at least one dimension.")
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ValueError(
                    "Batched tensors must share the same leading batch dimension."
                )
            flattened_tensors.append(tensor.reshape(tensor.shape[0], -1))
        return flattened_tensors

    @classmethod
    def _combined_batch_l2(
        cls,
        tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
        eps: float,
    ) -> torch.Tensor:
        """Compute one raw ``(B,)`` L2 norm across flattened selected tensors."""
        flattened_tensors = cls._flatten_batch_tensors(tensors=tensors)
        return torch.cat(flattened_tensors, dim=1).float().norm(dim=1).clamp_min(eps)

    @classmethod
    def _combined_feature_dimension(
        cls,
        tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> int:
        """Return the flattened non-batch dimension across selected tensors."""
        return sum(
            flattened_tensor.shape[1]
            for flattened_tensor in cls._flatten_batch_tensors(tensors=tensors)
        )

    @staticmethod
    def _dimension_ratio_scale(
        input_dimension: int,
        output_dimension: int,
        exponent: float,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Build a tensor scalar for dimension-ratio normalization."""
        if input_dimension < 1:
            raise ValueError(
                f"input_dimension must be positive, got {input_dimension}."
            )
        if output_dimension < 1:
            raise ValueError(
                f"output_dimension must be positive, got {output_dimension}."
            )
        return reference.new_tensor(
            float(input_dimension) / float(output_dimension)
        ).pow(exponent)

    def _forward_with_replacements(
        self,
        graph: PolicyRegularizationGraph,
        context: PolicyForwardContext,
        replacements: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Re-enter the policy graph with replacements at ``self.input_domain``."""
        return graph.evaluate(
            input_domain=self.input_domain.value,
            context=context,
            replacements=replacements,
        )

    def _differentiable_input_tensors(
        self,
        domain_inputs: dict[str, TensorTree],
    ) -> tuple[torch.Tensor, ...]:
        """Return selected input tensors prepared for Jacobian products.

        Returns the tensors in ``input_keys`` order. With ``detach_inputs=True``
        each tensor is a detached leaf requiring grad; with
        ``detach_inputs=False`` graph-connected tensors are used as-is, and
        grad-free tensors become fresh leaves so the shared context is never
        mutated in place.
        """
        input_tensors: list[torch.Tensor] = []
        for key in self.input_keys:
            value = domain_inputs[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"{type(self).__name__} can only perturb tensor inputs, "
                    f"got non-tensor input key: {key}."
                )
            value = value.float()
            if self.detach_inputs or not value.requires_grad:
                value = value.detach().requires_grad_(True)
            input_tensors.append(value)
        return tuple(input_tensors)

    def _disabled_output(
        self,
        device: torch.device,
    ) -> LossOutput:
        """Return a zero loss without diagnostics for disabled regularizers."""
        zero = torch.tensor(0.0, device=device)
        return LossOutput(total_loss=zero)


class FiniteDifferenceLipschitzRegularizer(PolicyRegularizer):
    """Estimate local directional sensitivity with finite-difference probes.

    Note:
        The regularizer estimates the local slope of the policy graph at the
        tensors selected by ``input_domain`` and ``input_keys``. For each
        selected tensor ``x_k`` with shape ``(B, ...)``, it draws a random
        direction according to ``perturbation_mode``, normalizes it per batch
        item to RMS one, and sets ``delta_k = noise_scale * rms(x_k) * n_k``.
        The ``PolicyRegularizationGraph`` then re-runs the same policy
        operation order with ``x_k + delta_k`` and ``x_k - delta_k`` (centered
        differences); the graph replays one RNG snapshot per batch, so both
        forwards share identical stochastic draws and the delta isolates input
        sensitivity.

        The selected output deltas are flattened and concatenated, so the
        numerator is the raw output norm ``||delta_y||_2``. The denominator is
        the raw concatenated input perturbation norm ``||delta_x||_2``. When
        enabled, ``scale_by_dimension_ratio`` multiplies this slope by
        ``sqrt(D_in) / sqrt(D_out)``. The raw loss is
        ``mean(max(slope - target, 0)^2)`` and ``weight`` scales that raw loss.

        In ``"gaussian_dense"`` mode the slope satisfies
        ``E[slope^2] = ||J||_F^2 / D_in`` at small ``noise_scale`` — an
        average-direction gain comparable to
        ``JacobianFrobeniusLipschitzRegularizer``, not a worst-case Lipschitz
        constant (use ``SpectralJacobianLipschitzRegularizer`` for that).
        ``"gaussian_channel_broadcast"`` instead probes spatially constant
        per-channel shifts, mimicking global illumination changes.
    """

    def __init__(
        self,
        input_keys: list[str],
        input_domain: str = PolicyGraphInputDomain.ENCODED_FEATURES.value,
        output_keys: list[str] | None = None,
        weight: float = 1e-3,
        target: float = 1.0,
        noise_scale: float = 1e-2,
        detach_inputs: bool = True,
        max_batch_size: int | None = None,
        apply_during_eval: bool = False,
        eps: float = 1e-12,
        disable_decoder_stochastic: bool = True,
        scale_by_dimension_ratio: bool = False,
        perturbation_mode: str = FiniteDifferencePerturbationMode.GAUSSIAN_DENSE.value,
    ) -> None:
        """Initialize finite-difference local sensitivity regularization.

        Args:
            input_keys: Tensor keys to perturb in ``input_domain``. Selected
                tensors must be floating point with shape ``(B, ...)``.
            input_domain: Graph boundary to perturb.
            output_keys: Prediction keys used for the numerator. Selected
                predictions must be floating point with shape ``(B, ...)``.
            weight: Multiplier applied to the raw hinge penalty.
            target: Hinge threshold for the local slope estimate.
            noise_scale: RMS-relative perturbation magnitude. A value of ``1e-2``
                samples perturbations with RMS equal to one percent of each
                input tensor's per-sample RMS.
            detach_inputs: Whether to detach the configured input-domain tensors
                before perturbation.
            max_batch_size: Optional leading-batch slice for cheaper regularizer
                passes.
            apply_during_eval: Whether to compute the regularizer from eval-mode
                policy graphs.
            eps: Minimum denominator/norm used for numerical stability.
            disable_decoder_stochastic: Whether to run decoder stochastic layers
                in eval mode during perturbed forwards.
            scale_by_dimension_ratio: Whether to multiply the raw L2 slope by
                ``sqrt(D_in) / sqrt(D_out)``.
            perturbation_mode: Random direction sampler. ``"gaussian_dense"``
                samples independent Gaussian values for every element.
                ``"gaussian_channel_broadcast"`` samples one Gaussian value per
                batch/time/channel and broadcasts over spatial axes for 4D/5D
                image-like tensors.

        Raises:
            ValueError: If scalar hyperparameters are outside valid ranges.
        """
        super().__init__(
            input_keys=input_keys,
            input_domain=input_domain,
            output_keys=output_keys,
            apply_during_eval=apply_during_eval,
            max_batch_size=max_batch_size,
            detach_inputs=detach_inputs,
            disable_decoder_stochastic=disable_decoder_stochastic,
        )
        if weight < 0.0:
            raise ValueError(f"weight must be non-negative, got {weight}.")
        if target < 0.0:
            raise ValueError(f"target must be non-negative, got {target}.")
        if noise_scale <= 0.0:
            raise ValueError(f"noise_scale must be positive, got {noise_scale}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")
        valid_modes = {mode.value for mode in FiniteDifferencePerturbationMode}
        if perturbation_mode not in valid_modes:
            raise ValueError(
                f"perturbation_mode must be one of {sorted(valid_modes)}, "
                f"got {perturbation_mode}."
            )
        self.weight = weight
        self.target = target
        self.noise_scale = noise_scale
        self.eps = eps
        self.scale_by_dimension_ratio = scale_by_dimension_ratio
        self.perturbation_mode = FiniteDifferencePerturbationMode(perturbation_mode)

    def _sample_random_direction(
        self,
        key: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a finite-difference direction with the configured structure."""
        match self.perturbation_mode:
            case FiniteDifferencePerturbationMode.GAUSSIAN_DENSE:
                return torch.randn_like(value)
            case FiniteDifferencePerturbationMode.GAUSSIAN_CHANNEL_BROADCAST:
                return self._sample_channel_broadcast_direction(key=key, value=value)

    @staticmethod
    def _sample_channel_broadcast_direction(
        key: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Sample per-channel Gaussian directions broadcast over spatial axes."""
        if value.ndim == 4:
            noise_shape = (value.shape[0], value.shape[1], 1, 1)
        elif value.ndim == 5:
            noise_shape = (value.shape[0], value.shape[1], value.shape[2], 1, 1)
        else:
            raise ValueError(
                "FiniteDifferenceLipschitzRegularizer "
                "perturbation_mode='gaussian_channel_broadcast' requires input "
                f"key '{key}' to have shape (B, C, H, W) or (B, T, C, H, W), "
                f"got {tuple(value.shape)}."
            )
        return torch.randn(
            noise_shape,
            device=value.device,
            dtype=value.dtype,
        ).expand_as(value)

    def _build_perturbations(
        self,
        domain_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample perturbations with per-sample RMS ``noise_scale * rms(input)``."""
        perturbations: dict[str, torch.Tensor] = {}
        for key in self.input_keys:
            value = domain_inputs[key]
            value_float = value.float()
            random_direction = self._sample_random_direction(
                key=key,
                value=value_float,
            )
            direction_rms = batch_rms(tensor=random_direction, eps=self.eps)
            normalized_direction = random_direction / reshape_batch_scale_for_broadcast(
                scale=direction_rms,
                tensor=value_float,
            )
            input_rms = batch_rms(tensor=value_float, eps=self.eps)
            perturbations[key] = (
                normalized_direction
                * self.noise_scale
                * reshape_batch_scale_for_broadcast(
                    scale=input_rms,
                    tensor=value_float,
                )
            )
        return perturbations

    def forward(
        self,
        graph: PolicyRegularizationGraph,
    ) -> LossOutput:
        """Compute a finite-difference hinge penalty for one policy graph.

        Args:
            graph: Batch-local policy graph with cached forward tensors and a
                deterministic re-entry callback.

        Returns:
            Loss output containing ``weight * mean(max(slope - target, 0)^2)``.
            Diagnostic components include the raw penalty, mean slope, and max
            slope. Slopes are raw L2/L2 when ``scale_by_dimension_ratio=False``
            and RMS/RMS when ``scale_by_dimension_ratio=True``.
        """
        context = graph.context
        device = next(iter(context.predictions.values())).device
        if not graph.training and not self.apply_during_eval:
            return self._disabled_output(device=device)

        regularizer_context = self._prepare_context(context=context)
        self._validate_input_keys(context=regularizer_context)
        output_keys = self._resolve_output_keys(
            graph=graph,
            predictions=regularizer_context.predictions,
        )
        domain_inputs = self._domain_inputs(context=regularizer_context)
        perturbations = self._build_perturbations(domain_inputs=domain_inputs)

        with graph.deterministic_scope(enabled=self.disable_decoder_stochastic):
            plus_replacements = {
                key: domain_inputs[key] + perturbation
                for key, perturbation in perturbations.items()
            }
            plus_predictions = self._forward_with_replacements(
                graph=graph,
                context=regularizer_context,
                replacements=plus_replacements,
            )
            minus_replacements = {
                key: domain_inputs[key] - perturbation
                for key, perturbation in perturbations.items()
            }
            minus_predictions = self._forward_with_replacements(
                graph=graph,
                context=regularizer_context,
                replacements=minus_replacements,
            )
        output_delta = {
            key: plus_predictions[key] - minus_predictions[key] for key in output_keys
        }
        input_deltas = [2.0 * perturbation for perturbation in perturbations.values()]

        flat_output_delta = self._flatten_outputs(
            predictions=output_delta,
            output_keys=output_keys,
        ).float()
        output_l2 = flat_output_delta.norm(dim=1)
        output_dimension = flat_output_delta.shape[1]
        input_l2 = self._combined_batch_l2(tensors=input_deltas, eps=self.eps)
        local_slope = output_l2 / input_l2
        if self.scale_by_dimension_ratio:
            input_dimension = self._combined_feature_dimension(tensors=input_deltas)
            local_slope = local_slope * self._dimension_ratio_scale(
                input_dimension=input_dimension,
                output_dimension=output_dimension,
                exponent=0.5,
                reference=local_slope,
            )
        raw_penalty = (
            torch.clamp(local_slope.float() - self.target, min=0.0).pow(2).mean()
        )
        return LossOutput(
            total_loss=self.weight * raw_penalty,
            component_losses={
                MetricKey.LIPSCHITZ_FINITE_DIFFERENCE_LOSS.value: raw_penalty,
                MetricKey.LIPSCHITZ_SLOPE_MEAN.value: local_slope.detach().mean(),
                MetricKey.LIPSCHITZ_SLOPE_MAX.value: local_slope.detach().max(),
            },
        )


class ImageAugmentationConsistencyRegularizer(PolicyRegularizer):
    """Penalize output changes under image augmentations of observations."""

    def __init__(
        self,
        input_keys: list[str],
        output_keys: list[str] | None = None,
        weight: float = 1e-3,
        color_augmentation: A.Compose | None = None,
        spatial_augmentation: A.Compose | None = None,
        loss_mode: str = ImageAugmentationConsistencyLossMode.POSITION_TRAJECTORY_L2.value,
        detach_inputs: bool = True,
        detach_targets: bool = True,
        max_batch_size: int | None = None,
        apply_during_eval: bool = False,
        input_min: float = -1.0,
        input_max: float = 1.0,
        max_pixel_value: float = 255.0,
    ) -> None:
        """Initialize image augmentation consistency regularization.

        Args:
            input_keys: Observation image keys to augment. Tensors must be
                ``(B, C, H, W)`` or ``(B, T, C, H, W)``.
            output_keys: Prediction keys whose consistency is penalized. If
                omitted, the graph default loss output keys are used.
            weight: Multiplier applied to the raw consistency MSE.
            color_augmentation: Optional Albumentations color transform. Applied
                before spatial augmentation, matching ``ImageProcessor``.
            spatial_augmentation: Optional Albumentations spatial transform.
            loss_mode: Consistency reduction. ``"position_trajectory_l2"`` uses
                normalized position action trajectory distance; ``"flat_output_mse"``
                preserves the previous flat output MSE behavior.
            detach_inputs: Whether to detach observations before augmentation.
            detach_targets: Whether clean predictions are stop-gradient targets.
            max_batch_size: Optional leading-batch slice for cheaper passes.
            apply_during_eval: Whether to compute the regularizer in eval mode.
            input_min: Normalized value corresponding to image intensity zero.
            input_max: Normalized value corresponding to ``max_pixel_value``.
            max_pixel_value: Pixel range used for Albumentations conversion.

        Raises:
            ValueError: If no augmentation is configured or scalar parameters are
                invalid.
        """
        super().__init__(
            input_keys=input_keys,
            input_domain=PolicyGraphInputDomain.OBSERVATION.value,
            output_keys=output_keys,
            apply_during_eval=apply_during_eval,
            max_batch_size=max_batch_size,
            detach_inputs=detach_inputs,
            disable_decoder_stochastic=True,
        )
        if weight < 0.0:
            raise ValueError(f"weight must be non-negative, got {weight}.")
        if input_max <= input_min:
            raise ValueError(
                f"input_max must be greater than input_min, got "
                f"input_min={input_min} and input_max={input_max}."
            )
        if max_pixel_value <= 0.0:
            raise ValueError(
                f"max_pixel_value must be positive, got {max_pixel_value}."
            )
        valid_loss_modes = {mode.value for mode in ImageAugmentationConsistencyLossMode}
        if loss_mode not in valid_loss_modes:
            raise ValueError(
                f"loss_mode must be one of {sorted(valid_loss_modes)}, got {loss_mode}."
            )
        self.color_augmentation = (
            color_augmentation
            if self._has_augmentation(augmentation=color_augmentation)
            else None
        )
        self.spatial_augmentation = (
            spatial_augmentation
            if self._has_augmentation(augmentation=spatial_augmentation)
            else None
        )
        if self.color_augmentation is None and self.spatial_augmentation is None:
            raise ValueError(
                "At least one of color_augmentation or spatial_augmentation must "
                "contain transforms."
            )
        self.weight = weight
        self.loss_mode = ImageAugmentationConsistencyLossMode(loss_mode)
        self.detach_targets = detach_targets
        self.input_min = input_min
        self.input_max = input_max
        self.max_pixel_value = max_pixel_value

    @staticmethod
    def _has_augmentation(augmentation: A.Compose | None) -> bool:
        """Return whether an Albumentations pipeline has work to do."""
        return augmentation is not None and len(augmentation.transforms) > 0

    @property
    def _input_range(self) -> float:
        return self.input_max - self.input_min

    def _to_uint8_frame(self, frame: torch.Tensor) -> np.ndarray:
        """Convert a normalized ``(C, H, W)`` frame to Albumentations format."""
        zero_to_one = ((frame - self.input_min) / self._input_range).clamp(0.0, 1.0)
        pixel_frame = (
            (zero_to_one * self.max_pixel_value).round().to(torch.uint8).numpy()
        )
        if pixel_frame.shape[0] == 1:
            return pixel_frame[0]
        return np.moveaxis(pixel_frame, 0, -1)

    def _from_augmented_frame(
        self,
        frame: np.ndarray,
        channel_count: int,
    ) -> torch.Tensor:
        """Convert an augmented Albumentations frame back to ``(C, H, W)``."""
        frame_array = np.asarray(frame)
        if frame_array.ndim == 2:
            channel_first = frame_array[None]
        else:
            channel_first = np.moveaxis(frame_array, -1, 0)
        if channel_first.shape[0] != channel_count:
            raise ValueError(
                f"Augmented image channel count changed from {channel_count} to "
                f"{channel_first.shape[0]}."
            )
        zero_to_one = (
            torch.from_numpy(channel_first.copy()).float() / self.max_pixel_value
        )
        return zero_to_one * self._input_range + self.input_min

    def _augment_sample(self, frames: torch.Tensor) -> torch.Tensor:
        """Augment one sample's ``(T, C, H, W)`` window with shared parameters.

        All frames of one temporal window pass through Albumentations as one
        ``images=`` batch, so the sampled transform parameters are identical
        across the window and the augmented sequence stays temporally coherent.
        """
        channel_count = frames.shape[1]
        if self.color_augmentation is not None and channel_count != 3:
            raise ValueError(
                "color_augmentation requires RGB image tensors with 3 channels, "
                f"got {channel_count} channels."
            )
        pixel_frames = [self._to_uint8_frame(frame=frame) for frame in frames]
        if self.color_augmentation is not None:
            pixel_frames = self.color_augmentation(images=pixel_frames)["images"]
        if self.spatial_augmentation is not None:
            pixel_frames = self.spatial_augmentation(images=pixel_frames)["images"]
        augmented_frames = [
            self._from_augmented_frame(frame=frame, channel_count=channel_count)
            for frame in pixel_frames
        ]
        return torch.stack(augmented_frames, dim=0)

    def _augment_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """Apply image augmentations per sample to a batched image tensor."""
        value_float = value.detach().float().cpu()
        if value_float.ndim == 4:
            sample_windows = value_float.unsqueeze(1)  # (B, 1, C, H, W)
        elif value_float.ndim == 5:
            sample_windows = value_float
        else:
            raise ValueError(
                "ImageAugmentationConsistencyRegularizer input tensors must have "
                f"shape (B, C, H, W) or (B, T, C, H, W), got {tuple(value.shape)}."
            )

        augmented = torch.stack(
            [self._augment_sample(frames=window) for window in sample_windows],
            dim=0,
        )
        if value_float.ndim == 4:
            augmented = augmented.squeeze(1)  # (B, 1, C, H, W) -> (B, C, H, W)
        return augmented.to(device=value.device, dtype=value.dtype)

    def _build_augmented_replacements(
        self,
        domain_inputs: dict[str, TensorTree],
    ) -> dict[str, torch.Tensor]:
        """Build augmented observation replacements for configured image keys."""
        replacements: dict[str, torch.Tensor] = {}
        for key in self.input_keys:
            value = domain_inputs[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    "ImageAugmentationConsistencyRegularizer can only augment "
                    f"tensor observations, got non-tensor input key: {key}."
                )
            replacements[key] = self._augment_tensor(value=value)
        return replacements

    def _input_delta_rms(
        self,
        domain_inputs: dict[str, TensorTree],
        replacements: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-sample RMS of the applied image augmentation delta."""
        deltas = [
            replacements[key] - domain_inputs[key]
            for key in self.input_keys
            if isinstance(domain_inputs[key], torch.Tensor)
        ]
        flat_delta = torch.cat(
            [delta.reshape(delta.shape[0], -1).float() for delta in deltas],
            dim=1,
        )
        return flat_delta.pow(2).mean(dim=1).sqrt()

    def _resolve_position_action_keys(
        self,
        graph: PolicyRegularizationGraph,
        predictions: dict[str, torch.Tensor],
    ) -> list[str]:
        """Resolve position action outputs for trajectory consistency."""
        keys = [
            key
            for key, metadata in graph.action_metadata.items()
            if metadata.action_type == ProprioceptiveType.POSITION.value
            and key in predictions
        ]
        if not keys:
            raise ValueError(
                "ImageAugmentationConsistencyRegularizer "
                "loss_mode='position_trajectory_l2' requires at least one "
                "position action output in policy action metadata."
            )
        return keys

    def _uses_delta_position_actions(
        self,
        graph: PolicyRegularizationGraph,
        key: str,
    ) -> bool:
        """Return whether a position action key represents per-step deltas."""
        metadata = graph.action_metadata.get(key)
        if metadata is None:
            return False
        if not isinstance(metadata, PositionActionMetadata | OnTheFlyActionMetadata):
            return False
        return metadata.computation_method == ActionComputationMethod.DELTA.value

    def _position_trajectory_components(
        self,
        graph: PolicyRegularizationGraph,
        clean_predictions: dict[str, torch.Tensor],
        augmented_predictions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute normalized position trajectory consistency diagnostics."""
        position_keys = self._resolve_position_action_keys(
            graph=graph,
            predictions=augmented_predictions,
        )
        per_key_per_step_l2: list[torch.Tensor] = []
        for key in position_keys:
            clean_position = clean_predictions[key]
            if self.detach_targets:
                clean_position = clean_position.detach()
            position_delta = (augmented_predictions[key] - clean_position).float()
            if position_delta.ndim != 3:
                raise ValueError(
                    "ImageAugmentationConsistencyRegularizer "
                    "loss_mode='position_trajectory_l2' expects position action "
                    f"key '{key}' to have shape (B, T, D), got "
                    f"{tuple(position_delta.shape)}."
                )
            if self._uses_delta_position_actions(graph=graph, key=key):
                position_delta = position_delta.cumsum(dim=1)
            per_key_per_step_l2.append(position_delta.norm(dim=-1))
        per_step_l2 = torch.stack(per_key_per_step_l2, dim=0).mean(dim=0)
        return {
            MetricKey.IMAGE_AUGMENTATION_POSITION_PER_STEP_L2.value: (
                per_step_l2.mean()
            ),
            MetricKey.IMAGE_AUGMENTATION_POSITION_PER_STEP_L2_MAX.value: (
                per_step_l2.max()
            ),
            MetricKey.IMAGE_AUGMENTATION_POSITION_FINAL_L2.value: (
                per_step_l2[:, -1].mean()
            ),
        }

    def _raw_penalty_and_components(
        self,
        graph: PolicyRegularizationGraph,
        clean_predictions: dict[str, torch.Tensor],
        augmented_predictions: dict[str, torch.Tensor],
        output_keys: list[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Reduce clean/augmented predictions to a raw consistency penalty."""
        clean_outputs = self._flatten_outputs(
            predictions=clean_predictions,
            output_keys=output_keys,
        ).float()
        if self.detach_targets:
            clean_outputs = clean_outputs.detach()
        augmented_outputs = self._flatten_outputs(
            predictions=augmented_predictions,
            output_keys=output_keys,
        ).float()
        output_delta = augmented_outputs - clean_outputs
        flat_output_mse = output_delta.pow(2).mean()
        components = {
            MetricKey.IMAGE_AUGMENTATION_FLAT_OUTPUT_MSE.value: flat_output_mse,
            MetricKey.IMAGE_AUGMENTATION_OUTPUT_DELTA_L2.value: (
                output_delta.detach().norm(dim=1).mean()
            ),
        }
        match self.loss_mode:
            case ImageAugmentationConsistencyLossMode.FLAT_OUTPUT_MSE:
                return flat_output_mse, components
            case ImageAugmentationConsistencyLossMode.POSITION_TRAJECTORY_L2:
                position_components = self._position_trajectory_components(
                    graph=graph,
                    clean_predictions=clean_predictions,
                    augmented_predictions=augmented_predictions,
                )
                components.update(position_components)
                return (
                    position_components[
                        MetricKey.IMAGE_AUGMENTATION_POSITION_PER_STEP_L2.value
                    ],
                    components,
                )

    def forward(
        self,
        graph: PolicyRegularizationGraph,
    ) -> LossOutput:
        """Compute output consistency under configured image augmentations."""
        context = graph.context
        device = next(iter(context.predictions.values())).device
        if not graph.training and not self.apply_during_eval:
            return self._disabled_output(device=device)

        regularizer_context = self._prepare_context(context=context)
        self._validate_input_keys(context=regularizer_context)
        output_keys = self._resolve_output_keys(
            graph=graph,
            predictions=regularizer_context.predictions,
        )
        domain_inputs = self._domain_inputs(context=regularizer_context)
        replacements = self._build_augmented_replacements(domain_inputs=domain_inputs)

        with graph.deterministic_scope(enabled=self.disable_decoder_stochastic):
            if self.detach_targets:
                with torch.no_grad():
                    clean_predictions = self._forward_with_replacements(
                        graph=graph,
                        context=regularizer_context,
                        replacements={},
                    )
            else:
                clean_predictions = self._forward_with_replacements(
                    graph=graph,
                    context=regularizer_context,
                    replacements={},
                )
            augmented_predictions = self._forward_with_replacements(
                graph=graph,
                context=regularizer_context,
                replacements=replacements,
            )

        raw_penalty, consistency_components = self._raw_penalty_and_components(
            graph=graph,
            clean_predictions=clean_predictions,
            augmented_predictions=augmented_predictions,
            output_keys=output_keys,
        )
        return LossOutput(
            total_loss=self.weight * raw_penalty,
            component_losses={
                MetricKey.IMAGE_AUGMENTATION_CONSISTENCY_LOSS.value: raw_penalty,
                **consistency_components,
                MetricKey.IMAGE_AUGMENTATION_INPUT_DELTA_RMS.value: (
                    self._input_delta_rms(
                        domain_inputs=domain_inputs,
                        replacements=replacements,
                    )
                    .detach()
                    .mean()
                ),
            },
        )


class JacobianFrobeniusLipschitzRegularizer(PolicyRegularizer):
    """Estimate a local Jacobian Frobenius penalty with Hutchinson probes.

    Note:
        The regularizer estimates ``||J||_F^2`` for the local Jacobian mapping
        selected input tensors to selected policy outputs. It creates
        differentiable input variables for ``input_keys``, evaluates the policy
        graph from ``input_domain``, flattens the selected output tensors to
        shape ``(B, D_out)``, and samples Rademacher probes ``r`` with entries
        in ``{-1, 1}``.

        For each probe, it computes ``s = sum(flat_outputs * r)`` and then
        ``torch.autograd.grad(s, inputs, create_graph=True)``. The per-sample
        estimate is the sum of squared gradients across all selected input
        tensors. Averaging over samples and probes gives
        ``E_r ||J^T r||_2^2 = ||J||_F^2`` for the selected graph boundary.
        When enabled, ``scale_by_dimension_ratio`` multiplies the squared
        estimate by ``D_in / D_out``. The raw loss is that squared Frobenius
        estimate, and ``weight`` scales the raw loss.
    """

    def __init__(
        self,
        input_keys: list[str],
        input_domain: str = PolicyGraphInputDomain.ENCODED_FEATURES.value,
        output_keys: list[str] | None = None,
        weight: float = 1e-4,
        number_of_probes: int = 1,
        detach_inputs: bool = True,
        max_batch_size: int | None = None,
        apply_during_eval: bool = False,
        disable_decoder_stochastic: bool = True,
        scale_by_dimension_ratio: bool = False,
    ) -> None:
        """Initialize Hutchinson Jacobian Frobenius regularization.

        Args:
            input_keys: Tensor keys forming the Jacobian input product space.
                Selected tensors must be floating point with shape ``(B, ...)``.
            input_domain: Graph boundary containing ``input_keys``.
            output_keys: Prediction keys flattened into the Jacobian output
                vector. Selected tensors must be floating point with shape
                ``(B, ...)``.
            weight: Multiplier applied to the raw Frobenius-squared estimate.
            number_of_probes: Number of independent Rademacher probes to average.
                Higher values reduce estimator variance and add VJP cost.
            detach_inputs: Whether to detach the selected graph boundary before
                constructing differentiable input variables.
            max_batch_size: Optional leading-batch slice for cheaper regularizer
                passes.
            apply_during_eval: Whether to compute the regularizer from eval-mode
                policy graphs.
            disable_decoder_stochastic: Whether probe evaluations should run
                decoder stochastic layers in eval mode.
            scale_by_dimension_ratio: Whether to multiply the Frobenius-squared
                estimate by ``D_in / D_out``.

        Raises:
            ValueError: If scalar hyperparameters are outside valid ranges.
        """
        super().__init__(
            input_keys=input_keys,
            input_domain=input_domain,
            output_keys=output_keys,
            apply_during_eval=apply_during_eval,
            max_batch_size=max_batch_size,
            detach_inputs=detach_inputs,
            disable_decoder_stochastic=disable_decoder_stochastic,
        )
        if weight < 0.0:
            raise ValueError(f"weight must be non-negative, got {weight}.")
        if number_of_probes < 1:
            raise ValueError(
                f"number_of_probes must be at least 1, got {number_of_probes}."
            )
        self.weight = weight
        self.number_of_probes = number_of_probes
        self.scale_by_dimension_ratio = scale_by_dimension_ratio

    def forward(
        self,
        graph: PolicyRegularizationGraph,
    ) -> LossOutput:
        """Compute a Hutchinson Frobenius-squared Jacobian penalty.

        Args:
            graph: Batch-local policy graph with cached forward tensors and a
                deterministic re-entry callback.

        Returns:
            Loss output containing ``weight * mean(||J^T r||_2^2)``. Diagnostic
            components include the raw Frobenius-squared estimate and its square
            root.
        """
        context = graph.context
        device = next(iter(context.predictions.values())).device
        if not graph.training and not self.apply_during_eval:
            return self._disabled_output(device=device)

        regularizer_context = self._prepare_context(context=context)
        self._validate_input_keys(context=regularizer_context)
        output_keys = self._resolve_output_keys(
            graph=graph,
            predictions=regularizer_context.predictions,
        )
        domain_inputs = self._domain_inputs(context=regularizer_context)
        input_tensors = self._differentiable_input_tensors(
            domain_inputs=domain_inputs,
        )

        def evaluate(input_values: tuple[torch.Tensor, ...]) -> torch.Tensor:
            replacements = dict(zip(self.input_keys, input_values))
            predictions = self._forward_with_replacements(
                graph=graph,
                context=regularizer_context,
                replacements=replacements,
            )
            return self._flatten_outputs(
                predictions=predictions,
                output_keys=output_keys,
            ).float()

        probe_penalties: list[torch.Tensor] = []
        with (
            graph.deterministic_scope(enabled=self.disable_decoder_stochastic),
            sdpa_kernel([SDPBackend.MATH]),
        ):
            flat_outputs = evaluate(input_values=input_tensors)
            output_dimension = flat_outputs.shape[1]
            dimension_scale = flat_outputs.new_tensor(1.0)
            if self.scale_by_dimension_ratio:
                input_dimension = self._combined_feature_dimension(
                    tensors=input_tensors,
                )
                dimension_scale = self._dimension_ratio_scale(
                    input_dimension=input_dimension,
                    output_dimension=output_dimension,
                    exponent=1.0,
                    reference=flat_outputs,
                )
            for _ in range(self.number_of_probes):
                probe = torch.empty_like(flat_outputs).bernoulli_(0.5)
                probe = probe.mul(2.0).sub(1.0)
                scalar = (flat_outputs * probe).sum()
                input_gradients = torch.autograd.grad(
                    scalar,
                    input_tensors,
                    create_graph=True,
                )
                per_sample_squared_norm = torch.zeros(
                    flat_outputs.shape[0],
                    device=flat_outputs.device,
                    dtype=flat_outputs.dtype,
                )
                for gradient in input_gradients:
                    per_sample_squared_norm = (
                        per_sample_squared_norm
                        + gradient.reshape(
                            gradient.shape[0],
                            -1,
                        )
                        .float()
                        .pow(2)
                        .sum(dim=1)
                    )
                per_sample_squared_norm = per_sample_squared_norm * dimension_scale
                probe_penalties.append(per_sample_squared_norm)

        raw_penalty = torch.stack(probe_penalties, dim=0).mean()
        return LossOutput(
            total_loss=self.weight * raw_penalty,
            component_losses={
                MetricKey.LIPSCHITZ_JACOBIAN_FROBENIUS_LOSS.value: raw_penalty,
                MetricKey.LIPSCHITZ_JACOBIAN_FROBENIUS_NORM.value: raw_penalty.detach().sqrt(),
            },
        )


class SpectralJacobianLipschitzRegularizer(PolicyRegularizer):
    """Estimate a local Jacobian spectral norm with power iteration.

    Note:
        The regularizer estimates the largest singular value of the local
        Jacobian of the selected output vector with respect to the selected input
        tensors. It first creates differentiable input variables for the tensors
        named by ``input_keys``. The local function ``evaluate(inputs)`` replaces
        those tensors inside the ``PolicyRegularizationGraph``, re-runs the same
        policy operation order from ``input_domain``, and returns the selected
        outputs flattened to shape ``(B, D_out)``.

        The implementation samples an input-space direction ``v`` across the
        tuple of selected tensors, then uses
        ``torch.func.jvp(evaluate, ..., v)`` to compute the Jacobian-vector
        product ``J v``. It normalizes that output direction to ``u``, evaluates
        ``(evaluate(inputs) * u).sum()``, and calls ``torch.autograd.grad`` with
        respect to the selected input tensors to compute the vector-Jacobian
        product ``J^T u``. Repeating these two products performs power iteration
        on ``J^T J``. After the configured iterations,
        ``sigma_hat = norm(J v)``. When enabled,
        ``scale_by_dimension_ratio`` multiplies ``sigma_hat`` by
        ``sqrt(D_in) / sqrt(D_out)``. The raw loss is
        ``max(sigma_hat - target, 0)^2`` and ``weight`` scales that raw loss.
    """

    def __init__(
        self,
        input_keys: list[str],
        input_domain: str = PolicyGraphInputDomain.ENCODED_FEATURES.value,
        output_keys: list[str] | None = None,
        weight: float = 1e-4,
        target: float = 1.0,
        number_of_power_iterations: int = 1,
        detach_inputs: bool = True,
        max_batch_size: int | None = None,
        apply_during_eval: bool = False,
        eps: float = 1e-12,
        disable_decoder_stochastic: bool = True,
        scale_by_dimension_ratio: bool = False,
    ) -> None:
        """Initialize local spectral-Jacobian Lipschitz regularization.

        Args:
            input_keys: Tensor keys forming the Jacobian input product space.
                Selected tensors must be floating point with shape ``(B, ...)``.
            input_domain: Graph boundary containing ``input_keys``.
            output_keys: Prediction keys flattened into the Jacobian output
                vector. Selected tensors must be floating point with shape
                ``(B, ...)``.
            weight: Multiplier applied to the raw hinge penalty.
            target: Hinge threshold for the estimated spectral norm.
            number_of_power_iterations: Number of local power iterations. Higher
                values improve the estimate and increase JVP/VJP cost.
            detach_inputs: Whether to detach the selected input tensors before
                constructing differentiable perturbation variables.
            max_batch_size: Optional leading-batch slice for cheaper regularizer
                passes.
            apply_during_eval: Whether to compute the regularizer from eval-mode
                policy graphs.
            eps: Minimum denominator/norm used for numerical stability.
            disable_decoder_stochastic: Whether to run decoder stochastic layers
                in eval mode during JVP/VJP evaluation.
            scale_by_dimension_ratio: Whether to multiply the spectral estimate
                by ``sqrt(D_in) / sqrt(D_out)``.

        Raises:
            ValueError: If scalar hyperparameters are outside valid ranges.
        """
        super().__init__(
            input_keys=input_keys,
            input_domain=input_domain,
            output_keys=output_keys,
            apply_during_eval=apply_during_eval,
            max_batch_size=max_batch_size,
            detach_inputs=detach_inputs,
            disable_decoder_stochastic=disable_decoder_stochastic,
        )
        if weight < 0.0:
            raise ValueError(f"weight must be non-negative, got {weight}.")
        if target < 0.0:
            raise ValueError(f"target must be non-negative, got {target}.")
        if number_of_power_iterations < 1:
            raise ValueError(
                "number_of_power_iterations must be at least 1, "
                f"got {number_of_power_iterations}."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.weight = weight
        self.target = target
        self.number_of_power_iterations = number_of_power_iterations
        self.eps = eps
        self.scale_by_dimension_ratio = scale_by_dimension_ratio

    def forward(
        self,
        graph: PolicyRegularizationGraph,
    ) -> LossOutput:
        """Compute a local spectral-Jacobian hinge penalty.

        Args:
            graph: Batch-local policy graph with cached forward tensors and a
                deterministic re-entry callback.

        Returns:
            Loss output containing ``weight * max(sigma - target, 0)^2`` and
            diagnostics for the raw penalty and detached ``sigma`` estimate.
            The Jacobian maps selected input tensors to the flattened selected
            outputs over the current regularizer batch.
        """
        context = graph.context
        device = next(iter(context.predictions.values())).device
        if not graph.training and not self.apply_during_eval:
            return self._disabled_output(device=device)

        regularizer_context = self._prepare_context(context=context)
        self._validate_input_keys(context=regularizer_context)
        output_keys = self._resolve_output_keys(
            graph=graph,
            predictions=regularizer_context.predictions,
        )
        domain_inputs = self._domain_inputs(context=regularizer_context)
        input_tensors = self._differentiable_input_tensors(
            domain_inputs=domain_inputs,
        )

        def evaluate(input_values: tuple[torch.Tensor, ...]) -> torch.Tensor:
            replacements = dict(zip(self.input_keys, input_values))
            predictions = self._forward_with_replacements(
                graph=graph,
                context=regularizer_context,
                replacements=replacements,
            )
            return self._flatten_outputs(
                predictions=predictions,
                output_keys=output_keys,
            ).float()

        with (
            graph.deterministic_scope(enabled=self.disable_decoder_stochastic),
            sdpa_kernel([SDPBackend.MATH]),
        ):
            direction = tuple(torch.randn_like(value) for value in input_tensors)
            direction = normalize_tensor_tuple(tensors=direction, eps=self.eps)

            for _ in range(self.number_of_power_iterations):
                _, jacobian_vector = jvp(evaluate, (input_tensors,), (direction,))
                output_direction = jacobian_vector.detach()
                output_direction = output_direction / output_direction.norm().clamp_min(
                    self.eps
                )
                predictions = evaluate(input_values=input_tensors)
                scalar = (predictions * output_direction).sum()
                vector_jacobian = torch.autograd.grad(
                    scalar,
                    input_tensors,
                    retain_graph=False,
                    create_graph=False,
                )
                direction = normalize_tensor_tuple(
                    tensors=tuple(value.detach() for value in vector_jacobian),
                    eps=self.eps,
                )

            _, jacobian_vector = jvp(evaluate, (input_tensors,), (direction,))

        sigma = jacobian_vector.float().norm()
        if self.scale_by_dimension_ratio:
            input_dimension = self._combined_feature_dimension(tensors=input_tensors)
            output_dimension = jacobian_vector.shape[1]
            sigma = sigma * self._dimension_ratio_scale(
                input_dimension=input_dimension,
                output_dimension=output_dimension,
                exponent=0.5,
                reference=sigma,
            )
        raw_penalty = torch.clamp(sigma - self.target, min=0.0).pow(2)
        return LossOutput(
            total_loss=self.weight * raw_penalty,
            component_losses={
                MetricKey.LIPSCHITZ_SPECTRAL_JACOBIAN_LOSS.value: raw_penalty,
                MetricKey.LIPSCHITZ_SIGMA.value: sigma.detach(),
            },
        )
