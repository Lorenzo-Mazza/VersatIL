"""Policy-level regularizers for local sensitivity control."""

import abc

import torch
import torch.nn as nn
from torch.func import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from versatil.common.tensor_ops import (
    TensorTree,
    batch_rms,
    combined_batch_rms,
    detach_floating_tensor_dictionary,
    normalize_tensor_tuple,
    reshape_batch_scale_for_broadcast,
    slice_tensor_dictionary,
)
from versatil.metrics.base import LossOutput
from versatil.metrics.constants import MetricKey
from versatil.metrics.regularization_context import (
    PolicyForwardContext,
    PolicyGraphInputDomain,
    PolicyRegularizationGraph,
)

RegularizerInputDomain = PolicyGraphInputDomain


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
        self.input_domain = RegularizerInputDomain(input_domain)
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
        """Apply sub-batching and optional detachment to graph-boundary tensors.

        Args:
            context: Original policy forward context. Tensor dictionaries are
                expected to share the same leading batch dimension ``B``.

        Returns:
            Context containing sliced tensors with at most ``max_batch_size``
            rows. Floating tensors in the configured input domain are detached
            when ``detach_inputs`` is enabled.

        Raises:
            ValueError: If a required context dictionary is missing.
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
                case RegularizerInputDomain.OBSERVATION:
                    observation = detach_floating_tensor_dictionary(values=observation)
                case RegularizerInputDomain.ENCODED_FEATURES:
                    encoded_features = detach_floating_tensor_dictionary(
                        values=encoded_features
                    )
                case RegularizerInputDomain.DECODER_FEATURES:
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
        """Return tensors at the configured graph boundary.

        Args:
            context: Policy forward context containing raw observations, encoded
                features, decoder-ready features, and predictions.

        Returns:
            Dictionary for ``self.input_domain``. Selected ``input_keys`` must
            resolve to batched tensors with shape ``(B, ...)`` for
            perturbation-based regularizers.
        """
        match self.input_domain:
            case RegularizerInputDomain.OBSERVATION:
                return context.observation
            case RegularizerInputDomain.ENCODED_FEATURES:
                return context.encoded_features
            case RegularizerInputDomain.DECODER_FEATURES:
                return context.decoder_features

    def _validate_input_keys(self, context: PolicyForwardContext) -> None:
        """Validate that configured input tensors can be perturbed.

        Args:
            context: Prepared policy forward context.

        Raises:
            ValueError: If an input key is absent from the configured domain or
                points to a non-floating tensor.
        """
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
        """Resolve prediction tensors used to measure sensitivity.

        Args:
            graph: Batch-local policy graph with default loss output keys.
            predictions: Prediction dictionary from the original forward pass.
                Each selected value must be floating point and batched as
                ``(B, ...)``.

        Returns:
            Ordered list of prediction keys to flatten and compare.

        Raises:
            ValueError: If selected output keys are missing, non-floating, or
                cannot be inferred.
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
        """Flatten selected predictions into one vector per batch item.

        Args:
            predictions: Prediction tensor dictionary. Selected tensors must have
                shape ``(B, ...)`` and matching batch size.
            output_keys: Keys in ``predictions`` to concatenate.

        Returns:
            Tensor with shape ``(B, D_out)`` where ``D_out`` is the total flattened
            size across selected prediction tensors.
        """
        return torch.cat(
            [
                predictions[key].reshape(predictions[key].shape[0], -1)
                for key in output_keys
            ],
            dim=1,
        )

    def _forward_with_replacements(
        self,
        graph: PolicyRegularizationGraph,
        context: PolicyForwardContext,
        replacements: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluate the policy graph after replacing configured-domain tensors.

        Args:
            graph: Batch-local graph re-entry interface owned by ``Policy``.
            context: Prepared forward context whose tensors form the base state
                for the re-entry.
            replacements: Tensor values keyed by ``input_keys``. Each replacement
                must have the same shape as the corresponding tensor in
                ``self.input_domain``.

        Returns:
            Prediction dictionary produced by the policy graph after applying the
            replacements.
        """
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

        Args:
            domain_inputs: Tensor dictionary for the configured graph boundary.

        Returns:
            Tuple of selected tensors in ``input_keys`` order. With
            ``detach_inputs=True``, each tensor is a detached leaf requiring grad.
            With ``detach_inputs=False``, tensors stay connected to the existing
            graph and are marked as requiring grad only when needed.
        """
        input_tensors: list[torch.Tensor] = []
        for key in self.input_keys:
            value = domain_inputs[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"{type(self).__name__} can only perturb tensor inputs, "
                    f"got non-tensor input key: {key}."
                )
            if self.detach_inputs:
                input_tensors.append(value.detach().requires_grad_(True))
                continue
            if not value.requires_grad:
                value = value.requires_grad_(True)
            input_tensors.append(value)
        return tuple(input_tensors)

    def _zero_output(
        self,
        device: torch.device,
        component_keys: list[str],
    ) -> LossOutput:
        """Return zero loss with stable component keys.

        Args:
            device: Device for the returned scalar tensors.
            component_keys: Diagnostic keys to include in ``component_losses``.

        Returns:
            Loss output with zero total loss and zero-valued components.
        """
        zero = torch.tensor(0.0, device=device)
        return LossOutput(
            total_loss=zero,
            component_losses=dict.fromkeys(component_keys, zero),
        )


class FiniteDifferenceLipschitzRegularizer(PolicyRegularizer):
    """Estimate local Lipschitz slopes with finite-difference perturbations.

    Note:
        The regularizer estimates the local slope of the policy graph at the
        tensors selected by ``input_domain`` and ``input_keys``. For each
        selected tensor ``x_k`` with shape ``(B, ...)``, it draws
        ``n_k = torch.randn_like(x_k)``, normalizes ``n_k`` per batch item to RMS
        one, and sets ``delta_k = noise_scale * rms(x_k) * n_k``. The
        ``PolicyRegularizationGraph`` then re-runs the same policy operation
        order with ``x_k + delta_k`` and, in symmetric mode, ``x_k - delta_k``.

        The selected output tensors are flattened and concatenated per batch
        item. The slope for each sample is
        ``rms(y_plus - y_minus) / rms(2 * delta)`` in symmetric mode, or
        ``rms(y_plus - y_base) / rms(delta)`` in one-sided mode. When multiple
        input keys are perturbed, the denominator RMS is computed after
        concatenating all flattened input deltas. The raw loss is
        ``mean(max(slope - target, 0)^2)`` and ``weight`` scales that raw loss.
    """

    def __init__(
        self,
        input_keys: list[str],
        input_domain: str = RegularizerInputDomain.ENCODED_FEATURES.value,
        output_keys: list[str] | None = None,
        weight: float = 1e-3,
        target: float = 1.0,
        noise_scale: float = 1e-2,
        symmetric: bool = True,
        detach_inputs: bool = True,
        max_batch_size: int | None = None,
        apply_during_eval: bool = False,
        eps: float = 1e-12,
        disable_decoder_stochastic: bool = True,
    ) -> None:
        """Initialize finite-difference local Lipschitz regularization.

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
            symmetric: Whether to use ``f(x + delta) - f(x - delta)``. If false,
                uses ``f(x + delta) - f(x)``.
            detach_inputs: Whether to detach the configured input-domain tensors
                before perturbation.
            max_batch_size: Optional leading-batch slice for cheaper regularizer
                passes.
            apply_during_eval: Whether to compute the regularizer from eval-mode
                policy graphs.
            eps: Minimum denominator/norm used for numerical stability.
            disable_decoder_stochastic: Whether to run decoder stochastic layers
                in eval mode during perturbed forwards.

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
        self.weight = weight
        self.target = target
        self.noise_scale = noise_scale
        self.symmetric = symmetric
        self.eps = eps

    def _build_perturbations(
        self,
        domain_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample per-sample RMS-normalized perturbations.

        Args:
            domain_inputs: Tensor dictionary for the configured graph boundary.
                Values for ``input_keys`` must have shape ``(B, ...)``.

        Returns:
            Perturbation dictionary keyed like ``input_keys``. Each tensor has the
            same shape as the corresponding input tensor and per-sample RMS
            ``noise_scale * rms(input)``.
        """
        perturbations: dict[str, torch.Tensor] = {}
        for key in self.input_keys:
            value = domain_inputs[key]
            random_direction = torch.randn_like(value)
            direction_rms = batch_rms(tensor=random_direction, eps=self.eps)
            normalized_direction = random_direction / reshape_batch_scale_for_broadcast(
                scale=direction_rms,
                tensor=value,
            )
            input_rms = batch_rms(tensor=value, eps=self.eps)
            perturbations[key] = (
                normalized_direction
                * self.noise_scale
                * reshape_batch_scale_for_broadcast(scale=input_rms, tensor=value)
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
            slope. Slopes are computed from RMS output and input differences.
        """
        context = graph.context
        device = next(iter(context.predictions.values())).device
        if not graph.training and not self.apply_during_eval:
            return self._zero_output(
                device=device,
                component_keys=[
                    MetricKey.LIPSCHITZ_FINITE_DIFFERENCE_LOSS.value,
                    MetricKey.LIPSCHITZ_SLOPE_MEAN.value,
                    MetricKey.LIPSCHITZ_SLOPE_MAX.value,
                ],
            )

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
            if self.symmetric:
                minus_replacements = {
                    key: domain_inputs[key] - perturbation
                    for key, perturbation in perturbations.items()
                }
                minus_predictions = self._forward_with_replacements(
                    graph=graph,
                    context=regularizer_context,
                    replacements=minus_replacements,
                )
                output_delta = self._flatten_outputs(
                    predictions=plus_predictions,
                    output_keys=output_keys,
                ) - self._flatten_outputs(
                    predictions=minus_predictions,
                    output_keys=output_keys,
                )
                input_deltas = [
                    2.0 * perturbation for perturbation in perturbations.values()
                ]
            else:
                base_predictions = self._forward_with_replacements(
                    graph=graph,
                    context=regularizer_context,
                    replacements={},
                )
                output_delta = self._flatten_outputs(
                    predictions=plus_predictions,
                    output_keys=output_keys,
                ) - self._flatten_outputs(
                    predictions=base_predictions,
                    output_keys=output_keys,
                )
                input_deltas = list(perturbations.values())

        output_rms = batch_rms(tensor=output_delta, eps=self.eps)
        input_rms = combined_batch_rms(tensors=input_deltas, eps=self.eps)
        local_slope = output_rms / input_rms
        raw_penalty = torch.clamp(local_slope - self.target, min=0.0).pow(2).mean()
        return LossOutput(
            total_loss=self.weight * raw_penalty,
            component_losses={
                MetricKey.LIPSCHITZ_FINITE_DIFFERENCE_LOSS.value: raw_penalty,
                MetricKey.LIPSCHITZ_SLOPE_MEAN.value: local_slope.detach().mean(),
                MetricKey.LIPSCHITZ_SLOPE_MAX.value: local_slope.detach().max(),
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
        The raw loss is that mean squared Frobenius estimate, and ``weight``
        scales the raw loss.
    """

    def __init__(
        self,
        input_keys: list[str],
        input_domain: str = RegularizerInputDomain.ENCODED_FEATURES.value,
        output_keys: list[str] | None = None,
        weight: float = 1e-4,
        number_of_probes: int = 1,
        detach_inputs: bool = True,
        max_batch_size: int | None = None,
        apply_during_eval: bool = False,
        disable_decoder_stochastic: bool = True,
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
            return self._zero_output(
                device=device,
                component_keys=[
                    MetricKey.LIPSCHITZ_JACOBIAN_FROBENIUS_LOSS.value,
                    MetricKey.LIPSCHITZ_JACOBIAN_FROBENIUS_NORM.value,
                ],
            )

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
            )

        probe_penalties: list[torch.Tensor] = []
        with (
            graph.deterministic_scope(enabled=self.disable_decoder_stochastic),
            sdpa_kernel([SDPBackend.MATH]),
        ):
            flat_outputs = evaluate(input_values=input_tensors)
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
                        .pow(2)
                        .sum(dim=1)
                    )
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
        ``sigma_hat = norm(J v)`` and the raw loss is
        ``max(sigma_hat - target, 0)^2``. ``weight`` scales that raw loss.
    """

    def __init__(
        self,
        input_keys: list[str],
        input_domain: str = RegularizerInputDomain.ENCODED_FEATURES.value,
        output_keys: list[str] | None = None,
        weight: float = 1e-4,
        target: float = 1.0,
        number_of_power_iterations: int = 1,
        detach_inputs: bool = True,
        max_batch_size: int | None = None,
        apply_during_eval: bool = False,
        eps: float = 1e-12,
        disable_decoder_stochastic: bool = True,
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
            return self._zero_output(
                device=device,
                component_keys=[
                    MetricKey.LIPSCHITZ_SPECTRAL_JACOBIAN_LOSS.value,
                    MetricKey.LIPSCHITZ_SIGMA.value,
                ],
            )

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
            )

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

        sigma = jacobian_vector.norm()
        raw_penalty = torch.clamp(sigma - self.target, min=0.0).pow(2)
        return LossOutput(
            total_loss=self.weight * raw_penalty,
            component_losses={
                MetricKey.LIPSCHITZ_SPECTRAL_JACOBIAN_LOSS.value: raw_penalty,
                MetricKey.LIPSCHITZ_SIGMA.value: sigma.detach(),
            },
        )
