"""Configuration objects for policy-level regularizers."""

from dataclasses import dataclass

from omegaconf import MISSING

from versatil.metrics.regularization_context import PolicyGraphInputDomain


@dataclass
class BaseRegularizerConfig:
    """Base configuration for policy regularizers.

    Attributes:
        _target_: Import path used by Hydra to instantiate the regularizer.
    """

    _target_: str = MISSING


@dataclass
class FiniteDifferenceLipschitzRegularizerConfig(BaseRegularizerConfig):
    """Configuration for finite-difference local Lipschitz regularization.

    Attributes:
        _target_: Import path for ``FiniteDifferenceLipschitzRegularizer``.
        input_keys: Tensor keys to perturb in ``input_domain``.
        input_domain: Policy graph boundary containing ``input_keys``.
        output_keys: Prediction keys used to measure output change. If omitted,
            the policy graph's default loss output keys are used.
        weight: Multiplier for the raw hinge penalty.
        target: Hinge threshold for the local slope estimate.
        noise_scale: RMS-relative perturbation magnitude.
        symmetric: Whether to use centered finite differences.
        detach_inputs: Whether to detach the selected graph boundary before
            perturbation.
        max_batch_size: Optional regularizer-only batch size cap.
        apply_during_eval: Whether to compute the regularizer in eval mode.
        eps: Minimum denominator/norm used for numerical stability.
        disable_decoder_stochastic: Whether perturbed forwards should run decoder
            stochastic layers in eval mode.
    """

    _target_: str = "versatil.metrics.regularizers.FiniteDifferenceLipschitzRegularizer"
    input_keys: list[str] = MISSING
    input_domain: str = PolicyGraphInputDomain.ENCODED_FEATURES.value
    output_keys: list[str] | None = None
    weight: float = 1e-3
    target: float = 1.0
    noise_scale: float = 1e-2
    symmetric: bool = True
    detach_inputs: bool = True
    max_batch_size: int | None = None
    apply_during_eval: bool = False
    eps: float = 1e-12
    disable_decoder_stochastic: bool = True


@dataclass
class JacobianFrobeniusLipschitzRegularizerConfig(BaseRegularizerConfig):
    """Configuration for Hutchinson Jacobian Frobenius regularization.

    Attributes:
        _target_: Import path for ``JacobianFrobeniusLipschitzRegularizer``.
        input_keys: Tensor keys forming the Jacobian input product space.
        input_domain: Policy graph boundary containing ``input_keys``.
        output_keys: Prediction keys flattened into the Jacobian output vector.
        weight: Multiplier for the raw Frobenius-squared estimate.
        number_of_probes: Number of Rademacher probes to average.
        detach_inputs: Whether to detach the selected graph boundary before
            constructing differentiable input variables.
        max_batch_size: Optional regularizer-only batch size cap.
        apply_during_eval: Whether to compute the regularizer in eval mode.
        disable_decoder_stochastic: Whether probe evaluations should run decoder
            stochastic layers in eval mode.
    """

    _target_: str = (
        "versatil.metrics.regularizers.JacobianFrobeniusLipschitzRegularizer"
    )
    input_keys: list[str] = MISSING
    input_domain: str = PolicyGraphInputDomain.ENCODED_FEATURES.value
    output_keys: list[str] | None = None
    weight: float = 1e-4
    number_of_probes: int = 1
    detach_inputs: bool = True
    max_batch_size: int | None = None
    apply_during_eval: bool = False
    disable_decoder_stochastic: bool = True


@dataclass
class SpectralJacobianLipschitzRegularizerConfig(BaseRegularizerConfig):
    """Configuration for local spectral-Jacobian Lipschitz regularization.

    Attributes:
        _target_: Import path for ``SpectralJacobianLipschitzRegularizer``.
        input_keys: Tensor keys forming the Jacobian input product space.
        input_domain: Policy graph boundary containing ``input_keys``.
        output_keys: Prediction keys flattened into the Jacobian output vector.
        weight: Multiplier for the raw hinge penalty.
        target: Hinge threshold for the spectral norm estimate.
        number_of_power_iterations: Number of JVP/VJP power iterations.
        detach_inputs: Whether to detach the selected graph boundary before
            constructing differentiable input variables.
        max_batch_size: Optional regularizer-only batch size cap.
        apply_during_eval: Whether to compute the regularizer in eval mode.
        eps: Minimum denominator/norm used for numerical stability.
        disable_decoder_stochastic: Whether JVP/VJP forwards should run decoder
            stochastic layers in eval mode.
    """

    _target_: str = "versatil.metrics.regularizers.SpectralJacobianLipschitzRegularizer"
    input_keys: list[str] = MISSING
    input_domain: str = PolicyGraphInputDomain.ENCODED_FEATURES.value
    output_keys: list[str] | None = None
    weight: float = 1e-4
    target: float = 1.0
    number_of_power_iterations: int = 1
    detach_inputs: bool = True
    max_batch_size: int | None = None
    apply_during_eval: bool = False
    eps: float = 1e-12
    disable_decoder_stochastic: bool = True
