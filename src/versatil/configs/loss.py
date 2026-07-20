"""Loss configuration for policy training."""

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from versatil.metrics.kernels import KernelType


@dataclass
class BaseLossConfig:
    """Base configuration for loss modules.

    Attributes:
        _target_: Import path instantiated by Hydra.
    """

    _target_: str = MISSING


@dataclass
class RegressionLossConfig(BaseLossConfig):
    """Configuration for regression loss (position, orientation).

    Attributes:
        _target_: Import path instantiated by Hydra.
        action_keys: List of action keys to compute loss for (e.g., ['position',
            'orientation']).
        mse_weight: Weight for MSE loss.
        l1_weight: Weight for L1 loss.
        huber_weight: Weight for Huber loss.
        huber_delta: Delta parameter for Huber loss.
        per_key_weights: Optional dictionary of per-key weights.
    """

    _target_: str = "versatil.metrics.losses.regression.RegressionLoss"
    action_keys: list[str] = MISSING
    mse_weight: float = 1.0
    l1_weight: float = 0.0
    huber_weight: float = 0.0
    huber_delta: float = 1.0
    per_key_weights: dict[str, float] | None = None


@dataclass
class GripperLossConfig(BaseLossConfig):
    """Configuration for gripper loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        key: Action key for gripper.
        actions_metadata: Dict of metadata of the action space.
        bce_weight: Weight for binary cross entropy (binary gripper).
        mse_weight: Weight for MSE loss (continuous gripper).
        pos_weight: Optional positive class weight for BCE.
    """

    _target_: str = "versatil.metrics.losses.gripper.GripperLoss"
    key: str = MISSING
    actions_metadata: Any = "${task.action_space.actions_metadata}"
    bce_weight: float = 1.0
    mse_weight: float = 1.0
    pos_weight: float | None = None


@dataclass
class KLDivergenceLossConfig(BaseLossConfig):
    """Configuration for KL divergence loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Weight for KL divergence loss KL(posterior || prior).
        prior_regularization_weight: Weight for KL(prior || N(0,I)) regularization. Only
            meaningful for learned priors. Pushes the learned prior towards a standard
            Gaussian.
    """

    _target_: str = "versatil.metrics.losses.divergence.KLDivergenceLoss"
    weight: float = 0.0001
    prior_regularization_weight: float = 0.0


@dataclass
class GaussianEntropyLossConfig(BaseLossConfig):
    """Configuration for entropy loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        key: Prediction key for logvar tensor to compute entropy over.
        weight: Loss weight. Positive values encourage higher entropy.
    """

    _target_: str = "versatil.metrics.losses.divergence.GaussianEntropyLoss"
    key: str = MISSING
    weight: float = 0.0


@dataclass
class BinaryKLDivergenceLossConfig(BaseLossConfig):
    """Configuration for binary KL divergence loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Weight for KL divergence loss.
        free_bits: Free bits threshold (only penalize KL above this value).
        latent_bits: Number of bits of the latent codes.
        entropy_weight: Weight for the entropy regularization term.
    """

    _target_: str = "versatil.metrics.losses.divergence.BinaryKLDivergenceLoss"
    weight: float = 0.0001
    free_bits: float = 0.0
    latent_bits: int = MISSING
    entropy_weight: float = 0.005


@dataclass
class MaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for Maximum Mean Discrepancy (MMD) loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Loss weight for MMD(posterior, prior).
        prior_regularization_weight: Weight for MMD(prior, N(0,I)) regularization. Only
            meaningful for learned priors.
        prior_target_key: Posterior output key used as aggregate prior-matching samples.
            Use ``LatentKey.POSTERIOR_MU`` for deterministic WAE-style matching.
        kernel_type: Kernel type for MMD computation (see KernelType enum).
        bandwidth_multipliers: Scale factors for bandwidth. When
            use_median_heuristic=True these scale the adaptive median. When False these
            are absolute bandwidth values. WAE recommends [2 * latent_dim] with
            use_median_heuristic=False.
        use_median_heuristic: Adaptive bandwidth via median heuristic (True) or fixed
            absolute bandwidths (False).
        use_fixed_gaussian_as_prior: If True, always use standard Gaussian as prior.
    """

    _target_: str = (
        "versatil.metrics.losses.maximum_mean_discrepancy.MaximumMeanDiscrepancyLoss"
    )
    weight: float = 1.0
    prior_regularization_weight: float = 0.0
    prior_target_key: str = "${latent_key:POSTERIOR_LATENT}"
    kernel_type: str = KernelType.RBF.value
    bandwidth_multipliers: list[float] | None = field(
        default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 5.0]
    )
    use_median_heuristic: bool = True
    use_fixed_gaussian_as_prior: bool = False


@dataclass
class ConditionalMaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for conditional state-latent MMD loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Scalar weight of this loss in the total loss.
        state_weight: Weight of the state kernel in the joint kernel.
        prior_target_key: Metadata key holding prior samples matched against the
            posterior.
        condition_key: Feature key used as the conditioning variable.
        kernel_type: Kernel for the latent term, rbf or imq.
        bandwidth_multipliers: Bandwidth multipliers of the latent kernel mixture.
        use_median_heuristic: Whether the latent bandwidth uses the median heuristic.
        condition_kernel_type: Kernel for the conditioning term, rbf or imq.
        condition_bandwidth_multipliers: Bandwidth multipliers of the conditioning
            kernel mixture.
        condition_use_median_heuristic: Whether the conditioning bandwidth uses the
            median heuristic.
        normalize_condition: Whether the conditioning variable is standardized before
            the kernel.
    """

    _target_: str = "versatil.metrics.losses.maximum_mean_discrepancy.ConditionalMaximumMeanDiscrepancyLoss"
    weight: float = 1.0
    state_weight: float = 1.0
    prior_target_key: str = "${latent_key:POSTERIOR_LATENT}"
    condition_key: str = "${latent_key:PRIOR_CONDITION}"
    kernel_type: str = KernelType.RBF.value
    bandwidth_multipliers: list[float] | None = field(
        default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 5.0]
    )
    use_median_heuristic: bool = True
    condition_kernel_type: str = KernelType.RBF.value
    condition_bandwidth_multipliers: list[float] | None = field(
        default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 5.0]
    )
    condition_use_median_heuristic: bool = True
    normalize_condition: bool = True


@dataclass
class VQCommitmentLossConfig(BaseLossConfig):
    """Configuration for VQ commitment loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        num_codes: Number of codebook entries per residual layer (K). Must match the VQ
            posterior's ResidualVQ configuration.
        num_residual_layers: Number of residual VQ layers. Must match the VQ posterior's
            ResidualVQ configuration.
        weight: Loss weight for the commitment term ||z_continuous -
            sg(z_quantized)||^2.
    """

    _target_: str = "versatil.metrics.losses.vector_quantization.VQCommitmentLoss"
    num_codes: int = MISSING
    num_residual_layers: int = MISSING
    weight: float = 1.0


@dataclass
class VQPriorCrossEntropyLossConfig(BaseLossConfig):
    """Configuration for VQ prior cross-entropy loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Loss weight for the cross-entropy term.
    """

    _target_: str = (
        "versatil.metrics.losses.vector_quantization.VQPriorCrossEntropyLoss"
    )
    weight: float = 1.0


@dataclass
class BinaryMaximumMeanDiscrepancyLossConfig(BaseLossConfig):
    """Configuration for Binary Maximum Mean Discrepancy (MMD) loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Loss weight.
    """

    _target_: str = "versatil.metrics.losses.maximum_mean_discrepancy.BinaryMaximumMeanDiscrepancyLoss"
    weight: float = 1.0


@dataclass
class TrajectoryLengthLossConfig(BaseLossConfig):
    """Configuration for trajectory length loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Weight for length loss.
        action_key: Action key to compute length for.
    """

    _target_: str = "versatil.metrics.losses.trajectory.TrajectoryLengthLoss"
    weight: float = 0.1
    action_key: str = MISSING


@dataclass
class TrajectorySmoothnessConfig(BaseLossConfig):
    """Configuration for trajectory smoothness loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Weight for smoothness loss.
        action_key: Action key to compute smoothness for.
    """

    _target_: str = "versatil.metrics.losses.trajectory.TrajectorySmoothness"
    weight: float = 0.01
    action_key: str = MISSING


@dataclass
class ActionTokenLossConfig(BaseLossConfig):
    """Configuration for action token cross-entropy loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Scalar multiplier applied to the cross-entropy term.
        label_smoothing: Label smoothing factor [0, 1].
    """

    _target_: str = "versatil.metrics.losses.classification.ActionTokenLoss"
    weight: float = 1.0
    label_smoothing: float = 0.2


@dataclass
class PhaseClassificationLossConfig(BaseLossConfig):
    """Configuration for phase classification loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        key: Key for phase labels.
        cross_entropy_weight: Weight for cross-entropy loss.
        entropy_weight: Weight for entropy regularization (Entropy maximization avoids
            experts collapse).
        label_smoothing: Label smoothing factor for cross-entropy.
    """

    _target_: str = "versatil.metrics.losses.classification.PhaseClassificationLoss"
    key: str = MISSING
    cross_entropy_weight: float = 1.0
    entropy_weight: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class GripperMixtureNLLossConfig(BaseLossConfig):
    """Configuration for gripper Mixture Negative Log-Likelihood loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        key: Key for gripper actions.
        actions_metadata: Dict of metadata of the action space.
        weight: Loss weight.
        learned_variance: If True, expects {key}_mean and {key}_logvar for continuous.
            If False, expects {key} (stacked means) and uses sigma.
        sigma: Fixed std for continuous gripper (only used when learned_variance=False).
        min_variance: Minimum variance for numerical stability (learned_variance=True).
    """

    _target_: str = "versatil.metrics.losses.mixture.GripperMixtureNLLoss"
    key: str = MISSING
    actions_metadata: Any = "${task.action_space.actions_metadata}"
    weight: float = 1.0
    learned_variance: bool = False
    sigma: float = 0.5
    min_variance: float = 1e-4


@dataclass
class CompositeLossConfig(BaseLossConfig):
    """Configuration for composite loss with custom modules.

    Attributes:
        _target_: Import path instantiated by Hydra.
        loss_modules: Dictionary of loss module names to loss instances.
        weights: Deprecated legacy composite weights. Kept only for config compatibility
            and ignored at runtime.
    """

    _target_: str = "versatil.metrics.losses.composite.CompositeLoss"
    loss_modules: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] | None = None


@dataclass
class PriorDenoisingLossConfig(BaseLossConfig):
    """Configuration for diffusion prior denoising loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Weight for this loss component.
    """

    _target_: str = "versatil.metrics.losses.prior_denoising.PriorDenoisingLoss"
    weight: float = 1.0


@dataclass
class MoELossConfig:
    """Configuration for Mixture of Experts (MoE) loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        base_loss: Any BaseLoss instance to wrap (e.g., RegressionLoss(...)).
        entropy_weight: Weight for per-example routing entropy. Penalizes peaky-per-
            example routing. Pushes each example's routing distribution toward uniform,
            which prevents one example from being routed to a single expert with
            probability 1.
        load_balance_weight: Weight for Switch-Transformer-style load-balancing term.
            Penalizes batch-level imbalance in expert usage. The term is ``K * sum_k f_k
            * P_k`` where ``f_k`` is the fraction of examples whose argmax routes to
            expert k and ``P_k`` is the mean routing weight for expert k across the
            batch. Minimum value 1.0 is reached when usage is uniform across the batch.
            Crucially, this allows per-example routing to be peaky (so experts can
            specialize) while still forcing every expert to be used by some examples (so
            no expert dies). Use this when entropy alone produces dead experts.
    """

    _target_: str = "versatil.metrics.losses.mixture_of_experts.MoELoss"
    base_loss: BaseLossConfig = MISSING
    entropy_weight: float = 0.0
    load_balance_weight: float = 0.0


@dataclass
class GaussianMixtureNLLossConfig(BaseLossConfig):
    """Configuration for Gaussian Mixture Negative Log-Likelihood loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        action_keys: List of continuous action keys.
        weight: Overall loss weight.
        per_key_weights: Optional per-key weights.
        learned_variance: If True, expects {action_key}_mean and {action_key}_logvar. If
            False, expects {action_key} (stacked means) and uses sigmas. Defaults to
            False.
        sigmas: Fixed stddev per action key (only used when learned_variance=False).
            The loss defaults missing keys to 0.5.
        min_variance: Minimum variance for numerical stability (learned_variance=True).
    """

    _target_: str = "versatil.metrics.losses.mixture.GaussianMixtureNLLoss"
    action_keys: list[str] = MISSING
    weight: float = 1.0
    per_key_weights: dict[str, float] | None = None
    learned_variance: bool = False
    sigmas: dict[str, float] | None = None
    min_variance: float = 1e-4


@dataclass
class VICLatentLossConfig(BaseLossConfig):
    """Configuration for VICReg-style covariance + variance loss.

    Attributes:
        _target_: Import path instantiated by Hydra.
        key: Prediction key for latent mu tensor.
        covariance_weight: Weight for off-diagonal covariance penalty.
        variance_weight: Weight for variance hinge loss.
        gamma: Hinge threshold for per-dimension standard deviation.
    """

    _target_: str = "versatil.metrics.losses.latent_geometry.VICLatentLoss"
    key: str = "${latent_key:POSTERIOR_MU}"
    covariance_weight: float = 3.0
    variance_weight: float = 10.0
    gamma: float = 0.3


@dataclass
class PosteriorGeometryLossConfig(BaseLossConfig):
    """Configuration for posterior latent moment regularization.

    Attributes:
        _target_: Import path instantiated by Hydra.
        key: Prediction key for latent vectors.
        mean_weight: Weight for squared batch-mean penalty.
        std_weight: Weight for squared deviation from ``target_std``.
        target_std: Desired per-dimension posterior standard deviation.
        max_std_weight: Weight for hinge penalty above ``max_std``.
        max_std: Maximum tolerated per-dimension standard deviation.
        covariance_weight: Weight for off-diagonal covariance penalty.
        epsilon: Numerical epsilon for standard deviation.
    """

    _target_: str = "versatil.metrics.losses.latent_geometry.PosteriorGeometryLoss"
    key: str = "${latent_key:POSTERIOR_MU}"
    mean_weight: float = 0.0
    std_weight: float = 0.0
    target_std: float = 1.0
    max_std_weight: float = 0.0
    max_std: float = 2.0
    covariance_weight: float = 0.0
    epsilon: float = 1e-6


@dataclass
class OptimalTransportLossConfig(BaseLossConfig):
    """Configuration for Optimal Transport loss using Sinkhorn divergence.

    Attributes:
        _target_: Import path instantiated by Hydra.
        action_keys: List of keys for action tensors in predictions and targets.
        weight: Scaling factor for the total loss.
        p: Exponent for the ground cost. p=1 gives ``||a - a'||_2``, p=2 gives ``(1/2) *
            ||a - a'||_2^2``.
        blur_fraction: Dimensionless Sinkhorn regularization, expressed as a fraction of
            the reference pairwise scale sqrt(2 * dim) * expected_std. GeomLoss
            recommends ~0.1.
        reach_multiplier: Unbalanced OT scale, as a multiple of the reference pairwise
            scale. ``None`` keeps balanced OT. Typical values for mild outlier tolerance
            are 3.0-10.0.
        expected_std: Expected per-dimension standard deviation of the action samples.
            For actions normalized to [-1, 1], use ~1/sqrt(3) ~ 0.577.
        time_scale: Scaling factor for the linear time embedding concatenated to
            actions. time_scale=0 gives permutation-invariant OT over the horizon.
    """

    _target_: str = "versatil.metrics.losses.optimal_transport.OptimalTransportLoss"
    action_keys: list[str] = MISSING
    weight: float = 1.0
    p: int = 2
    blur_fraction: float = 0.1
    reach_multiplier: float | None = None
    expected_std: float = 1.0
    time_scale: float = 1.0


@dataclass
class LatentOptimalTransportLossConfig(BaseLossConfig):
    """Configuration for latent Sinkhorn divergence between posterior and prior.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Scaling factor for the total loss.
        prior_target_key: Posterior output key used as aggregate prior-matching samples.
            Use ``LatentKey.POSTERIOR_MU`` for deterministic WAE-style matching.
        p: Exponent for the ground cost. p=2 is standard for W_2-style regularization of
            latent distributions.
        blur_fraction: Dimensionless Sinkhorn regularization, as a fraction of the
            reference pairwise scale sqrt(2 * dim).
        reach_multiplier: Unbalanced OT scale, as a multiple of the reference pairwise
            scale. ``None`` keeps balanced OT.
    """

    _target_: str = (
        "versatil.metrics.losses.optimal_transport.LatentOptimalTransportLoss"
    )
    weight: float = 1.0
    prior_target_key: str = "${latent_key:POSTERIOR_LATENT}"
    p: int = 2
    blur_fraction: float = 0.1
    reach_multiplier: float | None = None


@dataclass
class RelaxedConditionalLatentOptimalTransportLossConfig(BaseLossConfig):
    """Configuration for relaxed conditional latent Sinkhorn divergence.

    Attributes:
        _target_: Import path instantiated by Hydra.
        weight: Scalar weight of this loss in the total loss.
        prior_target_key: Metadata key holding prior samples matched against the
            posterior.
        condition_key: Feature key used as the conditioning variable.
        p: Order of the transport cost.
        blur_fraction: Sinkhorn blur as a fraction of the point-cloud diameter.
        reach_multiplier: Unbalanced-transport reach as a multiple of the diameter.
        state_weight: Weight of the state coordinates in the transport cost.
        normalize_condition: Whether the conditioning variable is standardized before
            transport.
    """

    _target_: str = "versatil.metrics.losses.optimal_transport.RelaxedConditionalLatentOptimalTransportLoss"
    weight: float = 1.0
    prior_target_key: str = "${latent_key:POSTERIOR_LATENT}"
    condition_key: str = "${latent_key:PRIOR_CONDITION}"
    p: int = 2
    blur_fraction: float = 0.1
    reach_multiplier: float | None = None
    state_weight: float = 1.0
    normalize_condition: bool = True
