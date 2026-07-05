"""Tests for the BaseLoss weights API across versatil.metrics.losses modules."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.metrics.base import BaseLoss
from versatil.metrics.losses.classification import (
    ActionTokenLoss,
    PhaseClassificationLoss,
)
from versatil.metrics.losses.composite import CompositeLoss
from versatil.metrics.losses.divergence import (
    BinaryKLDivergenceLoss,
    GaussianEntropyLoss,
    KLDivergenceLoss,
)
from versatil.metrics.losses.gripper import GripperLoss
from versatil.metrics.losses.latent_geometry import (
    PosteriorGeometryLoss,
    VICLatentLoss,
)
from versatil.metrics.losses.maximum_mean_discrepancy import (
    MaximumMeanDiscrepancyLoss,
)
from versatil.metrics.losses.mixture_of_experts import MoELoss
from versatil.metrics.losses.prior_denoising import PriorDenoisingLoss
from versatil.metrics.losses.regression import RegressionLoss
from versatil.metrics.losses.trajectory import (
    TrajectoryLengthLoss,
)
from versatil.metrics.losses.vector_quantization import (
    VQCommitmentLoss,
    VQPriorCrossEntropyLoss,
)
from versatil.models.decoding.constants import DecoderOutputKey


@pytest.fixture
def leaf_weight_spec_factory(
    binary_gripper_metadata_factory: Callable[..., dict],
) -> Callable[..., dict[str, Any]]:
    """Factory fixture: one `(loss, initial, set_to, partial, expected_after_partial)` per name."""

    def factory(name: str) -> dict[str, Any]:
        match name:
            case "regression":
                return {
                    "loss": RegressionLoss(
                        action_keys=["a"],
                        mse_weight=1.0,
                        l1_weight=0.5,
                        huber_weight=0.25,
                    ),
                    "initial_weights": {
                        "mse_weight": 1.0,
                        "l1_weight": 0.5,
                        "huber_weight": 0.25,
                    },
                    "set_to": {
                        "mse_weight": 0.2,
                        "l1_weight": 0.3,
                        "huber_weight": 0.4,
                    },
                    "partial_update": {"l1_weight": 0.1},
                    "expected_after_partial": {
                        "mse_weight": 1.0,
                        "l1_weight": 0.1,
                        "huber_weight": 0.25,
                    },
                }
            case "gripper":
                return {
                    "loss": GripperLoss(
                        key="gripper",
                        actions_metadata=binary_gripper_metadata_factory(),
                        bce_weight=0.01,
                        mse_weight=0.0,
                    ),
                    "initial_weights": {"bce_weight": 0.01, "mse_weight": 0.0},
                    "set_to": {"bce_weight": 0.2, "mse_weight": 0.3},
                    "partial_update": {"mse_weight": 0.8},
                    "expected_after_partial": {"bce_weight": 0.01, "mse_weight": 0.8},
                }
            case "kl_divergence":
                return {
                    "loss": KLDivergenceLoss(
                        weight=10.0,
                        prior_regularization_weight=0.2,
                    ),
                    "initial_weights": {
                        "weight": 10.0,
                        "prior_regularization_weight": 0.2,
                    },
                    "set_to": {
                        "weight": 1.0,
                        "prior_regularization_weight": 0.9,
                    },
                    "partial_update": {"prior_regularization_weight": 0.9},
                    "expected_after_partial": {
                        "weight": 10.0,
                        "prior_regularization_weight": 0.9,
                    },
                }
            case "binary_kl_divergence":
                return {
                    "loss": BinaryKLDivergenceLoss(weight=5.0, entropy_weight=0.001),
                    "initial_weights": {"weight": 5.0, "entropy_weight": 0.001},
                    "set_to": {"weight": 1.0, "entropy_weight": 0.0},
                    "partial_update": {"entropy_weight": 0.5},
                    "expected_after_partial": {"weight": 5.0, "entropy_weight": 0.5},
                }
            case "gaussian_entropy":
                return {
                    "loss": GaussianEntropyLoss(weight=0.02, bound_weight=0.5),
                    "initial_weights": {"weight": 0.02, "bound_weight": 0.5},
                    "set_to": {"weight": 1.0, "bound_weight": 2.0},
                    "partial_update": {"bound_weight": 0.0},
                    "expected_after_partial": {"weight": 0.02, "bound_weight": 0.0},
                }
            case "maximum_mean_discrepancy":
                return {
                    "loss": MaximumMeanDiscrepancyLoss(
                        weight=1.0, prior_regularization_weight=0.2
                    ),
                    "initial_weights": {
                        "weight": 1.0,
                        "prior_regularization_weight": 0.2,
                    },
                    "set_to": {"weight": 3.0, "prior_regularization_weight": 0.05},
                    "partial_update": {"prior_regularization_weight": 0.9},
                    "expected_after_partial": {
                        "weight": 1.0,
                        "prior_regularization_weight": 0.9,
                    },
                }
            case "phase_classification":
                return {
                    "loss": PhaseClassificationLoss(
                        key="phase",
                        cross_entropy_weight=0.1,
                        entropy_weight=0.05,
                    ),
                    "initial_weights": {
                        "cross_entropy_weight": 0.1,
                        "entropy_weight": 0.05,
                    },
                    "set_to": {"cross_entropy_weight": 0.3, "entropy_weight": 0.0},
                    "partial_update": {"entropy_weight": 0.2},
                    "expected_after_partial": {
                        "cross_entropy_weight": 0.1,
                        "entropy_weight": 0.2,
                    },
                }
            case "vic_latent":
                return {
                    "loss": VICLatentLoss(covariance_weight=3.0, variance_weight=10.0),
                    "initial_weights": {
                        "covariance_weight": 3.0,
                        "variance_weight": 10.0,
                    },
                    "set_to": {"covariance_weight": 1.0, "variance_weight": 2.0},
                    "partial_update": {"variance_weight": 5.0},
                    "expected_after_partial": {
                        "covariance_weight": 3.0,
                        "variance_weight": 5.0,
                    },
                }
            case "posterior_geometry":
                return {
                    "loss": PosteriorGeometryLoss(
                        mean_weight=0.1,
                        std_weight=0.2,
                        max_std_weight=0.3,
                        covariance_weight=0.4,
                    ),
                    "initial_weights": {
                        "mean_weight": 0.1,
                        "std_weight": 0.2,
                        "max_std_weight": 0.3,
                        "covariance_weight": 0.4,
                    },
                    "set_to": {
                        "mean_weight": 0.5,
                        "std_weight": 0.6,
                        "max_std_weight": 0.7,
                        "covariance_weight": 0.8,
                    },
                    "partial_update": {"max_std_weight": 0.9},
                    "expected_after_partial": {
                        "mean_weight": 0.1,
                        "std_weight": 0.2,
                        "max_std_weight": 0.9,
                        "covariance_weight": 0.4,
                    },
                }
            case "prior_denoising":
                return {
                    "loss": PriorDenoisingLoss(weight=0.03),
                    "initial_weights": {"weight": 0.03},
                    "set_to": {"weight": 0.5},
                    "partial_update": {"weight": 0.9},
                    "expected_after_partial": {"weight": 0.9},
                }
        raise ValueError(f"Unknown leaf_weight_spec_factory name: {name}")

    return factory


@pytest.mark.unit
class TestLossWeightsAPI:
    @pytest.mark.parametrize(
        "leaf_name",
        [
            "regression",
            "gripper",
            "kl_divergence",
            "binary_kl_divergence",
            "gaussian_entropy",
            "maximum_mean_discrepancy",
            "phase_classification",
            "vic_latent",
            "posterior_geometry",
            "prior_denoising",
        ],
    )
    def test_weights_returns_initial_values(
        self,
        leaf_weight_spec_factory: Callable[..., dict[str, Any]],
        leaf_name: str,
    ) -> None:
        spec = leaf_weight_spec_factory(leaf_name)
        assert spec["loss"].weights == spec["initial_weights"]

    @pytest.mark.parametrize(
        "leaf_name",
        [
            "regression",
            "gripper",
            "kl_divergence",
            "binary_kl_divergence",
            "gaussian_entropy",
            "maximum_mean_discrepancy",
            "phase_classification",
            "vic_latent",
            "posterior_geometry",
            "prior_denoising",
        ],
    )
    def test_set_weights_replaces_full_tree(
        self,
        leaf_weight_spec_factory: Callable[..., dict[str, Any]],
        leaf_name: str,
    ) -> None:
        spec = leaf_weight_spec_factory(leaf_name)
        loss = spec["loss"]
        loss.set_weights(spec["set_to"])
        assert loss.weights == spec["set_to"]

    @pytest.mark.parametrize(
        "leaf_name",
        [
            "regression",
            "gripper",
            "kl_divergence",
            "binary_kl_divergence",
            "gaussian_entropy",
            "maximum_mean_discrepancy",
            "phase_classification",
            "vic_latent",
            "posterior_geometry",
            "prior_denoising",
        ],
    )
    def test_update_weights_applies_partial_override(
        self,
        leaf_weight_spec_factory: Callable[..., dict[str, Any]],
        leaf_name: str,
    ) -> None:
        spec = leaf_weight_spec_factory(leaf_name)
        loss = spec["loss"]
        loss.update_weights(spec["partial_update"])
        assert loss.weights == spec["expected_after_partial"]

    def test_action_token_loss_weight_scales_forward_output(
        self, rng: np.random.Generator
    ) -> None:
        batch_size, horizon, vocab_size = 2, 4, 5
        logits = torch.from_numpy(
            rng.standard_normal((batch_size, horizon, vocab_size)).astype(np.float32)
        )
        targets = torch.from_numpy(rng.integers(0, vocab_size, (batch_size, horizon)))
        predictions = {DecoderOutputKey.ACTION_LOGITS.value: logits}
        target_dict = {SampleKey.TOKENIZED_ACTIONS.value: targets}

        output_unit = ActionTokenLoss(weight=1.0, label_smoothing=0.0)(
            predictions, target_dict
        )
        output_half = ActionTokenLoss(weight=0.5, label_smoothing=0.0)(
            predictions, target_dict
        )
        assert output_half.total_loss.item() == pytest.approx(
            output_unit.total_loss.item() * 0.5, rel=1e-5
        )

    def test_moe_loss_weights_includes_base_loss_subtree(self) -> None:
        base = RegressionLoss(
            action_keys=["a"], mse_weight=1.0, l1_weight=0.5, huber_weight=0.25
        )
        loss = MoELoss(base_loss=base, entropy_weight=0.01, load_balance_weight=0.2)
        assert loss.weights == {
            "entropy_weight": 0.01,
            "load_balance_weight": 0.2,
            "base_loss": {
                "mse_weight": 1.0,
                "l1_weight": 0.5,
                "huber_weight": 0.25,
            },
        }

    def test_moe_loss_set_weights_delegates_to_base_loss(self) -> None:
        base = RegressionLoss(
            action_keys=["a"], mse_weight=1.0, l1_weight=0.5, huber_weight=0.25
        )
        loss = MoELoss(base_loss=base, entropy_weight=0.01, load_balance_weight=0.2)
        loss.set_weights(
            {
                "entropy_weight": 0.05,
                "load_balance_weight": 0.3,
                "base_loss": {
                    "mse_weight": 0.0,
                    "l1_weight": 1.0,
                    "huber_weight": 0.0,
                },
            }
        )
        assert loss.entropy_weight == pytest.approx(0.05)
        assert loss.load_balance_weight == pytest.approx(0.3)
        assert base.mse_weight == pytest.approx(0.0)
        assert base.l1_weight == pytest.approx(1.0)

    def test_moe_loss_update_weights_targets_nested_leaf(self) -> None:
        base = RegressionLoss(
            action_keys=["a"], mse_weight=1.0, l1_weight=0.5, huber_weight=0.25
        )
        loss = MoELoss(base_loss=base, entropy_weight=0.01, load_balance_weight=0.2)
        loss.update_weights({"base_loss": {"l1_weight": 0.9}})
        assert base.l1_weight == pytest.approx(0.9)
        assert base.mse_weight == pytest.approx(1.0)
        assert loss.entropy_weight == pytest.approx(0.01)


@pytest.fixture(
    params=[
        "prior_denoising",
        "trajectory_length",
        "action_token",
        "vq_commitment",
        "vq_prior_ce",
        "regression",
        "gripper",
        "phase_classification",
        "kl_divergence",
        "binary_kl_divergence",
        "gaussian_entropy",
        "maximum_mean_discrepancy",
        "vic_latent",
        "posterior_geometry",
    ]
)
def leaf_loss_case(
    request: pytest.FixtureRequest,
    binary_gripper_metadata_factory: Callable[..., dict],
) -> tuple[BaseLoss, set[str]]:
    """Factory fixture: one ``(leaf_loss, expected_weight_keys)`` pair per param id."""
    match request.param:
        case "prior_denoising":
            return PriorDenoisingLoss(weight=0.5), {"weight"}
        case "trajectory_length":
            return TrajectoryLengthLoss(action_key="action"), {"weight"}
        case "action_token":
            return ActionTokenLoss(), {"weight"}
        case "vq_commitment":
            return (
                VQCommitmentLoss(num_codes=4, num_residual_layers=1),
                {"weight"},
            )
        case "vq_prior_ce":
            return VQPriorCrossEntropyLoss(), {"weight"}
        case "regression":
            return (
                RegressionLoss(action_keys=["action"]),
                {"mse_weight", "l1_weight", "huber_weight"},
            )
        case "gripper":
            return (
                GripperLoss(
                    key="gripper",
                    actions_metadata=binary_gripper_metadata_factory(),
                ),
                {"bce_weight", "mse_weight"},
            )
        case "phase_classification":
            return (
                PhaseClassificationLoss(key="phase"),
                {"cross_entropy_weight", "entropy_weight"},
            )
        case "kl_divergence":
            return (
                KLDivergenceLoss(),
                {"weight", "prior_regularization_weight"},
            )
        case "binary_kl_divergence":
            return BinaryKLDivergenceLoss(), {"weight", "entropy_weight"}
        case "gaussian_entropy":
            return GaussianEntropyLoss(), {"weight", "bound_weight"}
        case "maximum_mean_discrepancy":
            return (
                MaximumMeanDiscrepancyLoss(),
                {"weight", "prior_regularization_weight"},
            )
        case "vic_latent":
            return VICLatentLoss(), {"covariance_weight", "variance_weight"}
        case "posterior_geometry":
            return (
                PosteriorGeometryLoss(),
                {
                    "mean_weight",
                    "std_weight",
                    "max_std_weight",
                    "covariance_weight",
                },
            )
    raise ValueError(f"Unknown leaf_loss_case param: {request.param}")


@pytest.mark.unit
class TestSetWeightsValidation:
    def test_leaf_set_weights_rejects_missing_key(
        self, leaf_loss_case: tuple[BaseLoss, set[str]]
    ) -> None:
        loss, expected_keys = leaf_loss_case
        missing_key = sorted(expected_keys)[0]
        partial = {key: 0.1 for key in expected_keys if key != missing_key}
        with pytest.raises(
            KeyError,
            match=(
                f"{type(loss).__name__}.set_weights: "
                rf"missing=\['{missing_key}'\]"
            ),
        ):
            loss.set_weights(partial)

    def test_leaf_set_weights_rejects_extra_key(
        self, leaf_loss_case: tuple[BaseLoss, set[str]]
    ) -> None:
        loss, expected_keys = leaf_loss_case
        extra = dict.fromkeys(expected_keys, 0.1)
        extra["bogus_weight"] = 0.1
        with pytest.raises(
            KeyError,
            match=(
                f"{type(loss).__name__}.set_weights: "
                r"missing=\[\], extra=\['bogus_weight'\]"
            ),
        ):
            loss.set_weights(extra)

    def test_composite_set_weights_rejects_missing_child(self) -> None:
        composite = CompositeLoss(
            loss_modules={
                "a": PriorDenoisingLoss(weight=0.1),
                "b": PriorDenoisingLoss(weight=0.2),
            }
        )
        with pytest.raises(
            KeyError,
            match=r"CompositeLoss\.set_weights: missing=\['b'\]",
        ):
            composite.set_weights({"a": {"weight": 0.5}})

    def test_composite_set_weights_rejects_extra_child(self) -> None:
        composite = CompositeLoss(loss_modules={"a": PriorDenoisingLoss(weight=0.1)})
        with pytest.raises(
            KeyError,
            match=r"CompositeLoss\.set_weights: missing=\[\], extra=\['bogus'\]",
        ):
            composite.set_weights({"a": {"weight": 0.5}, "bogus": {"weight": 0.5}})

    def test_moe_set_weights_rejects_missing_base_loss(
        self,
        binary_gripper_metadata_factory: Callable[..., dict],
    ) -> None:
        inner = PriorDenoisingLoss(weight=0.1)
        moe = MoELoss(base_loss=inner)
        with pytest.raises(
            KeyError,
            match=r"MoELoss\.set_weights: missing=\['base_loss'\]",
        ):
            moe.set_weights({"entropy_weight": 0.0, "load_balance_weight": 0.0})

    def test_moe_set_weights_rejects_extra_key(self) -> None:
        inner = PriorDenoisingLoss(weight=0.1)
        moe = MoELoss(base_loss=inner)
        with pytest.raises(
            KeyError,
            match=r"MoELoss\.set_weights: missing=\[\], extra=\['bogus'\]",
        ):
            moe.set_weights(
                {
                    "entropy_weight": 0.0,
                    "load_balance_weight": 0.0,
                    "base_loss": {"weight": 0.3},
                    "bogus": 1.0,
                }
            )

    def test_update_weights_rejects_unknown_nested_key(self) -> None:
        composite = CompositeLoss(
            loss_modules={
                "regression": RegressionLoss(action_keys=["action"]),
            }
        )
        with pytest.raises(KeyError, match="Unknown weight key 'bogus'"):
            composite.update_weights({"regression": {"bogus": 0.1}})

    def test_update_weights_rejects_dict_for_scalar_leaf(self) -> None:
        composite = CompositeLoss(
            loss_modules={"denoising": PriorDenoisingLoss(weight=0.5)}
        )
        with pytest.raises(
            TypeError,
            match="Weight override for 'weight' expects a scalar",
        ):
            composite.update_weights({"denoising": {"weight": {"nested": 0.1}}})

    def test_update_weights_rejects_scalar_for_dict_subtree(self) -> None:
        composite = CompositeLoss(
            loss_modules={
                "regression": RegressionLoss(action_keys=["action"]),
            }
        )
        with pytest.raises(
            TypeError,
            match="Weight override for 'regression' expects a dict subtree",
        ):
            composite.update_weights({"regression": 0.5})
