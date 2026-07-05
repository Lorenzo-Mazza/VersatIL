"""Tests for versatil.metrics.losses.composite module."""

import warnings
from collections.abc import Callable
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch

from versatil.metrics.base import BaseLoss, LossOutput
from versatil.metrics.losses.composite import CompositeLoss
from versatil.metrics.losses.regression import RegressionLoss


@pytest.fixture
def mock_loss_factory() -> Callable[..., MagicMock]:
    def factory(
        required_keys: set[str] | None = None,
        loss_value: float = 1.0,
        component_name: str = "comp",
        metadata: dict | None = None,
        weights: dict[str, float] | None = None,
    ) -> MagicMock:
        mock = MagicMock(spec=BaseLoss)
        mock.get_required_keys.return_value = required_keys or set()
        mock.return_value = LossOutput(
            total_loss=torch.tensor(loss_value),
            component_losses={component_name: torch.tensor(loss_value)},
            metadata=metadata or {},
        )
        type(mock).weights = PropertyMock(
            return_value=weights if weights is not None else {"weight": 1.0}
        )
        return mock

    return factory


@pytest.fixture
def composite_loss_factory(
    mock_loss_factory: Callable[..., MagicMock],
) -> Callable[..., CompositeLoss]:
    def factory(
        loss_configs: list[tuple[str, float, str]] | None = None,
    ) -> CompositeLoss:
        if loss_configs is None:
            loss_configs = [
                ("loss_a", 2.0, "comp_a"),
                ("loss_b", 3.0, "comp_b"),
            ]
        modules = {}
        for name, value, component_name in loss_configs:
            modules[name] = mock_loss_factory(
                loss_value=value,
                component_name=component_name,
            )
        return CompositeLoss(loss_modules=modules)

    return factory


@pytest.fixture
def dummy_predictions() -> dict[str, torch.Tensor]:
    return {"dummy": torch.tensor([1.0])}


@pytest.fixture
def dummy_targets() -> dict[str, torch.Tensor]:
    return {"dummy": torch.tensor([1.0])}


class _ParametricLoss(BaseLoss):
    """Minimal parametric BaseLoss for composite-parameter tests."""

    def __init__(self, input_dimension: int = 3, output_dim: int = 1) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(input_dimension, output_dim, bias=False)

    def get_required_keys(self) -> set[str]:
        return {"input"}

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        output = self.projection(predictions["input"]).mean()
        return LossOutput(
            total_loss=output,
            component_losses={"param_loss": output},
        )


@pytest.fixture
def parametric_loss_factory() -> Callable[..., _ParametricLoss]:
    def factory(input_dimension: int = 3, output_dim: int = 1) -> _ParametricLoss:
        return _ParametricLoss(input_dimension=input_dimension, output_dim=output_dim)

    return factory


@pytest.mark.unit
class TestCompositeLossWeights:
    def test_weights_returns_each_child_weight_tree_keyed_by_name(
        self,
        mock_loss_factory: Callable[..., MagicMock],
    ):
        modules = {
            "a": mock_loss_factory(component_name="a", weights={"weight": 0.3}),
            "b": mock_loss_factory(
                component_name="b",
                weights={"mse_weight": 1.0, "l1_weight": 0.5},
            ),
        }
        composite = CompositeLoss(loss_modules=modules)
        assert composite.weights == {
            "a": {"weight": 0.3},
            "b": {"mse_weight": 1.0, "l1_weight": 0.5},
        }

    def test_set_weights_delegates_to_each_child(
        self,
        mock_loss_factory: Callable[..., MagicMock],
    ):
        child_a = mock_loss_factory(component_name="a", weights={"weight": 0.1})
        child_b = mock_loss_factory(component_name="b", weights={"weight": 0.2})
        composite = CompositeLoss(loss_modules={"a": child_a, "b": child_b})

        composite.set_weights({"a": {"weight": 0.7}, "b": {"weight": 0.9}})

        child_a.set_weights.assert_called_once_with({"weight": 0.7})
        child_b.set_weights.assert_called_once_with({"weight": 0.9})

    def test_update_weights_applies_partial_override_to_single_child(
        self,
        mock_loss_factory: Callable[..., MagicMock],
    ):
        child_a = mock_loss_factory(component_name="a", weights={"weight": 0.3})
        child_b = mock_loss_factory(
            component_name="b",
            weights={"mse_weight": 1.0, "l1_weight": 0.5},
        )
        composite = CompositeLoss(loss_modules={"a": child_a, "b": child_b})

        composite.update_weights({"b": {"mse_weight": 0.2}})

        # After the merge, set_weights is called with the fully merged tree.
        child_a.set_weights.assert_called_once_with({"weight": 0.3})
        child_b.set_weights.assert_called_once_with(
            {"mse_weight": 0.2, "l1_weight": 0.5}
        )

    def test_update_weights_unknown_key_raises(
        self,
        mock_loss_factory: Callable[..., MagicMock],
    ):
        child = mock_loss_factory(component_name="a", weights={"weight": 0.3})
        composite = CompositeLoss(loss_modules={"a": child})

        with pytest.raises(KeyError, match="Unknown weight key 'bogus'"):
            composite.update_weights({"bogus": {}})


@pytest.mark.unit
class TestCompositeLossGetRequiredKeys:
    def test_collects_union_of_all_sub_loss_keys(
        self,
        mock_loss_factory: Callable[..., MagicMock],
    ):
        loss_a = mock_loss_factory(required_keys={"position", "orientation"})
        loss_b = mock_loss_factory(required_keys={"gripper", "position"})
        composite = CompositeLoss(loss_modules={"a": loss_a, "b": loss_b})
        assert composite.get_required_keys() == {"position", "orientation", "gripper"}

    def test_returns_empty_set_when_all_sub_losses_have_no_keys(
        self,
        mock_loss_factory: Callable[..., MagicMock],
    ):
        composite = CompositeLoss(
            loss_modules={
                "a": mock_loss_factory(required_keys=set()),
                "b": mock_loss_factory(required_keys=set()),
            }
        )
        assert composite.get_required_keys() == set()


@pytest.mark.unit
class TestCompositeLossForward:
    def test_total_loss_is_plain_sum_of_sub_losses(
        self,
        composite_loss_factory: Callable[..., CompositeLoss],
        dummy_predictions: dict[str, torch.Tensor],
        dummy_targets: dict[str, torch.Tensor],
    ):
        composite = composite_loss_factory(
            loss_configs=[
                ("loss_a", 2.0, "comp_a"),
                ("loss_b", 3.0, "comp_b"),
            ],
        )
        output = composite(dummy_predictions, dummy_targets)
        # total = 2.0 + 3.0 = 5.0 (each sub-loss applies its own weight internally)
        assert output.total_loss.item() == pytest.approx(5.0)

    def test_component_losses_are_prefixed_with_module_name(
        self,
        composite_loss_factory: Callable[..., CompositeLoss],
        dummy_predictions: dict[str, torch.Tensor],
        dummy_targets: dict[str, torch.Tensor],
    ):
        composite = composite_loss_factory(
            loss_configs=[
                ("regression", 1.0, "mse"),
                ("gripper", 2.0, "bce"),
            ],
        )
        output = composite(dummy_predictions, dummy_targets)
        assert "regression/mse" in output.component_losses
        assert "gripper/bce" in output.component_losses
        assert output.component_losses["regression/mse"].item() == pytest.approx(1.0)
        assert output.component_losses["gripper/bce"].item() == pytest.approx(2.0)

    def test_metadata_from_all_sub_losses_is_collected(
        self,
        mock_loss_factory: Callable[..., MagicMock],
        dummy_predictions: dict[str, torch.Tensor],
        dummy_targets: dict[str, torch.Tensor],
    ):
        loss_a = mock_loss_factory(metadata={"key_a": "val_a"})
        loss_b = mock_loss_factory(metadata={"key_b": "val_b"})
        composite = CompositeLoss(loss_modules={"a": loss_a, "b": loss_b})
        output = composite(dummy_predictions, dummy_targets)
        assert output.metadata["key_a"] == "val_a"
        assert output.metadata["key_b"] == "val_b"

    def test_each_sub_loss_receives_predictions_and_targets(
        self,
        mock_loss_factory: Callable[..., MagicMock],
        dummy_predictions: dict[str, torch.Tensor],
        dummy_targets: dict[str, torch.Tensor],
    ):
        loss_a = mock_loss_factory(component_name="a")
        loss_b = mock_loss_factory(component_name="b")
        composite = CompositeLoss(loss_modules={"a": loss_a, "b": loss_b})
        composite(dummy_predictions, dummy_targets)
        loss_a.assert_called_once_with(dummy_predictions, dummy_targets, None)
        loss_b.assert_called_once_with(dummy_predictions, dummy_targets, None)

    def test_is_pad_forwarded_to_sub_losses(
        self,
        mock_loss_factory: Callable[..., MagicMock],
        dummy_predictions: dict[str, torch.Tensor],
        dummy_targets: dict[str, torch.Tensor],
    ):
        is_pad = torch.tensor([False, True])
        loss_a = mock_loss_factory(component_name="a")
        composite = CompositeLoss(loss_modules={"a": loss_a})
        composite(dummy_predictions, dummy_targets, is_pad=is_pad)
        loss_a.assert_called_once_with(dummy_predictions, dummy_targets, is_pad)


@pytest.mark.unit
class TestCompositeLossParameters:
    def test_parameters_includes_all_sub_loss_parameters(self):
        loss_a = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        loss_b = RegressionLoss(action_keys=["orientation"], mse_weight=1.0)
        composite = CompositeLoss(loss_modules={"a": loss_a, "b": loss_b})

        composite_params = list(composite.parameters())
        loss_a_params = list(loss_a.parameters())
        loss_b_params = list(loss_b.parameters())
        # RegressionLoss has no trainable parameters, but ModuleDict
        # should still work. Test with a parametric sub-loss:
        assert len(composite_params) == len(loss_a_params) + len(loss_b_params)

    def test_parameters_exposes_trainable_sub_loss_parameters(
        self,
        parametric_loss_factory: Callable[..., _ParametricLoss],
    ):
        parametric = parametric_loss_factory(input_dimension=4, output_dim=2)
        composite = CompositeLoss(loss_modules={"param": parametric})

        composite_parameters = list(composite.parameters())

        assert parametric.projection.weight in composite_parameters
        assert all(parameter.requires_grad for parameter in composite_parameters)

    def test_gradient_flows_through_composite_to_sub_loss(
        self,
        parametric_loss_factory: Callable[..., _ParametricLoss],
        rng: np.random.Generator,
    ):
        parametric = parametric_loss_factory(input_dimension=3, output_dim=1)
        composite = CompositeLoss(loss_modules={"param": parametric})

        input_data = rng.standard_normal((2, 3)).astype(np.float32)
        predictions = {"input": torch.from_numpy(input_data)}
        output = composite(predictions, {})
        output.total_loss.backward()

        assert parametric.projection.weight.grad is not None


@pytest.mark.unit
class TestCompositeLossLegacyWeightsKwarg:
    def test_all_ones_weights_do_not_emit_warning(
        self,
        mock_loss_factory: Callable[..., MagicMock],
    ) -> None:
        modules = {"a": mock_loss_factory(component_name="a")}
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            CompositeLoss(loss_modules=modules, weights={"a": 1.0})

    def test_non_one_weights_emit_deprecation_warning(
        self,
        mock_loss_factory: Callable[..., MagicMock],
    ) -> None:
        modules = {"a": mock_loss_factory(component_name="a")}
        with pytest.warns(
            DeprecationWarning,
            match="CompositeLoss.weights is deprecated and ignored at runtime",
        ):
            CompositeLoss(loss_modules=modules, weights={"a": 0.5})
