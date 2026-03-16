"""Tests for versatil.metrics.composite module."""
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.metrics.base import BaseLoss, LossOutput
from versatil.metrics.composite import CompositeLoss


@pytest.fixture
def mock_loss_factory() -> Callable[..., MagicMock]:
    def factory(
        required_keys: set[str] | None = None,
        loss_value: float = 1.0,
        component_name: str = "comp",
        metadata: dict | None = None,
    ) -> MagicMock:
        mock = MagicMock(spec=BaseLoss)
        mock.get_required_keys.return_value = required_keys or set()
        mock.return_value = LossOutput(
            total_loss=torch.tensor(loss_value),
            component_losses={component_name: torch.tensor(loss_value)},
            metadata=metadata or {},
        )
        return mock

    return factory


@pytest.fixture
def composite_loss_factory(
    mock_loss_factory: Callable[..., MagicMock],
) -> Callable[..., CompositeLoss]:
    def factory(
        loss_configs: list[tuple[str, float, str]] | None = None,
        weights: dict[str, float] | None = None,
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
        return CompositeLoss(loss_modules=modules, weights=weights)

    return factory


@pytest.fixture
def dummy_predictions() -> dict[str, torch.Tensor]:
    return {"dummy": torch.tensor([1.0])}


@pytest.fixture
def dummy_targets() -> dict[str, torch.Tensor]:
    return {"dummy": torch.tensor([1.0])}


@pytest.mark.unit
class TestCompositeLossInitialization:

    def test_default_weights_are_all_ones(
        self,
        mock_loss_factory: Callable[..., MagicMock],
    ):
        modules = {
            "a": mock_loss_factory(component_name="a"),
            "b": mock_loss_factory(component_name="b"),
        }
        composite = CompositeLoss(loss_modules=modules)
        assert composite.weights == {"a": 1.0, "b": 1.0}

    def test_explicit_weights_are_stored(
        self,
        mock_loss_factory: Callable[..., MagicMock],
    ):
        modules = {
            "a": mock_loss_factory(component_name="a"),
            "b": mock_loss_factory(component_name="b"),
        }
        weights = {"a": 0.5, "b": 2.0}
        composite = CompositeLoss(loss_modules=modules, weights=weights)
        assert composite.weights == {"a": 0.5, "b": 2.0}


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

    def test_total_loss_is_weighted_sum_of_sub_losses(
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
            weights={"loss_a": 0.5, "loss_b": 2.0},
        )
        output = composite(dummy_predictions, dummy_targets)
        # total = 0.5 * 2.0 + 2.0 * 3.0 = 1.0 + 6.0 = 7.0
        assert output.total_loss.item() == pytest.approx(7.0)

    def test_default_weights_produce_unweighted_sum(
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
            weights=None,
        )
        output = composite(dummy_predictions, dummy_targets)
        # total = 1.0 * 2.0 + 1.0 * 3.0 = 5.0
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

    def test_missing_weight_defaults_to_one(
        self,
        mock_loss_factory: Callable[..., MagicMock],
        dummy_predictions: dict[str, torch.Tensor],
        dummy_targets: dict[str, torch.Tensor],
    ):
        composite = CompositeLoss(
            loss_modules={
                "present": mock_loss_factory(loss_value=4.0, component_name="p"),
                "absent": mock_loss_factory(loss_value=5.0, component_name="a"),
            },
            weights={"present": 2.0},
        )
        output = composite(dummy_predictions, dummy_targets)
        # total = 2.0 * 4.0 + 1.0 * 5.0 = 13.0
        assert output.total_loss.item() == pytest.approx(13.0)

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
