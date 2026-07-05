"""Tests for versatil.metrics.losses.gripper module."""

import re

import pytest
import torch
import torch.nn.functional as F

from versatil.data.constants import BinaryGripperRange, GripperType
from versatil.data.metadata import (
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
)
from versatil.metrics.constants import MetricKey
from versatil.metrics.losses.gripper import GripperLoss


@pytest.mark.unit
class TestGripperLossInit:
    def test_binary_gripper_stores_type(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata)
        assert loss.gripper_type == GripperType.BINARY.value
        assert loss.binary_gripper_range == BinaryGripperRange.ZERO_ONE.value

    def test_continuous_gripper_stores_type(self, continuous_gripper_metadata_factory):
        metadata = continuous_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata)
        assert loss.gripper_type == GripperType.CONTINUOUS.value

    def test_raises_on_missing_key(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        with pytest.raises(
            ValueError,
            match=re.escape("wrong_key is not available to the action space"),
        ):
            GripperLoss(key="wrong_key", actions_metadata=metadata)

    def test_float_pos_weight_is_converted_to_buffer_tensor(
        self, binary_gripper_metadata_factory
    ):
        metadata = binary_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata, pos_weight=2.0)
        assert isinstance(loss.pos_weight, torch.Tensor)
        assert loss.pos_weight.item() == pytest.approx(2.0)

    def test_on_the_fly_metadata_extracts_gripper_type(self):
        source = GripperObservationMetadata(
            raw_data_column_keys=["gripper"],
            dimension=1,
            dtype="int32",
            needs_normalization=False,
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value,
        )
        on_the_fly = OnTheFlyActionMetadata(source_metadata=source)
        loss = GripperLoss(key="gripper", actions_metadata={"gripper": on_the_fly})
        assert loss.gripper_type == GripperType.BINARY.value
        assert loss.binary_gripper_range == BinaryGripperRange.MINUS_ONE_ONE.value


@pytest.mark.unit
class TestGripperLossRequiresActionSpaceTargets:
    def test_true_when_bce_weight_positive(self, binary_gripper_metadata_factory):
        loss = GripperLoss(
            key="gripper",
            actions_metadata=binary_gripper_metadata_factory(),
            bce_weight=0.05,
        )
        assert loss.requires_action_space_targets is True

    def test_false_when_bce_weight_zero(self, binary_gripper_metadata_factory):
        loss = GripperLoss(
            key="gripper",
            actions_metadata=binary_gripper_metadata_factory(),
            bce_weight=0.0,
            mse_weight=1.0,
        )
        assert loss.requires_action_space_targets is False


class TestGripperLossGetRequiredKeys:
    def test_returns_gripper_key(self, binary_gripper_metadata_factory):
        loss = GripperLoss(
            key="gripper", actions_metadata=binary_gripper_metadata_factory()
        )
        assert loss.get_required_keys() == {"gripper"}


@pytest.mark.unit
class TestGripperLossForward:
    def test_binary_gripper_computes_bce(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata, bce_weight=1.0)
        predictions = {"gripper": torch.tensor([[[0.0]]])}
        targets = {"gripper": torch.tensor([[[1.0]]])}
        output = loss(predictions, targets)
        expected_bce = F.binary_cross_entropy_with_logits(
            predictions["gripper"],
            targets["gripper"].float(),
        )
        assert output.total_loss.item() == pytest.approx(expected_bce.item())
        assert MetricKey.GRIPPER_BCE.value in output.component_losses

    def test_binary_gripper_minus_one_one_range_normalizes_targets(
        self, binary_gripper_metadata_factory
    ):
        metadata = binary_gripper_metadata_factory(
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value
        )
        loss = GripperLoss(key="gripper", actions_metadata=metadata, bce_weight=1.0)
        predictions = {"gripper": torch.tensor([[[0.0]]])}
        targets = {"gripper": torch.tensor([[[-1.0]]])}
        output = loss(predictions, targets)
        # -1 should be mapped to 0.0: (-1 + 1) / 2 = 0
        expected_bce = F.binary_cross_entropy_with_logits(
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
        )
        assert output.total_loss.item() == pytest.approx(expected_bce.item())

    def test_continuous_gripper_computes_mse(self, continuous_gripper_metadata_factory):
        metadata = continuous_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata, mse_weight=1.0)
        predictions = {"gripper": torch.tensor([[[0.5]]])}
        targets = {"gripper": torch.tensor([[[1.0]]])}
        output = loss(predictions, targets)
        expected_mse = F.mse_loss(predictions["gripper"], targets["gripper"])
        assert output.total_loss.item() == pytest.approx(expected_mse.item())
        assert MetricKey.GRIPPER_MSE.value in output.component_losses

    def test_raises_on_missing_key(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions and targets must contain key 'gripper' for GripperLoss."
            ),
        ):
            loss({"wrong": torch.tensor(1.0)}, {"wrong": torch.tensor(1.0)})
