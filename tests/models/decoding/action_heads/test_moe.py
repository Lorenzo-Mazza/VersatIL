"""Tests for versatil.models.decoding.action_heads.moe module."""

import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.models.decoding.action_heads.moe import MoEHead
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey, MoERoutingType


@pytest.fixture
def moe_head_factory(
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., MoEHead]:
    """Factory for MoEHead instances with configurable initialization mode."""

    def factory(
        mode: str = "explicit",
        num_experts: int = 3,
        input_dim: int = 64,
        output_dim: int = 3,
        gating_input_dim: int = 32,
        routing_type: str = MoERoutingType.SOFT.value,
    ) -> MoEHead:
        if mode == "explicit":
            experts = [
                action_head_factory(input_dim=input_dim, output_dim=output_dim)
                for _ in range(num_experts)
            ]
            return MoEHead(
                experts=experts,
                gating_input_dim=gating_input_dim,
                routing_type=routing_type,
            )
        elif mode == "base_with_count":
            base = action_head_factory(input_dim=input_dim, output_dim=output_dim)
            return MoEHead(
                base_expert=base,
                num_experts=num_experts,
                gating_input_dim=gating_input_dim,
                routing_type=routing_type,
            )
        elif mode == "lazy":
            base = action_head_factory(input_dim=input_dim, output_dim=output_dim)
            return MoEHead(
                base_expert=base,
                gating_input_dim=gating_input_dim,
                routing_type=routing_type,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return factory


@pytest.fixture
def gating_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for gating input tensors."""

    def factory(
        batch_size: int = 2,
        prediction_horizon: int = 8,
        gating_input_dim: int = 32,
    ) -> torch.Tensor:
        shape = (batch_size, prediction_horizon, gating_input_dim)
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    return factory


class TestMoEHeadInitialization:
    def test_explicit_experts_mode(
        self,
        moe_head_factory: Callable[..., MoEHead],
    ):
        head = moe_head_factory(mode="explicit", num_experts=4)
        assert head.is_initialized is True
        assert len(head.experts) == 4
        assert head._base_expert_template is None

    def test_base_expert_with_num_experts_mode(
        self,
        moe_head_factory: Callable[..., MoEHead],
    ):
        head = moe_head_factory(mode="base_with_count", num_experts=3)
        assert head.is_initialized is True
        assert len(head.experts) == 3
        assert head._base_expert_template is None

    def test_lazy_mode_base_expert_only(
        self,
        moe_head_factory: Callable[..., MoEHead],
    ):
        head = moe_head_factory(mode="lazy")
        assert head.is_initialized is False
        assert head.experts is None
        assert head._base_expert_template is not None

    def test_no_args_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape("Must provide 'experts' or 'base_expert'"),
        ):
            MoEHead()


class TestMoEHeadSetNumExperts:
    def test_creates_experts_from_template(
        self,
        moe_head_factory: Callable[..., MoEHead],
    ):
        head = moe_head_factory(mode="lazy", output_dim=5)
        head.set_output_dim(dim=5)
        head.set_num_experts(num_experts=4)
        assert head.is_initialized is True
        assert len(head.experts) == 4
        assert head._base_expert_template is None

    def test_raises_if_already_initialized(
        self,
        moe_head_factory: Callable[..., MoEHead],
    ):
        head = moe_head_factory(mode="explicit", num_experts=3)
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "MoEHead already initialized. Cannot call set_num_experts twice."
            ),
        ):
            head.set_num_experts(num_experts=5)

    def test_raises_if_no_template(
        self,
        moe_head_factory: Callable[..., MoEHead],
    ):
        head = moe_head_factory(mode="lazy")
        # Remove the template to simulate missing state
        head._base_expert_template = None
        with pytest.raises(
            RuntimeError,
            match=re.escape("No base_expert template stored. Cannot create experts."),
        ):
            head.set_num_experts(num_experts=3)

    def test_raises_if_no_lazy_init_params(
        self,
        moe_head_factory: Callable[..., MoEHead],
    ):
        head = moe_head_factory(mode="lazy")
        # Remove lazy init params to simulate missing state
        head._lazy_init_params = None
        with pytest.raises(
            RuntimeError,
            match=re.escape("No lazy init params stored."),
        ):
            head.set_num_experts(num_experts=3)

    def test_raises_if_no_output_dim(
        self,
        action_head_factory: Callable[..., ActionHead],
    ):
        base = action_head_factory(input_dim=64, output_dim=3)
        head = MoEHead(
            base_expert=base,
            gating_input_dim=32,
        )
        # output_dim is None by default in lazy mode
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Output dimension is not set for MoE Head. Call set_output_dim() first."
            ),
        ):
            head.set_num_experts(num_experts=3)


class TestMoEHeadOutputDim:
    def test_raises_when_not_set(
        self,
        moe_head_factory: Callable[..., MoEHead],
    ):
        head = moe_head_factory(mode="lazy")
        with pytest.raises(
            RuntimeError,
            match=re.escape("output_dim not set. Call set_output_dim() first."),
        ):
            _ = head.output_dim

    def test_set_output_dim_propagates_to_experts(
        self,
        moe_head_factory: Callable[..., MoEHead],
    ):
        head = moe_head_factory(mode="explicit", num_experts=3, output_dim=3)
        head.set_output_dim(dim=7)
        assert head.output_dim == 7
        for expert in head.experts:
            assert expert.output_dim == 7


class TestMoEHeadForward:
    def test_raises_when_not_initialized(
        self,
        moe_head_factory: Callable[..., MoEHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
        gating_tensor_factory: Callable[..., torch.Tensor],
    ):
        head = moe_head_factory(mode="lazy")
        features = embedding_tensor_factory()
        gating = gating_tensor_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape("MoEHead not initialized. Call set_num_experts() first."),
        ):
            head.forward(features=features, gating_feature=gating)

    def test_returns_expected_keys(
        self,
        moe_head_factory: Callable[..., MoEHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
        gating_tensor_factory: Callable[..., torch.Tensor],
    ):
        head = moe_head_factory(
            mode="explicit",
            num_experts=3,
            input_dim=64,
            output_dim=5,
            gating_input_dim=32,
        )
        features = embedding_tensor_factory(embedding_dimension=64)
        gating = gating_tensor_factory(gating_input_dim=32)
        result = head.forward(features=features, gating_feature=gating)
        assert SampleKey.ACTION.value in result
        assert DecoderOutputKey.ROUTING_WEIGHTS.value in result
        assert DecoderOutputKey.EXPERT_OUTPUTS.value in result

    @pytest.mark.parametrize("num_experts", [2, 4])
    @pytest.mark.parametrize("output_dim", [3, 7])
    def test_output_shape(
        self,
        moe_head_factory: Callable[..., MoEHead],
        embedding_tensor_factory: Callable[..., torch.Tensor],
        gating_tensor_factory: Callable[..., torch.Tensor],
        num_experts: int,
        output_dim: int,
    ):
        batch_size = 2
        prediction_horizon = 8
        input_dim = 64
        gating_input_dim = 32
        head = moe_head_factory(
            mode="explicit",
            num_experts=num_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            gating_input_dim=gating_input_dim,
        )
        features = embedding_tensor_factory(
            batch_size=batch_size,
            prediction_horizon=prediction_horizon,
            embedding_dimension=input_dim,
        )
        gating = gating_tensor_factory(
            batch_size=batch_size,
            prediction_horizon=prediction_horizon,
            gating_input_dim=gating_input_dim,
        )
        result = head.forward(features=features, gating_feature=gating)
        action = result[SampleKey.ACTION.value]
        routing_weights = result[DecoderOutputKey.ROUTING_WEIGHTS.value]
        expert_outputs = result[DecoderOutputKey.EXPERT_OUTPUTS.value]
        assert action.shape == (batch_size, prediction_horizon, output_dim)
        assert routing_weights.shape[-1] == num_experts
        assert expert_outputs.shape[-2] == num_experts
        assert expert_outputs.shape[-1] == output_dim
