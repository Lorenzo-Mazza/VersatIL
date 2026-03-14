"""Tests for versatil.models.decoding.mixture_of_experts module."""
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch

from versatil.models.decoding.constants import DecoderOutputKey, MoERoutingType
from versatil.models.decoding.mixture_of_experts import BaseMixtureOfExperts


@pytest.fixture
def moe_factory() -> Callable[..., BaseMixtureOfExperts]:
    """Factory for BaseMixtureOfExperts with configurable parameters."""
    def factory(
        num_experts: int = 4,
        device: str = "cpu",
        gating_input_dim: int | None = 64,
        routing_type: str = MoERoutingType.SOFT.value,
        top_k: int = 2,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
    ) -> BaseMixtureOfExperts:
        return BaseMixtureOfExperts(
            num_experts=num_experts,
            device=device,
            gating_input_dim=gating_input_dim,
            routing_type=routing_type,
            top_k=top_k,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )
    return factory


@pytest.fixture
def routing_weights_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for routing weight tensors normalized along the expert dimension."""
    def factory(
        batch_size: int = 2,
        num_experts: int = 4,
        horizon: int | None = None,
    ) -> torch.Tensor:
        if horizon is not None:
            shape = (batch_size, horizon, num_experts)
        else:
            shape = (batch_size, num_experts)
        raw = torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32)
        )
        return torch.softmax(raw, dim=-1)
    return factory


@pytest.fixture
def expert_outputs_factory(
    rng: np.random.Generator,
) -> Callable[..., list[torch.Tensor]]:
    """Factory for lists of expert output tensors."""
    def factory(
        num_experts: int = 4,
        batch_size: int = 2,
        prediction_horizon: int = 8,
        action_dim: int = 7,
    ) -> list[torch.Tensor]:
        return [
            torch.from_numpy(
                rng.standard_normal(
                    (batch_size, prediction_horizon, action_dim)
                ).astype(np.float32)
            )
            for _ in range(num_experts)
        ]
    return factory


class TestBaseMixtureOfExpertsInitialization:

    @pytest.mark.parametrize("num_experts", [2, 8])
    @pytest.mark.parametrize("routing_type", [
        MoERoutingType.SOFT.value,
        MoERoutingType.TOP_K.value,
    ])
    @pytest.mark.parametrize("top_k", [1, 3])
    @pytest.mark.parametrize("temperature", [0.5, 2.0])
    def test_stores_configuration(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        num_experts: int,
        routing_type: str,
        top_k: int,
        temperature: float,
    ):
        moe = moe_factory(
            num_experts=num_experts,
            routing_type=routing_type,
            top_k=top_k,
            temperature=temperature,
        )
        assert moe.num_experts == num_experts
        assert moe.routing_type == routing_type
        assert moe.top_k == min(top_k, num_experts)
        assert moe.temperature.item() == pytest.approx(temperature)

    @pytest.mark.parametrize("num_experts, expectation", [
        (0, pytest.raises(ValueError, match="Must provide at least one expert")),
        (1, does_not_raise()),
        (4, does_not_raise()),
    ])
    def test_num_experts_validation(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        num_experts: int,
        expectation,
    ):
        with expectation:
            moe_factory(num_experts=num_experts)

    def test_invalid_routing_type_raises(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
    ):
        valid_routing_types = [e.value for e in MoERoutingType]
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Invalid routing_type: invalid_routing. Expected one of {valid_routing_types}"
            ),
        ):
            moe_factory(routing_type="invalid_routing")

    def test_builds_gating_network_when_input_dim_provided(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
    ):
        moe = moe_factory(gating_input_dim=64)
        assert moe.has_gating_network is True
        assert moe.gating_network is not None

    def test_no_gating_network_when_input_dim_none(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
    ):
        moe = moe_factory(gating_input_dim=None)
        assert moe.has_gating_network is False
        assert moe.gating_network is None

    def test_learnable_temperature_is_parameter(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
    ):
        moe = moe_factory(
            learnable_temperature=True,
            temperature=2.0,
        )
        assert isinstance(moe.temperature, torch.nn.Parameter)
        assert moe.temperature.requires_grad is True
        assert moe.temperature.item() == pytest.approx(2.0)

    def test_buffer_temperature_is_not_parameter(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
    ):
        moe = moe_factory(
            learnable_temperature=False,
            temperature=3.0,
        )
        assert not isinstance(moe.temperature, torch.nn.Parameter)
        assert moe.temperature.item() == pytest.approx(3.0)
        parameter_names = [name for name, _ in moe.named_parameters()]
        assert "temperature" not in parameter_names


class TestComputeRoutingWeights:

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("num_experts", [2, 6])
    def test_output_shape(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        input_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        num_experts: int,
    ):
        input_dim = 64
        moe = moe_factory(
            num_experts=num_experts,
            gating_input_dim=input_dim,
        )
        gating_feature = input_tensor_factory(
            batch_size=batch_size,
            input_dimension=input_dim,
        )
        weights = moe.compute_routing_weights(features=gating_feature)
        assert weights.shape == (batch_size, num_experts)

    def test_weights_sum_to_one(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        input_dim = 64
        moe = moe_factory(gating_input_dim=input_dim)
        gating_feature = input_tensor_factory(input_dimension=input_dim)
        weights = moe.compute_routing_weights(features=gating_feature)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_higher_temperature_produces_more_uniform_weights(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        num_experts = 4
        input_dim = 64
        gating_feature = input_tensor_factory(
            batch_size=2,
            input_dimension=input_dim,
        )
        moe_low_temp = moe_factory(
            num_experts=num_experts,
            gating_input_dim=input_dim,
            temperature=0.1,
        )
        moe_high_temp = moe_factory(
            num_experts=num_experts,
            gating_input_dim=input_dim,
            temperature=100.0,
        )
        # Copy gating network weights so both see the same logits
        moe_high_temp.gating_network.load_state_dict(
            moe_low_temp.gating_network.state_dict()
        )
        weights_low = moe_low_temp.compute_routing_weights(features=gating_feature)
        weights_high = moe_high_temp.compute_routing_weights(features=gating_feature)
        # High temperature should produce weights closer to uniform (1/num_experts)
        uniform = torch.full_like(weights_high, 1.0 / num_experts)
        distance_low = (weights_low - uniform).abs().mean()
        distance_high = (weights_high - uniform).abs().mean()
        assert distance_high < distance_low

    def test_uses_raw_features_without_gating_network(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        num_experts = 4
        moe = moe_factory(
            num_experts=num_experts,
            gating_input_dim=None,
        )
        # When no gating network, features must already have num_experts dim
        raw_logits = input_tensor_factory(
            batch_size=2,
            input_dimension=num_experts,
        )
        weights = moe.compute_routing_weights(features=raw_logits)
        expected = torch.softmax(raw_logits / moe.temperature, dim=-1)
        assert torch.allclose(weights, expected, atol=1e-5)


class TestGetExpertSpecialization:

    def test_returns_expected_keys(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        moe = moe_factory(gating_input_dim=64)
        gating_feature = input_tensor_factory(input_dimension=64)
        result = moe.get_expert_specialization(gating_feature=gating_feature)
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            DecoderOutputKey.EXPERT_USAGE.value,
            DecoderOutputKey.ROUTING_ENTROPY.value,
            DecoderOutputKey.TOP_EXPERT_CONFIDENCE.value,
        }
        assert isinstance(result[DecoderOutputKey.EXPERT_USAGE.value], torch.Tensor)
        assert isinstance(result[DecoderOutputKey.ROUTING_ENTROPY.value], torch.Tensor)
        assert isinstance(result[DecoderOutputKey.TOP_EXPERT_CONFIDENCE.value], torch.Tensor)

    def test_expert_usage_sums_to_one(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        moe = moe_factory(gating_input_dim=64)
        gating_feature = input_tensor_factory(input_dimension=64)
        result = moe.get_expert_specialization(gating_feature=gating_feature)
        usage = result[DecoderOutputKey.EXPERT_USAGE.value]
        assert usage.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_entropy_non_negative(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        moe = moe_factory(gating_input_dim=64)
        gating_feature = input_tensor_factory(input_dimension=64)
        result = moe.get_expert_specialization(gating_feature=gating_feature)
        entropy = result[DecoderOutputKey.ROUTING_ENTROPY.value]
        assert entropy.item() >= 0.0


class TestApplyRouting:

    def test_soft_routing_output_shape(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        expert_outputs_factory: Callable[..., list[torch.Tensor]],
        routing_weights_factory: Callable[..., torch.Tensor],
    ):
        num_experts = 4
        batch_size = 2
        prediction_horizon = 8
        action_dim = 7
        moe = moe_factory(
            num_experts=num_experts,
            routing_type=MoERoutingType.SOFT.value,
        )
        outputs = expert_outputs_factory(
            num_experts=num_experts,
            batch_size=batch_size,
            prediction_horizon=prediction_horizon,
            action_dim=action_dim,
        )
        weights = routing_weights_factory(
            batch_size=batch_size,
            num_experts=num_experts,
        )
        result = moe._apply_routing(
            expert_outputs=outputs,
            weights=weights,
        )
        assert result.shape == (batch_size, prediction_horizon, action_dim)

    def test_topk_routing_output_shape(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        expert_outputs_factory: Callable[..., list[torch.Tensor]],
        routing_weights_factory: Callable[..., torch.Tensor],
    ):
        num_experts = 4
        batch_size = 2
        prediction_horizon = 8
        action_dim = 7
        moe = moe_factory(
            num_experts=num_experts,
            routing_type=MoERoutingType.TOP_K.value,
            top_k=2,
        )
        outputs = expert_outputs_factory(
            num_experts=num_experts,
            batch_size=batch_size,
            prediction_horizon=prediction_horizon,
            action_dim=action_dim,
        )
        weights = routing_weights_factory(
            batch_size=batch_size,
            num_experts=num_experts,
        )
        result = moe._apply_routing(
            expert_outputs=outputs,
            weights=weights,
        )
        assert result.shape == (batch_size, prediction_horizon, action_dim)

    def test_soft_routing_with_3d_weights(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        expert_outputs_factory: Callable[..., list[torch.Tensor]],
        routing_weights_factory: Callable[..., torch.Tensor],
    ):
        num_experts = 4
        batch_size = 2
        prediction_horizon = 8
        action_dim = 7
        moe = moe_factory(
            num_experts=num_experts,
            routing_type=MoERoutingType.SOFT.value,
        )
        outputs = expert_outputs_factory(
            num_experts=num_experts,
            batch_size=batch_size,
            prediction_horizon=prediction_horizon,
            action_dim=action_dim,
        )
        # 3D weights: (B, horizon, num_experts)
        weights = routing_weights_factory(
            batch_size=batch_size,
            num_experts=num_experts,
            horizon=prediction_horizon,
        )
        result = moe._apply_routing(
            expert_outputs=outputs,
            weights=weights,
        )
        assert result.shape == (batch_size, prediction_horizon, action_dim)

    def test_soft_routing_value_correctness(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
    ):
        num_experts = 3
        batch_size = 2
        output_dim = 4
        moe = moe_factory(
            num_experts=num_experts,
            routing_type=MoERoutingType.SOFT.value,
            gating_input_dim=None,
        )
        # Create expert outputs where each expert outputs a distinct constant
        outputs = [
            torch.full((batch_size, output_dim), fill_value=float(i))
            for i in range(num_experts)
        ]
        # Manually set routing weights: batch 0 -> all weight on expert 1,
        # batch 1 -> equal weights
        weights = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
            dtype=torch.float32,
        )
        result = moe._apply_routing(
            expert_outputs=outputs,
            weights=weights,
        )
        # Batch 0: 0*0.0 + 1*1.0 + 2*0.0 = 1.0
        assert torch.allclose(
            result[0], torch.full((output_dim,), 1.0), atol=1e-5,
        )
        # Batch 1: 0*(1/3) + 1*(1/3) + 2*(1/3) = 1.0
        assert torch.allclose(
            result[1], torch.full((output_dim,), 1.0), atol=1e-5,
        )

    def test_topk_routing_selects_fewer_experts(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
    ):
        num_experts = 4
        top_k = 2
        batch_size = 2
        output_dim = 3
        moe = moe_factory(
            num_experts=num_experts,
            routing_type=MoERoutingType.TOP_K.value,
            top_k=top_k,
        )
        # Create expert outputs where each expert outputs a distinct constant
        outputs = [
            torch.full((batch_size, output_dim), fill_value=float(i))
            for i in range(num_experts)
        ]
        # Routing weights strongly favoring experts 0 and 1
        weights = torch.tensor(
            [[0.9, 0.1, 0.0, 0.0], [0.0, 0.1, 0.0, 0.9]],
            dtype=torch.float32,
        )
        result = moe._apply_routing(
            expert_outputs=outputs,
            weights=weights,
        )
        # For batch 0: top-2 are experts 0 (w=0.9) and 1 (w=0.1)
        # Renormalized: 0.9/1.0=0.9, 0.1/1.0=0.1
        expected_batch_0 = 0.0 * 0.9 + 1.0 * 0.1  # = 0.1
        assert result[0, 0].item() == pytest.approx(expected_batch_0, abs=1e-5)
        # For batch 1: top-2 are experts 3 (w=0.9) and 1 (w=0.1)
        # Renormalized: 0.9/1.0=0.9, 0.1/1.0=0.1
        expected_batch_1 = 3.0 * 0.9 + 1.0 * 0.1  # = 2.8
        assert result[1, 0].item() == pytest.approx(expected_batch_1, abs=1e-5)

    def test_topk_routing_with_3d_weights(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        expert_outputs_factory: Callable[..., list[torch.Tensor]],
        routing_weights_factory: Callable[..., torch.Tensor],
    ):
        num_experts = 4
        batch_size = 2
        prediction_horizon = 8
        action_dim = 7
        moe = moe_factory(
            num_experts=num_experts,
            routing_type=MoERoutingType.TOP_K.value,
            top_k=2,
        )
        outputs = expert_outputs_factory(
            num_experts=num_experts,
            batch_size=batch_size,
            prediction_horizon=prediction_horizon,
            action_dim=action_dim,
        )
        weights = routing_weights_factory(
            batch_size=batch_size,
            num_experts=num_experts,
            horizon=prediction_horizon,
        )
        result = moe._apply_routing(
            expert_outputs=outputs,
            weights=weights,
        )
        assert result.shape == (batch_size, prediction_horizon, action_dim)

    def test_topk_clamps_to_num_experts(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        expert_outputs_factory: Callable[..., list[torch.Tensor]],
        routing_weights_factory: Callable[..., torch.Tensor],
    ):
        num_experts = 3
        moe = moe_factory(
            num_experts=num_experts,
            routing_type=MoERoutingType.TOP_K.value,
            top_k=10,
        )
        # top_k should be clamped to num_experts
        assert moe.top_k == num_experts
        outputs = expert_outputs_factory(num_experts=num_experts)
        weights = routing_weights_factory(num_experts=num_experts)
        result = moe._apply_routing(
            expert_outputs=outputs,
            weights=weights,
        )
        assert result.shape == outputs[0].shape

    def test_unknown_routing_type_raises(
        self,
        moe_factory: Callable[..., BaseMixtureOfExperts],
        expert_outputs_factory: Callable[..., list[torch.Tensor]],
        routing_weights_factory: Callable[..., torch.Tensor],
    ):
        num_experts = 4
        moe = moe_factory(
            num_experts=num_experts,
            routing_type=MoERoutingType.SOFT.value,
        )
        # Override routing_type after construction to bypass __init__ validation
        moe.routing_type = "unknown_type"
        outputs = expert_outputs_factory(num_experts=num_experts)
        weights = routing_weights_factory(num_experts=num_experts)
        with pytest.raises(
            ValueError,
            match="Unknown routing type: unknown_type",
        ):
            moe._apply_routing(
                expert_outputs=outputs,
                weights=weights,
            )
