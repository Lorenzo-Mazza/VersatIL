"""Tests for versatil.models.decoding.decoders.moe module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import (
    ActionHeadLayout,
    DecoderOutputKey,
    MoERoutingType,
)
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.decoding.decoders.moe import MoEDecoder
from versatil.models.decoding.mixture_of_experts import BaseMixtureOfExperts
from versatil.models.layers.activation import ActivationFunction

EMBEDDING_DIMENSION = 32
BATCH_SIZE = 2
POSITION_DIM = 3
PREDICTION_HORIZON = 4
NUM_EXPERTS = 3
GATING_FEATURE_KEY = "gating_feature"
INFERENCE_GATING_KEY = "inference_gating"


class MinimalDecoder(ActionDecoder):
    """Minimal concrete ActionDecoder for testing MoEDecoder."""

    def __init__(
        self,
        decoder_input: DecoderInput,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        action_heads: dict,
        device: str,
        observation_horizon: int,
        prediction_horizon: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
        )
        self.embedding_dimension = embedding_dimension
        self.linear = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        feature = next(iter(features.values()))
        projected = self.linear(feature)
        results = {}
        for key, head in self.action_heads.items():
            expanded = projected.unsqueeze(1).expand(-1, self.prediction_horizon, -1)
            results[key] = head(expanded)
        return results


class MinimalDecoderWithInitWeights(MinimalDecoder):
    """MinimalDecoder with _init_weights that zeros all Linear layers."""

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class MinimalJointDecoder(MinimalDecoder):
    """Minimal joint-head decoder that returns action-space component outputs."""

    action_head_layout: ActionHeadLayout = ActionHeadLayout.JOINT

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del actions
        feature = next(iter(features.values()))
        projected = self.linear(feature)
        expanded = projected.unsqueeze(1).expand(-1, self.prediction_horizon, -1)
        joint_action_head = next(iter(self.action_heads.values()))
        joint_action_output = joint_action_head(expanded)
        return self.action_space.split_action_tensor(
            action_tensor=joint_action_output,
            owner_name=type(self).__name__,
        )


class FeaturelessDecoder(ActionDecoder):
    """Minimal decoder that does not require expert features."""

    action_head_layout: ActionHeadLayout = ActionHeadLayout.NONE

    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        prediction_horizon: int,
    ) -> None:
        super().__init__(
            decoder_input=DecoderInput(keys=[]),
            observation_space=observation_space,
            action_space=action_space,
            action_heads={},
            device="cpu",
            observation_horizon=1,
            prediction_horizon=prediction_horizon,
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del features, actions
        return {
            action_key: torch.zeros(
                BATCH_SIZE,
                self.prediction_horizon,
                action_dimension,
            )
            for action_key, action_dimension in (
                self.action_space.predicted_action_dimensions.items()
            )
        }


@pytest.fixture
def base_expert_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., MinimalDecoder]:
    """Factory for MinimalDecoder base expert instances."""

    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        position_dim: int = POSITION_DIM,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        prediction_horizon: int = PREDICTION_HORIZON,
        with_init_weights: bool = False,
        joint_layout: bool = False,
    ) -> MinimalDecoder:
        action_space = mock_action_space_factory(
            position_dim=position_dim,
            has_orientation=has_orientation,
            orientation_dim=orientation_dim,
        )
        observation_space = mock_observation_space_factory()
        if joint_layout:
            action_heads = {
                "joint_action": action_head_factory(input_dim=embedding_dimension),
            }
            decoder_class = MinimalJointDecoder
        else:
            action_heads = {
                "position_action": action_head_factory(input_dim=embedding_dimension),
            }
            decoder_class = (
                MinimalDecoderWithInitWeights if with_init_weights else MinimalDecoder
            )
        decoder_input = DecoderInput(keys=["rgb_features"])
        return decoder_class(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device="cpu",
            observation_horizon=1,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
        )

    return factory


@pytest.fixture
def featureless_expert_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
) -> Callable[..., FeaturelessDecoder]:
    """Factory for decoders that only use gating features."""

    def factory() -> FeaturelessDecoder:
        return FeaturelessDecoder(
            observation_space=mock_observation_space_factory(),
            action_space=mock_action_space_factory(position_dim=POSITION_DIM),
            prediction_horizon=PREDICTION_HORIZON,
        )

    return factory


@pytest.fixture
def moe_decoder_factory(
    base_expert_factory: Callable[..., MinimalDecoder],
) -> Callable[..., MoEDecoder]:
    """Factory for MoEDecoder instances."""

    def factory(
        num_experts: int = NUM_EXPERTS,
        gating_feature_key: str = GATING_FEATURE_KEY,
        inference_gating_key: str | None = None,
        gating_input_dim: int = EMBEDDING_DIMENSION,
        gating_hidden_dims: list[int] | None = None,
        gating_activation: str = ActivationFunction.SILU.value,
        routing_type: str = MoERoutingType.SOFT.value,
        top_k: int = 2,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        gating_dropout: float = 0.1,
        gating_normalization: bool = True,
        with_init_weights: bool = False,
    ) -> MoEDecoder:
        base_expert = base_expert_factory(with_init_weights=with_init_weights)
        return MoEDecoder(
            base_expert=base_expert,
            num_experts=num_experts,
            gating_feature_key=gating_feature_key,
            inference_gating_key=inference_gating_key,
            gating_input_dim=gating_input_dim,
            gating_hidden_dims=gating_hidden_dims,
            gating_activation=gating_activation,
            routing_type=routing_type,
            top_k=top_k,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            gating_dropout=gating_dropout,
            gating_normalization=gating_normalization,
        )

    return factory


@pytest.mark.unit
class TestMoEDecoderInitialization:
    def test_inherits_from_action_decoder_and_base_moe(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
    ):
        decoder = moe_decoder_factory()
        assert isinstance(decoder, ActionDecoder)
        assert isinstance(decoder, BaseMixtureOfExperts)

    @pytest.mark.parametrize("num_experts", [2, 4])
    @pytest.mark.parametrize("gating_feature_key", ["gating_a", "gating_b"])
    def test_stores_configuration(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        num_experts: int,
        gating_feature_key: str,
    ):
        decoder = moe_decoder_factory(
            num_experts=num_experts,
            gating_feature_key=gating_feature_key,
        )
        assert decoder.num_experts == num_experts
        assert decoder.gating_feature_key == gating_feature_key
        assert decoder.action_keys == ["position_action"]

    def test_joint_layout_routes_action_space_keys(
        self,
        base_expert_factory: Callable[..., MinimalDecoder],
    ):
        base_expert = base_expert_factory(
            joint_layout=True,
            has_orientation=True,
            orientation_dim=2,
        )
        decoder = MoEDecoder(
            base_expert=base_expert,
            num_experts=NUM_EXPERTS,
            gating_feature_key=GATING_FEATURE_KEY,
            gating_input_dim=EMBEDDING_DIMENSION,
        )

        assert decoder.action_keys == ["position_action", "orientation_action"]

    @pytest.mark.parametrize(
        "inference_gating_key, expected_key",
        [
            (None, GATING_FEATURE_KEY),
            (INFERENCE_GATING_KEY, INFERENCE_GATING_KEY),
        ],
    )
    def test_inference_gating_key(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        inference_gating_key: str | None,
        expected_key: str,
    ):
        decoder = moe_decoder_factory(
            gating_feature_key=GATING_FEATURE_KEY,
            inference_gating_key=inference_gating_key,
        )
        assert decoder.inference_gating_key == expected_key

    def test_creates_expert_decoders_module_list(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
    ):
        num_experts = 4
        decoder = moe_decoder_factory(num_experts=num_experts)
        assert isinstance(decoder.expert_decoders, nn.ModuleList)
        assert len(decoder.expert_decoders) == num_experts

    def test_action_heads_alias_first_expert_heads(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
    ):
        decoder = moe_decoder_factory()
        assert decoder.action_heads is decoder.expert_decoders[0].action_heads

    def test_base_expert_is_not_registered_as_submodule(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
    ):
        decoder = moe_decoder_factory()

        # Every parameter must belong to state that participates in forward:
        # the experts, the gating network, the aliased expert-0 heads, or the
        # routing temperature.
        live_prefixes = (
            "expert_decoders.",
            "gating_network.",
            "action_heads.",
            "temperature",
        )
        for key in decoder.state_dict():
            assert key.startswith(live_prefixes), key


@pytest.mark.unit
class TestCreateExpertsFromConfig:
    @pytest.mark.parametrize("num_experts", [2, 5])
    def test_creates_correct_number_of_experts(
        self,
        base_expert_factory: Callable[..., MinimalDecoder],
        num_experts: int,
    ):
        base = base_expert_factory()
        experts = MoEDecoder._create_experts_from_config(
            base_expert=base, num_experts=num_experts
        )
        assert len(experts) == num_experts

    def test_experts_are_independent_objects(
        self,
        base_expert_factory: Callable[..., MinimalDecoder],
    ):
        base = base_expert_factory()
        experts = MoEDecoder._create_experts_from_config(
            base_expert=base, num_experts=NUM_EXPERTS
        )
        for expert in experts:
            assert expert is not base
        for i in range(len(experts)):
            for j in range(i + 1, len(experts)):
                assert experts[i] is not experts[j]

    def test_experts_have_reset_parameters(
        self,
        base_expert_factory: Callable[..., MinimalDecoder],
    ):
        base = base_expert_factory()
        original_weight = base.linear.weight.data.clone()
        experts = MoEDecoder._create_experts_from_config(
            base_expert=base, num_experts=2
        )
        # After reset_parameters, expert weights differ from original
        for expert in experts:
            assert not torch.equal(expert.linear.weight.data, original_weight)

    def test_applies_init_weights_when_available(
        self,
        base_expert_factory: Callable[..., MinimalDecoder],
    ):
        base = base_expert_factory(with_init_weights=True)
        experts = MoEDecoder._create_experts_from_config(
            base_expert=base, num_experts=2
        )
        # MinimalDecoderWithInitWeights zeros all Linear weights after reset
        for expert in experts:
            torch.testing.assert_close(
                expert.linear.weight.data,
                torch.zeros_like(expert.linear.weight.data),
                atol=0,
                rtol=0,
            )

    def test_works_without_init_weights(
        self,
        base_expert_factory: Callable[..., MinimalDecoder],
    ):
        base = base_expert_factory(with_init_weights=False)
        assert not hasattr(base, "_init_weights")
        experts = MoEDecoder._create_experts_from_config(
            base_expert=base, num_experts=2
        )
        assert len(experts) == 2
        # Weights are non-zero (from reset_parameters, no zeroing _init_weights)
        for expert in experts:
            assert not torch.all(expert.linear.weight.data == 0)


@pytest.mark.integration
class TestMoEDecoderForward:
    def test_training_uses_gating_feature_key(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_decoder_factory(
            gating_feature_key=GATING_FEATURE_KEY,
            inference_gating_key=INFERENCE_GATING_KEY,
        )
        decoder.train()
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["rgb_features", GATING_FEATURE_KEY],
        )
        # No KeyError means training path correctly used gating_feature_key
        outputs = decoder(features=features, actions=None)
        assert DecoderOutputKey.ROUTING_WEIGHTS.value in outputs

    def test_inference_uses_inference_gating_key(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_decoder_factory(
            gating_feature_key=GATING_FEATURE_KEY,
            inference_gating_key=INFERENCE_GATING_KEY,
        )
        decoder.eval()
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["rgb_features", INFERENCE_GATING_KEY],
        )
        with torch.no_grad():
            outputs = decoder(features=features, actions=None)
        assert DecoderOutputKey.ROUTING_WEIGHTS.value in outputs

    def test_output_contains_expert_outputs(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_decoder_factory()
        decoder.train()
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["rgb_features", GATING_FEATURE_KEY],
        )
        outputs = decoder(features=features, actions=None)
        assert DecoderOutputKey.EXPERT_OUTPUTS.value in outputs
        assert len(outputs[DecoderOutputKey.EXPERT_OUTPUTS.value]) == NUM_EXPERTS

    def test_output_contains_combined_action_keys(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_decoder_factory()
        decoder.train()
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["rgb_features", GATING_FEATURE_KEY],
        )
        outputs = decoder(features=features, actions=None)
        assert "position_action" in outputs
        assert outputs["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            POSITION_DIM,
        )

    def test_routing_weights_shape(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        num_experts = 4
        decoder = moe_decoder_factory(num_experts=num_experts)
        decoder.train()
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["rgb_features", GATING_FEATURE_KEY],
        )
        outputs = decoder(features=features, actions=None)
        routing_weights = outputs[DecoderOutputKey.ROUTING_WEIGHTS.value]
        assert routing_weights.shape == (BATCH_SIZE, num_experts)

    def test_forward_does_not_mutate_features(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = moe_decoder_factory()
        decoder.train()
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["rgb_features", GATING_FEATURE_KEY],
        )
        assert GATING_FEATURE_KEY in features
        decoder(features=features, actions=None)
        assert GATING_FEATURE_KEY in features

    def test_only_gating_feature_does_not_require_expert_features(
        self,
        featureless_expert_factory: Callable[..., FeaturelessDecoder],
    ):
        decoder = MoEDecoder(
            base_expert=featureless_expert_factory(),
            num_experts=NUM_EXPERTS,
            gating_feature_key=GATING_FEATURE_KEY,
            gating_input_dim=EMBEDDING_DIMENSION,
        )
        features = {
            GATING_FEATURE_KEY: torch.ones(BATCH_SIZE, EMBEDDING_DIMENSION),
        }

        outputs = decoder(features=features, actions=None)

        assert outputs["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            POSITION_DIM,
        )

    @pytest.mark.parametrize(
        "routing_type",
        [
            MoERoutingType.SOFT.value,
            MoERoutingType.TOP_K.value,
        ],
    )
    def test_routing_type_produces_valid_output(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        routing_type: str,
    ):
        decoder = moe_decoder_factory(routing_type=routing_type, top_k=2)
        decoder.train()
        features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=["rgb_features", GATING_FEATURE_KEY],
        )
        outputs = decoder(features=features, actions=None)
        assert outputs["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            POSITION_DIM,
        )


@pytest.fixture
def expert_action_outputs_factory(
    rng: np.random.Generator,
) -> Callable[..., list[dict[str, torch.Tensor]]]:
    """Factory for lists of expert output dictionaries."""

    def factory(
        num_experts: int = NUM_EXPERTS,
        batch_size: int = BATCH_SIZE,
        prediction_horizon: int = PREDICTION_HORIZON,
        position_dim: int = POSITION_DIM,
    ) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "position_action": torch.from_numpy(
                    rng.standard_normal(
                        (batch_size, prediction_horizon, position_dim)
                    ).astype(np.float32)
                )
            }
            for _ in range(num_experts)
        ]

    return factory


@pytest.fixture
def moe_routing_weights_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for softmax-normalized routing weight tensors."""

    def factory(
        batch_size: int = BATCH_SIZE,
        num_experts: int = NUM_EXPERTS,
    ) -> torch.Tensor:
        raw = torch.from_numpy(
            rng.standard_normal((batch_size, num_experts)).astype(np.float32)
        )
        return torch.softmax(raw, dim=-1)

    return factory


@pytest.mark.unit
class TestCombineExpertOutputs:
    def test_combines_all_action_keys(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        expert_action_outputs_factory: Callable[..., list[dict[str, torch.Tensor]]],
        moe_routing_weights_factory: Callable[..., torch.Tensor],
    ):
        decoder = moe_decoder_factory()
        expert_outputs = expert_action_outputs_factory()
        weights = moe_routing_weights_factory()
        combined = decoder._combine_expert_outputs(
            expert_outputs=expert_outputs, weights=weights
        )
        assert "position_action" in combined

    def test_output_shape_preserves_dimensions(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        expert_action_outputs_factory: Callable[..., list[dict[str, torch.Tensor]]],
        moe_routing_weights_factory: Callable[..., torch.Tensor],
    ):
        decoder = moe_decoder_factory()
        expert_outputs = expert_action_outputs_factory()
        weights = moe_routing_weights_factory()
        combined = decoder._combine_expert_outputs(
            expert_outputs=expert_outputs, weights=weights
        )
        assert combined["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            POSITION_DIM,
        )


@pytest.mark.unit
def test_auxiliary_output_keys(
    moe_decoder_factory: Callable[..., MoEDecoder],
):
    decoder = moe_decoder_factory()
    assert decoder.get_auxiliary_output_keys() == {
        DecoderOutputKey.ROUTING_WEIGHTS.value,
        DecoderOutputKey.EXPERT_OUTPUTS.value,
    }
