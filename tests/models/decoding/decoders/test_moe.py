"""Tests for versatil.models.decoding.decoders.moe module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey, MoERoutingType
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
    ):
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

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


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
        prediction_horizon: int = PREDICTION_HORIZON,
        with_init_weights: bool = False,
    ) -> MinimalDecoder:
        action_space = mock_action_space_factory(position_dim=position_dim)
        observation_space = mock_observation_space_factory()
        action_heads = {
            "position_action": action_head_factory(input_dim=embedding_dimension),
        }
        decoder_input = DecoderInput(keys=["rgb_features"])
        decoder_class = (
            MinimalDecoderWithInitWeights if with_init_weights else MinimalDecoder
        )
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


@pytest.fixture
def mock_cuda():
    """Mock CUDA stream operations for CPU testing."""
    with (
        patch("torch.cuda.Stream", return_value=MagicMock()),
        patch("torch.cuda.stream"),
        patch("torch.cuda.synchronize"),
    ):
        yield


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
        assert isinstance(decoder.base_expert, MinimalDecoder)

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

    def test_action_heads_shared_with_base_expert(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
    ):
        decoder = moe_decoder_factory()
        # Mutating base_expert.action_heads must be visible through decoder.action_heads
        decoder.base_expert.action_heads["sentinel"] = nn.Identity()
        assert "sentinel" in decoder.action_heads
        del decoder.base_expert.action_heads["sentinel"]
        assert "sentinel" not in decoder.action_heads


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


class TestMoEDecoderForward:
    def test_training_uses_gating_feature_key(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        mock_cuda: None,
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
        mock_cuda: None,
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
        mock_cuda: None,
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
        mock_cuda: None,
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
        mock_cuda: None,
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

    def test_pops_gating_key_from_features(
        self,
        moe_decoder_factory: Callable[..., MoEDecoder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        mock_cuda: None,
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
        assert GATING_FEATURE_KEY not in features

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
        mock_cuda: None,
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
