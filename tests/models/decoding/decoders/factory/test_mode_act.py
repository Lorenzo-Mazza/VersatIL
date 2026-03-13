"""Tests for versatil.models.decoding.decoders.factory.mode_act module."""
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

from versatil.models.decoding.action_heads.gaussian import GaussianHead
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import (
    DecoderOutputKey,
    FeatureType,
    GMMInitStrategy,
)
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.mode_act import (
    MixtureOfDensitiesActionTransformer,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.bidirectional_decoder import (
    BidirectionalDecoder,
)


EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_LAYERS = 1
FEEDFORWARD_DIMENSION = 64
SPATIAL_HEIGHT = 4
SPATIAL_WIDTH = 4
BATCH_SIZE = 2
POSITION_DIM = 3
PREDICTION_HORIZON = 4
NUM_MIXTURE_COMPONENTS = 3


@pytest.fixture
def mode_act_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_heads_factory: Callable[..., dict[str, ActionHead]],
) -> Callable[..., MixtureOfDensitiesActionTransformer]:
    """Factory for MixtureOfDensitiesActionTransformer with small dimensions."""

    def factory(
        input_keys: list[str] | None = None,
        position_dim: int = POSITION_DIM,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
        observation_horizon: int = 1,
        prediction_horizon: int = PREDICTION_HORIZON,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = FEEDFORWARD_DIMENSION,
        number_of_layers: int = NUMBER_OF_LAYERS,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.LAYER_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = PositionalEncodingType.ROPE.value,
        num_mixture_components: int = NUM_MIXTURE_COMPONENTS,
        gating_hidden_dims: list[int] | None = None,
        gating_activation: str = ActivationFunction.SILU.value,
        gating_dropout: float = 0.1,
        gating_normalization: bool = True,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        gating_feature_key: str | None = None,
        gmm_init_strategy: str = GMMInitStrategy.KMEANS_PLUS_PLUS.value,
        deterministic_inference: bool = True,
        device: str = "cpu",
    ) -> MixtureOfDensitiesActionTransformer:
        if input_keys is None:
            input_keys = ["rgb_features"]
        action_space = mock_action_space_factory(
            position_dim=position_dim,
            has_orientation=has_orientation,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            gripper_dim=gripper_dim,
        )
        action_heads = action_heads_factory(
            action_space=action_space,
            input_dim=embedding_dimension,
        )
        observation_space = mock_observation_space_factory()
        return MixtureOfDensitiesActionTransformer(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_layers=number_of_layers,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            positional_encoding_type=positional_encoding_type,
            num_mixture_components=num_mixture_components,
            gating_hidden_dims=gating_hidden_dims,
            gating_activation=gating_activation,
            gating_dropout=gating_dropout,
            gating_normalization=gating_normalization,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            gating_feature_key=gating_feature_key,
            gmm_init_strategy=gmm_init_strategy,
            deterministic_inference=deterministic_inference,
        )

    return factory


@pytest.fixture
def gaussian_mode_act_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    gaussian_head_factory: Callable[..., GaussianHead],
) -> Callable[..., MixtureOfDensitiesActionTransformer]:
    """Factory for MixtureOfDensitiesActionTransformer using GaussianHead."""

    def factory(
        input_keys: list[str] | None = None,
        position_dim: int = POSITION_DIM,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
        observation_horizon: int = 1,
        prediction_horizon: int = PREDICTION_HORIZON,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_layers: int = NUMBER_OF_LAYERS,
        feedforward_dimension: int | None = FEEDFORWARD_DIMENSION,
        num_mixture_components: int = NUM_MIXTURE_COMPONENTS,
        gating_normalization: bool = True,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        gating_feature_key: str | None = None,
        gmm_init_strategy: str = GMMInitStrategy.KMEANS_PLUS_PLUS.value,
        deterministic_inference: bool = True,
        device: str = "cpu",
    ) -> MixtureOfDensitiesActionTransformer:
        if input_keys is None:
            input_keys = ["rgb_features"]
        action_space = mock_action_space_factory(
            position_dim=position_dim,
            has_orientation=has_orientation,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            gripper_dim=gripper_dim,
        )
        action_heads = {}
        for key, meta in action_space.actions_metadata.items():
            if meta.requires_prediction_head:
                action_heads[key] = gaussian_head_factory(
                    input_dim=embedding_dimension,
                )
        observation_space = mock_observation_space_factory()
        return MixtureOfDensitiesActionTransformer(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_layers=number_of_layers,
            feedforward_dimension=feedforward_dimension,
            num_mixture_components=num_mixture_components,
            gating_normalization=gating_normalization,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            gating_feature_key=gating_feature_key,
            gmm_init_strategy=gmm_init_strategy,
            deterministic_inference=deterministic_inference,
        )

    return factory


class TestModeACTInitialization:

    def test_inherits_from_action_decoder(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
    ):
        decoder = mode_act_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("num_mixture_components", [3, 5])
    def test_stores_configuration(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        embedding_dimension: int,
        num_mixture_components: int,
    ):
        decoder = mode_act_factory(
            embedding_dimension=embedding_dimension,
            num_mixture_components=num_mixture_components,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_layers=NUMBER_OF_LAYERS,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            activation=ActivationFunction.GELU.value,
            normalization_type=NormalizationType.LAYER_NORM.value,
            attention_type=AttentionType.MULTI_HEAD.value,
            dropout_rate=0.05,
            attention_dropout=0.02,
            positional_encoding_type=PositionalEncodingType.ROPE.value,
            gmm_init_strategy=GMMInitStrategy.UNIFORM.value,
            deterministic_inference=False,
        )
        assert decoder.embedding_dimension == embedding_dimension
        assert decoder.num_mixture_components == num_mixture_components
        assert decoder.number_of_heads == NUMBER_OF_HEADS
        assert decoder.number_of_layers == NUMBER_OF_LAYERS
        assert decoder.feedforward_dimension == FEEDFORWARD_DIMENSION
        assert decoder.activation == ActivationFunction.GELU.value
        assert decoder.normalization_type == NormalizationType.LAYER_NORM.value
        assert decoder.attention_type == AttentionType.MULTI_HEAD.value
        assert decoder.dropout_rate == 0.05
        assert decoder.attention_dropout == 0.02
        assert decoder.positional_encoding_type == PositionalEncodingType.ROPE.value
        assert decoder.gmm_init_strategy == GMMInitStrategy.UNIFORM.value
        assert decoder.deterministic_inference is False

    def test_decoder_input_requires_spatial(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
    ):
        decoder = mode_act_factory()
        assert FeatureType.SPATIAL.value in decoder.decoder_input.required_types
        assert decoder.decoder_input.requires_actions is False

    def test_creates_transformer_components(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
    ):
        decoder = mode_act_factory()
        assert isinstance(decoder.input_sequence_builder, TransformerInputBuilder)
        assert isinstance(decoder.action_decoder, BidirectionalDecoder)
        assert isinstance(decoder.action_queries, nn.Parameter)
        assert isinstance(decoder.mode_query, nn.Parameter)

    def test_creates_mixture_heads(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
    ):
        num_components = 5
        decoder = mode_act_factory(num_mixture_components=num_components)
        for action_key in decoder.action_heads:
            assert action_key in decoder.mixture_heads
            assert len(decoder.mixture_heads[action_key]) == num_components

    @pytest.mark.parametrize("gating_normalization, expect_layer_norm", [
        (True, True),
        (False, False),
    ])
    def test_gating_network_normalization(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        gating_normalization: bool,
        expect_layer_norm: bool,
    ):
        decoder = mode_act_factory(gating_normalization=gating_normalization)
        assert isinstance(decoder.gating_network, nn.Sequential)
        contains_layer_norm = any(
            isinstance(module, nn.LayerNorm)
            for module in decoder.gating_network.modules()
        )
        assert contains_layer_norm is expect_layer_norm

    def test_temperature_as_buffer(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
    ):
        decoder = mode_act_factory(
            learnable_temperature=False,
            temperature=2.0,
        )
        assert not isinstance(decoder.temperature, nn.Parameter)
        assert isinstance(decoder.temperature, torch.Tensor)
        assert decoder.temperature.item() == pytest.approx(2.0)
        assert "temperature" in dict(decoder.named_buffers())

    def test_temperature_as_parameter(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
    ):
        decoder = mode_act_factory(
            learnable_temperature=True,
            temperature=0.5,
        )
        assert isinstance(decoder.temperature, nn.Parameter)
        assert decoder.temperature.requires_grad is True
        assert decoder.temperature.item() == pytest.approx(0.5)

    def test_action_queries_shape(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
    ):
        prediction_horizon = 6
        embedding_dimension = 32
        decoder = mode_act_factory(
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
        )
        assert decoder.action_queries.shape == (prediction_horizon, embedding_dimension)

    def test_mode_query_shape(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
    ):
        embedding_dimension = 32
        decoder = mode_act_factory(embedding_dimension=embedding_dimension)
        assert decoder.mode_query.shape == (1, embedding_dimension)

    def test_mixture_heads_are_independent_copies(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
    ):
        decoder = mode_act_factory(num_mixture_components=NUM_MIXTURE_COMPONENTS)
        for action_key in decoder.action_heads:
            heads = decoder.mixture_heads[action_key]
            for index in range(1, len(heads)):
                for parameter_a, parameter_b in zip(
                    heads[0].parameters(), heads[index].parameters()
                ):
                    assert not torch.equal(parameter_a, parameter_b) or parameter_a.numel() == 0


class TestModeACTForwardWithGaussianHead:

    def test_training_returns_mixture_outputs(
        self,
        gaussian_mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gaussian_mode_act_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = noisy_actions_factory(
            prediction_horizon=PREDICTION_HORIZON,
            action_keys_to_dims={"position_action": POSITION_DIM},
        )
        predictions = decoder(features=features, actions=actions)
        assert f"position_action_{DecoderOutputKey.MEAN.value}" in predictions
        assert f"position_action_{DecoderOutputKey.LOGVAR.value}" in predictions
        assert DecoderOutputKey.ROUTING_WEIGHTS.value in predictions

    def test_training_output_shapes(
        self,
        gaussian_mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gaussian_mode_act_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = noisy_actions_factory(
            prediction_horizon=PREDICTION_HORIZON,
            action_keys_to_dims={"position_action": POSITION_DIM},
        )
        predictions = decoder(features=features, actions=actions)
        mean_key = f"position_action_{DecoderOutputKey.MEAN.value}"
        logvar_key = f"position_action_{DecoderOutputKey.LOGVAR.value}"
        assert predictions[mean_key].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            NUM_MIXTURE_COMPONENTS,
            POSITION_DIM,
        )
        assert predictions[logvar_key].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            NUM_MIXTURE_COMPONENTS,
            POSITION_DIM,
        )
        assert predictions[DecoderOutputKey.ROUTING_WEIGHTS.value].shape == (
            BATCH_SIZE,
            NUM_MIXTURE_COMPONENTS,
        )

    def test_inference_returns_sampled_actions(
        self,
        gaussian_mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gaussian_mode_act_factory(deterministic_inference=True)
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features, actions=None)
        assert "position_action" in predictions
        assert DecoderOutputKey.ROUTING_WEIGHTS.value not in predictions

    def test_inference_output_shape(
        self,
        gaussian_mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gaussian_mode_act_factory(deterministic_inference=True)
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features, actions=None)
        assert predictions["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            POSITION_DIM,
        )


class TestModeACTForwardWithActionHead:

    def test_training_returns_stacked_outputs(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = mode_act_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = noisy_actions_factory(
            prediction_horizon=PREDICTION_HORIZON,
            action_keys_to_dims={"position_action": POSITION_DIM},
        )
        predictions = decoder(features=features, actions=actions)
        assert "position_action" in predictions
        assert predictions["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            NUM_MIXTURE_COMPONENTS,
            POSITION_DIM,
        )
        assert DecoderOutputKey.ROUTING_WEIGHTS.value in predictions

    def test_inference_returns_sampled_actions(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = mode_act_factory(deterministic_inference=True)
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features, actions=None)
        assert "position_action" in predictions
        assert predictions["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            POSITION_DIM,
        )

    def test_training_with_multiple_action_heads(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = mode_act_factory(
            position_dim=3,
            has_orientation=True,
            orientation_dim=4,
            has_gripper=True,
            gripper_dim=1,
        )
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = noisy_actions_factory(
            prediction_horizon=PREDICTION_HORIZON,
            action_keys_to_dims={
                "position_action": 3,
                "orientation_action": 4,
                "gripper_action": 1,
            },
        )
        predictions = decoder(features=features, actions=actions)
        assert predictions["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            NUM_MIXTURE_COMPONENTS,
            3,
        )
        assert predictions["orientation_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            NUM_MIXTURE_COMPONENTS,
            4,
        )
        assert predictions["gripper_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            NUM_MIXTURE_COMPONENTS,
            1,
        )


class TestModeACTSampling:

    def test_sample_from_gaussian_mixture_deterministic(
        self,
        rng: np.random.Generator,
    ):
        batch_size = 4
        prediction_horizon = 6
        num_components = 3
        action_dim = 5
        mean = torch.from_numpy(
            rng.standard_normal(
                (batch_size, prediction_horizon, num_components, action_dim)
            ).astype(np.float32)
        )
        logvar = torch.from_numpy(
            rng.standard_normal(
                (batch_size, prediction_horizon, num_components, action_dim)
            ).astype(np.float32)
        )
        routing_weights = torch.softmax(
            torch.from_numpy(
                rng.standard_normal((batch_size, num_components)).astype(np.float32)
            ),
            dim=-1,
        )
        result = MixtureOfDensitiesActionTransformer._sample_from_gaussian_mixture(
            mean=mean,
            logvar=logvar,
            routing_weights=routing_weights,
            deterministic=True,
        )
        assert result.shape == (batch_size, prediction_horizon, action_dim)
        # Deterministic selects argmax component and returns its mean
        selected_indices = torch.argmax(routing_weights, dim=-1)
        for batch_index in range(batch_size):
            expected = mean[batch_index, :, selected_indices[batch_index], :]
            torch.testing.assert_close(
                result[batch_index], expected, atol=0, rtol=0
            )

    def test_sample_from_gaussian_mixture_stochastic(
        self,
        rng: np.random.Generator,
    ):
        batch_size = 4
        prediction_horizon = 6
        num_components = 3
        action_dim = 5
        mean = torch.from_numpy(
            rng.standard_normal(
                (batch_size, prediction_horizon, num_components, action_dim)
            ).astype(np.float32)
        )
        logvar = torch.zeros(
            batch_size, prediction_horizon, num_components, action_dim
        )
        routing_weights = torch.softmax(
            torch.from_numpy(
                rng.standard_normal((batch_size, num_components)).astype(np.float32)
            ),
            dim=-1,
        )
        result = MixtureOfDensitiesActionTransformer._sample_from_gaussian_mixture(
            mean=mean,
            logvar=logvar,
            routing_weights=routing_weights,
            deterministic=False,
        )
        assert result.shape == (batch_size, prediction_horizon, action_dim)
        # Stochastic adds noise, so result differs from mean (with high probability)
        # We just verify the shape is correct and it ran without error

    def test_sample_from_mixture_deterministic(
        self,
        rng: np.random.Generator,
    ):
        batch_size = 4
        prediction_horizon = 6
        num_components = 3
        action_dim = 5
        stacked = torch.from_numpy(
            rng.standard_normal(
                (batch_size, prediction_horizon, num_components, action_dim)
            ).astype(np.float32)
        )
        routing_weights = torch.softmax(
            torch.from_numpy(
                rng.standard_normal((batch_size, num_components)).astype(np.float32)
            ),
            dim=-1,
        )
        result = MixtureOfDensitiesActionTransformer._sample_from_mixture(
            stacked=stacked,
            routing_weights=routing_weights,
            deterministic=True,
        )
        assert result.shape == (batch_size, prediction_horizon, action_dim)
        selected_indices = torch.argmax(routing_weights, dim=-1)
        for batch_index in range(batch_size):
            expected = stacked[batch_index, :, selected_indices[batch_index], :]
            torch.testing.assert_close(
                result[batch_index], expected, atol=0, rtol=0
            )

    def test_sample_from_mixture_stochastic(
        self,
        rng: np.random.Generator,
    ):
        batch_size = 4
        prediction_horizon = 6
        num_components = 3
        action_dim = 5
        stacked = torch.from_numpy(
            rng.standard_normal(
                (batch_size, prediction_horizon, num_components, action_dim)
            ).astype(np.float32)
        )
        routing_weights = torch.softmax(
            torch.from_numpy(
                rng.standard_normal((batch_size, num_components)).astype(np.float32)
            ),
            dim=-1,
        )
        result = MixtureOfDensitiesActionTransformer._sample_from_mixture(
            stacked=stacked,
            routing_weights=routing_weights,
            deterministic=False,
        )
        assert result.shape == (batch_size, prediction_horizon, action_dim)


class TestModeACTGMMInitialization:

    def test_compute_uniform_centers_shape_and_range(
        self,
        rng: np.random.Generator,
    ):
        out_dim = 5
        num_components = 4
        data_min = torch.from_numpy(
            rng.standard_normal((out_dim,)).astype(np.float32)
        )
        data_max = data_min + torch.abs(
            torch.from_numpy(
                rng.standard_normal((out_dim,)).astype(np.float32)
            )
        ) + 0.1
        centers = MixtureOfDensitiesActionTransformer._compute_uniform_centers(
            data_min=data_min,
            data_max=data_max,
            number_of_mixture_components=num_components,
        )
        assert centers.shape == (num_components, out_dim)
        # First center should be at data_min, last at data_max
        torch.testing.assert_close(centers[0], data_min, atol=1e-6, rtol=0)
        torch.testing.assert_close(centers[-1], data_max, atol=1e-6, rtol=0)
        # All centers should be within [data_min, data_max]
        for component_index in range(num_components):
            assert torch.all(centers[component_index] >= data_min - 1e-6)
            assert torch.all(centers[component_index] <= data_max + 1e-6)

    def test_compute_kmeans_plus_plus_centers_shape(
        self,
        rng: np.random.Generator,
    ):
        out_dim = 5
        num_components = 4
        data_min = torch.from_numpy(
            rng.standard_normal((out_dim,)).astype(np.float32)
        )
        data_max = data_min + torch.abs(
            torch.from_numpy(
                rng.standard_normal((out_dim,)).astype(np.float32)
            )
        ) + 0.1
        centers = MixtureOfDensitiesActionTransformer._compute_kmeans_plus_plus_centers(
            data_min=data_min,
            data_max=data_max,
            number_of_mixture_components=num_components,
            out_dim=out_dim,
        )
        assert centers.shape == (num_components, out_dim)
        # All centers should be within [data_min, data_max]
        for component_index in range(num_components):
            assert torch.all(centers[component_index] >= data_min - 1e-6)
            assert torch.all(centers[component_index] <= data_max + 1e-6)

    def test_gating_feature_key_uses_external_feature(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        gating_key = "gating_input"
        decoder = mode_act_factory(
            input_keys=["rgb_features"],
            gating_feature_key=gating_key,
        )
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        gating_features = flat_feature_factory(
            batch_size=BATCH_SIZE,
            feature_dim=EMBEDDING_DIMENSION,
            feature_keys=[gating_key],
        )
        features.update(gating_features)
        actions = noisy_actions_factory(
            prediction_horizon=PREDICTION_HORIZON,
            action_keys_to_dims={"position_action": POSITION_DIM},
        )
        predictions = decoder(features=features, actions=actions)
        assert DecoderOutputKey.ROUTING_WEIGHTS.value in predictions
        assert predictions[DecoderOutputKey.ROUTING_WEIGHTS.value].shape == (
            BATCH_SIZE,
            NUM_MIXTURE_COMPONENTS,
        )

    def test_compute_uniform_centers_single_component(
        self,
        rng: np.random.Generator,
    ):
        out_dim = 3
        data_min = torch.from_numpy(
            rng.standard_normal((out_dim,)).astype(np.float32)
        )
        data_max = data_min + 1.0
        centers = MixtureOfDensitiesActionTransformer._compute_uniform_centers(
            data_min=data_min,
            data_max=data_max,
            number_of_mixture_components=1,
        )
        assert centers.shape == (1, out_dim)
        # Single component should be at midpoint (alpha=0.5)
        expected = data_min + 0.5 * (data_max - data_min)
        torch.testing.assert_close(centers[0], expected, atol=1e-6, rtol=0)


class TestModeACTInferenceMode:

    def test_stochastic_gaussian_inference(
        self,
        gaussian_mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gaussian_mode_act_factory(deterministic_inference=False)
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features, actions=None)
        assert "position_action" in predictions
        assert predictions["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            POSITION_DIM,
        )

    def test_stochastic_action_head_inference(
        self,
        mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = mode_act_factory(deterministic_inference=False)
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features, actions=None)
        assert "position_action" in predictions
        assert predictions["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            POSITION_DIM,
        )

    def test_deterministic_inference_is_reproducible(
        self,
        gaussian_mode_act_factory: Callable[..., MixtureOfDensitiesActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gaussian_mode_act_factory(deterministic_inference=True)
        decoder.eval()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions_first = decoder(features=features, actions=None)
        predictions_second = decoder(features=features, actions=None)
        torch.testing.assert_close(
            predictions_first["position_action"],
            predictions_second["position_action"],
            atol=1e-6,
            rtol=1e-6,
        )
