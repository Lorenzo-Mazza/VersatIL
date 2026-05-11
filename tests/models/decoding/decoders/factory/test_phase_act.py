"""Tests for versatil.models.decoding.decoders.factory.phase_act module."""

import re
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from versatil_constants.tso import TSOObsKey

from versatil.configs.experiment import ExperimentConfig
from versatil.models.decoding.action_heads.moe import MoEHead
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.factory.act import ACT
from versatil.models.decoding.decoders.factory.phase_act import PhaseACT
from versatil.models.feature_meta import FeatureType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.training.callbacks.confusion_matrix import ConfusionMatrixCallback

EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_ENCODER_LAYERS = 1
NUMBER_OF_DECODER_LAYERS = 1
FEEDFORWARD_DIMENSION = 64
SPATIAL_HEIGHT = 4
SPATIAL_WIDTH = 4
BATCH_SIZE = 2
POSITION_DIM = 3
PREDICTION_HORIZON = 4
NUM_PHASES = 3


@pytest.fixture
def phase_action_space_factory(
    mock_action_space_factory: Callable[..., MagicMock],
) -> Callable[..., MagicMock]:
    """Factory for mock ActionSpace with phase_label metadata entry."""

    def factory(
        position_dim: int = POSITION_DIM,
        num_phases: int = NUM_PHASES,
    ) -> MagicMock:
        action_space = mock_action_space_factory(position_dim=position_dim)
        action_space.actions_metadata[TSOObsKey.PHASE_LABEL.value] = SimpleNamespace(
            requires_prediction_head=True,
            prediction_dimension=num_phases,
        )
        action_space.get_total_action_dim.return_value = position_dim + num_phases
        return action_space

    return factory


@pytest.fixture
def lazy_moe_head_factory(
    action_head_factory: Callable[..., ActionHead],
) -> Callable[..., MoEHead]:
    """Factory for lazy-initialized MoEHead (no num_experts set yet)."""

    def factory(
        input_dim: int = EMBEDDING_DIMENSION,
    ) -> MoEHead:
        base_expert = action_head_factory(input_dim=input_dim)
        return MoEHead(base_expert=base_expert)

    return factory


@pytest.fixture
def phase_act_factory(
    phase_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    action_head_factory: Callable[..., ActionHead],
    lazy_moe_head_factory: Callable[..., MoEHead],
) -> Callable[..., PhaseACT]:
    """Factory for PhaseACT instances with small dimensions."""

    def factory(
        embedding_dimension: int = EMBEDDING_DIMENSION,
        position_dim: int = POSITION_DIM,
        num_phases: int = NUM_PHASES,
        prediction_horizon: int = PREDICTION_HORIZON,
        observation_horizon: int = 1,
        phase_routing_key: str = TSOObsKey.PHASE_LABEL.value,
    ) -> PhaseACT:
        action_space = phase_action_space_factory(
            position_dim=position_dim,
            num_phases=num_phases,
        )
        observation_space = mock_observation_space_factory()
        phase_head = action_head_factory(input_dim=embedding_dimension)
        moe_position_head = lazy_moe_head_factory(input_dim=embedding_dimension)
        action_heads = {
            TSOObsKey.PHASE_LABEL.value: phase_head,
            "position_action": moe_position_head,
        }
        return PhaseACT(
            input_keys=["rgb_features"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device="cpu",
            embedding_dimension=embedding_dimension,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_encoder_layers=NUMBER_OF_ENCODER_LAYERS,
            number_of_decoder_layers=NUMBER_OF_DECODER_LAYERS,
            feedforward_dimension=FEEDFORWARD_DIMENSION,
            phase_routing_key=phase_routing_key,
        )

    return factory


class TestPhaseACTInitialization:
    def test_inherits_from_act(
        self,
        phase_act_factory: Callable[..., PhaseACT],
    ):
        decoder = phase_act_factory()
        assert isinstance(decoder, ACT)

    @pytest.mark.parametrize("num_phases", [3, 5])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    def test_stores_configuration(
        self,
        phase_act_factory: Callable[..., PhaseACT],
        num_phases: int,
        embedding_dimension: int,
    ):
        decoder = phase_act_factory(
            num_phases=num_phases,
            embedding_dimension=embedding_dimension,
        )
        assert decoder.phase_routing_key == TSOObsKey.PHASE_LABEL.value
        assert decoder.embedding_dimension == embedding_dimension

    def test_raises_without_phase_head(
        self,
        phase_action_space_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
        lazy_moe_head_factory: Callable[..., MoEHead],
    ):
        action_space = phase_action_space_factory()
        observation_space = mock_observation_space_factory()
        moe_position_head = lazy_moe_head_factory()
        action_heads = {
            "position_action": moe_position_head,
        }
        missing_heads = {TSOObsKey.PHASE_LABEL.value}
        configured_heads = {"position_action"}
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Action space requires heads for {missing_heads}, but they are not configured. "
                f"Configured heads: {configured_heads}"
            ),
        ):
            PhaseACT(
                input_keys=["rgb_features"],
                action_space=action_space,
                action_heads=action_heads,
                observation_space=observation_space,
                observation_horizon=1,
                prediction_horizon=PREDICTION_HORIZON,
                device="cpu",
                embedding_dimension=EMBEDDING_DIMENSION,
                number_of_heads=NUMBER_OF_HEADS,
                number_of_encoder_layers=NUMBER_OF_ENCODER_LAYERS,
                number_of_decoder_layers=NUMBER_OF_DECODER_LAYERS,
                feedforward_dimension=FEEDFORWARD_DIMENSION,
                phase_routing_key=TSOObsKey.PHASE_LABEL.value,
            )

    def test_raises_without_moe_head(
        self,
        phase_action_space_factory: Callable[..., MagicMock],
        mock_observation_space_factory: Callable[..., MagicMock],
        action_head_factory: Callable[..., ActionHead],
    ):
        action_space = phase_action_space_factory()
        observation_space = mock_observation_space_factory()
        phase_head = action_head_factory(input_dim=EMBEDDING_DIMENSION)
        position_head = action_head_factory(input_dim=EMBEDDING_DIMENSION)
        action_heads = {
            TSOObsKey.PHASE_LABEL.value: phase_head,
            "position_action": position_head,
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                "PhaseACT requires at least one MoE action head for phase-based routing."
            ),
        ):
            PhaseACT(
                input_keys=["rgb_features"],
                action_space=action_space,
                action_heads=action_heads,
                observation_space=observation_space,
                observation_horizon=1,
                prediction_horizon=PREDICTION_HORIZON,
                device="cpu",
                embedding_dimension=EMBEDDING_DIMENSION,
                number_of_heads=NUMBER_OF_HEADS,
                number_of_encoder_layers=NUMBER_OF_ENCODER_LAYERS,
                number_of_decoder_layers=NUMBER_OF_DECODER_LAYERS,
                feedforward_dimension=FEEDFORWARD_DIMENSION,
                phase_routing_key=TSOObsKey.PHASE_LABEL.value,
            )

    def test_initializes_lazy_moe_experts(
        self,
        phase_act_factory: Callable[..., PhaseACT],
    ):
        decoder = phase_act_factory(num_phases=NUM_PHASES)
        moe_head = decoder.action_heads["position_action"]
        assert isinstance(moe_head, MoEHead)
        assert moe_head.is_initialized is True
        assert len(moe_head.experts) == NUM_PHASES

    def test_decoder_input_requires_spatial(
        self,
        phase_act_factory: Callable[..., PhaseACT],
    ):
        decoder = phase_act_factory()
        assert FeatureType.SPATIAL.value in decoder.decoder_input.required_types
        assert decoder.decoder_input.requires_actions is False


class TestPhaseACTForward:
    def test_output_keys(
        self,
        phase_act_factory: Callable[..., PhaseACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = phase_act_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features)
        expected_keys = {
            TSOObsKey.PHASE_LABEL.value,
            "position_action",
            DecoderOutputKey.ROUTING_WEIGHTS.value,
            f"position_action_{DecoderOutputKey.EXPERT_OUTPUTS.value}",
        }
        assert set(predictions.keys()) == expected_keys

    @pytest.mark.parametrize("prediction_horizon", [4, 8])
    def test_output_shapes(
        self,
        phase_act_factory: Callable[..., PhaseACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        prediction_horizon: int,
    ):
        decoder = phase_act_factory(prediction_horizon=prediction_horizon)
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features)
        assert predictions[TSOObsKey.PHASE_LABEL.value].shape == (
            BATCH_SIZE,
            prediction_horizon,
            NUM_PHASES,
        )
        assert predictions["position_action"].shape == (
            BATCH_SIZE,
            prediction_horizon,
            POSITION_DIM,
        )
        expert_outputs_key = f"position_action_{DecoderOutputKey.EXPERT_OUTPUTS.value}"
        assert predictions[expert_outputs_key].shape == (
            BATCH_SIZE,
            prediction_horizon,
            NUM_PHASES,
            POSITION_DIM,
        )

    def test_phase_logits_used_as_routing_weights(
        self,
        phase_act_factory: Callable[..., PhaseACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = phase_act_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features)
        routing_weights = predictions[DecoderOutputKey.ROUTING_WEIGHTS.value]
        # Routing weights are derived from phase logits via softmax, so last dim == NUM_PHASES
        assert routing_weights.shape[-1] == NUM_PHASES

    def test_routing_selects_expert_matching_dominant_phase(
        self,
        phase_act_factory: Callable[..., PhaseACT],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = phase_act_factory()
        decoder.eval()
        phase_head = decoder.action_heads[TSOObsKey.PHASE_LABEL.value]
        with torch.no_grad():
            phase_head.output_proj.weight.data.zero_()
            phase_head.output_proj.bias.data.zero_()
            phase_head.output_proj.bias.data[0] = 100.0
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        with torch.no_grad():
            predictions = decoder(features=features)
        expert_outputs_key = f"position_action_{DecoderOutputKey.EXPERT_OUTPUTS.value}"
        expert_outputs = predictions[expert_outputs_key]
        routed_output = predictions["position_action"]
        expert_0_output = expert_outputs[:, :, 0, :]
        torch.testing.assert_close(routed_output, expert_0_output, atol=1e-5, rtol=1e-5)


class TestPhaseACTTemporalObservation:
    @pytest.mark.parametrize(
        "observation_horizon, expects_temporal_pe",
        [
            (1, False),
            (3, True),
        ],
    )
    def test_temporal_pe_created_based_on_observation_horizon(
        self,
        phase_act_factory: Callable[..., PhaseACT],
        observation_horizon: int,
        expects_temporal_pe: bool,
    ):
        decoder = phase_act_factory(observation_horizon=observation_horizon)
        layer = decoder.input_sequence_builder.temporal_positional_encoding_layer
        if expects_temporal_pe:
            assert isinstance(layer, LearnedPositionalEncoding1D)
        else:
            assert layer is None


def test_auxiliary_output_keys(
    phase_act_factory: Callable[..., PhaseACT],
):
    decoder = phase_act_factory()
    assert decoder.get_auxiliary_output_keys() == {
        DecoderOutputKey.ROUTING_WEIGHTS.value,
    }


def test_get_callbacks_returns_confusion_matrix(
    phase_act_factory: Callable[..., PhaseACT],
):
    decoder = phase_act_factory()
    experiment_config = MagicMock(spec=ExperimentConfig)
    experiment_config.val_every = 5
    callbacks = decoder.get_callbacks(experiment_config=experiment_config)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], ConfusionMatrixCallback)
    assert callbacks[0].log_every_n_epochs == 5
