"""Tests for versatil.models.decoding.latent.posterior.vq_encoder module."""

import logging
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.posterior.vq_encoder import VQPosteriorEncoder


@pytest.fixture
def vq_encoder_factory() -> Callable[..., VQPosteriorEncoder]:

    def factory(
        latent_dimension: int = 8,
        num_codes: int = 4,
        num_residual_layers: int = 1,
        embedding_dimension: int = 16,
        prediction_horizon: int = 4,
        observation_horizon: int = 1,
        attention_dropout: float = 0.0,
        normalization_type: str = "rmsnorm",
        attention_type: str = "mha",
        positional_encoding_type: str | None = None,
    ) -> VQPosteriorEncoder:
        return VQPosteriorEncoder(
            latent_dimension=latent_dimension,
            num_codes=num_codes,
            num_residual_layers=num_residual_layers,
            embedding_dimension=embedding_dimension,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            device="cpu",
            number_of_heads=2,
            feedforward_dimension=32,
            number_of_encoder_layers=1,
            dropout_rate=0.0,
            attention_dropout=attention_dropout,
            normalization_type=normalization_type,
            attention_type=attention_type,
            positional_encoding_type=positional_encoding_type,
        )

    return factory


@pytest.fixture
def mock_residual_vq_factory(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:

    def factory(
        batch_size: int = 4,
        latent_dimension: int = 8,
        num_codes: int = 4,
        num_layers: int = 1,
    ) -> MagicMock:
        z_q = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        )
        all_indices = [
            torch.from_numpy(
                rng.integers(0, num_codes, size=(batch_size,)).astype(np.int64)
            )
            for _ in range(num_layers)
        ]
        z_e_per_layer = torch.from_numpy(
            rng.standard_normal((num_layers, batch_size, latent_dimension)).astype(
                np.float32
            )
        )
        z_q_per_layer = torch.from_numpy(
            rng.standard_normal((num_layers, batch_size, latent_dimension)).astype(
                np.float32
            )
        )
        return MagicMock(
            spec=torch.nn.Module,
            return_value=(z_q, all_indices, z_e_per_layer, z_q_per_layer),
        )

    return factory


class TestVQPosteriorEncoderInit:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "latent_dimension, num_codes, num_residual_layers",
        [(8, 4, 1), (16, 16, 2), (32, 2, 3)],
    )
    def test_stores_configuration(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
        latent_dimension: int,
        num_codes: int,
        num_residual_layers: int,
    ) -> None:
        encoder = vq_encoder_factory(
            latent_dimension=latent_dimension,
            num_codes=num_codes,
            num_residual_layers=num_residual_layers,
        )
        assert encoder.code_dim == latent_dimension
        assert encoder.num_codes == num_codes
        assert encoder.num_residual_layers == num_residual_layers
        assert encoder.latent_dimension == latent_dimension
        assert len(encoder.residual_vq.layers) == num_residual_layers

    @pytest.mark.parametrize("positional_encoding_type", [None, "rope"])
    def test_positional_encoding_type_forwarded_to_transformer(
        self,
        positional_encoding_type: str | None,
    ) -> None:
        with patch(
            "versatil.models.decoding.latent.posterior.vq_encoder.TransformerEncoder"
        ) as mock_encoder_cls:
            VQPosteriorEncoder(
                latent_dimension=8,
                num_codes=4,
                num_residual_layers=1,
                embedding_dimension=16,
                prediction_horizon=4,
                observation_horizon=1,
                device="cpu",
                number_of_heads=2,
                feedforward_dimension=32,
                number_of_encoder_layers=1,
                positional_encoding_type=positional_encoding_type,
            )
        assert (
            mock_encoder_cls.call_args.kwargs["positional_encoding_type"]
            == positional_encoding_type
        )


class TestVQPosteriorEncoderGetAuxiliaryOutputKeys:
    @pytest.mark.unit
    def test_returns_vq_specific_keys(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
    ) -> None:
        encoder = vq_encoder_factory(latent_dimension=8)
        keys = encoder.get_auxiliary_output_keys()
        assert keys == {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.VQ_INDICES.value,
            LatentKey.VQ_Z_CONTINUOUS.value,
            LatentKey.VQ_QUANTIZED.value,
        }

    @pytest.mark.unit
    def test_does_not_contain_gaussian_keys(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
    ) -> None:
        encoder = vq_encoder_factory(latent_dimension=8)
        keys = encoder.get_auxiliary_output_keys()
        assert LatentKey.POSTERIOR_MU.value not in keys
        assert LatentKey.POSTERIOR_LOGVAR.value not in keys


class TestVQPosteriorEncoderEncode:
    @pytest.mark.unit
    def test_calls_residual_vq_with_projected_output(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        mock_residual_vq_factory: Callable[..., MagicMock],
    ) -> None:
        latent_dimension = 8
        batch_size = 4
        encoder = vq_encoder_factory(latent_dimension=latent_dimension)
        actions = action_dictionary_factory(
            batch_size=batch_size, prediction_horizon=4, action_dimension=2
        )
        mock_rvq = mock_residual_vq_factory(
            batch_size=batch_size, latent_dimension=latent_dimension
        )
        encoder.residual_vq = mock_rvq

        encoder.encode(actions)

        mock_rvq.assert_called_once()
        call_input = mock_rvq.call_args[0][0]
        assert call_input.shape == (batch_size, latent_dimension)

    @pytest.mark.unit
    def test_output_keys_match_get_auxiliary_output_keys(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        mock_residual_vq_factory: Callable[..., MagicMock],
    ) -> None:
        latent_dimension = 8
        batch_size = 4
        encoder = vq_encoder_factory(latent_dimension=latent_dimension)
        actions = action_dictionary_factory(
            batch_size=batch_size, prediction_horizon=4, action_dimension=2
        )
        encoder.residual_vq = mock_residual_vq_factory(
            batch_size=batch_size, latent_dimension=latent_dimension
        )

        result = encoder.encode(actions)

        assert encoder.get_auxiliary_output_keys().issubset(result.keys())

    @pytest.mark.unit
    def test_logs_warning_when_padding_key_missing(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        mock_residual_vq_factory: Callable[..., MagicMock],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        latent_dimension = 8
        batch_size = 4
        encoder = vq_encoder_factory(latent_dimension=latent_dimension)
        actions = action_dictionary_factory(
            batch_size=batch_size,
            prediction_horizon=4,
            action_dimension=2,
            include_padding_mask=False,
        )
        encoder.residual_vq = mock_residual_vq_factory(
            batch_size=batch_size, latent_dimension=latent_dimension
        )

        with caplog.at_level(logging.WARNING):
            encoder.encode(actions)

        assert "No padding key found in actions; assuming no padding." in caplog.text

    @pytest.mark.unit
    def test_no_warning_when_padding_key_present(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        mock_residual_vq_factory: Callable[..., MagicMock],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        latent_dimension = 8
        batch_size = 4
        encoder = vq_encoder_factory(latent_dimension=latent_dimension)
        actions = action_dictionary_factory(
            batch_size=batch_size,
            prediction_horizon=4,
            action_dimension=2,
            include_padding_mask=True,
        )
        assert SampleKey.IS_PAD_ACTION.value in actions
        encoder.residual_vq = mock_residual_vq_factory(
            batch_size=batch_size, latent_dimension=latent_dimension
        )

        with caplog.at_level(logging.WARNING):
            encoder.encode(actions)

        assert (
            "No padding key found in actions; assuming no padding." not in caplog.text
        )


class TestVQPosteriorEncoderEncodeIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("latent_dimension", [4, 16])
    @pytest.mark.parametrize("num_residual_layers", [1, 2])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_output_shapes(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        latent_dimension: int,
        num_residual_layers: int,
        batch_size: int,
    ) -> None:
        encoder = vq_encoder_factory(
            latent_dimension=latent_dimension, num_residual_layers=num_residual_layers
        )
        encoder.eval()
        actions = action_dictionary_factory(
            batch_size=batch_size, prediction_horizon=4, action_dimension=2
        )

        result = encoder.encode(actions)

        assert result[LatentKey.POSTERIOR_LATENT.value].shape == (
            batch_size,
            latent_dimension,
        )
        # VQ_Z_CONTINUOUS / VQ_QUANTIZED are stacked per-layer in code space.
        assert result[LatentKey.VQ_Z_CONTINUOUS.value].shape == (
            num_residual_layers,
            batch_size,
            latent_dimension,
        )
        assert result[LatentKey.VQ_QUANTIZED.value].shape == (
            num_residual_layers,
            batch_size,
            latent_dimension,
        )
        assert len(result[LatentKey.VQ_INDICES.value]) == num_residual_layers
        for indices in result[LatentKey.VQ_INDICES.value]:
            assert indices.shape == (batch_size,)

    @pytest.mark.integration
    def test_quantized_is_detached(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        encoder = vq_encoder_factory(latent_dimension=8)
        encoder.eval()
        actions = action_dictionary_factory(
            batch_size=4, prediction_horizon=4, action_dimension=2
        )

        result = encoder.encode(actions)

        assert not result[LatentKey.VQ_QUANTIZED.value].requires_grad

    @pytest.mark.integration
    def test_z_continuous_carries_gradient(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        encoder = vq_encoder_factory(latent_dimension=8)
        encoder.train()
        actions = action_dictionary_factory(
            batch_size=4, prediction_horizon=4, action_dimension=2
        )

        result = encoder.encode(actions)

        # VQ_Z_CONTINUOUS carries the encoder's gradient so the commitment
        # loss can push the encoder toward the codebook.
        assert result[LatentKey.VQ_Z_CONTINUOUS.value].requires_grad is True

    @pytest.mark.integration
    def test_straight_through_gradient_reaches_projection(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        encoder = vq_encoder_factory(latent_dimension=8)
        encoder.train()
        actions = action_dictionary_factory(
            batch_size=4, prediction_horizon=4, action_dimension=2
        )

        result = encoder.encode(actions)
        result[LatentKey.POSTERIOR_LATENT.value].sum().backward()

        assert encoder.latent_projection.weight.grad is not None

    @pytest.mark.integration
    def test_encode_with_observations(
        self,
        vq_encoder_factory: Callable[..., VQPosteriorEncoder],
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        latent_dimension = 8
        batch_size = 4
        encoder = vq_encoder_factory(latent_dimension=latent_dimension)
        encoder.eval()
        actions = action_dictionary_factory(
            batch_size=batch_size, prediction_horizon=4, action_dimension=2
        )
        observations = flat_feature_factory(
            batch_size=batch_size, feature_dim=16, feature_keys=["state_features"]
        )

        result = encoder.encode(actions=actions, observations=observations)

        assert result[LatentKey.POSTERIOR_LATENT.value].shape == (
            batch_size,
            latent_dimension,
        )

    @pytest.mark.integration
    def test_exclude_keys_filters_observations(
        self,
        action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        latent_dimension = 8
        batch_size = 4
        embedding_dimension = 16
        encoder = VQPosteriorEncoder(
            latent_dimension=latent_dimension,
            num_codes=4,
            num_residual_layers=1,
            embedding_dimension=embedding_dimension,
            prediction_horizon=4,
            observation_horizon=1,
            device="cpu",
            number_of_heads=2,
            feedforward_dimension=32,
            number_of_encoder_layers=1,
            dropout_rate=0.0,
            exclude_keys=["excluded_feature"],
        )
        encoder.eval()
        actions = action_dictionary_factory(
            batch_size=batch_size, prediction_horizon=4, action_dimension=2
        )
        observations = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=embedding_dimension,
            feature_keys=["state_features", "excluded_feature"],
        )

        result = encoder.encode(actions=actions, observations=observations)

        assert result[LatentKey.POSTERIOR_LATENT.value].shape == (
            batch_size,
            latent_dimension,
        )
