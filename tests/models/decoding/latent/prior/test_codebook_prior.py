"""Tests for versatil.models.decoding.latent.prior.codebook_prior module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.prior.codebook_prior import CodebookPrior


@pytest.fixture
def codebook_prior_factory() -> Callable[..., CodebookPrior]:

    def factory(
        latent_dimension: int = 8,
        num_codes: int = 4,
        num_residual_layers: int = 1,
        embedding_dimension: int = 16,
        observation_horizon: int = 1,
        temperature: float = 1.0,
        attention_dropout: float = 0.0,
        normalization_type: str = "rmsnorm",
        attention_type: str = "mha",
        positional_encoding_type: str | None = None,
    ) -> CodebookPrior:
        return CodebookPrior(
            latent_dimension=latent_dimension,
            num_codes=num_codes,
            num_residual_layers=num_residual_layers,
            embedding_dimension=embedding_dimension,
            observation_horizon=observation_horizon,
            device="cpu",
            number_of_heads=2,
            feedforward_dimension=32,
            number_of_encoder_layers=1,
            dropout_rate=0.0,
            temperature=temperature,
            attention_dropout=attention_dropout,
            normalization_type=normalization_type,
            attention_type=attention_type,
            positional_encoding_type=positional_encoding_type,
        )

    return factory


class TestCodebookPriorInit:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "latent_dimension, num_codes, num_residual_layers",
        [(8, 4, 1), (16, 16, 2), (32, 2, 3)],
    )
    def test_stores_configuration(
        self,
        latent_dimension: int,
        num_codes: int,
        num_residual_layers: int,
    ) -> None:
        prior = CodebookPrior(
            latent_dimension=latent_dimension,
            num_codes=num_codes,
            num_residual_layers=num_residual_layers,
            embedding_dimension=16,
            observation_horizon=1,
            device="cpu",
            number_of_heads=2,
            feedforward_dimension=32,
            number_of_encoder_layers=1,
        )
        assert prior.code_dim == latent_dimension
        assert prior.num_codes == num_codes
        assert prior.num_residual_layers == num_residual_layers
        assert prior.latent_dimension == latent_dimension
        assert len(prior.code_heads) == num_residual_layers

    @pytest.mark.parametrize("positional_encoding_type", [None, "rope"])
    def test_positional_encoding_type_forwarded_to_transformer(
        self,
        positional_encoding_type: str | None,
    ) -> None:
        with patch(
            "versatil.models.decoding.latent.prior.codebook_prior.TransformerEncoder"
        ) as mock_encoder_cls:
            CodebookPrior(
                latent_dimension=8,
                num_codes=4,
                num_residual_layers=1,
                embedding_dimension=16,
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


class TestCodebookPriorWirePosterior:
    @pytest.mark.unit
    def test_sets_residual_vq_reference(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = codebook_prior_factory(latent_dimension=8)
        mock_posterior = mock_vq_posterior_factory(code_dim=8)
        prior.wire_posterior(mock_posterior)
        assert prior.residual_vq is mock_posterior.residual_vq

    @pytest.mark.unit
    def test_raises_on_code_dim_mismatch(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = codebook_prior_factory(latent_dimension=8)
        mock_posterior = mock_vq_posterior_factory(code_dim=16)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "ResidualVQ code_dim (16) does not match CodebookPrior code_dim (8)"
            ),
        ):
            prior.wire_posterior(mock_posterior)


class TestCodebookPriorGetAuxiliaryOutputKeys:
    @pytest.mark.unit
    def test_returns_codebook_specific_keys(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
    ) -> None:
        prior = codebook_prior_factory(latent_dimension=8)
        keys = prior.get_auxiliary_output_keys()
        assert keys == {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.VQ_PRIOR_INDICES.value,
            LatentKey.PRIOR_CODE_LOGITS.value,
        }

    @pytest.mark.unit
    def test_does_not_contain_gaussian_posterior_or_logprob_keys(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
    ) -> None:
        prior = codebook_prior_factory(latent_dimension=8)
        keys = prior.get_auxiliary_output_keys()
        assert LatentKey.PRIOR_MU.value not in keys
        assert LatentKey.PRIOR_LOGVAR.value not in keys
        assert LatentKey.VQ_INDICES.value not in keys
        assert LatentKey.PRIOR_LOG_PROB.value not in keys


class TestCodebookPriorForward:
    @pytest.mark.unit
    def test_raises_without_residual_vq(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        target_latents_factory: Callable[..., torch.Tensor],
    ) -> None:
        prior = codebook_prior_factory(latent_dimension=8)
        observations = observation_dictionary_factory(batch_size=4)
        target_latents = target_latents_factory(batch_size=4, latent_dim=8)
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "CodebookPrior.residual_vq is not set. "
                "Call set_residual_vq() before forward()."
            ),
        ):
            prior.forward(target_latents=target_latents, observations=observations)


class TestCodebookPriorForwardIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("latent_dimension", [4, 16])
    @pytest.mark.parametrize("num_residual_layers", [1, 2])
    def test_output_shapes(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        target_latents_factory: Callable[..., torch.Tensor],
        latent_dimension: int,
        num_residual_layers: int,
    ) -> None:
        batch_size = 4
        prior = codebook_prior_factory(
            latent_dimension=latent_dimension,
            num_residual_layers=num_residual_layers,
        )
        prior.wire_posterior(
            mock_vq_posterior_factory(
                code_dim=latent_dimension, num_layers=num_residual_layers
            )
        )
        prior.eval()
        observations = observation_dictionary_factory(batch_size=batch_size)
        target_latents = target_latents_factory(
            batch_size=batch_size, latent_dim=latent_dimension
        )

        result = prior.forward(target_latents=target_latents, observations=observations)

        assert result[LatentKey.PRIOR_LATENT.value].shape == (
            batch_size,
            latent_dimension,
        )
        assert LatentKey.VQ_INDICES.value not in result
        assert LatentKey.PRIOR_LOG_PROB.value not in result
        assert len(result[LatentKey.VQ_PRIOR_INDICES.value]) == num_residual_layers
        for indices in result[LatentKey.VQ_PRIOR_INDICES.value]:
            assert indices.shape == (batch_size,)

    @pytest.mark.integration
    def test_output_contains_all_required_keys(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        target_latents_factory: Callable[..., torch.Tensor],
    ) -> None:
        latent_dimension = 8
        num_residual_layers = 2
        prior = codebook_prior_factory(
            latent_dimension=latent_dimension,
            num_residual_layers=num_residual_layers,
        )
        prior.wire_posterior(
            mock_vq_posterior_factory(
                code_dim=latent_dimension, num_layers=num_residual_layers
            )
        )
        prior.eval()
        observations = observation_dictionary_factory(batch_size=4)
        target_latents = target_latents_factory(
            batch_size=4, latent_dim=latent_dimension
        )

        result = prior.forward(target_latents=target_latents, observations=observations)

        expected_keys = {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.VQ_PRIOR_INDICES.value,
            LatentKey.PRIOR_CODE_LOGITS.value,
        }
        assert expected_keys.issubset(result.keys())

    @pytest.mark.integration
    def test_sample_prior_returns_correct_shape(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        batch_size = 6
        latent_dimension = 8
        prior = codebook_prior_factory(latent_dimension=latent_dimension)
        prior.wire_posterior(mock_vq_posterior_factory(code_dim=latent_dimension))
        prior.eval()
        observations = observation_dictionary_factory(batch_size=batch_size)

        z = prior.sample_prior(batch_size=batch_size, observations=observations)

        assert z.shape == (batch_size, latent_dimension)

    @pytest.mark.integration
    def test_output_contains_logits_with_correct_shape(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
        observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
        target_latents_factory: Callable[..., torch.Tensor],
    ) -> None:
        latent_dimension = 8
        num_codes = 4
        num_residual_layers = 2
        batch_size = 4
        prior = codebook_prior_factory(
            latent_dimension=latent_dimension,
            num_codes=num_codes,
            num_residual_layers=num_residual_layers,
        )
        prior.wire_posterior(
            mock_vq_posterior_factory(
                code_dim=latent_dimension,
                num_codes=num_codes,
                num_layers=num_residual_layers,
            )
        )
        prior.eval()
        observations = observation_dictionary_factory(batch_size=batch_size)
        target_latents = target_latents_factory(
            batch_size=batch_size, latent_dim=latent_dimension
        )

        result = prior.forward(target_latents=target_latents, observations=observations)

        logits = result[LatentKey.PRIOR_CODE_LOGITS.value]
        assert len(logits) == num_residual_layers
        for layer_logits in logits:
            assert layer_logits.shape == (batch_size, num_codes)

    @pytest.mark.integration
    def test_sample_prior_raises_without_observations(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = codebook_prior_factory(latent_dimension=8)
        prior.wire_posterior(mock_vq_posterior_factory(code_dim=8))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "CodebookPrior requires observations for conditional sampling."
            ),
        ):
            prior.sample_prior(batch_size=4, observations=None)
