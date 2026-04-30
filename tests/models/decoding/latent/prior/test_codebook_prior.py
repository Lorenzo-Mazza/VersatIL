"""Tests for versatil.models.decoding.latent.prior.codebook_prior module."""

import copy
import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.variational import VariationalAlgorithm
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.prior.codebook_prior import CodebookPrior
from versatil.models.decoding.latent.vq.residual_vq import ResidualVQ


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


class _PosteriorOwner(torch.nn.Module):
    def __init__(self, residual_vq: ResidualVQ) -> None:
        super().__init__()
        self.latent_dimension = residual_vq.code_dim
        self.residual_vq = residual_vq


@pytest.mark.integration
def test_algorithm_deepcopy_rewires_prior_to_copied_codebook(
    codebook_prior_factory: Callable[..., CodebookPrior],
) -> None:
    prior = codebook_prior_factory(latent_dimension=8, num_codes=4)
    posterior = _PosteriorOwner(
        ResidualVQ(
            input_dim=8,
            code_dim=8,
            num_codes=prior.num_codes,
            num_layers=prior.num_residual_layers,
            kmeans_init=True,
        )
    )
    algorithm = VariationalAlgorithm(
        base_algorithm=MagicMock(spec=DecodingAlgorithm),
        posterior_encoder=posterior,
        prior=prior,
    )
    cloned_algorithm = copy.deepcopy(algorithm)

    original_embedding = algorithm.posterior_encoder.residual_vq.layers[
        0
    ].codebook.embed
    cloned_embedding = cloned_algorithm.posterior_encoder.residual_vq.layers[
        0
    ].codebook.embed
    cloned_algorithm.prior.residual_vq.layers[0].codebook.embed.fill_(7.0)

    torch.testing.assert_close(cloned_embedding, torch.full_like(cloned_embedding, 7.0))
    torch.testing.assert_close(original_embedding, torch.zeros_like(original_embedding))


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

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "latent_dimension, num_codes, num_residual_layers, temperature, expected_message",
        [
            (0, 4, 1, 1.0, "latent_dimension must be positive, got 0."),
            (8, 0, 1, 1.0, "num_codes must be positive, got 0."),
            (8, 4, 0, 1.0, "num_residual_layers must be positive, got 0."),
            (8, 4, 1, 0.0, "temperature must be positive, got 0.0."),
        ],
    )
    def test_rejects_invalid_configuration(
        self,
        latent_dimension: int,
        num_codes: int,
        num_residual_layers: int,
        temperature: float,
        expected_message: str,
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            CodebookPrior(
                latent_dimension=latent_dimension,
                num_codes=num_codes,
                num_residual_layers=num_residual_layers,
                embedding_dimension=16,
                observation_horizon=1,
                device="cpu",
                temperature=temperature,
            )

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
    def test_shared_residual_vq_is_not_registered_as_prior_child(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = codebook_prior_factory(latent_dimension=8)
        mock_posterior = mock_vq_posterior_factory(code_dim=8)
        prior.wire_posterior(mock_posterior)

        assert prior.residual_vq is mock_posterior.residual_vq
        module_names = {name for name, _ in prior.named_modules()}
        assert "residual_vq" not in module_names
        assert all(not name.startswith("residual_vq.") for name in module_names)
        prior.eval()
        assert mock_posterior.residual_vq.training is True

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

    @pytest.mark.unit
    def test_raises_on_num_codes_mismatch(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = codebook_prior_factory(latent_dimension=8, num_codes=4)
        mock_posterior = mock_vq_posterior_factory(code_dim=8, num_codes=8)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "ResidualVQ num_codes (8) does not match CodebookPrior num_codes (4)"
            ),
        ):
            prior.wire_posterior(mock_posterior)

    @pytest.mark.unit
    def test_raises_on_num_layers_mismatch(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = codebook_prior_factory(latent_dimension=8, num_residual_layers=2)
        mock_posterior = mock_vq_posterior_factory(code_dim=8, num_layers=3)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "ResidualVQ num_layers (3) does not match "
                "CodebookPrior num_residual_layers (2)"
            ),
        ):
            prior.wire_posterior(mock_posterior)

    @pytest.mark.unit
    def test_raises_on_missing_residual_vq_attribute(
        self,
        codebook_prior_factory: Callable[..., CodebookPrior],
    ) -> None:
        prior = codebook_prior_factory(latent_dimension=8)
        mock_posterior = MagicMock(spec=[])
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "Posterior MagicMock does not expose a "
                "residual_vq attribute required by CodebookPrior."
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
                "CodebookPrior.residual_vq is not set or has been garbage-collected. "
                "Call wire_posterior() before forward(), and keep the posterior alive."
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
