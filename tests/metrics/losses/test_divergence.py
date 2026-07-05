"""Tests for versatil.metrics.losses.divergence module."""

import math
import re

import numpy as np
import pytest
import torch

from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.losses.divergence import (
    BinaryKLDivergenceLoss,
    GaussianEntropyLoss,
    KLDivergenceLoss,
)
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey


@pytest.mark.unit
class TestGaussianEntropyLossInit:
    def test_raises_if_key_does_not_contain_logvar(self):
        with pytest.raises(
            ValueError,
            match=re.escape("GaussianEntropyLoss expects a logvar key, got 'mu'."),
        ):
            GaussianEntropyLoss(key="mu")

    def test_stores_parameters(self):
        loss = GaussianEntropyLoss(
            key=LatentKey.PRIOR_LOGVAR.value,
            weight=0.05,
            logvar_min=-3.0,
            logvar_max=3.0,
            bound_weight=2.0,
        )
        assert loss.weight == 0.05
        assert loss.logvar_min == -3.0
        assert loss.logvar_max == 3.0
        assert loss.bound_weight == 2.0


@pytest.mark.unit
class TestGaussianEntropyLossForward:
    def test_entropy_computation_is_correct(self):
        latent_dim = 4
        logvar = torch.zeros(2, latent_dim)  # logvar=0 => variance=1
        # H(N(mu, sigma^2)) = 0.5 * sum_d (1 + log(2*pi) + logvar_d)
        # With logvar=0: per-sample entropy = 0.5 * latent_dim * (1 + log(2*pi))
        expected_entropy_per_sample = 0.5 * latent_dim * (1 + math.log(2 * math.pi))
        expected_mean_entropy = expected_entropy_per_sample
        loss = GaussianEntropyLoss(
            key=LatentKey.PRIOR_LOGVAR.value,
            weight=1.0,
            bound_weight=0.0,
        )
        predictions = {LatentKey.PRIOR_LOGVAR.value: logvar}
        output = loss(predictions, {})
        # total_loss = -weight * entropy_mean + 0.0 * bound
        assert output.total_loss.item() == pytest.approx(-expected_mean_entropy)

    def test_bound_violation_penalizes_out_of_range_logvar(self):
        logvar_max = 2.0
        logvar = torch.tensor([[3.0, 3.0]])  # exceeds logvar_max by 1.0
        loss = GaussianEntropyLoss(
            key=LatentKey.PRIOR_LOGVAR.value,
            weight=0.0,  # disable entropy contribution
            logvar_max=logvar_max,
            logvar_min=-4.0,
            bound_weight=1.0,
        )
        output = loss({LatentKey.PRIOR_LOGVAR.value: logvar}, {})
        # bound_violation = relu(3 - 2)^2.mean() + relu(-4 - 3)^2.mean() = 1.0 + 0 = 1.0
        assert output.total_loss.item() == pytest.approx(1.0)

    def test_raises_on_missing_key(self):
        loss = GaussianEntropyLoss(key=LatentKey.PRIOR_LOGVAR.value)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{LatentKey.PRIOR_LOGVAR.value}' for GaussianEntropyLoss."
            ),
        ):
            loss({}, {})

    def test_component_loss_reports_entropy(self):
        loss = GaussianEntropyLoss(key=LatentKey.PRIOR_LOGVAR.value, weight=1.0)
        logvar = torch.zeros(2, 4)
        output = loss({LatentKey.PRIOR_LOGVAR.value: logvar}, {})
        component_key = f"{LatentKey.PRIOR_LOGVAR.value}_{MetricKey.ENTROPY.value}"
        assert component_key in output.component_losses


@pytest.mark.unit
class TestKLDivergenceLossGetRequiredKeys:
    def test_returns_latent_keys(self):
        loss = KLDivergenceLoss()
        keys = loss.get_required_keys()
        assert LatentKey.POSTERIOR_LATENT.value in keys
        assert LatentKey.PRIOR_LATENT.value in keys
        assert LatentKey.POSTERIOR_MU.value in keys
        assert LatentKey.POSTERIOR_LOGVAR.value in keys


@pytest.mark.unit
class TestKLDivergenceLossForwardClosedForm:
    def test_kl_is_zero_for_identical_distributions(self):
        batch_size, latent_dim = 4, 8
        mu = torch.zeros(batch_size, latent_dim)
        logvar = torch.zeros(batch_size, latent_dim)
        z = torch.zeros(batch_size, latent_dim)
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu,
            LatentKey.POSTERIOR_LOGVAR.value: logvar,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_MU.value: mu.clone(),
            LatentKey.PRIOR_LOGVAR.value: logvar.clone(),
            LatentKey.PRIOR_LATENT.value: z.clone(),
        }
        loss = KLDivergenceLoss(weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_is_positive_for_different_distributions(self, rng):
        batch_size, latent_dim = 4, 8
        mu_post = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dim)).astype(np.float32)
        )
        logvar_post = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dim)).astype(np.float32)
        )
        mu_prior = torch.zeros(batch_size, latent_dim)
        logvar_prior = torch.zeros(batch_size, latent_dim)
        z = mu_post + torch.exp(0.5 * logvar_post)
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu_post,
            LatentKey.POSTERIOR_LOGVAR.value: logvar_post,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_MU.value: mu_prior,
            LatentKey.PRIOR_LOGVAR.value: logvar_prior,
            LatentKey.PRIOR_LATENT.value: torch.zeros(batch_size, latent_dim),
        }
        loss = KLDivergenceLoss(weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() > 0

    def test_weight_scales_kl(self):
        batch_size, latent_dim = 2, 4
        mu_post = torch.ones(batch_size, latent_dim)
        logvar_post = torch.zeros(batch_size, latent_dim)
        mu_prior = torch.zeros(batch_size, latent_dim)
        logvar_prior = torch.zeros(batch_size, latent_dim)
        z = mu_post.clone()
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu_post,
            LatentKey.POSTERIOR_LOGVAR.value: logvar_post,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_MU.value: mu_prior,
            LatentKey.PRIOR_LOGVAR.value: logvar_prior,
            LatentKey.PRIOR_LATENT.value: torch.zeros_like(z),
        }
        loss_w1 = KLDivergenceLoss(weight=1.0)
        loss_w5 = KLDivergenceLoss(weight=5.0)
        output_w1 = loss_w1(predictions, {})
        output_w5 = loss_w5(predictions, {})
        assert output_w5.total_loss.item() == pytest.approx(
            5.0 * output_w1.total_loss.item(), rel=1e-5
        )

    def test_prior_regularization_adds_kl_to_standard_gaussian(self):
        batch_size, latent_dim = 4, 8
        # Prior = N(2, 1) so KL(prior || N(0,I)) > 0
        mu_post = torch.zeros(batch_size, latent_dim)
        logvar_post = torch.zeros(batch_size, latent_dim)
        mu_prior = 2.0 * torch.ones(batch_size, latent_dim)
        logvar_prior = torch.zeros(batch_size, latent_dim)
        z = torch.zeros(batch_size, latent_dim)
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu_post,
            LatentKey.POSTERIOR_LOGVAR.value: logvar_post,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_MU.value: mu_prior,
            LatentKey.PRIOR_LOGVAR.value: logvar_prior,
            LatentKey.PRIOR_LATENT.value: z.clone(),
        }
        loss_no_reg = KLDivergenceLoss(weight=1.0, prior_regularization_weight=0.0)
        loss_with_reg = KLDivergenceLoss(weight=1.0, prior_regularization_weight=1.0)
        output_no_reg = loss_no_reg(predictions, {})
        output_with_reg = loss_with_reg(predictions, {})
        assert output_with_reg.total_loss.item() > output_no_reg.total_loss.item()
        assert (
            MetricKey.HYPERPRIOR_KL_REGULARIZATION.value
            in output_with_reg.component_losses
        )

    def test_metadata_includes_latent_variables(self):
        batch_size, latent_dim = 2, 4
        mu = torch.zeros(batch_size, latent_dim)
        logvar = torch.zeros(batch_size, latent_dim)
        z = torch.zeros(batch_size, latent_dim)
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu,
            LatentKey.POSTERIOR_LOGVAR.value: logvar,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_MU.value: mu.clone(),
            LatentKey.PRIOR_LOGVAR.value: logvar.clone(),
            LatentKey.PRIOR_LATENT.value: z.clone(),
        }
        loss = KLDivergenceLoss(weight=1.0)
        output = loss(predictions, {})
        assert MetadataKey.POSTERIOR_Z.value in output.metadata
        assert MetadataKey.POSTERIOR_MU.value in output.metadata
        assert MetadataKey.POSTERIOR_LOGVAR.value in output.metadata
        assert MetadataKey.PRIOR_Z.value in output.metadata
        assert MetadataKey.PRIOR_MU.value in output.metadata
        assert MetadataKey.PRIOR_LOGVAR.value in output.metadata


@pytest.mark.unit
class TestKLDivergenceLossForwardWithLogProb:
    def test_uses_log_prob_path_when_prior_log_prob_present(self):
        batch_size, latent_dim = 4, 8
        mu = torch.zeros(batch_size, latent_dim)
        logvar = torch.zeros(batch_size, latent_dim)
        z = torch.zeros(batch_size, latent_dim)
        # log_prob of z under N(0,I) prior
        prior_log_prob = (
            torch.distributions.Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
            .log_prob(z)
            .sum(dim=-1)
        )
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu,
            LatentKey.POSTERIOR_LOGVAR.value: logvar,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_LOG_PROB.value: prior_log_prob,
        }
        loss = KLDivergenceLoss(weight=1.0)
        output = loss(predictions, {})
        # KL = E_q[log q - log p], both N(0,I), evaluated at z=0 => KL ≈ 0
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_omits_prior_z_metadata_when_prior_latent_absent(self):
        batch_size, latent_dim = 4, 8
        z = torch.zeros(batch_size, latent_dim)
        predictions = {
            LatentKey.POSTERIOR_MU.value: torch.zeros(batch_size, latent_dim),
            LatentKey.POSTERIOR_LOGVAR.value: torch.zeros(batch_size, latent_dim),
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_LOG_PROB.value: torch.zeros(batch_size),
        }
        loss = KLDivergenceLoss(weight=1.0)
        output = loss(predictions, {})
        # A None entry here used to crash MetricsAccumulator's torch.cat at
        # epoch end for VampPrior runs with sampling_from_prior_probability=0.
        assert MetadataKey.PRIOR_Z.value not in output.metadata
        assert all(value is not None for value in output.metadata.values())


@pytest.mark.unit
class TestBinaryKLDivergenceLossGetRequiredKeys:
    def test_returns_binary_logits_key(self):
        loss = BinaryKLDivergenceLoss()
        assert loss.get_required_keys() == {DecoderOutputKey.BINARY_LOGITS.value}


@pytest.mark.unit
class TestBinaryKLDivergenceLossForward:
    def test_uniform_logits_produce_near_zero_kl(self):
        # sigmoid(0) = 0.5 => Bernoulli(0.5) matches uniform prior
        logits = torch.zeros(4, 3, 8)  # (B, T, H)
        loss = BinaryKLDivergenceLoss(weight=1.0, entropy_weight=0.0, free_bits=0.0)
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output = loss(predictions, {})
        assert output.component_losses[
            MetricKey.RAW_KL_DIVERGENCE.value
        ].item() == pytest.approx(0.0, abs=1e-5)

    def test_padded_tokens_do_not_influence_kl(self):
        logits = torch.zeros(1, 2, 8)  # (B, T, H)
        logits[:, 1] = 20.0  # padded token with extreme logits
        is_pad = torch.tensor([[False, True]])
        loss = BinaryKLDivergenceLoss(weight=1.0, entropy_weight=0.0, free_bits=0.0)
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output = loss(predictions, {}, is_pad=is_pad)
        assert output.component_losses[
            MetricKey.RAW_KL_DIVERGENCE.value
        ].item() == pytest.approx(0.0, abs=1e-5)

    def test_none_logits_return_zero_without_crashing(self):
        predictions = {
            DecoderOutputKey.BINARY_LOGITS.value: None,
        }
        loss = BinaryKLDivergenceLoss(weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0)

    def test_extreme_logits_produce_positive_kl(self):
        logits = 10.0 * torch.ones(4, 3, 8)  # sigmoid(10) ≈ 1
        loss = BinaryKLDivergenceLoss(weight=1.0, entropy_weight=0.0, free_bits=0.0)
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output = loss(predictions, {})
        assert output.component_losses[MetricKey.RAW_KL_DIVERGENCE.value].item() > 0.1

    def test_free_bits_clamps_kl(self):
        logits = torch.zeros(4, 3, 8)  # KL ≈ 0
        free_bits = 1.0
        loss = BinaryKLDivergenceLoss(
            weight=1.0, entropy_weight=0.0, free_bits=free_bits
        )
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output = loss(predictions, {})
        assert output.component_losses[
            MetricKey.CLAMPED_KL_DIVERGENCE.value
        ].item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("free_bits", [0.5, 1.0, 2.0])
    def test_free_bits_reduces_effective_kl_below_threshold(self, free_bits):
        # Extreme logits: sigmoid(5) close to 1 => large per-bit KL
        logits = 5.0 * torch.ones(4, 3, 8)  # (B, T, H)
        loss_no_free = BinaryKLDivergenceLoss(
            weight=1.0, entropy_weight=0.0, free_bits=0.0
        )
        loss_with_free = BinaryKLDivergenceLoss(
            weight=1.0, entropy_weight=0.0, free_bits=free_bits
        )
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output_no_free = loss_no_free(predictions, {})
        output_with_free = loss_with_free(predictions, {})

        raw_kl = output_no_free.component_losses[
            MetricKey.RAW_KL_DIVERGENCE.value
        ].item()
        clamped_kl = output_with_free.component_losses[
            MetricKey.CLAMPED_KL_DIVERGENCE.value
        ].item()
        # Raw KL should exceed free_bits for extreme logits
        assert raw_kl > free_bits
        # Clamped KL = mean(max(0, kl_per_token - free_bits)) < raw_kl
        assert clamped_kl < raw_kl
        # Mathematically: clamped = raw - free_bits (since all tokens have same KL)
        assert clamped_kl == pytest.approx(raw_kl - free_bits, abs=1e-4)

    def test_raises_on_missing_key(self):
        loss = BinaryKLDivergenceLoss()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain key '{DecoderOutputKey.BINARY_LOGITS.value}' for BinaryKLDivergenceLoss."
            ),
        ):
            loss({}, {})

    def test_latent_code_usage_tracked_when_available(self):
        logits = torch.zeros(2, 3, 4)  # (B, T, H)
        latent_codes = torch.zeros(2, 3, 16)  # (B, T, 2^H)
        latent_codes[:, :, 0] = 1.0  # all use code 0
        predictions = {
            DecoderOutputKey.BINARY_LOGITS.value: logits,
            DecoderOutputKey.LATENT_CODES.value: latent_codes,
        }
        loss = BinaryKLDivergenceLoss(weight=1.0, latent_bits=4)
        output = loss(predictions, {})
        assert MetricKey.LATENT_CODE_USAGE.value in output.component_losses
