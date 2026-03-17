"""Tests for versatil.models.layers.free_transformer.binary_mapper module."""

import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.free_transformer.binary_mapper import BinaryMapper


class TestBinaryMapperInitialization:
    @pytest.mark.parametrize("latent_bits", [3, 5])
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    def test_stores_configuration(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        latent_bits: int,
        embedding_dimension: int,
    ):
        mapper = binary_mapper_factory(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
        )
        assert mapper.latent_bits == latent_bits
        assert mapper.latent_dim == 2**latent_bits
        assert mapper.embedding_dimension == embedding_dimension

    def test_logit_projection_maps_embedding_to_bits(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        latent_bits = 4
        embedding_dimension = 32
        mapper = binary_mapper_factory(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
        )
        assert mapper.logit_projection.in_features == embedding_dimension
        assert mapper.logit_projection.out_features == latent_bits

    def test_bit_patterns_buffer_has_correct_shape(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        latent_bits = 4
        mapper = binary_mapper_factory(latent_bits=latent_bits)
        assert mapper.bit_patterns.shape == (2**latent_bits, latent_bits)

    def test_bit_patterns_contain_only_zeros_and_ones(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        mapper = binary_mapper_factory(latent_bits=4)
        unique_values = torch.unique(mapper.bit_patterns)
        assert torch.equal(unique_values, torch.tensor([0.0, 1.0]))

    def test_bit_patterns_encode_binary_representations(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        latent_bits = 3
        mapper = binary_mapper_factory(latent_bits=latent_bits)
        # Index 5 in binary (little-endian): 5 = 1*2^0 + 0*2^1 + 1*2^2 => [1, 0, 1]
        expected_pattern_for_index_5 = torch.tensor([1.0, 0.0, 1.0])
        assert torch.equal(mapper.bit_patterns[5], expected_pattern_for_index_5)
        # Index 0 should be all zeros
        expected_pattern_for_index_0 = torch.tensor([0.0, 0.0, 0.0])
        assert torch.equal(mapper.bit_patterns[0], expected_pattern_for_index_0)
        # Index 7 (2^3 - 1) should be all ones
        expected_pattern_for_index_7 = torch.tensor([1.0, 1.0, 1.0])
        assert torch.equal(mapper.bit_patterns[7], expected_pattern_for_index_7)

    def test_bit_patterns_not_updated_by_gradient(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        mapper = binary_mapper_factory(latent_bits=4, embedding_dimension=32)
        bit_patterns_before = mapper.bit_patterns.clone()
        features = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        one_hot, logits = mapper(features=features)
        loss = one_hot.sum()
        loss.backward()
        assert torch.equal(mapper.bit_patterns, bit_patterns_before)


class TestBinaryMapperForward:
    @pytest.mark.parametrize(
        "batch_size, sequence_length",
        [
            (2, 4),
            (1, 8),
            (3, 1),
        ],
    )
    def test_output_shapes(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        sequence_length: int,
    ):
        latent_bits = 4
        embedding_dimension = 32
        mapper = binary_mapper_factory(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
        )
        features = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=embedding_dimension,
        )
        one_hot, logits = mapper(features=features)
        assert one_hot.shape == (batch_size, sequence_length, 2**latent_bits)
        assert logits.shape == (batch_size, sequence_length, latent_bits)

    def test_two_dimensional_input_without_sequence_dimension(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        latent_bits = 4
        embedding_dimension = 32
        batch_size = 3
        mapper = binary_mapper_factory(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
        )
        # (B, embedding_dimension) without sequence dim
        features = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=1,
            embedding_dimension=embedding_dimension,
        ).squeeze(1)
        one_hot, logits = mapper(features=features)
        assert one_hot.shape == (batch_size, 2**latent_bits)
        assert logits.shape == (batch_size, latent_bits)

    def test_one_hot_output_sums_to_one_in_forward_pass(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        # In forward pass value, one_hot = y_hard + g_soft - g_soft.detach()
        # which equals y_hard (since g_soft - g_soft.detach() is zero in forward).
        # y_hard is a proper one-hot vector, so sum should be 1.0.
        mapper = binary_mapper_factory(latent_bits=4, embedding_dimension=32)
        features = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        with torch.no_grad():
            one_hot, _ = mapper(features=features)
        sums = one_hot.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))

    def test_deterministic_mode_produces_consistent_results(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        mapper = binary_mapper_factory(latent_bits=4, embedding_dimension=32)
        features = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        with torch.no_grad():
            one_hot_first, logits_first = mapper(features=features, deterministic=True)
            one_hot_second, logits_second = mapper(
                features=features, deterministic=True
            )
        assert torch.equal(one_hot_first, one_hot_second)
        assert torch.equal(logits_first, logits_second)

    def test_stochastic_mode_uses_bernoulli_sampling(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        # Stochastic forward passes with the same input should sometimes differ.
        # Use a large batch/sequence to make it statistically unlikely to be identical.
        mapper = binary_mapper_factory(latent_bits=4, embedding_dimension=32)
        mapper.logit_projection.weight.data.fill_(0.0)
        mapper.logit_projection.bias.data.fill_(0.0)
        # With logits=0, sigmoid=0.5, so Bernoulli(0.5) should produce variation
        features = sequence_tensor_factory(
            batch_size=8, sequence_length=16, embedding_dimension=32
        )
        with torch.no_grad():
            one_hot_first, _ = mapper(features=features, deterministic=False)
            one_hot_second, _ = mapper(features=features, deterministic=False)
        assert not torch.equal(one_hot_first, one_hot_second)

    def test_different_inputs_produce_different_codes(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        mapper = binary_mapper_factory(latent_bits=4, embedding_dimension=32)
        features_a = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        features_b = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        with torch.no_grad():
            one_hot_a, logits_a = mapper(features=features_a, deterministic=True)
            one_hot_b, logits_b = mapper(features=features_b, deterministic=True)
        assert not torch.equal(logits_a, logits_b)

    def test_deterministic_maps_high_logits_to_bit_one(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        latent_bits = 3
        embedding_dimension = 16
        mapper = binary_mapper_factory(
            latent_bits=latent_bits, embedding_dimension=embedding_dimension
        )
        # Force logits to be large positive => sigmoid > 0.5 => all bits = 1
        # All bits = 1 means index 2^H - 1 = 7 for 3 bits
        mapper.logit_projection.weight.data.fill_(0.0)
        mapper.logit_projection.bias.data.fill_(10.0)
        features = torch.zeros(1, 1, embedding_dimension)
        with torch.no_grad():
            one_hot, logits = mapper(features=features, deterministic=True)
        expected_index = 2**latent_bits - 1  # 7 for 3 bits
        assert one_hot[0, 0, expected_index] == 1.0
        assert one_hot[0, 0].sum() == 1.0

    def test_deterministic_maps_low_logits_to_bit_zero(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        latent_bits = 3
        embedding_dimension = 16
        mapper = binary_mapper_factory(
            latent_bits=latent_bits, embedding_dimension=embedding_dimension
        )
        # Force logits to be large negative => sigmoid < 0.5 => all bits = 0
        # All bits = 0 means index 0
        mapper.logit_projection.weight.data.fill_(0.0)
        mapper.logit_projection.bias.data.fill_(-10.0)
        features = torch.zeros(1, 1, embedding_dimension)
        with torch.no_grad():
            one_hot, logits = mapper(features=features, deterministic=True)
        assert one_hot[0, 0, 0] == 1.0
        assert one_hot[0, 0].sum() == 1.0


class TestBinaryMapperSoftDistribution:
    def test_soft_distribution_sums_to_one(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        rng: np.random.Generator,
    ):
        latent_bits = 3
        mapper = binary_mapper_factory(latent_bits=latent_bits, embedding_dimension=32)
        logits = torch.from_numpy(
            rng.standard_normal((2, 4, latent_bits)).astype(np.float32)
        )
        soft_distribution = mapper._compute_soft_distribution(logits)
        sums = soft_distribution.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_soft_distribution_shape(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        rng: np.random.Generator,
    ):
        latent_bits = 4
        mapper = binary_mapper_factory(latent_bits=latent_bits, embedding_dimension=32)
        logits = torch.from_numpy(
            rng.standard_normal((2, 4, latent_bits)).astype(np.float32)
        )
        soft_distribution = mapper._compute_soft_distribution(logits)
        assert soft_distribution.shape == (2, 4, 2**latent_bits)

    def test_soft_distribution_all_non_negative(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        rng: np.random.Generator,
    ):
        latent_bits = 4
        mapper = binary_mapper_factory(latent_bits=latent_bits, embedding_dimension=32)
        logits = torch.from_numpy(
            rng.standard_normal((2, 4, latent_bits)).astype(np.float32)
        )
        soft_distribution = mapper._compute_soft_distribution(logits)
        assert (soft_distribution >= 0).all()

    def test_soft_distribution_peaks_at_deterministic_code(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        latent_bits = 3
        mapper = binary_mapper_factory(latent_bits=latent_bits, embedding_dimension=32)
        # Large positive logits => all bits = 1 => code index 7
        logits = torch.full((1, 1, latent_bits), 10.0)
        soft_distribution = mapper._compute_soft_distribution(logits)
        expected_peak_index = 2**latent_bits - 1
        assert soft_distribution.argmax(dim=-1).item() == expected_peak_index

    def test_soft_distribution_rejects_wrong_logit_dimension(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        latent_bits = 4
        mapper = binary_mapper_factory(latent_bits=latent_bits, embedding_dimension=32)
        wrong_bits = latent_bits + 1
        logits = torch.zeros(2, 4, wrong_bits)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Logits last dimension {wrong_bits} does not match latent_bits {latent_bits}"
            ),
        ):
            mapper._compute_soft_distribution(logits)

    def test_uniform_logits_produce_uniform_distribution(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        latent_bits = 3
        mapper = binary_mapper_factory(latent_bits=latent_bits, embedding_dimension=32)
        # Logits = 0 => sigmoid = 0.5 => each bit equally likely => uniform over codes
        logits = torch.zeros(1, 1, latent_bits)
        soft_distribution = mapper._compute_soft_distribution(logits)
        expected_uniform = 1.0 / (2**latent_bits)
        assert torch.allclose(
            soft_distribution,
            torch.full_like(soft_distribution, expected_uniform),
            atol=1e-5,
        )


class TestBinaryMapperGradientFlow:
    def test_gradient_flows_through_straight_through_estimator(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        mapper = binary_mapper_factory(latent_bits=4, embedding_dimension=32)
        features = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        features.requires_grad_(True)
        one_hot, logits = mapper(features=features)
        loss = one_hot.sum()
        loss.backward()
        assert features.grad is not None
        assert torch.all(torch.isfinite(features.grad))

    def test_gradient_flows_to_logit_projection_weights(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        mapper = binary_mapper_factory(latent_bits=4, embedding_dimension=32)
        features = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        one_hot, logits = mapper(features=features)
        loss = one_hot.sum()
        loss.backward()
        assert mapper.logit_projection.weight.grad is not None
        assert torch.all(torch.isfinite(mapper.logit_projection.weight.grad))

    def test_straight_through_forward_equals_hard_one_hot(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
        sequence_tensor_factory: Callable[..., torch.Tensor],
    ):
        # The straight-through estimator: forward pass = hard one-hot.
        # y_hard + g_soft - g_soft.detach() = y_hard in forward, g_soft in backward.
        mapper = binary_mapper_factory(latent_bits=4, embedding_dimension=32)
        features = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=32
        )
        with torch.no_grad():
            one_hot, _ = mapper(features=features, deterministic=True)
        # Each position should have exactly one non-zero entry equal to 1.0
        non_zero_counts = (one_hot != 0).sum(dim=-1)
        assert torch.all(non_zero_counts == 1)
        assert torch.allclose(one_hot.sum(dim=-1), torch.ones_like(one_hot.sum(dim=-1)))

    def test_straight_through_gradient_matches_soft_distribution(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        # Verify the STE: gradient of one_hot w.r.t. logits should equal
        # gradient of g_soft (soft distribution), not of y_hard (which is zero).
        latent_bits = 3
        embedding_dimension = 16
        mapper = binary_mapper_factory(
            latent_bits=latent_bits, embedding_dimension=embedding_dimension
        )
        # Use deterministic mode to get consistent bit patterns
        features = torch.zeros(1, 1, embedding_dimension, requires_grad=True)
        one_hot, logits = mapper(features=features, deterministic=True)
        # Pick a specific code index and compute gradient
        target_index = one_hot.argmax(dim=-1).item()
        loss = one_hot[0, 0, target_index]
        loss.backward()
        # The gradient must be non-zero (comes from g_soft, not y_hard)
        assert features.grad is not None
        assert not torch.all(features.grad == 0), (
            "Gradient is all zeros, meaning the straight-through estimator "
            "is not passing gradients from the soft distribution"
        )

    def test_gradient_nonzero_for_non_selected_codes(
        self,
        binary_mapper_factory: Callable[..., BinaryMapper],
    ):
        # The soft distribution has mass on all codes, so gradients should
        # flow even when the loss targets a code that wasn't selected
        latent_bits = 3
        embedding_dimension = 16
        mapper = binary_mapper_factory(
            latent_bits=latent_bits, embedding_dimension=embedding_dimension
        )
        features = torch.zeros(1, 1, embedding_dimension, requires_grad=True)
        one_hot, logits = mapper(features=features, deterministic=True)
        # Pick a code that is NOT the selected one
        selected_index = one_hot.argmax(dim=-1).item()
        other_index = (selected_index + 1) % (2**latent_bits)
        loss = one_hot[0, 0, other_index]
        loss.backward()
        # Despite the forward value being 0, gradient flows via g_soft
        assert features.grad is not None
        assert not torch.all(features.grad == 0), (
            "Gradient is zero for non-selected code, but the STE should "
            "route gradients through the soft distribution to all codes"
        )
