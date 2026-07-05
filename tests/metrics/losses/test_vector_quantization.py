"""Tests for versatil.metrics.losses.vector_quantization module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.losses.vector_quantization import (
    VQCommitmentLoss,
    VQPriorCrossEntropyLoss,
)
from versatil.models.decoding.constants import LatentKey


class TestVQCommitmentLoss:
    @pytest.fixture
    def vq_predictions_factory(
        self, rng: np.random.Generator
    ) -> Callable[..., dict[str, torch.Tensor]]:

        def factory(
            batch_size: int = 8,
            code_dim: int = 16,
            num_codes: int = 4,
            num_layers: int = 1,
        ) -> dict[str, torch.Tensor]:
            z_continuous = torch.from_numpy(
                rng.standard_normal((num_layers, batch_size, code_dim)).astype(
                    np.float32
                )
            )
            z_quantized = torch.from_numpy(
                rng.standard_normal((num_layers, batch_size, code_dim)).astype(
                    np.float32
                )
            )
            all_indices = [
                torch.from_numpy(
                    rng.integers(0, num_codes, size=(batch_size,)).astype(np.int64)
                )
                for _ in range(num_layers)
            ]
            return {
                LatentKey.VQ_Z_CONTINUOUS.value: z_continuous,
                LatentKey.VQ_QUANTIZED.value: z_quantized,
                LatentKey.VQ_INDICES.value: all_indices,
            }

        return factory

    @pytest.mark.unit
    @pytest.mark.parametrize("weight", [0.5, 1.0, 10.0])
    def test_stores_weight(self, weight: float) -> None:
        loss = VQCommitmentLoss(num_codes=4, num_residual_layers=1, weight=weight)
        assert loss.weight == weight

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_codes, num_residual_layers",
        [(4, 1), (16, 2), (8, 4)],
    )
    def test_stores_codebook_dimensions(
        self, num_codes: int, num_residual_layers: int
    ) -> None:
        loss = VQCommitmentLoss(
            num_codes=num_codes,
            num_residual_layers=num_residual_layers,
        )
        assert loss.num_codes == num_codes
        assert loss.num_residual_layers == num_residual_layers

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_codes, num_residual_layers, expectation",
        [
            (4, 1, does_not_raise()),
            (
                0,
                1,
                pytest.raises(
                    ValueError,
                    match=re.escape("num_codes must be positive, got 0."),
                ),
            ),
            (
                -1,
                1,
                pytest.raises(
                    ValueError,
                    match=re.escape("num_codes must be positive, got -1."),
                ),
            ),
            (
                4,
                0,
                pytest.raises(
                    ValueError,
                    match=re.escape("num_residual_layers must be positive, got 0."),
                ),
            ),
            (
                4,
                -2,
                pytest.raises(
                    ValueError,
                    match=re.escape("num_residual_layers must be positive, got -2."),
                ),
            ),
        ],
    )
    def test_rejects_invalid_codebook_dimensions(
        self,
        num_codes: int,
        num_residual_layers: int,
        expectation: AbstractContextManager,
    ) -> None:
        with expectation:
            VQCommitmentLoss(
                num_codes=num_codes,
                num_residual_layers=num_residual_layers,
            )

    @pytest.mark.unit
    def test_raises_on_missing_keys(self) -> None:
        loss = VQCommitmentLoss(num_codes=4, num_residual_layers=1, weight=1.0)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain {loss.get_required_keys()} for VQCommitmentLoss."
            ),
        ):
            loss.forward(predictions={}, targets={})

    @pytest.mark.unit
    @pytest.mark.parametrize("code_dim", [4, 16, 64])
    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_returns_nonnegative_scalar(
        self,
        vq_predictions_factory: Callable[..., dict[str, torch.Tensor]],
        code_dim: int,
        batch_size: int,
        num_layers: int,
    ) -> None:
        loss = VQCommitmentLoss(num_codes=4, num_residual_layers=num_layers, weight=1.0)
        predictions = vq_predictions_factory(
            batch_size=batch_size, code_dim=code_dim, num_layers=num_layers
        )
        result = loss.forward(predictions=predictions, targets={})
        assert result.total_loss.dim() == 0
        assert result.total_loss.item() >= 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize("code_dim", [4, 16])
    def test_weight_scales_total_loss(
        self,
        vq_predictions_factory: Callable[..., dict[str, torch.Tensor]],
        code_dim: int,
    ) -> None:
        predictions = vq_predictions_factory(batch_size=8, code_dim=code_dim)
        result_w1 = VQCommitmentLoss(
            num_codes=4, num_residual_layers=1, weight=1.0
        ).forward(predictions=predictions, targets={})
        result_w5 = VQCommitmentLoss(
            num_codes=4, num_residual_layers=1, weight=5.0
        ).forward(predictions=predictions, targets={})
        assert torch.isclose(
            result_w5.total_loss, result_w1.total_loss * 5.0, rtol=1e-5
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_zero_loss_when_continuous_equals_quantized(
        self, rng: np.random.Generator, num_layers: int
    ) -> None:
        z = torch.from_numpy(
            rng.standard_normal((num_layers, 8, 16)).astype(np.float32)
        )
        predictions = {
            LatentKey.VQ_Z_CONTINUOUS.value: z,
            LatentKey.VQ_QUANTIZED.value: z.clone(),
        }
        result = VQCommitmentLoss(
            num_codes=4, num_residual_layers=num_layers, weight=1.0
        ).forward(predictions=predictions, targets={})
        assert torch.isclose(result.total_loss, torch.tensor(0.0), atol=1e-6)

    @pytest.mark.unit
    def test_averages_commitment_across_layers(self, rng: np.random.Generator) -> None:
        num_layers = 3
        batch_size = 8
        code_dim = 4
        z_continuous = torch.from_numpy(
            rng.standard_normal((num_layers, batch_size, code_dim)).astype(np.float32)
        )
        z_quantized = torch.from_numpy(
            rng.standard_normal((num_layers, batch_size, code_dim)).astype(np.float32)
        )
        predictions = {
            LatentKey.VQ_Z_CONTINUOUS.value: z_continuous,
            LatentKey.VQ_QUANTIZED.value: z_quantized,
        }
        total = (
            VQCommitmentLoss(num_codes=4, num_residual_layers=num_layers, weight=1.0)
            .forward(predictions=predictions, targets={})
            .total_loss
        )

        per_layer_mse = torch.stack(
            [
                F.mse_loss(z_continuous[layer_index], z_quantized[layer_index])
                for layer_index in range(num_layers)
            ]
        )
        assert torch.isclose(total, per_layer_mse.mean(), atol=1e-6)

    @pytest.mark.unit
    def test_component_losses_contains_commitment_key(
        self,
        vq_predictions_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        result = VQCommitmentLoss(
            num_codes=4, num_residual_layers=1, weight=1.0
        ).forward(
            predictions=vq_predictions_factory(batch_size=8, code_dim=16), targets={}
        )
        assert MetricKey.VQ_COMMITMENT_LOSS.value in result.component_losses

    @pytest.mark.unit
    def test_codebook_usage_in_metadata(
        self,
        vq_predictions_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        result = VQCommitmentLoss(
            num_codes=4, num_residual_layers=1, weight=1.0
        ).forward(
            predictions=vq_predictions_factory(batch_size=8, code_dim=16, num_codes=4),
            targets={},
        )
        assert MetricKey.VQ_CODEBOOK_USAGE.value in result.metadata
        assert MetadataKey.VQ_CODE_INDICES.value in result.metadata
        assert MetadataKey.VQ_NUM_CODES.value in result.metadata

    @pytest.mark.unit
    def test_codebook_usage_uses_k_times_l_denominator(self) -> None:
        num_codes = 8
        num_layers = 2
        batch_size = 16
        z = torch.zeros((num_layers, batch_size, 4))
        # Layer 0: all 8 distinct codes used (0..7 twice). Layer 1: only 4
        # distinct codes used. Total distinct = 8 + 4 = 12; capacity = K*L = 16.
        layer_0 = torch.arange(batch_size) % num_codes
        layer_1 = torch.arange(batch_size) % 4
        predictions = {
            LatentKey.VQ_Z_CONTINUOUS.value: z,
            LatentKey.VQ_QUANTIZED.value: z.clone(),
            LatentKey.VQ_INDICES.value: [layer_0, layer_1],
        }
        result = VQCommitmentLoss(
            num_codes=num_codes, num_residual_layers=num_layers, weight=1.0
        ).forward(predictions=predictions, targets={})
        expected_usage = (num_codes + 4) / (num_codes * num_layers)
        assert result.metadata[MetricKey.VQ_CODEBOOK_USAGE.value] == pytest.approx(
            expected_usage
        )

    @pytest.mark.unit
    def test_codebook_usage_absent_when_indices_not_provided(
        self, rng: np.random.Generator
    ) -> None:
        z = torch.from_numpy(rng.standard_normal((1, 8, 16)).astype(np.float32))
        predictions = {
            LatentKey.VQ_Z_CONTINUOUS.value: z,
            LatentKey.VQ_QUANTIZED.value: z.clone(),
        }
        result = VQCommitmentLoss(
            num_codes=4, num_residual_layers=1, weight=1.0
        ).forward(predictions=predictions, targets={})
        assert MetricKey.VQ_CODEBOOK_USAGE.value not in result.metadata


class TestVQPriorCrossEntropyLoss:
    @pytest.fixture
    def prior_ce_predictions_factory(
        self, rng: np.random.Generator
    ) -> Callable[..., dict[str, torch.Tensor]]:

        def factory(
            batch_size: int = 8,
            num_codes: int = 4,
            num_layers: int = 1,
        ) -> dict[str, torch.Tensor]:
            all_logits = [
                torch.from_numpy(
                    rng.standard_normal((batch_size, num_codes)).astype(np.float32)
                )
                for _ in range(num_layers)
            ]
            all_indices = [
                torch.from_numpy(
                    rng.integers(0, num_codes, size=(batch_size,)).astype(np.int64)
                )
                for _ in range(num_layers)
            ]
            return {
                LatentKey.PRIOR_CODE_LOGITS.value: all_logits,
                LatentKey.VQ_INDICES.value: all_indices,
            }

        return factory

    @pytest.mark.unit
    @pytest.mark.parametrize("weight", [0.5, 1.0, 10.0])
    def test_stores_weight(self, weight: float) -> None:
        loss = VQPriorCrossEntropyLoss(weight=weight)
        assert loss.weight == weight

    @pytest.mark.unit
    def test_raises_on_missing_keys(self) -> None:
        loss = VQPriorCrossEntropyLoss(weight=1.0)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain {loss.get_required_keys()} for VQPriorCrossEntropyLoss."
            ),
        ):
            loss.forward(predictions={}, targets={})

    @pytest.mark.unit
    def test_raises_when_prior_logits_are_empty(self) -> None:
        predictions = {
            LatentKey.PRIOR_CODE_LOGITS.value: [],
            LatentKey.VQ_INDICES.value: [],
        }
        with pytest.raises(
            ValueError,
            match=re.escape("VQPriorCrossEntropyLoss received no prior logits."),
        ):
            VQPriorCrossEntropyLoss(weight=1.0).forward(
                predictions=predictions, targets={}
            )

    @pytest.mark.unit
    def test_raises_on_layer_count_mismatch(self) -> None:
        predictions = {
            LatentKey.PRIOR_CODE_LOGITS.value: [torch.zeros(4, 8)],
            LatentKey.VQ_INDICES.value: [
                torch.zeros(4, dtype=torch.long),
                torch.zeros(4, dtype=torch.long),
            ],
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                "VQPriorCrossEntropyLoss expected the same number of prior logit "
                "layers and posterior index layers, got 1 and 2."
            ),
        ):
            VQPriorCrossEntropyLoss(weight=1.0).forward(
                predictions=predictions, targets={}
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prior_logits, posterior_indices, expected_message",
        [
            (
                torch.zeros(4, 2, 8),
                torch.zeros(4, dtype=torch.long),
                "Prior logits for VQ layer 0 must have shape (B, K), got (4, 2, 8).",
            ),
            (
                torch.zeros(4, 8),
                torch.zeros(4, 1, dtype=torch.long),
                "Posterior indices for VQ layer 0 must have shape (B,), got (4, 1).",
            ),
            (
                torch.zeros(4, 8),
                torch.zeros(3, dtype=torch.long),
                "Prior logits and posterior indices for VQ layer 0 must have the "
                "same batch size, got 4 and 3.",
            ),
        ],
    )
    def test_raises_on_invalid_layer_shapes(
        self,
        prior_logits: torch.Tensor,
        posterior_indices: torch.Tensor,
        expected_message: str,
    ) -> None:
        predictions = {
            LatentKey.PRIOR_CODE_LOGITS.value: [prior_logits],
            LatentKey.VQ_INDICES.value: [posterior_indices],
        }
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            VQPriorCrossEntropyLoss(weight=1.0).forward(
                predictions=predictions, targets={}
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("num_codes", [2, 4, 16])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_returns_nonnegative_scalar(
        self,
        prior_ce_predictions_factory: Callable[..., dict[str, torch.Tensor]],
        num_codes: int,
        num_layers: int,
        batch_size: int,
    ) -> None:
        loss = VQPriorCrossEntropyLoss(weight=1.0)
        predictions = prior_ce_predictions_factory(
            batch_size=batch_size, num_codes=num_codes, num_layers=num_layers
        )
        result = loss.forward(predictions=predictions, targets={})
        assert result.total_loss.dim() == 0
        assert result.total_loss.item() >= 0.0

    @pytest.mark.unit
    def test_weight_scales_total_loss(
        self,
        prior_ce_predictions_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        predictions = prior_ce_predictions_factory(batch_size=8, num_codes=4)
        result_w1 = VQPriorCrossEntropyLoss(weight=1.0).forward(
            predictions=predictions, targets={}
        )
        result_w5 = VQPriorCrossEntropyLoss(weight=5.0).forward(
            predictions=predictions, targets={}
        )
        assert torch.isclose(
            result_w5.total_loss, result_w1.total_loss * 5.0, rtol=1e-5
        )

    @pytest.mark.unit
    def test_perfect_prior_gives_low_loss(self, rng: np.random.Generator) -> None:
        batch_size = 8
        num_codes = 4
        indices = torch.from_numpy(
            rng.integers(0, num_codes, size=(batch_size,)).astype(np.int64)
        )  # (B,)
        logits = torch.full((batch_size, num_codes), -10.0)  # (B, K)
        for i in range(batch_size):
            logits[i, indices[i]] = 10.0
        predictions = {
            LatentKey.PRIOR_CODE_LOGITS.value: [logits],
            LatentKey.VQ_INDICES.value: [indices],
        }
        result = VQPriorCrossEntropyLoss(weight=1.0).forward(
            predictions=predictions, targets={}
        )
        assert result.total_loss.item() < 0.01

    @pytest.mark.unit
    def test_uniform_prior_gives_log_k_loss(self) -> None:
        batch_size = 64
        num_codes = 4
        logits = torch.zeros(batch_size, num_codes)  # uniform logits
        indices = torch.zeros(batch_size, dtype=torch.long)
        predictions = {
            LatentKey.PRIOR_CODE_LOGITS.value: [logits],
            LatentKey.VQ_INDICES.value: [indices],
        }
        result = VQPriorCrossEntropyLoss(weight=1.0).forward(
            predictions=predictions, targets={}
        )
        expected_ce = torch.log(torch.tensor(float(num_codes)))
        assert torch.isclose(result.total_loss, expected_ce, atol=0.01)

    @pytest.mark.unit
    def test_component_losses_contains_ce_key(
        self,
        prior_ce_predictions_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        result = VQPriorCrossEntropyLoss(weight=1.0).forward(
            predictions=prior_ce_predictions_factory(batch_size=8, num_codes=4),
            targets={},
        )
        assert MetricKey.VQ_PRIOR_CROSS_ENTROPY.value in result.component_losses
