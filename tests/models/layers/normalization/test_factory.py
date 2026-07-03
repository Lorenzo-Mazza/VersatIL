"""Tests for versatil.models.layers.normalization.factory module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch

from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import (
    create_block_normalization,
    create_normalization_layer,
)
from versatil.models.layers.normalization.unconditioned_norm import UnconditionedNorm


@pytest.fixture
def norm_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for normalization input tensors."""

    def factory(
        batch_size: int = 2,
        channels: int = 64,
        spatial: bool = False,
        height: int = 4,
        width: int = 4,
        sequence_length: int = 8,
    ) -> torch.Tensor:
        if spatial:
            shape = (batch_size, channels, height, width)
        else:
            shape = (batch_size, sequence_length, channels)
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    return factory


class TestCreateNormalizationLayer:
    @pytest.mark.parametrize(
        "normalization_type",
        [
            NormalizationType.LAYER_NORM.value,
            NormalizationType.RMS_NORM.value,
        ],
    )
    def test_created_layer_produces_valid_output(
        self,
        norm_input_factory: Callable[..., torch.Tensor],
        normalization_type: str,
    ):
        dimension = 64
        layer = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=dimension,
        )
        tensor = norm_input_factory(batch_size=2, channels=dimension)
        output = layer(tensor)
        assert output.shape == tensor.shape
        assert torch.all(torch.isfinite(output))

    @pytest.mark.parametrize(
        "normalization_type",
        [
            NormalizationType.LAYER_NORM.value,
            NormalizationType.RMS_NORM.value,
        ],
    )
    def test_conditioned_layer_produces_valid_output(
        self,
        norm_input_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        normalization_type: str,
    ):
        dimension = 64
        conditioning_dimension = 32
        layer = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=dimension,
            conditioning_dimension=conditioning_dimension,
        )
        tensor = norm_input_factory(batch_size=2, channels=dimension)
        condition = condition_factory(
            batch_size=2, conditioning_dimension=conditioning_dimension
        )
        output, _ = layer(tensor, condition)
        assert output.shape == tensor.shape
        assert torch.all(torch.isfinite(output))

    @pytest.mark.parametrize(
        "normalization_type, conditioning_dimension, expectation",
        [
            (NormalizationType.LAYER_NORM.value, 32, does_not_raise()),
            (NormalizationType.RMS_NORM.value, 32, does_not_raise()),
            (NormalizationType.LAYER_NORM.value, None, does_not_raise()),
            (NormalizationType.RMS_NORM.value, None, does_not_raise()),
        ],
    )
    def test_condition_dim_combinations(
        self,
        normalization_type: str,
        conditioning_dimension: int | None,
        expectation,
    ):
        with expectation:
            create_normalization_layer(
                normalization_type=normalization_type,
                dimension=64,
                conditioning_dimension=conditioning_dimension,
            )

    def test_invalid_type_raises_with_supported_types_message(self):
        invalid_type = "nonexistent_norm"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unsupported normalization type: {invalid_type}. "
                f"Must be one of {[e.value for e in NormalizationType]}."
            ),
        ):
            create_normalization_layer(
                normalization_type=invalid_type,
                dimension=64,
            )

    @pytest.mark.parametrize(
        "normalization_type",
        [NormalizationType.LAYER_NORM.value, NormalizationType.RMS_NORM.value],
    )
    def test_adaptive_base_norm_has_no_learnable_affine_parameters(
        self, normalization_type: str
    ):
        layer = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=64,
            conditioning_dimension=32,
        )
        assert len(list(layer.norm.parameters())) == 0

    @pytest.mark.parametrize("epsilon", [1e-5, 1e-8])
    def test_epsilon_is_forwarded_to_layer(self, epsilon: float):
        layer = create_normalization_layer(
            normalization_type=NormalizationType.LAYER_NORM.value,
            dimension=64,
            epsilon=epsilon,
        )
        assert layer.eps == epsilon

    def test_layer_norm_output_is_normalized(
        self,
        norm_input_factory: Callable[..., torch.Tensor],
    ):
        dimension = 64
        layer = create_normalization_layer(
            normalization_type=NormalizationType.LAYER_NORM.value,
            dimension=dimension,
        )
        tensor = norm_input_factory(batch_size=4, channels=dimension)
        output = layer(tensor)
        assert torch.allclose(
            output.mean(dim=-1),
            torch.zeros_like(output.mean(dim=-1)),
            atol=1e-5,
        )


class TestCreateBlockNormalization:
    @pytest.mark.parametrize(
        "normalization_type, conditioning_dimension, use_gating, expected_type",
        [
            (NormalizationType.RMS_NORM.value, 32, False, AdaNorm),
            (NormalizationType.LAYER_NORM.value, 32, True, AdaNorm),
            (NormalizationType.RMS_NORM.value, None, False, UnconditionedNorm),
            (NormalizationType.LAYER_NORM.value, None, False, UnconditionedNorm),
        ],
        ids=[
            "rms_conditioned",
            "layernorm_gated",
            "rms_unconditioned",
            "layernorm_unconditioned",
        ],
    )
    def test_creates_correct_normalization_type(
        self,
        normalization_type: str,
        conditioning_dimension: int | None,
        use_gating: bool,
        expected_type: type,
    ):
        norm = create_block_normalization(
            normalization_type=normalization_type,
            dimension=64,
            conditioning_dimension=conditioning_dimension,
            use_gating=use_gating,
        )
        assert isinstance(norm, expected_type)

    def test_invalid_type_raises(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unsupported normalization type: invalid_type. "
                f"Must be one of {[e.value for e in NormalizationType]}."
            ),
        ):
            create_block_normalization(
                normalization_type="invalid_type",
                dimension=64,
            )

    def test_adaptive_norm_output_changes_with_condition(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        dimension = 32
        conditioning_dimension = 16
        norm = create_block_normalization(
            normalization_type=NormalizationType.RMS_NORM.value,
            dimension=dimension,
            conditioning_dimension=conditioning_dimension,
            init_strategy="xavier",
        )
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=dimension
        )
        condition_a = condition_factory(
            batch_size=2, conditioning_dimension=conditioning_dimension
        )
        condition_b = condition_a * 5.0
        output_a, gate_a = norm(x=tensor, condition=condition_a)
        output_b, gate_b = norm(x=tensor, condition=condition_b)
        assert output_a.shape == tensor.shape
        assert not torch.allclose(output_a, output_b)

    def test_unconditioned_norm_ignores_condition(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        dimension = 32
        norm = create_block_normalization(
            normalization_type=NormalizationType.RMS_NORM.value,
            dimension=dimension,
        )
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=dimension
        )
        condition = condition_factory(batch_size=2, conditioning_dimension=16)
        output_no_cond, gate_no_cond = norm(x=tensor, condition=None)
        output_with_cond, gate_with_cond = norm(x=tensor, condition=condition)
        assert torch.allclose(output_no_cond, output_with_cond)
        assert torch.equal(gate_no_cond, torch.ones(1))
        assert torch.equal(gate_with_cond, torch.ones(1))

    def test_gated_adaptive_norm_gate_is_zero_at_init(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        dimension = 32
        conditioning_dimension = 16
        norm = create_block_normalization(
            normalization_type=NormalizationType.RMS_NORM.value,
            dimension=dimension,
            conditioning_dimension=conditioning_dimension,
            use_gating=True,
            init_strategy="zero",
        )
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=dimension
        )
        condition = condition_factory(
            batch_size=2, conditioning_dimension=conditioning_dimension
        )
        _, gate = norm(x=tensor, condition=condition)
        assert torch.allclose(gate, torch.zeros_like(gate), atol=1e-6)

    def test_ungated_adaptive_norm_gate_is_one_at_init(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        dimension = 32
        conditioning_dimension = 16
        norm = create_block_normalization(
            normalization_type=NormalizationType.LAYER_NORM.value,
            dimension=dimension,
            conditioning_dimension=conditioning_dimension,
            use_gating=False,
            init_strategy="zero",
        )
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=dimension
        )
        condition = condition_factory(
            batch_size=2, conditioning_dimension=conditioning_dimension
        )
        _, gate = norm(x=tensor, condition=condition)
        assert torch.allclose(gate, torch.ones_like(gate))

    def test_xavier_init_produces_nonzero_conditioning_effect(
        self,
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        dimension = 64
        conditioning_dimension = 32
        zero_norm = create_block_normalization(
            normalization_type=NormalizationType.RMS_NORM.value,
            dimension=dimension,
            conditioning_dimension=conditioning_dimension,
            init_strategy="zero",
        )
        xavier_norm = create_block_normalization(
            normalization_type=NormalizationType.RMS_NORM.value,
            dimension=dimension,
            conditioning_dimension=conditioning_dimension,
            init_strategy="xavier",
        )
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=4, embedding_dimension=dimension
        )
        condition = condition_factory(
            batch_size=2, conditioning_dimension=conditioning_dimension
        )
        output_zero, _ = zero_norm(x=tensor, condition=condition)
        output_xavier, _ = xavier_norm(x=tensor, condition=condition)
        # Zero-init modulation is identity (scale=1, shift=0), xavier is not
        assert not torch.allclose(output_zero, output_xavier)
