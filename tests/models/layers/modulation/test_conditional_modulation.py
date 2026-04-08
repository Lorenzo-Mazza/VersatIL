"""Tests for versatil.models.layers.modulation.conditional_modulation module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)


@pytest.fixture
def modulation_factory() -> Callable[..., ConditionalModulation]:
    """Factory for ConditionalModulation instances."""

    def factory(
        condition_dim: int = 32,
        feature_dim: int = 16,
        use_shift: bool = True,
        use_gate: bool = False,
        activation: str = ActivationFunction.SILU.value,
        init_strategy: str = "zero",
    ) -> ConditionalModulation:
        return ConditionalModulation(
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_shift=use_shift,
            use_gate=use_gate,
            activation=activation,
            init_strategy=init_strategy,
        )

    return factory


class TestConditionalModulationInitialization:
    @pytest.mark.parametrize("condition_dim", [32, 64])
    @pytest.mark.parametrize("feature_dim", [16, 32])
    @pytest.mark.parametrize("use_shift", [True, False])
    @pytest.mark.parametrize("use_gate", [True, False])
    def test_stores_configuration(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        condition_dim: int,
        feature_dim: int,
        use_shift: bool,
        use_gate: bool,
    ):
        module = modulation_factory(
            condition_dim=condition_dim,
            feature_dim=feature_dim,
            use_shift=use_shift,
            use_gate=use_gate,
        )
        assert module.feature_dim == feature_dim
        assert module.use_shift == use_shift
        assert module.use_gate == use_gate
        assert module.output_dim == feature_dim * (1 + use_shift + use_gate)

    @pytest.mark.parametrize(
        "init_strategy, expectation",
        [
            ("zero", does_not_raise()),
            ("xavier", does_not_raise()),
            (
                "unknown_strategy",
                pytest.raises(
                    ValueError,
                    match=re.escape("Unknown init_strategy: unknown_strategy"),
                ),
            ),
        ],
    )
    def test_init_strategy_validation(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        init_strategy: str,
        expectation,
    ):
        with expectation:
            module = modulation_factory(init_strategy=init_strategy)
            assert module.init_strategy == init_strategy

    def test_zero_init_zeroes_all_projection_weights(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
    ):
        module = modulation_factory(init_strategy="zero")
        for layer in module.projection.modules():
            if isinstance(layer, nn.Linear):
                assert torch.all(layer.weight == 0)
                assert torch.all(layer.bias == 0)

    def test_xavier_init_produces_nonzero_weights_with_zero_biases(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
    ):
        module = modulation_factory(init_strategy="xavier")
        for layer in module.projection.modules():
            if isinstance(layer, nn.Linear):
                assert torch.any(layer.weight != 0)
                assert torch.all(layer.bias == 0)

    def test_init_marks_linear_layers_for_modulation(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
    ):
        module = modulation_factory(init_strategy="zero")
        linear_layers = [
            m for m in module.projection.modules() if isinstance(m, nn.Linear)
        ]
        assert len(linear_layers) > 0
        for layer in linear_layers:
            assert getattr(layer, "_is_modulation_layer", False) is True

    def test_swiglu_activation_forward_produces_valid_output(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            activation=ActivationFunction.SWIGLU.value,
            init_strategy="xavier",
        )
        tensor = nchw_tensor_factory(batch_size=2, channels=feature_dim)
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output, _ = module(x=tensor, condition=condition)
        assert output.shape == tensor.shape


class TestConditionalModulationForward:
    @pytest.mark.parametrize("use_shift", [True, False])
    def test_identity_init_produces_no_modulation_effect(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        use_shift: bool,
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_shift=use_shift,
            init_strategy="zero",
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output, _ = module(x=tensor, condition=condition)
        # gamma=0, beta=0 → x * (1+0) + 0 = x
        assert torch.allclose(output, tensor, atol=1e-6)

    def test_xavier_init_produces_modulation_effect(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            init_strategy="xavier",
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output, _ = module(x=tensor, condition=condition)
        assert not torch.allclose(output, tensor)

    def test_different_conditions_produce_different_outputs(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            init_strategy="xavier",
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=feature_dim,
        )
        condition_a = condition_factory(batch_size=2, condition_dim=32)
        condition_b = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            output_a, _ = module(x=tensor, condition=condition_a)
            output_b, _ = module(x=tensor, condition=condition_b)
        assert not torch.allclose(output_a, output_b)

    def test_scale_only_without_shift(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_shift=False,
            init_strategy="xavier",
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            projected = module.projection(condition)
            gamma = projected.split(feature_dim, dim=-1)[0]
            gamma_reshaped = gamma.unsqueeze(1)
            expected = tensor * (1 + gamma_reshaped)
            output, _ = module(x=tensor, condition=condition)
        assert torch.allclose(output, expected, atol=1e-5)

    def test_film_formula_with_shift(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_shift=True,
            init_strategy="xavier",
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        with torch.no_grad():
            projected = module.projection(condition)
            chunks = projected.split(feature_dim, dim=-1)
            gamma = chunks[0].unsqueeze(1)
            beta = chunks[1].unsqueeze(1)
            expected = tensor * (1 + gamma) + beta
            output, _ = module(x=tensor, condition=condition)
        assert torch.allclose(output, expected, atol=1e-5)

    def test_4d_cnn_path_preserves_shape(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
        )
        tensor = nchw_tensor_factory(
            batch_size=2, channels=feature_dim, height=8, width=8
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        output, _ = module(x=tensor, condition=condition)
        assert output.shape == tensor.shape

    def test_3d_conv1d_path_when_feature_dim_in_dim_1(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
        )
        # (B=2, C=16, T=20) — C matches feature_dim → Conv1D path
        tensor = conv1d_tensor_factory(
            batch_size=2,
            channels=feature_dim,
            sequence_length=20,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        output, _ = module(x=tensor, condition=condition)
        assert output.shape == tensor.shape

    def test_3d_transformer_path_when_feature_dim_in_dim_2(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
        )
        # (B=2, S=10, D=16) — S != feature_dim, D matches → Transformer path
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        output, _ = module(x=tensor, condition=condition)
        assert output.shape == tensor.shape

    def test_3d_batch_in_dim_1_path(
        self,
        rng: np.random.Generator,
        modulation_factory: Callable[..., ConditionalModulation],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        batch_size = 2
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
        )
        # (S=5, B=2, D=16) — x.size(1)==condition.size(0)
        data = rng.standard_normal((5, batch_size, feature_dim)).astype(np.float32)
        tensor = torch.from_numpy(data)
        condition = condition_factory(batch_size=batch_size, condition_dim=32)
        output, _ = module(x=tensor, condition=condition)
        assert output.shape == tensor.shape

    def test_gate_returns_tuple_with_correct_shapes(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        nchw_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=True,
        )
        tensor = nchw_tensor_factory(
            batch_size=2, channels=feature_dim, height=8, width=8
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        result = module(x=tensor, condition=condition)
        assert len(result) == 2
        modulated, gate = result
        assert modulated.shape == tensor.shape
        assert gate.shape == (2, feature_dim, 1, 1)

    def test_no_gate_returns_ones_gate(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=False,
        )
        tensor = sequence_tensor_factory(
            batch_size=2,
            sequence_length=10,
            embedding_dimension=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        output, gate = module(x=tensor, condition=condition)
        assert output.shape == tensor.shape
        assert torch.equal(gate, torch.ones(1, dtype=tensor.dtype))

    def test_batch_dimension_mismatch_raises(
        self,
        rng: np.random.Generator,
        modulation_factory: Callable[..., ConditionalModulation],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(condition_dim=32, feature_dim=feature_dim)
        data = rng.standard_normal((5, 7, feature_dim)).astype(np.float32)
        tensor = torch.from_numpy(data)
        condition = condition_factory(batch_size=3, condition_dim=32)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Cannot match batch dimension: x.shape={tensor.shape}, "
                f"condition.shape={condition.shape}. "
                f"Expected x.size(0) or x.size(1) to equal "
                f"condition.size(0)={condition.size(0)}"
            ),
        ):
            module(x=tensor, condition=condition)

    def test_unsupported_2d_input_raises(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        flat_tensor_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        module = modulation_factory(condition_dim=32, feature_dim=16)
        tensor = flat_tensor_factory(batch_size=2, feature_dimension=16)
        condition = condition_factory(batch_size=2, condition_dim=32)
        with pytest.raises(
            ValueError,
            match=re.escape(f"Unsupported input shape: {tensor.shape}"),
        ):
            module(x=tensor, condition=condition)
