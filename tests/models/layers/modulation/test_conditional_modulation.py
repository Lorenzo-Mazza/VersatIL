"""Tests for versatil.models.layers.modulation.conditional_modulation module."""
import re
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)


@pytest.fixture
def cnn_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for CNN feature maps (B, C, H, W)."""
    def factory(
        batch_size: int = 2,
        channels: int = 16,
        height: int = 8,
        width: int = 8,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, channels, height, width)
        ).astype(np.float32)
        return torch.from_numpy(data)
    return factory


@pytest.fixture
def transformer_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for transformer inputs (B, S, D)."""
    def factory(
        batch_size: int = 2,
        sequence_length: int = 10,
        feature_dim: int = 16,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, sequence_length, feature_dim)
        ).astype(np.float32)
        return torch.from_numpy(data)
    return factory


@pytest.fixture
def conv1d_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for Conv1D inputs (B, C, T)."""
    def factory(
        batch_size: int = 2,
        channels: int = 16,
        time_steps: int = 20,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, channels, time_steps)
        ).astype(np.float32)
        return torch.from_numpy(data)
    return factory


@pytest.fixture
def modulation_factory() -> Callable[..., ConditionalModulation]:
    """Factory for ConditionalModulation instances."""
    def factory(
        condition_dim: int = 32,
        feature_dim: int = 16,
        use_shift: bool = True,
        use_gate: bool = False,
        activation: str = "silu",
        init_strategy: str = "identity",
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

    @pytest.mark.parametrize("use_shift, use_gate, expected_multiplier", [
        (False, False, 1),
        (True, False, 2),
        (False, True, 2),
        (True, True, 3),
    ])
    def test_output_dim_calculation(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        use_shift: bool,
        use_gate: bool,
        expected_multiplier: int,
    ):
        feature_dim = 16
        module = modulation_factory(
            feature_dim=feature_dim,
            use_shift=use_shift,
            use_gate=use_gate,
        )
        assert module.output_dim == feature_dim * expected_multiplier

    @pytest.mark.parametrize("init_strategy", ["identity", "xavier", "zero"])
    def test_valid_init_strategies(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        init_strategy: str,
    ):
        module = modulation_factory(init_strategy=init_strategy)
        assert module.init_strategy == init_strategy

    def test_invalid_init_strategy_raises(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
    ):
        invalid_strategy = "unknown_strategy"
        with pytest.raises(
            ValueError,
            match=re.escape(f"Unknown init_strategy: {invalid_strategy}"),
        ):
            modulation_factory(init_strategy=invalid_strategy)

    def test_inherits_nn_module(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
    ):
        module = modulation_factory(
            condition_dim=32,
            feature_dim=16,
            use_shift=True,
            use_gate=False,
        )
        assert isinstance(module, nn.Module)


class TestConditionalModulationForward:

    def test_4d_cnn_output_shape(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        cnn_input_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=False,
        )
        tensor = cnn_input_factory(
            batch_size=2,
            channels=feature_dim,
            height=8,
            width=8,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        output = module(tensor, condition)
        assert output.shape == tensor.shape

    def test_3d_transformer_output_shape(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        transformer_input_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=False,
        )
        tensor = transformer_input_factory(
            batch_size=2,
            sequence_length=10,
            feature_dim=feature_dim,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        output = module(tensor, condition)
        assert output.shape == tensor.shape

    def test_3d_conv1d_output_shape(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        conv1d_input_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=False,
        )
        tensor = conv1d_input_factory(
            batch_size=2,
            channels=feature_dim,
            time_steps=20,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        output = module(tensor, condition)
        assert output.shape == tensor.shape

    def test_use_gate_returns_tuple(
        self,
        modulation_factory: Callable[..., ConditionalModulation],
        cnn_input_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
            use_gate=True,
        )
        tensor = cnn_input_factory(
            batch_size=2,
            channels=feature_dim,
            height=8,
            width=8,
        )
        condition = condition_factory(batch_size=2, condition_dim=32)
        result = module(tensor, condition)
        assert isinstance(result, tuple)
        assert len(result) == 2
        modulated, gate = result
        assert modulated.shape == tensor.shape
        # Gate is broadcast-shaped: (B, C, 1, 1) for 4D CNN inputs
        assert gate.shape == (2, feature_dim, 1, 1)

    def test_unsupported_input_shape_raises(
        self,
        rng: np.random.Generator,
        modulation_factory: Callable[..., ConditionalModulation],
        condition_factory: Callable[..., torch.Tensor],
    ):
        module = modulation_factory(
            condition_dim=32,
            feature_dim=16,
        )
        # 2D input is not supported
        data = rng.standard_normal((2, 16)).astype(np.float32)
        tensor = torch.from_numpy(data)
        condition = condition_factory(batch_size=2, condition_dim=32)
        with pytest.raises(
            ValueError,
            match=re.escape(f"Unsupported input shape: {tensor.shape}"),
        ):
            module(tensor, condition)

    def test_batch_dimension_mismatch_raises(
        self,
        rng: np.random.Generator,
        modulation_factory: Callable[..., ConditionalModulation],
        condition_factory: Callable[..., torch.Tensor],
    ):
        feature_dim = 16
        module = modulation_factory(
            condition_dim=32,
            feature_dim=feature_dim,
        )
        # 3D tensor where neither dim 0 nor dim 1 matches condition batch size
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
            module(tensor, condition)
