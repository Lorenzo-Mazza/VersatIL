"""Tests for versatil.models.encoding.fusion.mlp module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.encoding.fusion.base import FusionOutput, SequentialFusion
from versatil.models.encoding.fusion.mlp import MLPFusion
from versatil.models.layers.activation import ActivationFunction


@pytest.fixture
def mlp_fusion_factory() -> Callable[..., MLPFusion]:
    """Factory for MLPFusion instances."""
    def factory(
        input_features: list[str] | None = None,
        output_name: str = "mlp_fused",
        hidden_dim: int = 32,
        mlp_hidden_dims: list[int] | None = None,
        activation_name: str = ActivationFunction.GELU.value,
        dropout: float = 0.1,
    ) -> MLPFusion:
        if input_features is None:
            input_features = ["rgb_features", "depth_features"]
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [64, 32]
        return MLPFusion(
            input_features=input_features,
            output_name=output_name,
            hidden_dim=hidden_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            activation_name=activation_name,
            dropout=dropout,
        )
    return factory


class TestMLPFusionInitialization:

    def test_inherits_from_sequential_fusion(
        self,
        mlp_fusion_factory: Callable[..., MLPFusion],
    ):
        module = mlp_fusion_factory()
        assert isinstance(module, SequentialFusion)

    @pytest.mark.parametrize("hidden_dim", [32, 128])
    @pytest.mark.parametrize("mlp_hidden_dims", [[64, 32], [128, 64, 32]])
    @pytest.mark.parametrize("output_name", ["mlp_fused", "my_output"])
    def test_stores_configuration(
        self,
        mlp_fusion_factory: Callable[..., MLPFusion],
        hidden_dim: int,
        mlp_hidden_dims: list[int],
        output_name: str,
    ):
        module = mlp_fusion_factory(
            hidden_dim=hidden_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            output_name=output_name,
        )
        assert module.hidden_dim == hidden_dim
        assert module.output_dim == mlp_hidden_dims[-1]
        assert module.output_name == output_name
        assert module.mlp is not None


class TestMLPFusionForward:

    def test_raises_if_projections_not_set_up(
        self,
        mlp_fusion_factory: Callable[..., MLPFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = mlp_fusion_factory()
        features = [
            input_tensor_factory(input_dim=64),
            input_tensor_factory(input_dim=128),
        ]
        with pytest.raises(RuntimeError, match="Projections must be set up"):
            module(features)

    @pytest.mark.parametrize("time_steps", [None, 4])
    def test_output_shape_with_and_without_time(
        self,
        mlp_fusion_factory: Callable[..., MLPFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        time_steps: int | None,
    ):
        batch_size = 2
        mlp_hidden_dims = [64, 48]
        module = mlp_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dim=32,
            mlp_hidden_dims=mlp_hidden_dims,
        )
        module.setup(feature_keys_to_dims={"feat_a": 64, "feat_b": 128})
        features = [
            input_tensor_factory(
                batch_size=batch_size,
                input_dim=64,
                sequence_length=time_steps,
            ),
            input_tensor_factory(
                batch_size=batch_size,
                input_dim=128,
                sequence_length=time_steps,
            ),
        ]
        output = module(features)
        expected_output_dim = mlp_hidden_dims[-1]
        if time_steps is not None:
            assert output.shape == (batch_size, time_steps, expected_output_dim)
        else:
            assert output.shape == (batch_size, expected_output_dim)

    @pytest.mark.parametrize("activation_name", [
        ActivationFunction.GELU.value,
        ActivationFunction.RELU.value,
        ActivationFunction.SILU.value,
    ])
    def test_forward_with_different_activations(
        self,
        mlp_fusion_factory: Callable[..., MLPFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        activation_name: str,
    ):
        module = mlp_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dim=16,
            mlp_hidden_dims=[32, 16],
            activation_name=activation_name,
        )
        module.setup(feature_keys_to_dims={"feat_a": 32, "feat_b": 64})
        features = [
            input_tensor_factory(input_dim=32),
            input_tensor_factory(input_dim=64),
        ]
        output = module(features)
        assert output.shape == (2, 16)

    @pytest.mark.parametrize("num_features", [2, 3])
    def test_forward_with_varying_input_count(
        self,
        mlp_fusion_factory: Callable[..., MLPFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        num_features: int,
    ):
        hidden_dim = 16
        feature_names = [f"feat_{i}" for i in range(num_features)]
        module = mlp_fusion_factory(
            input_features=feature_names,
            hidden_dim=hidden_dim,
            mlp_hidden_dims=[hidden_dim * num_features, 24],
        )
        dims = {name: 32 for name in feature_names}
        module.setup(feature_keys_to_dims=dims)
        features = [
            input_tensor_factory(input_dim=32)
            for _ in range(num_features)
        ]
        output = module(features)
        assert output.shape == (2, 24)


class TestMLPFusionGetOutputSpecification:

    @pytest.mark.parametrize("mlp_hidden_dims, expected_dim", [
        ([64, 32], 32),
        ([128, 64, 16], 16),
    ])
    def test_output_dim_equals_last_mlp_layer(
        self,
        mlp_fusion_factory: Callable[..., MLPFusion],
        mlp_hidden_dims: list[int],
        expected_dim: int,
    ):
        module = mlp_fusion_factory(mlp_hidden_dims=mlp_hidden_dims)
        spec = module.get_output_specification()
        assert spec.output_dim == expected_dim

    def test_output_name_matches(
        self,
        mlp_fusion_factory: Callable[..., MLPFusion],
    ):
        module = mlp_fusion_factory(output_name="test_mlp")
        spec = module.get_output_specification()
        assert spec.output_name == "test_mlp"

    def test_returns_fusion_output_type(
        self,
        mlp_fusion_factory: Callable[..., MLPFusion],
    ):
        module = mlp_fusion_factory()
        spec = module.get_output_specification()
        assert isinstance(spec, FusionOutput)
