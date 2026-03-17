"""Tests for versatil.models.encoding.fusion.concat module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.encoding.fusion.concat import ConcatFusion


@pytest.fixture
def concat_fusion_factory() -> Callable[..., ConcatFusion]:
    """Factory for ConcatFusion instances."""

    def factory(
        input_features: list[str] | None = None,
        output_name: str = "concat_fused",
        hidden_dim: int = 32,
    ) -> ConcatFusion:
        if input_features is None:
            input_features = ["rgb_features", "depth_features"]
        return ConcatFusion(
            input_features=input_features,
            output_name=output_name,
            hidden_dim=hidden_dim,
        )

    return factory


class TestConcatFusionInitialization:
    def test_has_sequential_fusion_interface(
        self,
        concat_fusion_factory: Callable[..., ConcatFusion],
    ):
        module = concat_fusion_factory()
        assert hasattr(module, "hidden_dim")
        assert hasattr(module, "projections")
        assert hasattr(module, "setup")
        assert hasattr(module, "get_output_specification")

    @pytest.mark.parametrize(
        "input_features",
        [
            ["feat_a", "feat_b"],
            ["feat_a", "feat_b", "feat_c"],
        ],
    )
    @pytest.mark.parametrize("hidden_dim", [32, 128])
    @pytest.mark.parametrize("output_name", ["concat_fused", "my_output"])
    def test_stores_configuration(
        self,
        concat_fusion_factory: Callable[..., ConcatFusion],
        input_features: list[str],
        hidden_dim: int,
        output_name: str,
    ):
        module = concat_fusion_factory(
            input_features=input_features,
            hidden_dim=hidden_dim,
            output_name=output_name,
        )
        assert module.input_features == input_features
        assert module.hidden_dim == hidden_dim
        assert module.output_name == output_name


class TestConcatFusionForward:
    def test_raises_if_projections_not_set_up(
        self,
        concat_fusion_factory: Callable[..., ConcatFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = concat_fusion_factory()
        features = [
            input_tensor_factory(input_dimension=64),
            input_tensor_factory(input_dimension=128),
        ]
        with pytest.raises(
            RuntimeError,
            match="Projections must be set up before forward pass",
        ):
            module(features)

    @pytest.mark.parametrize("time_steps", [None, 4])
    def test_output_shape_with_and_without_time(
        self,
        concat_fusion_factory: Callable[..., ConcatFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        time_steps: int | None,
    ):
        hidden_dim = 32
        batch_size = 2
        module = concat_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dim=hidden_dim,
        )
        module.setup(feature_keys_to_dims={"feat_a": 64, "feat_b": 128})
        features = [
            input_tensor_factory(
                batch_size=batch_size,
                input_dimension=64,
                sequence_length=time_steps,
            ),
            input_tensor_factory(
                batch_size=batch_size,
                input_dimension=128,
                sequence_length=time_steps,
            ),
        ]
        output = module(features)
        expected_dim = hidden_dim * 2
        if time_steps is not None:
            assert output.shape == (batch_size, time_steps, expected_dim)
        else:
            assert output.shape == (batch_size, expected_dim)

    @pytest.mark.parametrize("num_features", [2, 4])
    def test_output_dim_scales_with_feature_count(
        self,
        concat_fusion_factory: Callable[..., ConcatFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        num_features: int,
    ):
        hidden_dim = 16
        feature_names = [f"feat_{i}" for i in range(num_features)]
        module = concat_fusion_factory(
            input_features=feature_names,
            hidden_dim=hidden_dim,
        )
        dims = dict.fromkeys(feature_names, 32)
        module.setup(feature_keys_to_dims=dims)
        features = [
            input_tensor_factory(input_dimension=32) for _ in range(num_features)
        ]
        output = module(features)
        assert output.shape[-1] == hidden_dim * num_features


class TestConcatFusionGetOutputSpecification:
    @pytest.mark.parametrize(
        "num_features, hidden_dim, expected_dim",
        [
            (2, 32, 64),
            (3, 16, 48),
        ],
    )
    def test_output_dim_equals_hidden_dim_times_feature_count(
        self,
        concat_fusion_factory: Callable[..., ConcatFusion],
        num_features: int,
        hidden_dim: int,
        expected_dim: int,
    ):
        feature_names = [f"feat_{i}" for i in range(num_features)]
        module = concat_fusion_factory(
            input_features=feature_names,
            hidden_dim=hidden_dim,
        )
        spec = module.get_output_specification()
        assert spec.output_dim == expected_dim

    def test_output_name_matches(
        self,
        concat_fusion_factory: Callable[..., ConcatFusion],
    ):
        module = concat_fusion_factory(output_name="test_fused")
        spec = module.get_output_specification()
        assert spec.output_name == "test_fused"

    def test_returns_specification_with_expected_fields(
        self,
        concat_fusion_factory: Callable[..., ConcatFusion],
    ):
        module = concat_fusion_factory()
        spec = module.get_output_specification()
        assert hasattr(spec, "output_name")
        assert hasattr(spec, "output_dim")
