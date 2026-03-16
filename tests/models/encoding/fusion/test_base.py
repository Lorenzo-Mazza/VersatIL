"""Tests for versatil.models.encoding.fusion.base module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.encoding.fusion.base import (
    FusionInput,
    FusionModule,
    FusionOutput,
    SequentialFusion,
)


class ConcreteFusionModule(FusionModule):
    """Concrete implementation for testing abstract FusionModule."""

    def __init__(
        self,
        input_features: list[str],
        output_name: str,
    ):
        input_specification = FusionInput(input_features=input_features)
        super().__init__(
            input_specification=input_specification,
            output_name=output_name,
        )
        self._output_dim = 64

    def _setup_layers(self, feature_keys_to_dims: dict[str, int | tuple]):
        pass

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return features[0]

    def get_output_specification(self) -> FusionOutput:
        return FusionOutput(
            output_name=self.output_name,
            output_dim=self._output_dim,
        )


class ConcreteSequentialFusion(SequentialFusion):
    """Concrete implementation for testing abstract SequentialFusion."""

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if self.projections is None:
            raise RuntimeError("Projections not set up")
        projected = [proj(feat) for feat, proj in zip(features, self.projections)]
        return torch.cat(projected, dim=-1)

    def get_output_specification(self) -> FusionOutput:
        return FusionOutput(
            output_name=self.output_name,
            output_dim=self.hidden_dim * len(self.input_features),
        )


@pytest.fixture
def fusion_module_factory() -> Callable[..., ConcreteFusionModule]:
    """Factory for ConcreteFusionModule instances."""
    def factory(
        input_features: list[str] | None = None,
        output_name: str = "fused_output",
    ) -> ConcreteFusionModule:
        if input_features is None:
            input_features = ["rgb_features", "depth_features"]
        return ConcreteFusionModule(
            input_features=input_features,
            output_name=output_name,
        )
    return factory


@pytest.fixture
def sequential_fusion_factory() -> Callable[..., ConcreteSequentialFusion]:
    """Factory for ConcreteSequentialFusion instances."""
    def factory(
        input_features: list[str] | None = None,
        output_name: str = "fused_output",
        hidden_dim: int = 64,
    ) -> ConcreteSequentialFusion:
        if input_features is None:
            input_features = ["rgb_features", "depth_features"]
        return ConcreteSequentialFusion(
            input_features=input_features,
            output_name=output_name,
            hidden_dim=hidden_dim,
        )
    return factory


class TestFusionInputDataclass:

    @pytest.mark.parametrize("input_features", [["rgb", "depth"], ["a", "b", "c"]])
    @pytest.mark.parametrize("required_count", [1, 3])
    @pytest.mark.parametrize("max_count", [None, 5])
    def test_stores_configuration(
        self,
        input_features: list[str],
        required_count: int,
        max_count: int | None,
    ):
        spec = FusionInput(
            input_features=input_features,
            required_count=required_count,
            max_count=max_count,
        )
        assert spec.input_features == input_features
        assert spec.required_count == required_count
        assert spec.max_count == max_count


class TestFusionOutputDataclass:

    @pytest.mark.parametrize("output_name", ["fused", "combined"])
    @pytest.mark.parametrize("output_dim", [64, (16, 32)])
    def test_stores_configuration(
        self,
        output_name: str,
        output_dim: int | tuple[int, ...],
    ):
        spec = FusionOutput(output_name=output_name, output_dim=output_dim)
        assert spec.output_name == output_name
        assert spec.output_dim == output_dim


class TestFusionModuleInitialization:

    @pytest.mark.parametrize("input_features", [
        ["rgb_features", "depth_features"],
        ["feat_a", "feat_b", "feat_c"],
    ])
    @pytest.mark.parametrize("output_name", ["fused_output", "my_fused"])
    def test_stores_configuration(
        self,
        fusion_module_factory: Callable[..., ConcreteFusionModule],
        input_features: list[str],
        output_name: str,
    ):
        module = fusion_module_factory(
            input_features=input_features,
            output_name=output_name,
        )
        assert module.input_specification.input_features == input_features
        assert module.output_name == output_name
        assert module._initialized is False

    def test_has_nn_module_interface(
        self,
        fusion_module_factory: Callable[..., ConcreteFusionModule],
    ):
        module = fusion_module_factory()
        assert hasattr(module, "forward")
        assert hasattr(module, "parameters")
        assert hasattr(module, "state_dict")


class TestFusionModuleInputFeaturesProperty:

    def test_getter_returns_input_features(
        self,
        fusion_module_factory: Callable[..., ConcreteFusionModule],
    ):
        module = fusion_module_factory(input_features=["a", "b"])
        assert module.input_features == ["a", "b"]

    def test_setter_updates_specification(
        self,
        fusion_module_factory: Callable[..., ConcreteFusionModule],
    ):
        module = fusion_module_factory(input_features=["a", "b"])
        module.input_features = ["x", "y", "z"]
        assert module.input_specification.input_features == ["x", "y", "z"]


class TestFusionModuleSetup:

    def test_setup_sets_initialized_true(
        self,
        fusion_module_factory: Callable[..., ConcreteFusionModule],
    ):
        module = fusion_module_factory()
        module.setup(feature_keys_to_dims={"rgb_features": 64, "depth_features": 32})
        assert module._initialized is True

    def test_setup_skips_if_already_initialized(
        self,
        fusion_module_factory: Callable[..., ConcreteFusionModule],
    ):
        module = fusion_module_factory()
        module.setup(feature_keys_to_dims={"rgb_features": 64, "depth_features": 32})
        # Calling again should not raise or re-initialize
        module.setup(feature_keys_to_dims={"rgb_features": 128})
        assert module._initialized is True


class TestFusionModuleGetOutputDim:

    def test_returns_output_dim_from_specification(
        self,
        fusion_module_factory: Callable[..., ConcreteFusionModule],
    ):
        module = fusion_module_factory()
        assert module.get_output_dim() == 64


class TestSequentialFusionInitialization:

    @pytest.mark.parametrize("input_features", [
        ["rgb_features", "depth_features"],
        ["feat_a", "feat_b", "feat_c"],
    ])
    @pytest.mark.parametrize("hidden_dim", [64, 128])
    @pytest.mark.parametrize("output_name", ["fused_output", "seq_fused"])
    def test_stores_configuration(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
        input_features: list[str],
        hidden_dim: int,
        output_name: str,
    ):
        module = sequential_fusion_factory(
            input_features=input_features,
            hidden_dim=hidden_dim,
            output_name=output_name,
        )
        assert module.input_features == input_features
        assert module.hidden_dim == hidden_dim
        assert module.output_name == output_name
        assert module.projections is None

    def test_has_fusion_module_interface(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
    ):
        module = sequential_fusion_factory()
        assert hasattr(module, "input_features")
        assert hasattr(module, "output_name")
        assert hasattr(module, "setup")
        assert hasattr(module, "get_output_dim")


class TestSequentialFusionSetupLayers:

    @pytest.mark.parametrize("hidden_dim", [32, 128])
    def test_creates_projection_per_input_feature(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
        hidden_dim: int,
    ):
        module = sequential_fusion_factory(
            input_features=["feat_a", "feat_b", "feat_c"],
            hidden_dim=hidden_dim,
        )
        dims = {"feat_a": 64, "feat_b": 128, "feat_c": 256}
        module.setup(feature_keys_to_dims=dims)
        assert module.projections is not None
        assert len(module.projections) == 3
        for proj in module.projections:
            assert proj.out_features == hidden_dim

    def test_projection_input_dims_match_feature_dims(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
    ):
        module = sequential_fusion_factory(
            input_features=["feat_a", "feat_b"],
        )
        dims = {"feat_a": 64, "feat_b": 128}
        module.setup(feature_keys_to_dims=dims)
        assert module.projections[0].in_features == 64
        assert module.projections[1].in_features == 128

    def test_handles_tuple_dimensions(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
    ):
        module = sequential_fusion_factory(
            input_features=["seq_feat"],
            hidden_dim=32,
        )
        dims = {"seq_feat": (10, 64)}
        module.setup(feature_keys_to_dims=dims)
        assert module.projections[0].in_features == 64

    def test_rejects_spatial_tuple_dimensions(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
    ):
        module = sequential_fusion_factory(
            input_features=["spatial_feat"],
        )
        dims = {"spatial_feat": (512, 7, 7)}
        with pytest.raises(ValueError, match="SequentialFusion requires flat or sequential"):
            module.setup(feature_keys_to_dims=dims)


class TestSequentialFusionForward:

    @pytest.mark.parametrize("time_steps", [None, 3])
    def test_forward_produces_correct_output(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        time_steps: int | None,
    ):
        hidden_dim = 32
        module = sequential_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dim=hidden_dim,
        )
        dims = {"feat_a": 64, "feat_b": 128}
        module.setup(feature_keys_to_dims=dims)
        features = [
            input_tensor_factory(
                input_dimension=dim,
                sequence_length=time_steps,
            )
            for dim in [64, 128]
        ]
        output = module(features)
        batch_size = 2
        if time_steps is not None:
            assert output.shape == (batch_size, time_steps, hidden_dim * 2)
        else:
            assert output.shape == (batch_size, hidden_dim * 2)
