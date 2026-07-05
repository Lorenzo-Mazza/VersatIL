"""Tests for versatil.models.encoding.fusion.base module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.encoding.fusion.base import (
    FusionInput,
    FusionModule,
    SequentialFusion,
)
from versatil.models.feature_meta import FeatureMetadata, FeatureType


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

    def _setup_layers(self, feature_registry: dict[str, FeatureMetadata]):
        pass

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return features[0]

    def get_output_specification(self) -> FeatureMetadata:
        return FeatureMetadata(
            key=self.output_name,
            feature_type=FeatureType.FLAT.value,
            dimension=(self._output_dim,),
        )


class ConcreteSequentialFusion(SequentialFusion):
    """Concrete implementation for testing abstract SequentialFusion."""

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if self.projections is None:
            raise RuntimeError("Projections not set up")
        projected = [
            proj(feat) for feat, proj in zip(features, self.projections, strict=True)
        ]
        return torch.cat(projected, dim=-1)

    def get_output_specification(self) -> FeatureMetadata:
        output_dim = self.hidden_dimension * len(self.input_features)
        dimension: tuple[int, ...] = (output_dim,)
        if self._output_feature_type == FeatureType.SEQUENTIAL.value:
            dimension = (self._output_sequence_length, output_dim)
        return FeatureMetadata(
            key=self.output_name,
            feature_type=self._output_feature_type,
            dimension=dimension,
        )


def _make_feature_registry(
    dims: dict[str, tuple[int, ...]],
) -> dict[str, FeatureMetadata]:
    """Helper to build a feature registry from dimension tuples."""
    return {
        name: FeatureMetadata(
            key=name,
            feature_type=FeatureType.FLAT.value
            if len(dim) == 1
            else FeatureType.SEQUENTIAL.value,
            dimension=dim,
        )
        for name, dim in dims.items()
    }


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
        hidden_dimension: int = 64,
    ) -> ConcreteSequentialFusion:
        if input_features is None:
            input_features = ["rgb_features", "depth_features"]
        return ConcreteSequentialFusion(
            input_features=input_features,
            output_name=output_name,
            hidden_dimension=hidden_dimension,
        )

    return factory


class TestFusionInputDataclass:
    @pytest.mark.parametrize("input_features", [["rgb", "depth"], ["a", "b", "c"]])
    def test_stores_configuration(self, input_features: list[str]):
        spec = FusionInput(input_features=input_features)
        assert spec.input_features == input_features


class TestFusionModuleInitialization:
    @pytest.mark.parametrize(
        "input_features",
        [
            ["rgb_features", "depth_features"],
            ["feat_a", "feat_b", "feat_c"],
        ],
    )
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
        registry = _make_feature_registry(
            {"rgb_features": (64,), "depth_features": (32,)}
        )
        module.setup(feature_registry=registry)
        assert module._initialized is True

    def test_setup_skips_if_already_initialized(
        self,
        fusion_module_factory: Callable[..., ConcreteFusionModule],
    ):
        module = fusion_module_factory()
        registry = _make_feature_registry(
            {"rgb_features": (64,), "depth_features": (32,)}
        )
        module.setup(feature_registry=registry)
        registry_changed = _make_feature_registry({"rgb_features": (128,)})
        module.setup(feature_registry=registry_changed)
        assert module._initialized is True


class TestFusionModuleGetOutputSpecification:
    def test_returns_feature_metadata(
        self,
        fusion_module_factory: Callable[..., ConcreteFusionModule],
    ):
        module = fusion_module_factory()
        spec = module.get_output_specification()
        assert spec.key == "fused_output"
        assert spec.dimension == (64,)
        assert spec.feature_type == FeatureType.FLAT.value


class TestSequentialFusionInitialization:
    @pytest.mark.parametrize(
        "input_features",
        [
            ["rgb_features", "depth_features"],
            ["feat_a", "feat_b", "feat_c"],
        ],
    )
    @pytest.mark.parametrize("hidden_dimension", [64, 128])
    @pytest.mark.parametrize("output_name", ["fused_output", "seq_fused"])
    def test_stores_configuration(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
        input_features: list[str],
        hidden_dimension: int,
        output_name: str,
    ):
        module = sequential_fusion_factory(
            input_features=input_features,
            hidden_dimension=hidden_dimension,
            output_name=output_name,
        )
        assert module.input_features == input_features
        assert module.hidden_dimension == hidden_dimension
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
        assert hasattr(module, "get_output_specification")


class TestSequentialFusionSetupLayers:
    @pytest.mark.parametrize("hidden_dimension", [32, 128])
    def test_creates_projection_per_input_feature(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
        hidden_dimension: int,
    ):
        module = sequential_fusion_factory(
            input_features=["feat_a", "feat_b", "feat_c"],
            hidden_dimension=hidden_dimension,
        )
        registry = _make_feature_registry(
            {"feat_a": (64,), "feat_b": (128,), "feat_c": (256,)}
        )
        module.setup(feature_registry=registry)
        assert module.projections is not None
        assert len(module.projections) == 3
        for proj in module.projections:
            assert proj.out_features == hidden_dimension

    def test_projection_input_dims_match_feature_dims(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
    ):
        module = sequential_fusion_factory(
            input_features=["feat_a", "feat_b"],
        )
        registry = _make_feature_registry({"feat_a": (64,), "feat_b": (128,)})
        module.setup(feature_registry=registry)
        assert module.projections[0].in_features == 64
        assert module.projections[1].in_features == 128

    def test_handles_sequential_dimensions(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
    ):
        module = sequential_fusion_factory(
            input_features=["seq_feat"],
            hidden_dimension=32,
        )
        registry = {
            "seq_feat": FeatureMetadata(
                key="seq_feat",
                feature_type=FeatureType.SEQUENTIAL.value,
                dimension=(10, 64),
            )
        }
        module.setup(feature_registry=registry)
        assert module.projections[0].in_features == 64

    @pytest.mark.parametrize(
        "registry, expected_type",
        [
            (
                {"feat_a": (64,), "feat_b": (128,)},
                FeatureType.FLAT.value,
            ),
            (
                {
                    "feat_a": (
                        10,
                        64,
                    ),
                    "feat_b": (
                        10,
                        128,
                    ),
                },
                FeatureType.SEQUENTIAL.value,
            ),
        ],
    )
    def test_output_feature_type_matches_inputs(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
        registry: dict[str, tuple[int, ...]],
        expected_type: str,
    ):
        module = sequential_fusion_factory(
            input_features=list(registry.keys()),
        )
        module.setup(feature_registry=_make_feature_registry(registry))
        spec = module.get_output_specification()
        assert spec.feature_type == expected_type

    def test_rejects_mixed_flat_and_sequential_features(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
    ):
        # Mixed inputs used to pass setup and crash with an opaque shape
        # error inside torch.cat on the first forward pass.
        module = sequential_fusion_factory(input_features=["feat_a", "feat_b"])
        registry = _make_feature_registry({"feat_a": (64,), "feat_b": (10, 128)})

        with pytest.raises(ValueError, match="share one feature type"):
            module.setup(feature_registry=registry)

    def test_rejects_sequential_features_with_mismatched_lengths(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
    ):
        module = sequential_fusion_factory(input_features=["feat_a", "feat_b"])
        registry = _make_feature_registry({"feat_a": (10, 64), "feat_b": (12, 128)})

        with pytest.raises(ValueError, match="equal sequence lengths"):
            module.setup(feature_registry=registry)

    def test_sequential_output_metadata_carries_sequence_length(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
    ):
        module = sequential_fusion_factory(input_features=["feat_a", "feat_b"])
        registry = _make_feature_registry({"feat_a": (10, 64), "feat_b": (10, 128)})
        module.setup(feature_registry=registry)

        spec = module.get_output_specification()

        assert spec.feature_type == FeatureType.SEQUENTIAL.value
        assert spec.dimension[0] == 10

    def test_rejects_spatial_features(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
    ):
        module = sequential_fusion_factory(
            input_features=["spatial_feat"],
        )
        registry = {
            "spatial_feat": FeatureMetadata(
                key="spatial_feat",
                feature_type=FeatureType.SPATIAL.value,
                dimension=(512, 7, 7),
            )
        }
        with pytest.raises(
            ValueError, match="SequentialFusion requires flat or sequential"
        ):
            module.setup(feature_registry=registry)


class TestSequentialFusionForward:
    @pytest.mark.parametrize("time_steps", [None, 3])
    def test_forward_produces_correct_output(
        self,
        sequential_fusion_factory: Callable[..., ConcreteSequentialFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        time_steps: int | None,
    ):
        hidden_dimension = 32
        module = sequential_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dimension=hidden_dimension,
        )
        registry = _make_feature_registry({"feat_a": (64,), "feat_b": (128,)})
        module.setup(feature_registry=registry)
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
            assert output.shape == (batch_size, time_steps, hidden_dimension * 2)
        else:
            assert output.shape == (batch_size, hidden_dimension * 2)
