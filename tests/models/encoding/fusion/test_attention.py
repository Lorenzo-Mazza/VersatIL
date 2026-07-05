"""Tests for versatil.models.encoding.fusion.attention module."""

import re
from collections.abc import Callable

import pytest
import torch

from versatil.models.encoding.fusion.attention import AttentionFusion
from versatil.models.feature_meta import FeatureMetadata, FeatureType


def _make_registry(
    dims: dict[str, tuple[int, ...]],
) -> dict[str, FeatureMetadata]:
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
def attention_fusion_factory() -> Callable[..., AttentionFusion]:
    """Factory for AttentionFusion instances."""

    def factory(
        input_features: list[str] | None = None,
        output_name: str = "attention_fused",
        hidden_dimension: int = 32,
        input_feature_query: str | None = None,
        number_of_heads: int = 4,
        dropout: float = 0.0,
        use_residual: bool = True,
        use_norm: bool = True,
    ) -> AttentionFusion:
        if input_features is None:
            input_features = ["rgb_features", "depth_features"]
        return AttentionFusion(
            input_features=input_features,
            output_name=output_name,
            hidden_dimension=hidden_dimension,
            input_feature_query=input_feature_query,
            number_of_heads=number_of_heads,
            dropout=dropout,
            use_residual=use_residual,
            use_norm=use_norm,
        )

    return factory


class TestAttentionFusionInitialization:
    def test_has_sequential_fusion_interface(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
    ):
        module = attention_fusion_factory()
        assert hasattr(module, "hidden_dimension")
        assert hasattr(module, "projections")
        assert hasattr(module, "setup")
        assert hasattr(module, "get_output_specification")

    @pytest.mark.parametrize("hidden_dimension", [32, 128])
    @pytest.mark.parametrize("output_name", ["attention_fused", "my_attn"])
    @pytest.mark.parametrize("use_residual", [True, False])
    @pytest.mark.parametrize("use_norm", [True, False])
    @pytest.mark.parametrize("input_feature_query", [None, "depth_features"])
    def test_stores_configuration(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        hidden_dimension: int,
        output_name: str,
        use_residual: bool,
        use_norm: bool,
        input_feature_query: str | None,
    ):
        module = attention_fusion_factory(
            hidden_dimension=hidden_dimension,
            output_name=output_name,
            use_residual=use_residual,
            use_norm=use_norm,
            input_feature_query=input_feature_query,
        )
        assert module.hidden_dimension == hidden_dimension
        assert module.output_name == output_name
        assert module.use_residual is use_residual
        assert module.use_norm is use_norm
        assert module.input_feature_query == input_feature_query

    def test_norms_created_when_use_norm_true(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
    ):
        module = attention_fusion_factory(
            input_features=["a", "b", "c"],
            use_norm=True,
        )
        assert module.norms is not None
        assert len(module.norms) == 3

    def test_norms_none_when_use_norm_false(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
    ):
        module = attention_fusion_factory(use_norm=False)
        assert module.norms is None

    def test_attention_module_exists(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
    ):
        module = attention_fusion_factory()
        assert module.attention is not None


class TestAttentionFusionForward:
    def test_raises_if_projections_not_set_up(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = attention_fusion_factory()
        features = [
            input_tensor_factory(input_dimension=64),
            input_tensor_factory(input_dimension=128),
        ]
        with pytest.raises(RuntimeError, match="Projections must be set up"):
            module(features)

    @pytest.mark.parametrize("time_steps", [None, 4])
    def test_output_shape_with_and_without_time(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        time_steps: int | None,
    ):
        batch_size = 2
        hidden_dimension = 32
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dimension=hidden_dimension,
        )
        module.setup(
            feature_registry=_make_registry({"feat_a": (64,), "feat_b": (128,)})
        )
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
        if time_steps is not None:
            assert output.shape == (batch_size, time_steps, hidden_dimension)
        else:
            assert output.shape == (batch_size, hidden_dimension)

    def test_sequential_features_with_time_dimension(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        rng,
    ):
        # The pipeline runs fusion before its T=1 squeeze, so sequential
        # features arrive as (B, T, S, D).
        batch_size, time_steps, sequence_length, hidden_dimension = 2, 1, 8, 32
        module = attention_fusion_factory(
            input_features=["tokens_a", "tokens_b"],
            hidden_dimension=hidden_dimension,
        )
        module.setup(
            feature_registry=_make_registry(
                {
                    "tokens_a": (sequence_length, 64),
                    "tokens_b": (sequence_length, 128),
                }
            )
        )
        features = [
            torch.from_numpy(
                rng.standard_normal(
                    (batch_size, time_steps, sequence_length, 64)
                ).astype("float32")
            ),
            torch.from_numpy(
                rng.standard_normal(
                    (batch_size, time_steps, sequence_length, 128)
                ).astype("float32")
            ),
        ]
        output = module(features)
        assert output.shape == (
            batch_size,
            time_steps,
            sequence_length,
            hidden_dimension,
        )
        specification = module.get_output_specification()
        assert specification.dimension == (sequence_length, hidden_dimension)

    def test_single_feature_returns_projected_query(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        hidden_dimension = 32
        module = attention_fusion_factory(
            input_features=["only_feat"],
            hidden_dimension=hidden_dimension,
            use_norm=False,
        )
        module.setup(feature_registry=_make_registry({"only_feat": (64,)}))
        features = [input_tensor_factory(input_dimension=64)]
        output = module(features)
        assert output.shape == (2, hidden_dimension)

    def test_query_feature_selection(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        hidden_dimension = 32
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dimension=hidden_dimension,
            input_feature_query="feat_b",
        )
        module.setup(
            feature_registry=_make_registry({"feat_a": (64,), "feat_b": (128,)})
        )
        features = [
            input_tensor_factory(input_dimension=64),
            input_tensor_factory(input_dimension=128),
        ]
        output = module(features)
        assert output.shape == (2, hidden_dimension)

    @pytest.mark.parametrize("use_residual", [True, False])
    def test_residual_connection_toggle(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        use_residual: bool,
    ):
        hidden_dimension = 32
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dimension=hidden_dimension,
            use_residual=use_residual,
        )
        module.setup(
            feature_registry=_make_registry({"feat_a": (64,), "feat_b": (128,)})
        )
        features = [
            input_tensor_factory(input_dimension=64),
            input_tensor_factory(input_dimension=128),
        ]
        output = module(features)
        assert output.shape == (2, hidden_dimension)

    @pytest.mark.parametrize("use_norm", [True, False])
    def test_norm_toggle(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        use_norm: bool,
    ):
        hidden_dimension = 32
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dimension=hidden_dimension,
            use_norm=use_norm,
        )
        module.setup(
            feature_registry=_make_registry({"feat_a": (64,), "feat_b": (128,)})
        )
        features = [
            input_tensor_factory(input_dimension=64),
            input_tensor_factory(input_dimension=128),
        ]
        output = module(features)
        assert output.shape == (2, hidden_dimension)

    def test_three_features_fused(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        hidden_dimension = 32
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b", "feat_c"],
            hidden_dimension=hidden_dimension,
        )
        module.setup(
            feature_registry=_make_registry(
                {"feat_a": (32,), "feat_b": (64,), "feat_c": (128,)}
            ),
        )
        features = [
            input_tensor_factory(input_dimension=32),
            input_tensor_factory(input_dimension=64),
            input_tensor_factory(input_dimension=128),
        ]
        output = module(features)
        assert output.shape == (2, hidden_dimension)

    def test_norms_none_with_use_norm_true_raises(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b"],
            use_norm=True,
        )
        module.setup(
            feature_registry=_make_registry({"feat_a": (64,), "feat_b": (128,)})
        )
        # Force norms to None while use_norm remains True
        module.norms = None
        features = [
            input_tensor_factory(input_dimension=64),
            input_tensor_factory(input_dimension=128),
        ]
        with pytest.raises(
            RuntimeError,
            match=re.escape("Norms should be initialized when use_norm is True"),
        ):
            module(features)


class TestAttentionFusionGetOutputSpecification:
    @pytest.mark.parametrize("hidden_dimension", [32, 128])
    def test_output_dim_equals_hidden_dim(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        hidden_dimension: int,
    ):
        module = attention_fusion_factory(hidden_dimension=hidden_dimension)
        spec = module.get_output_specification()
        assert spec.dimension[0] == hidden_dimension

    def test_output_name_matches(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
    ):
        module = attention_fusion_factory(output_name="test_attn")
        spec = module.get_output_specification()
        assert spec.key == "test_attn"

    def test_returns_specification_with_expected_fields(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
    ):
        module = attention_fusion_factory()
        spec = module.get_output_specification()
        assert hasattr(spec, "key")
        assert hasattr(spec, "dimension")
