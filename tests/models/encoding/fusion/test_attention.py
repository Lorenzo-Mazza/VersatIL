"""Tests for versatil.models.encoding.fusion.attention module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.encoding.fusion.attention import AttentionFusion
from versatil.models.encoding.fusion.base import FusionOutput, SequentialFusion


@pytest.fixture
def attention_fusion_factory() -> Callable[..., AttentionFusion]:
    """Factory for AttentionFusion instances."""
    def factory(
        input_features: list[str] | None = None,
        output_name: str = "attention_fused",
        hidden_dim: int = 32,
        input_feature_query: str | None = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_residual: bool = True,
        use_norm: bool = True,
    ) -> AttentionFusion:
        if input_features is None:
            input_features = ["rgb_features", "depth_features"]
        return AttentionFusion(
            input_features=input_features,
            output_name=output_name,
            hidden_dim=hidden_dim,
            input_feature_query=input_feature_query,
            num_heads=num_heads,
            dropout=dropout,
            use_residual=use_residual,
            use_norm=use_norm,
        )
    return factory


class TestAttentionFusionInitialization:

    def test_inherits_from_sequential_fusion(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
    ):
        module = attention_fusion_factory()
        assert isinstance(module, SequentialFusion)

    @pytest.mark.parametrize("hidden_dim", [32, 128])
    @pytest.mark.parametrize("output_name", ["attention_fused", "my_attn"])
    @pytest.mark.parametrize("use_residual", [True, False])
    @pytest.mark.parametrize("use_norm", [True, False])
    @pytest.mark.parametrize("input_feature_query", [None, "depth_features"])
    def test_stores_configuration(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        hidden_dim: int,
        output_name: str,
        use_residual: bool,
        use_norm: bool,
        input_feature_query: str | None,
    ):
        module = attention_fusion_factory(
            hidden_dim=hidden_dim,
            output_name=output_name,
            use_residual=use_residual,
            use_norm=use_norm,
            input_feature_query=input_feature_query,
        )
        assert module.hidden_dim == hidden_dim
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
            input_tensor_factory(input_dim=64),
            input_tensor_factory(input_dim=128),
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
        hidden_dim = 32
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dim=hidden_dim,
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
        if time_steps is not None:
            assert output.shape == (batch_size, time_steps, hidden_dim)
        else:
            assert output.shape == (batch_size, hidden_dim)

    def test_single_feature_returns_projected_query(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        """When only one feature is provided, attention is skipped."""
        hidden_dim = 32
        module = attention_fusion_factory(
            input_features=["only_feat"],
            hidden_dim=hidden_dim,
            use_norm=False,
        )
        module.setup(feature_keys_to_dims={"only_feat": 64})
        features = [input_tensor_factory(input_dim=64)]
        output = module(features)
        assert output.shape == (2, hidden_dim)

    def test_query_feature_selection(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        """Specifying input_feature_query selects that feature as query."""
        hidden_dim = 32
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dim=hidden_dim,
            input_feature_query="feat_b",
        )
        module.setup(feature_keys_to_dims={"feat_a": 64, "feat_b": 128})
        features = [
            input_tensor_factory(input_dim=64),
            input_tensor_factory(input_dim=128),
        ]
        output = module(features)
        assert output.shape == (2, hidden_dim)

    @pytest.mark.parametrize("use_residual", [True, False])
    def test_residual_connection_toggle(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        use_residual: bool,
    ):
        hidden_dim = 32
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dim=hidden_dim,
            use_residual=use_residual,
        )
        module.setup(feature_keys_to_dims={"feat_a": 64, "feat_b": 128})
        features = [
            input_tensor_factory(input_dim=64),
            input_tensor_factory(input_dim=128),
        ]
        output = module(features)
        assert output.shape == (2, hidden_dim)

    @pytest.mark.parametrize("use_norm", [True, False])
    def test_norm_toggle(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
        use_norm: bool,
    ):
        hidden_dim = 32
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b"],
            hidden_dim=hidden_dim,
            use_norm=use_norm,
        )
        module.setup(feature_keys_to_dims={"feat_a": 64, "feat_b": 128})
        features = [
            input_tensor_factory(input_dim=64),
            input_tensor_factory(input_dim=128),
        ]
        output = module(features)
        assert output.shape == (2, hidden_dim)

    def test_three_features_fused(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        hidden_dim = 32
        module = attention_fusion_factory(
            input_features=["feat_a", "feat_b", "feat_c"],
            hidden_dim=hidden_dim,
        )
        module.setup(
            feature_keys_to_dims={"feat_a": 32, "feat_b": 64, "feat_c": 128},
        )
        features = [
            input_tensor_factory(input_dim=32),
            input_tensor_factory(input_dim=64),
            input_tensor_factory(input_dim=128),
        ]
        output = module(features)
        assert output.shape == (2, hidden_dim)


class TestAttentionFusionGetOutputSpecification:

    @pytest.mark.parametrize("hidden_dim", [32, 128])
    def test_output_dim_equals_hidden_dim(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
        hidden_dim: int,
    ):
        module = attention_fusion_factory(hidden_dim=hidden_dim)
        spec = module.get_output_specification()
        assert spec.output_dim == hidden_dim

    def test_output_name_matches(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
    ):
        module = attention_fusion_factory(output_name="test_attn")
        spec = module.get_output_specification()
        assert spec.output_name == "test_attn"

    def test_returns_fusion_output_type(
        self,
        attention_fusion_factory: Callable[..., AttentionFusion],
    ):
        module = attention_fusion_factory()
        spec = module.get_output_specification()
        assert isinstance(spec, FusionOutput)
