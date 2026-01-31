"""Comprehensive tests for fusion modules."""
import pytest
import torch

from versatil.models.encoding.fusion.base import FusionInput, FusionOutput, FusionModule
from versatil.models.encoding.fusion.concat import ConcatFusion
from versatil.models.encoding.fusion.mlp import MLPFusion
from versatil.models.encoding.fusion.attention import AttentionFusion


RGB_FEATURES = "rgb_features"
DEPTH_FEATURES = "depth_features"
SEGMENTATION_FEATURES = "segmentation_features"
LANGUAGE_FEATURES = "language_features"
PROPRIO_FEATURES = "proprio_features"
GRIPPER_FEATURES = "gripper_features"
JOINT_ANGLES_FEATURES = "joint_angles_features"
FUSED_FEATURES = "fused_features"

SPATIAL_FEATURES = [RGB_FEATURES, DEPTH_FEATURES]


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 4


@pytest.fixture
def temporal_length():
    """Default temporal length for sequence tests."""
    return 10


@pytest.fixture
def spatial_feature_dims():
    """Feature dimensions for spatial features (C, H, W)."""
    return {
        RGB_FEATURES: (256, 7, 7),
        DEPTH_FEATURES: (128, 7, 7),
        SEGMENTATION_FEATURES: (64, 7, 7),
    }


@pytest.fixture
def sequence_feature_dims():
    """Feature dimensions for sequence features (flat)."""
    return {
        LANGUAGE_FEATURES: 512,
        PROPRIO_FEATURES: 128,
        GRIPPER_FEATURES: 2,
        JOINT_ANGLES_FEATURES: 7,
    }


@pytest.fixture
def mixed_feature_dims(spatial_feature_dims, sequence_feature_dims):
    """Mixed spatial and sequence feature dimensions."""
    return {**spatial_feature_dims, **sequence_feature_dims}


@pytest.fixture
def spatial_features_4d(batch_size):
    """4D spatial features (B, C, H, W)."""
    return [
        torch.randn(batch_size, 256, 7, 7),
        torch.randn(batch_size, 128, 7, 7),
    ]


@pytest.fixture
def sequence_features_2d(batch_size):
    """2D sequence features (B, D)."""
    return [
        torch.randn(batch_size, 512),
        torch.randn(batch_size, 128),
    ]


@pytest.fixture
def sequence_features_3d(batch_size, temporal_length):
    """3D sequence features with temporal dimension (B, T, D)."""
    return [
        torch.randn(batch_size, temporal_length, 512),
        torch.randn(batch_size, temporal_length, 128),
    ]


class TestConcatFusionInitialization:
    """Test ConcatFusion initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        fusion = ConcatFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )

        assert fusion.input_features == [LANGUAGE_FEATURES, PROPRIO_FEATURES]
        assert fusion.output_name == FUSED_FEATURES
        assert fusion.hidden_dim == 256

    def test_setup_layers(self, sequence_feature_dims):
        """Test layer setup with sequence dimensions."""
        fusion = ConcatFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(sequence_feature_dims)

        assert len(fusion.projections) == 2

    def test_setup_with_spatial_dims_fails(self, spatial_feature_dims):
        """Test setup fails with spatial dimensions."""
        fusion = ConcatFusion(
            input_features=[RGB_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )

        with pytest.raises(ValueError, match="requires flat dimensions"):
            fusion.setup(spatial_feature_dims)


class TestConcatFusionForward:
    """Test ConcatFusion forward pass."""

    def test_forward_2d_features(self, sequence_features_2d, sequence_feature_dims):
        """Test forward pass with 2D features (B, D)."""
        fusion = ConcatFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(sequence_feature_dims)

        output = fusion(sequence_features_2d)

        batch_size = sequence_features_2d[0].shape[0]
        assert output.shape == (batch_size, 256 * 2)
        assert output.dtype == torch.float32

    def test_forward_3d_features(self, sequence_features_3d, sequence_feature_dims):
        """Test forward pass with 3D features (B, T, D)."""
        fusion = ConcatFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(sequence_feature_dims)

        output = fusion(sequence_features_3d)

        batch_size, temporal_length = sequence_features_3d[0].shape[:2]
        assert output.shape == (batch_size, temporal_length, 256 * 2)

    def test_get_output_specification(self, sequence_feature_dims):
        """Test output specification is correct."""
        fusion = ConcatFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(sequence_feature_dims)

        spec = fusion.get_output_specification()
        assert spec.output_name == FUSED_FEATURES
        assert spec.output_dim == 512  # 256 * 2


class TestMLPFusionInitialization:
    """Test MLPFusion initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        fusion = MLPFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            mlp_hidden_dims=[512, 256],
        )

        assert fusion.input_features == [LANGUAGE_FEATURES, PROPRIO_FEATURES]
        assert fusion.output_name == FUSED_FEATURES
        assert fusion.hidden_dim == 256
        assert fusion.output_dim == 256  # Last hidden dim

    def test_setup_layers(self, sequence_feature_dims):
        """Test layer setup with sequence dimensions."""
        fusion = MLPFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            mlp_hidden_dims=[512, 256],
        )
        fusion.setup(sequence_feature_dims)

        assert len(fusion.projections) == 2


class TestMLPFusionForward:
    """Test MLPFusion forward pass."""

    def test_forward_2d_features(self, sequence_features_2d, sequence_feature_dims):
        """Test forward pass with 2D features (B, D)."""
        fusion = MLPFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            mlp_hidden_dims=[512, 256],
        )
        fusion.setup(sequence_feature_dims)

        output = fusion(sequence_features_2d)

        batch_size = sequence_features_2d[0].shape[0]
        assert output.shape == (batch_size, 256)

    def test_forward_3d_features(self, sequence_features_3d, sequence_feature_dims):
        """Test forward pass with 3D features (B, T, D)."""
        fusion = MLPFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            mlp_hidden_dims=[512, 256],
        )
        fusion.setup(sequence_feature_dims)

        output = fusion(sequence_features_3d)

        batch_size, temporal_length = sequence_features_3d[0].shape[:2]
        assert output.shape == (batch_size, temporal_length, 256)

    def test_get_output_specification(self, sequence_feature_dims):
        """Test output specification matches final MLP dimension."""
        fusion = MLPFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            mlp_hidden_dims=[512, 128],
        )
        fusion.setup(sequence_feature_dims)

        spec = fusion.get_output_specification()
        assert spec.output_name == FUSED_FEATURES
        assert spec.output_dim == 128  # Last hidden dim

    def test_different_mlp_architectures(self, sequence_feature_dims):
        """Test MLPFusion with different MLP architectures."""
        mlp_configs = [
            [256],
            [512, 256],
            [512, 256, 128],
        ]

        for mlp_hidden_dims in mlp_configs:
            fusion = MLPFusion(
                input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
                output_name=FUSED_FEATURES,
                hidden_dim=256,
                mlp_hidden_dims=mlp_hidden_dims,
            )
            fusion.setup(sequence_feature_dims)

            features = [torch.randn(2, 512), torch.randn(2, 128)]
            output = fusion(features)

            assert output.shape == (2, mlp_hidden_dims[-1])


class TestAttentionFusionInitialization:
    """Test AttentionFusion initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )

        assert fusion.input_features == [LANGUAGE_FEATURES, PROPRIO_FEATURES]
        assert fusion.output_name == FUSED_FEATURES
        assert fusion.hidden_dim == 256

    def test_setup_layers(self, sequence_feature_dims):
        """Test layer setup with sequence dimensions."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            use_norm=True,
        )
        fusion.setup(sequence_feature_dims)

        assert len(fusion.projections) == 2
        assert len(fusion.norms) == 2

    def test_setup_without_norm(self, sequence_feature_dims):
        """Test layer setup without normalization."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            use_norm=False,
        )
        fusion.setup(sequence_feature_dims)

        assert len(fusion.projections) == 2


class TestAttentionFusionForward:
    """Test AttentionFusion forward pass."""

    def test_forward_single_feature(self, sequence_feature_dims, batch_size):
        """Test forward with single feature returns projected feature."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            use_norm=False,
        )
        fusion.setup(sequence_feature_dims)

        features = [torch.randn(batch_size, 512)]
        output = fusion(features)

        assert output.shape == (batch_size, 256)

    def test_forward_2d_features(self, sequence_features_2d, sequence_feature_dims):
        """Test forward pass with 2D features (B, D)."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            use_norm=False,
        )
        fusion.setup(sequence_feature_dims)

        output = fusion(sequence_features_2d)

        batch_size = sequence_features_2d[0].shape[0]
        assert output.shape == (batch_size, 256)

    def test_forward_3d_features(self, sequence_features_3d, sequence_feature_dims):
        """Test forward pass with 3D features (B, T, D)."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            use_norm=False,
        )
        fusion.setup(sequence_feature_dims)

        output = fusion(sequence_features_3d)

        batch_size, temporal_length = sequence_features_3d[0].shape[:2]
        assert output.shape == (batch_size, temporal_length, 256)

    def test_custom_query_feature(self, sequence_feature_dims, batch_size):
        """Test using custom query feature for attention."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES, GRIPPER_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            input_feature_query=PROPRIO_FEATURES,
            use_norm=False,
        )
        fusion.setup(sequence_feature_dims)

        features = [
            torch.randn(batch_size, 512),
            torch.randn(batch_size, 128),
            torch.randn(batch_size, 2),
        ]
        output = fusion(features)

        assert output.shape == (batch_size, 256)

    def test_get_output_specification(self, sequence_feature_dims):
        """Test output specification is correct."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(sequence_feature_dims)

        spec = fusion.get_output_specification()
        assert spec.output_name == FUSED_FEATURES
        assert spec.output_dim == 256

    @pytest.mark.parametrize("use_residual,use_norm", [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ])
    def test_residual_and_norm_combinations(self, sequence_features_2d, sequence_feature_dims,
                                            use_residual, use_norm):
        """Test all combinations of residual and normalization."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            use_residual=use_residual,
            use_norm=use_norm,
        )
        fusion.setup(sequence_feature_dims)

        output = fusion(sequence_features_2d)

        batch_size = sequence_features_2d[0].shape[0]
        assert output.shape == (batch_size, 256)

    @pytest.mark.parametrize("num_heads", [1, 4, 8])
    def test_different_num_heads(self, num_heads, sequence_features_2d, sequence_feature_dims):
        """Test AttentionFusion with different number of attention heads."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            num_heads=num_heads,
        )
        fusion.setup(sequence_feature_dims)

        output = fusion(sequence_features_2d)

        batch_size = sequence_features_2d[0].shape[0]
        assert output.shape == (batch_size, 256)


class TestFusionModuleBase:
    """Tests for FusionModule abstract base class."""

    def test_nn_module(self):
        """Test that FusionModule is instance of nn.Module."""
        module = FusionModule(
                input_specification=FusionInput(input_features=["test"]),
                output_name="test",
            )
        assert isinstance(module, torch.nn.Module)

    def test_input_features_property(self):
        """Test input_features property getter."""
        fusion = ConcatFusion(
            input_features=["feat1", "feat2"],
            output_name="test",
            hidden_dim=64,
        )
        assert fusion.input_features == ["feat1", "feat2"]

    def test_input_features_setter(self):
        """Test input_features property setter."""
        fusion = ConcatFusion(
            input_features=["feat1", "feat2"],
            output_name="test",
            hidden_dim=64,
        )
        fusion.input_features = ["new_feat1", "new_feat2"]
        assert fusion.input_features == ["new_feat1", "new_feat2"]

    def test_get_output_dim_backward_compatibility(self, sequence_feature_dims):
        """Test get_output_dim for backward compatibility."""
        fusion = ConcatFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name="test",
            hidden_dim=64,
        )
        fusion.setup(sequence_feature_dims)

        output_dim = fusion.get_output_dim()
        assert output_dim == 128  # 64 * 2


class TestGradientFlow:
    """Test gradient flow through fusion modules."""

    def test_concat_fusion_gradients(self, sequence_features_2d, sequence_feature_dims):
        """Test gradients flow through ConcatFusion."""
        fusion = ConcatFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(sequence_feature_dims)

        features = [f.requires_grad_(True) for f in sequence_features_2d]
        output = fusion(features)
        loss = output.sum()
        loss.backward()

        for feat in features:
            assert feat.grad is not None

    def test_attention_fusion_gradients(self, sequence_features_2d, sequence_feature_dims):
        """Test gradients flow through AttentionFusion."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(sequence_feature_dims)

        features = [f.requires_grad_(True) for f in sequence_features_2d]
        output = fusion(features)
        loss = output.sum()
        loss.backward()

        for feat in features:
            assert feat.grad is not None

    def test_mlp_fusion_gradients(self, sequence_features_2d, sequence_feature_dims):
        """Test gradients flow through MLPFusion."""
        fusion = MLPFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            mlp_hidden_dims=[512, 256],
        )
        fusion.setup(sequence_feature_dims)

        features = [f.requires_grad_(True) for f in sequence_features_2d]
        output = fusion(features)
        loss = output.sum()
        loss.backward()

        for feat in features:
            assert feat.grad is not None


class TestFusionIntegration:
    """Integration tests for fusion modules."""


    def test_multiple_attention_fusion_layers(self, sequence_feature_dims, batch_size):
        """Test stacking multiple attention fusion layers."""
        # First layer
        fusion1 = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name="fused1",
            hidden_dim=256,
            use_residual=True,
        )
        fusion1.setup(sequence_feature_dims)

        language = torch.randn(batch_size, 512)
        proprio = torch.randn(batch_size, 128)
        output1 = fusion1([language, proprio])

        # Second layer
        feature_dims_2 = {
            "fused1": 256,
            GRIPPER_FEATURES: 2,
        }
        fusion2 = AttentionFusion(
            input_features=["fused1", GRIPPER_FEATURES],
            output_name="fused2",
            hidden_dim=128,
            use_residual=True,
        )
        fusion2.setup(feature_dims_2)

        gripper = torch.randn(batch_size, 2)
        output2 = fusion2([output1, gripper])

        assert output2.shape == (batch_size, 128)

    def test_concat_fusion_deterministic_eval(self, sequence_feature_dims):
        """Test ConcatFusion produces deterministic output in eval mode."""
        fusion = ConcatFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(sequence_feature_dims)
        fusion.eval()

        features = [torch.randn(2, 512), torch.randn(2, 128)]

        with torch.no_grad():
            output1 = fusion(features)
            output2 = fusion(features)

        torch.testing.assert_close(output1, output2)