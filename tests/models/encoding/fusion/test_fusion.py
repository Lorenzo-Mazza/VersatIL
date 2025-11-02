"""Comprehensive tests for fusion modules."""
import pytest
import torch

from refactoring.models.encoding.fusion.constants import FeatureType
from refactoring.models.encoding.fusion.base import (
    FusionInput,
    FusionOutput,
    FusionModule,
)
from refactoring.models.encoding.fusion.spatial import SpatialFusion
from refactoring.models.encoding.fusion.concat import ConcatFusion
from refactoring.models.encoding.fusion.mlp import MLPFusion
from refactoring.models.encoding.fusion.attention import AttentionFusion


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


class TestFusionInput:
    """Tests for FusionInput validation."""
    @pytest.mark.parametrize("feature_type", [
        FeatureType.SPATIAL,
        FeatureType.SEQUENTIAL,
        FeatureType.ANY,
    ])
    def test_feature_type_enum(self, feature_type):
        """Test that feature type enums are valid."""
        fusion_input = FusionInput(
            input_features=[RGB_FEATURES],
            feature_type=feature_type.value,
        )
        assert fusion_input.feature_type == feature_type.value


class TestFusionOutput:
    """Tests for FusionOutput properties."""

    def test_is_spatial(self):
        """Test spatial dimension detection (C, H, W)."""
        output = FusionOutput(output_name="fused", output_dim=(128, 32, 32))
        assert output.is_spatial
        assert not output.is_sequence
        assert not output.is_flat

    def test_is_sequence(self):
        """Test sequence dimension detection (T, D)."""
        output = FusionOutput(output_name="fused", output_dim=(512, 256))
        assert output.is_sequence
        assert not output.is_spatial
        assert not output.is_flat

    def test_is_flat(self):
        """Test flat dimension detection (D,)."""
        output = FusionOutput(output_name="fused", output_dim=(256,))
        assert output.is_flat
        assert not output.is_spatial
        assert not output.is_sequence

    def test_is_flat_with_int(self):
        """Test flat dimension detection with int instead of tuple."""
        output = FusionOutput(output_name="fused", output_dim=256)
        assert output.is_flat
        assert not output.is_spatial
        assert not output.is_sequence


class TestSpatialFusionInitialization:
    """Test SpatialFusion initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )

        assert fusion.input_features == [RGB_FEATURES, DEPTH_FEATURES]
        assert fusion.output_name == FUSED_FEATURES
        assert fusion.hidden_dim == 256
        assert fusion._initialized is False

    def test_setup_layers(self, spatial_feature_dims):
        """Test layer setup with spatial dimensions."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(spatial_feature_dims)

        assert fusion._initialized is True
        assert len(fusion.projections) == 2
        assert fusion.spatial_dims == (7, 7)

    def test_setup_mismatched_spatial_dims(self):
        """Test setup fails with mismatched spatial dimensions."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        feature_dims = {
            RGB_FEATURES: (256, 7, 7),
            DEPTH_FEATURES: (128, 14, 14),
        }

        with pytest.raises(ValueError, match="same spatial dimensions"):
            fusion.setup(feature_dims)

    def test_setup_idempotent(self, spatial_feature_dims):
        """Test that calling setup multiple times doesn't cause issues."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(spatial_feature_dims)
        fusion.setup(spatial_feature_dims)  # Should not error

        assert fusion._initialized


class TestSpatialFusionForward:
    """Test SpatialFusion forward pass."""

    def test_forward_4d_spatial_features(self, spatial_features_4d, spatial_feature_dims):
        """Test forward pass with 4D spatial features."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(spatial_feature_dims)

        output = fusion(spatial_features_4d)

        assert output.dim() == 4
        batch_size = spatial_features_4d[0].shape[0]
        assert output.shape == (batch_size, 256 * 2, 7, 7)
        assert output.dtype == torch.float32

    def test_forward_preserves_spatial_structure(self, spatial_features_4d, spatial_feature_dims):
        """Test forward pass preserves spatial structure."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(spatial_feature_dims)

        output = fusion(spatial_features_4d)

        assert output.shape[2:] == spatial_features_4d[0].shape[2:]

    def test_forward_single_feature(self, spatial_feature_dims):
        """Test forward pass with single spatial feature."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup({RGB_FEATURES: spatial_feature_dims[RGB_FEATURES]})

        features = [torch.randn(2, 256, 7, 7)]
        output = fusion(features)

        assert output.shape == (2, 256, 7, 7)

    def test_get_output_specification(self, spatial_feature_dims):
        """Test output specification is correct."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(spatial_feature_dims)

        spec = fusion.get_output_specification()
        assert spec.output_name == FUSED_FEATURES
        assert spec.output_dim == (512, 7, 7)  # 256 * 2 channels
        assert spec.is_spatial

    @pytest.mark.parametrize("num_features,expected_channels", [
        (2, 512),  # 256 * 2
        (3, 768),  # 256 * 3
    ])
    def test_different_feature_counts(self, num_features, expected_channels, batch_size):
        """Test fusion with different numbers of features."""
        feature_names = [RGB_FEATURES, DEPTH_FEATURES, SEGMENTATION_FEATURES][:num_features]
        feature_dims = {
            RGB_FEATURES: (256, 7, 7),
            DEPTH_FEATURES: (128, 7, 7),
            SEGMENTATION_FEATURES: (64, 7, 7),
        }

        fusion = SpatialFusion(
            input_features=feature_names,
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(feature_dims)

        features = [torch.randn(batch_size, feature_dims[name][0], 7, 7) for name in feature_names]
        output = fusion(features)

        assert output.shape == (batch_size, expected_channels, 7, 7)


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
        assert spec.is_flat


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

    def test_spatial_fusion_gradients(self, spatial_features_4d, spatial_feature_dims):
        """Test gradients flow through SpatialFusion."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(spatial_feature_dims)

        features = [f.requires_grad_(True) for f in spatial_features_4d]
        output = fusion(features)
        loss = output.sum()
        loss.backward()

        for feat in features:
            assert feat.grad is not None
            assert not torch.isnan(feat.grad).any()

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

    def test_spatial_to_sequential_pipeline(self, spatial_feature_dims, batch_size):
        """Test pipeline from spatial fusion to sequential fusion."""
        # Spatial fusion
        spatial_fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name="spatial_fused",
            hidden_dim=256,
        )
        spatial_fusion.setup(spatial_feature_dims)

        rgb = torch.randn(batch_size, 256, 7, 7)
        depth = torch.randn(batch_size, 128, 7, 7)
        spatial_output = spatial_fusion([rgb, depth])

        # Global average pooling to flatten
        pooled = spatial_output.mean(dim=[2, 3])  # (B, 512)

        # Sequential fusion
        sequential_dims = {
            "spatial_fused": pooled.shape[1],
            PROPRIO_FEATURES: 128,
        }
        sequential_fusion = ConcatFusion(
            input_features=["spatial_fused", PROPRIO_FEATURES],
            output_name="final_fused",
            hidden_dim=256,
        )
        sequential_fusion.setup(sequential_dims)

        proprio = torch.randn(batch_size, 128)
        final_output = sequential_fusion([pooled, proprio])

        assert final_output.shape == (batch_size, 512)  # 256 * 2

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

    def test_fusion_modules_no_nan_outputs(self, spatial_features_4d, sequence_features_2d,
                                          spatial_feature_dims, sequence_feature_dims):
        """Test all fusion modules produce valid outputs without NaN."""
        spatial_fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        spatial_fusion.setup(spatial_feature_dims)

        concat_fusion = ConcatFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        concat_fusion.setup(sequence_feature_dims)

        attention_fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        attention_fusion.setup(sequence_feature_dims)

        mlp_fusion = MLPFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
            mlp_hidden_dims=[512, 256],
        )
        mlp_fusion.setup(sequence_feature_dims)

        spatial_output = spatial_fusion(spatial_features_4d)
        concat_output = concat_fusion(sequence_features_2d)
        attention_output = attention_fusion(sequence_features_2d)
        mlp_output = mlp_fusion(sequence_features_2d)

        assert not torch.isnan(spatial_output).any()
        assert not torch.isnan(concat_output).any()
        assert not torch.isnan(attention_output).any()
        assert not torch.isnan(mlp_output).any()

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size, spatial_feature_dims):
        """Test fusion works with different batch sizes."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(spatial_feature_dims)

        features = [
            torch.randn(batch_size, 256, 7, 7),
            torch.randn(batch_size, 128, 7, 7),
        ]
        output = fusion(features)

        assert output.shape[0] == batch_size

    @pytest.mark.parametrize("hidden_dim", [128, 256, 512])
    def test_spatial_fusion_different_hidden_dims(self, hidden_dim, spatial_feature_dims):
        """Test SpatialFusion with different hidden dimensions."""
        fusion = SpatialFusion(
            input_features=[RGB_FEATURES, DEPTH_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=hidden_dim,
        )
        fusion.setup(spatial_feature_dims)

        features = [
            torch.randn(2, 256, 7, 7),
            torch.randn(2, 128, 7, 7),
        ]
        output = fusion(features)

        assert output.shape[1] == hidden_dim * 2

    def test_fusion_eval_mode(self, sequence_features_2d, sequence_feature_dims):
        """Test fusion modules in eval mode."""
        fusion = AttentionFusion(
            input_features=[LANGUAGE_FEATURES, PROPRIO_FEATURES],
            output_name=FUSED_FEATURES,
            hidden_dim=256,
        )
        fusion.setup(sequence_feature_dims)
        fusion.eval()

        with torch.no_grad():
            output = fusion(sequence_features_2d)

        assert not output.requires_grad

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