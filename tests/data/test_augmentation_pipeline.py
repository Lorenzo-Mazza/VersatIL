import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from omegaconf import OmegaConf

from refactoring.data.augmentation_pipeline import AugmentationPipeline


@pytest.fixture
def sample_rgb_images():
    """Generate sample RGB images (T=3, H=64, W=64, C=3)."""
    np.random.seed(42)
    return np.random.rand(3, 64, 64, 3).astype(np.float32)


@pytest.fixture
def sample_depth_images():
    """Generate sample depth images (T=3, H=64, W=64)."""
    np.random.seed(42)
    return np.random.rand(3, 64, 64).astype(np.float32)


@pytest.fixture
def sample_proprio_data():
    """Generate sample proprioceptive data (T=5, features=7)."""
    np.random.seed(42)
    return np.random.rand(5, 7).astype(np.float32)


class TestAugmentationPipelineInitialization:
    """Test initialization and configuration."""


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_init_no_augmentations(self, mock_instantiate):
        """Test initialization with no augmentations."""
        pipeline = AugmentationPipeline(train=True)

        assert not pipeline.use_color
        assert not pipeline.use_spatial
        assert not pipeline.use_rotation
        assert not pipeline.use_resize
        assert pipeline.photometric_transform is None
        assert pipeline.spatial_transform is None
        assert pipeline.rotation_transform is None
        assert pipeline.resize_transform_rgb is None
        assert pipeline.resize_transform_depth is None
        mock_instantiate.assert_not_called()


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_init_with_color_train(self, mock_instantiate):
        """Test that color augmentation is enabled in train mode."""
        mock_transform = MagicMock()
        mock_instantiate.return_value = mock_transform
        mock_config = MagicMock()

        pipeline = AugmentationPipeline(
            color_augmentation=mock_config,
            train=True
        )

        assert pipeline.use_color
        assert pipeline.photometric_transform == mock_transform
        mock_instantiate.assert_called_once_with(mock_config)


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_init_with_color_eval(self, mock_instantiate):
        """Test that color augmentation is disabled in eval mode."""
        mock_config = MagicMock()

        pipeline = AugmentationPipeline(
            color_augmentation=mock_config,
            train=False
        )

        assert not pipeline.use_color
        assert pipeline.photometric_transform is None
        mock_instantiate.assert_not_called()


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_init_with_spatial_train(self, mock_instantiate):
        """Test that spatial augmentation is enabled in train mode."""
        mock_transform = MagicMock()
        mock_instantiate.return_value = mock_transform
        mock_config = MagicMock()

        pipeline = AugmentationPipeline(
            spatial_augmentation=mock_config,
            train=True
        )

        assert pipeline.use_spatial
        assert pipeline.spatial_transform == mock_transform
        mock_instantiate.assert_called_once_with(mock_config)


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_init_with_rotation_train(self, mock_instantiate):
        """Test that rotation augmentation is enabled in train mode."""
        mock_transform = MagicMock()
        mock_instantiate.return_value = mock_transform
        mock_config = MagicMock()

        pipeline = AugmentationPipeline(
            rotation_augmentation=mock_config,
            train=True
        )

        assert pipeline.use_rotation
        assert pipeline.rotation_transform == mock_transform
        mock_instantiate.assert_called_once_with(mock_config)


    @patch('refactoring.data.augmentation_pipeline.A.Resize')
    def test_init_with_resize(self, mock_resize_class):
        """Test initialization with resize parameters."""
        mock_rgb_resize = MagicMock()
        mock_depth_resize = MagicMock()
        mock_resize_class.side_effect = [mock_rgb_resize, mock_depth_resize]

        pipeline = AugmentationPipeline(
            target_height=224,
            target_width=224,
            train=True
        )

        assert pipeline.use_resize
        assert pipeline.resize_transform_rgb == mock_rgb_resize
        assert pipeline.resize_transform_depth == mock_depth_resize
        mock_resize_class.assert_has_calls([
            call(height=224, width=224, interpolation=1, p=1.0),
            call(height=224, width=224, interpolation=0, p=1.0)
        ])


class TestSetupRotation:
    """Test rotation setup and matrix computation."""


    def test_setup_rotation_disabled(self):
        """Test that no rotation is returned when not configured."""
        pipeline = AugmentationPipeline(train=True)

        angle, matrix = pipeline.setup_rotation()

        assert angle == 0
        assert matrix is None


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_setup_rotation_probability_miss(self, mock_instantiate):
        """Test that rotation is skipped when probability check fails."""
        mock_transform = MagicMock()
        mock_transform.p = 0.5
        mock_transform.limit = [-30, 30]
        mock_instantiate.return_value = mock_transform

        pipeline = AugmentationPipeline(
            rotation_augmentation=MagicMock(),
            train=True
        )

        with patch('random.random', return_value=0.6):  # > p=0.5
            angle, matrix = pipeline.setup_rotation()

        assert angle == 0
        assert matrix is None


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_setup_rotation_creates_matrix(self, mock_instantiate):
        """Test rotation matrix creation for positive angle."""
        mock_transform = MagicMock()
        mock_transform.p = 0.5
        mock_transform.limit = [-30, 30]
        mock_instantiate.return_value = mock_transform

        pipeline = AugmentationPipeline(
            rotation_augmentation=MagicMock(),
            train=True
        )

        with patch('random.random', return_value=0.3), \
                patch('random.uniform', return_value=45.0):
            angle, matrix = pipeline.setup_rotation()

        assert angle == 45.0
        assert matrix is not None
        assert matrix.shape == (3, 3)

        # Verify rotation matrix properties for 45 degrees
        expected_cos = np.cos(np.deg2rad(45))
        expected_sin = np.sin(np.deg2rad(45))

        np.testing.assert_allclose(matrix[0, 0], expected_cos, rtol=1e-5)
        np.testing.assert_allclose(matrix[0, 1], -expected_sin, rtol=1e-5)
        np.testing.assert_allclose(matrix[1, 0], expected_sin, rtol=1e-5)
        np.testing.assert_allclose(matrix[1, 1], expected_cos, rtol=1e-5)
        np.testing.assert_array_equal(matrix[2, :], [0, 0, 1])


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_setup_rotation_negative_angle(self, mock_instantiate):
        """Test rotation matrix creation for negative angle."""
        mock_transform = MagicMock()
        mock_transform.p = 1.0
        mock_transform.limit = [-30, 30]
        mock_instantiate.return_value = mock_transform

        pipeline = AugmentationPipeline(
            rotation_augmentation=MagicMock(),
            train=True
        )

        with patch('random.uniform', return_value=-45.0):
            angle, matrix = pipeline.setup_rotation()

        assert angle == -45.0
        assert matrix is not None

        expected_cos = np.cos(np.deg2rad(-45))
        expected_sin = np.sin(np.deg2rad(-45))

        np.testing.assert_allclose(matrix[0, 0], expected_cos, rtol=1e-5)
        np.testing.assert_allclose(matrix[0, 1], -expected_sin, rtol=1e-5)
        np.testing.assert_allclose(matrix[1, 0], expected_sin, rtol=1e-5)
        np.testing.assert_allclose(matrix[1, 1], expected_cos, rtol=1e-5)


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_setup_rotation_zero_angle_returns_none(self, mock_instantiate):
        """Test that zero angle returns None for matrix."""
        mock_transform = MagicMock()
        mock_transform.p = 1.0
        mock_transform.limit = [0, 0]
        mock_instantiate.return_value = mock_transform

        pipeline = AugmentationPipeline(
            rotation_augmentation=MagicMock(),
            train=True
        )

        with patch('random.uniform', return_value=0.0):
            angle, matrix = pipeline.setup_rotation()

        assert angle == 0
        assert matrix is None


class TestApplyRGBAugmentations:
    """Test RGB image augmentation application."""


    def test_apply_rgb_no_transforms(self, sample_rgb_images):
        """Test that images are unchanged when no transforms configured."""
        pipeline = AugmentationPipeline(train=True)

        result = pipeline.apply_rgb_augmentations(sample_rgb_images)

        np.testing.assert_array_equal(result, sample_rgb_images)


    @patch('refactoring.data.augmentation_pipeline.A.Resize')
    def test_apply_rgb_with_resize(self, mock_resize_class, sample_rgb_images):
        """Test resize is applied to RGB images."""
        mock_resize = MagicMock()
        resized_frame = np.ones((224, 224, 3), dtype=np.float32) * 0.5
        mock_resize.return_value = {'image': resized_frame}
        mock_resize_class.return_value = mock_resize

        pipeline = AugmentationPipeline(
            target_height=224,
            target_width=224,
            train=True
        )

        result = pipeline.apply_rgb_augmentations(sample_rgb_images)

        assert result.shape == (3, 224, 224, 3)
        np.testing.assert_allclose(result, np.stack([resized_frame] * 3))
        assert mock_resize.call_count == 3


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_apply_rgb_with_color(self, mock_instantiate, sample_rgb_images):
        """Test color augmentation is applied."""
        mock_transform = MagicMock()
        modified_frame = sample_rgb_images[0] * 1.1
        mock_transform.side_effect = lambda image: {'image': image * 1.1}
        mock_instantiate.return_value = mock_transform

        pipeline = AugmentationPipeline(
            color_augmentation=MagicMock(),
            train=True
        )

        result = pipeline.apply_rgb_augmentations(sample_rgb_images)

        assert result.shape == sample_rgb_images.shape
        np.testing.assert_allclose(result[0], modified_frame)
        assert mock_transform.call_count == 3


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_apply_rgb_with_spatial(self, mock_instantiate, sample_rgb_images):
        """Test spatial augmentation is applied."""
        mock_transform = MagicMock()
        modified_frame = sample_rgb_images[0] + 0.1
        mock_transform.side_effect = lambda image: {'image': image + 0.1}
        mock_instantiate.return_value = mock_transform

        pipeline = AugmentationPipeline(
            spatial_augmentation=MagicMock(),
            train=True
        )

        result = pipeline.apply_rgb_augmentations(sample_rgb_images)

        assert result.shape == sample_rgb_images.shape
        np.testing.assert_allclose(result[0], modified_frame)
        assert mock_transform.call_count == 3


    @patch('refactoring.data.augmentation_pipeline.A.Rotate')
    def test_apply_rgb_with_rotation(self, mock_rotate_class, sample_rgb_images):
        """Test rotation is applied with correct interpolation."""
        pipeline = AugmentationPipeline(train=True)

        mock_rotate = MagicMock()
        rotated_frame = np.rot90(sample_rgb_images[0])
        mock_rotate.return_value = {'image': rotated_frame}
        mock_rotate_class.return_value = mock_rotate

        result = pipeline.apply_rgb_augmentations(sample_rgb_images, angle=90.0)

        assert result.shape == sample_rgb_images.shape
        np.testing.assert_allclose(result[0], rotated_frame)
        mock_rotate_class.assert_called_once_with(limit=(90.0, 90.0), p=1.0, interpolation=1)
        assert mock_rotate.call_count == 3


    def test_apply_rgb_skip_rotation_when_zero(self, sample_rgb_images):
        """Test rotation is skipped when angle is 0."""
        pipeline = AugmentationPipeline(train=True)

        with patch('refactoring.data.augmentation_pipeline.A.Rotate') as mock_rotate:
            result = pipeline.apply_rgb_augmentations(sample_rgb_images, angle=0)

            mock_rotate.assert_not_called()
            np.testing.assert_array_equal(result, sample_rgb_images)


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    @patch('refactoring.data.augmentation_pipeline.A.Resize')
    def test_apply_rgb_augmentation_order(self, mock_resize_class, mock_instantiate, sample_rgb_images):
        """Test augmentations are applied in correct order: resize -> color -> spatial -> rotate."""
        mock_resize = MagicMock()
        mock_resize.side_effect = lambda image: {'image': image + 0.1}  # Simulate resize
        mock_resize_class.return_value = mock_resize

        mock_color = MagicMock()
        mock_color.side_effect = lambda image: {'image': image * 1.1}  # Simulate color

        mock_spatial = MagicMock()
        mock_spatial.side_effect = lambda image: {'image': image - 0.05}  # Simulate spatial

        mock_instantiate.side_effect = [mock_color, mock_spatial]

        pipeline = AugmentationPipeline(
            color_augmentation=MagicMock(),
            spatial_augmentation=MagicMock(),
            target_height=64,  # Same size to keep shape
            target_width=64,
            train=True
        )

        # Apply without rotation
        result = pipeline.apply_rgb_augmentations(sample_rgb_images, angle=0)

        # Expected: original +0.1 (resize) *1.1 (color) -0.05 (spatial)
        expected = ((sample_rgb_images + 0.1) * 1.1) - 0.05
        np.testing.assert_allclose(result, expected)

        # Verify call counts (3 frames each)
        assert mock_resize.call_count == 3
        assert mock_color.call_count == 3
        assert mock_spatial.call_count == 3


class TestApplyDepthAugmentations:
    """Test depth image augmentation application."""


    def test_apply_depth_no_transforms(self, sample_depth_images):
        """Test that depth images are unchanged when no transforms configured."""
        pipeline = AugmentationPipeline(train=True)

        result = pipeline.apply_depth_augmentations(sample_depth_images)

        np.testing.assert_array_equal(result, sample_depth_images)


    @patch('refactoring.data.augmentation_pipeline.A.Resize')
    def test_apply_depth_with_resize(self, mock_resize_class, sample_depth_images):
        """Test resize is applied to depth images."""
        mock_resize = MagicMock()
        resized_frame = np.ones((128, 128), dtype=np.float32) * 5.0
        mock_resize.return_value = {'image': resized_frame}
        mock_resize_class.return_value = mock_resize

        pipeline = AugmentationPipeline(
            target_height=128,
            target_width=128,
            train=True
        )

        result = pipeline.apply_depth_augmentations(sample_depth_images)

        assert result.shape == (3, 128, 128)
        np.testing.assert_allclose(result, np.stack([resized_frame] * 3))
        assert mock_resize.call_count == 3


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_apply_depth_with_spatial(self, mock_instantiate, sample_depth_images):
        """Test spatial augmentation is applied to depth."""
        mock_transform = MagicMock()
        modified_frame = sample_depth_images[0] + 0.1
        mock_transform.side_effect = lambda image: {'image': image + 0.1}
        mock_instantiate.return_value = mock_transform

        pipeline = AugmentationPipeline(
            spatial_augmentation=MagicMock(),
            train=True
        )

        result = pipeline.apply_depth_augmentations(sample_depth_images)

        assert result.shape == sample_depth_images.shape
        np.testing.assert_allclose(result[0], modified_frame)
        assert mock_transform.call_count == 3


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_apply_depth_no_color_augmentation(self, mock_instantiate, sample_depth_images):
        """Test that color augmentation is NOT applied to depth."""
        mock_color_transform = MagicMock()
        mock_instantiate.return_value = mock_color_transform

        pipeline = AugmentationPipeline(
            color_augmentation=MagicMock(),
            train=True
        )

        result = pipeline.apply_depth_augmentations(sample_depth_images)

        mock_color_transform.assert_not_called()
        np.testing.assert_array_equal(result, sample_depth_images)


    @patch('refactoring.data.augmentation_pipeline.A.Rotate')
    def test_apply_depth_rotation_uses_nearest_neighbor(self, mock_rotate_class, sample_depth_images):
        """Test depth rotation uses nearest neighbor interpolation."""
        pipeline = AugmentationPipeline(train=True)

        mock_rotate = MagicMock()
        rotated_frame = np.rot90(sample_depth_images[0])
        mock_rotate.return_value = {'image': rotated_frame}
        mock_rotate_class.return_value = mock_rotate

        result = pipeline.apply_depth_augmentations(sample_depth_images, angle=90.0)

        assert result.shape == sample_depth_images.shape
        np.testing.assert_allclose(result[0], rotated_frame)
        mock_rotate_class.assert_called_once_with(limit=(90.0, 90.0), p=1.0, interpolation=0)
        assert mock_rotate.call_count == 3


    def test_apply_depth_skip_rotation_when_zero(self, sample_depth_images):
        """Test rotation is skipped when angle is 0 for depth."""
        pipeline = AugmentationPipeline(train=True)

        with patch('refactoring.data.augmentation_pipeline.A.Rotate') as mock_rotate:
            result = pipeline.apply_depth_augmentations(sample_depth_images, angle=0)

            mock_rotate.assert_not_called()
            np.testing.assert_array_equal(result, sample_depth_images)


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    @patch('refactoring.data.augmentation_pipeline.A.Resize')
    def test_apply_depth_augmentation_order(self, mock_resize_class, mock_instantiate, sample_depth_images):
        """Test augmentations are applied in correct order: resize -> spatial -> rotate."""
        mock_resize = MagicMock()
        mock_resize.side_effect = lambda image: {'image': image + 0.1}  # Simulate resize
        mock_resize_class.return_value = mock_resize

        mock_spatial = MagicMock()
        mock_spatial.side_effect = lambda image: {'image': image - 0.05}  # Simulate spatial

        mock_instantiate.return_value = mock_spatial

        pipeline = AugmentationPipeline(
            spatial_augmentation=MagicMock(),
            target_height=64,  # Same size
            target_width=64,
            train=True
        )

        # Apply without rotation
        result = pipeline.apply_depth_augmentations(sample_depth_images, angle=0)

        # Expected: (original +0.1) -0.05
        expected = (sample_depth_images + 0.1) - 0.05
        np.testing.assert_allclose(result, expected)

        assert mock_resize.call_count == 3
        assert mock_spatial.call_count == 3


class TestRotateProprioData:
    """Test proprioceptive data rotation."""


    def test_rotate_proprio_identity(self, sample_proprio_data):
        """Test rotation with identity matrix."""
        pipeline = AugmentationPipeline(train=True)
        R = np.eye(3)

        result = pipeline.rotate_proprio_data(sample_proprio_data, R)

        np.testing.assert_allclose(result[:, :3], sample_proprio_data[:, :3], rtol=1e-5)
        np.testing.assert_array_equal(result[:, 3:], sample_proprio_data[:, 3:])


    def test_rotate_proprio_90_degrees(self, sample_proprio_data):
        """Test 90-degree rotation around Z-axis."""
        pipeline = AugmentationPipeline(train=True)

        # 90-degree rotation: (action_embedding, y, z) -> (-y, action_embedding, z)
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        original_pos = sample_proprio_data[:, :3].copy()
        result = pipeline.rotate_proprio_data(sample_proprio_data, R)

        expected_pos = original_pos[:, [1, 0, 2]] * np.array([-1, 1, 1])
        np.testing.assert_allclose(result[:, :3], expected_pos, rtol=1e-5)
        np.testing.assert_array_equal(result[:, 3:], sample_proprio_data[:, 3:])


    def test_rotate_proprio_180_degrees(self, sample_proprio_data):
        """Test 180-degree rotation."""
        pipeline = AugmentationPipeline(train=True)

        R = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        original_pos = sample_proprio_data[:, :3].copy()
        result = pipeline.rotate_proprio_data(sample_proprio_data, R)

        expected_pos = original_pos * np.array([-1, -1, 1])
        np.testing.assert_allclose(result[:, :3], expected_pos, rtol=1e-5)


    def test_rotate_proprio_preserves_original(self, sample_proprio_data):
        """Test that rotation doesn't modify original array."""
        pipeline = AugmentationPipeline(train=True)
        original_copy = sample_proprio_data.copy()

        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        pipeline.rotate_proprio_data(sample_proprio_data, R)

        np.testing.assert_array_equal(sample_proprio_data, original_copy)


    @pytest.mark.parametrize("num_timesteps,num_features", [
        (1, 7),
        (5, 3),
        (10, 10),
        (100, 20),
    ])
    def test_rotate_proprio_various_shapes(self, num_timesteps, num_features):
        """Test rotation with various data shapes."""
        pipeline = AugmentationPipeline(train=True)

        proprio_data = np.random.rand(num_timesteps, num_features).astype(np.float32)
        R = np.eye(3)

        result = pipeline.rotate_proprio_data(proprio_data, R)

        assert result.shape == proprio_data.shape
        np.testing.assert_allclose(result[:, :3], proprio_data[:, :3], rtol=1e-5)
        if num_features > 3:
            np.testing.assert_array_equal(result[:, 3:], proprio_data[:, 3:])


    def test_rotate_proprio_less_than_3_features(self, sample_proprio_data):
        """Test with proprio data having less than 3 position features."""
        pipeline = AugmentationPipeline(train=True)
        proprio_data = sample_proprio_data[:, :2]  # Only 2 features

        R = np.eye(3)

        with pytest.raises(ValueError):
            pipeline.rotate_proprio_data(proprio_data, R)


class TestIntegration:
    """Integration tests for complete pipelines."""


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_full_pipeline_workflow_rgb(self, mock_instantiate, sample_rgb_images):
        """Test complete augmentation workflow for RGB."""
        mock_color = MagicMock(side_effect=lambda image: {'image': image * 1.1})
        mock_spatial = MagicMock(side_effect=lambda image: {'image': image - 0.05})
        mock_rotation = MagicMock(p=1.0, limit=[-30, 30])
        mock_instantiate.side_effect = [mock_color, mock_spatial, mock_rotation]

        pipeline = AugmentationPipeline(
            color_augmentation=MagicMock(),
            spatial_augmentation=MagicMock(),
            rotation_augmentation=MagicMock(),
            train=True
        )

        with patch('random.random', return_value=0.0), \
                patch('random.uniform', return_value=30.0):
            angle, R = pipeline.setup_rotation()

        assert angle == 30.0
        assert R is not None

        with patch('refactoring.data.augmentation_pipeline.A.Rotate') as mock_rotate:
            mock_rotate.return_value.side_effect = lambda image: {'image': image + 0.01}
            result = pipeline.apply_rgb_augmentations(sample_rgb_images, angle=angle)

        expected = (((sample_rgb_images * 1.1) - 0.05) + 0.01)
        np.testing.assert_allclose(result, expected)
        assert mock_color.call_count == 3
        assert mock_spatial.call_count == 3
        mock_rotate.assert_called_once_with(limit=(30.0, 30.0), p=1.0, interpolation=1)


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_full_pipeline_workflow_depth(self, mock_instantiate, sample_depth_images):
        """Test complete augmentation workflow for depth."""
        mock_spatial = MagicMock(side_effect=lambda image: {'image': image - 0.05})
        mock_rotation = MagicMock(p=1.0, limit=[-30, 30])
        mock_instantiate.side_effect = [mock_spatial, mock_rotation]

        pipeline = AugmentationPipeline(
            spatial_augmentation=MagicMock(),
            rotation_augmentation=MagicMock(),
            train=True
        )

        with patch('random.random', return_value=0.0), \
                patch('random.uniform', return_value=30.0):
            angle, R = pipeline.setup_rotation()

        with patch('refactoring.data.augmentation_pipeline.A.Rotate') as mock_rotate:
            mock_rotate.return_value.side_effect = lambda image: {'image': image + 0.01}
            result = pipeline.apply_depth_augmentations(sample_depth_images, angle=angle)

        expected = ((sample_depth_images - 0.05) + 0.01)
        np.testing.assert_allclose(result, expected)
        assert mock_spatial.call_count == 3
        mock_rotate.assert_called_once_with(limit=(30.0, 30.0), p=1.0, interpolation=0)


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_eval_mode_disables_training_augmentations(self, mock_instantiate):
        """Test that eval mode disables training-only augmentations."""
        pipeline = AugmentationPipeline(
            color_augmentation=MagicMock(),
            spatial_augmentation=MagicMock(),
            rotation_augmentation=MagicMock(),
            train=False
        )

        assert not pipeline.use_color
        assert not pipeline.use_spatial
        assert not pipeline.use_rotation

        mock_instantiate.assert_not_called()


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_rotation_matrix_deterministic(self, mock_instantiate):
        """Test rotation matrix computation is deterministic."""
        mock_transform = MagicMock()
        mock_transform.p = 1.0
        mock_transform.limit = [0, 90]
        mock_instantiate.return_value = mock_transform

        pipeline = AugmentationPipeline(
            rotation_augmentation=MagicMock(),
            train=True
        )

        angle = 45.0
        theta_rad = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

        expected_R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ])

        with patch('random.uniform', return_value=angle):
            _, R = pipeline.setup_rotation()

        np.testing.assert_allclose(R, expected_R, rtol=1e-5)


    @patch('refactoring.data.augmentation_pipeline.instantiate')
    def test_full_pipeline_with_resize_and_rotation(self, mock_instantiate, sample_rgb_images, sample_proprio_data):
        """Test integration with resize, rotation, and proprio."""
        mock_rotation = MagicMock(p=1.0, limit=[-30, 30])
        mock_instantiate.return_value = mock_rotation

        pipeline = AugmentationPipeline(
            rotation_augmentation=MagicMock(),
            target_height=64,
            target_width=64,
            train=True
        )

        with patch('random.uniform', return_value=90.0):
            angle, R = pipeline.setup_rotation()

        rgb_result = pipeline.apply_rgb_augmentations(sample_rgb_images, angle)
        proprio_result = pipeline.rotate_proprio_data(sample_proprio_data, R)

        # Since resize is identity (same size), rgb unchanged before rotation
        # But rotation at 90 degrees
        expected_rgb = np.stack([np.rot90(frame, k=1) for frame in sample_rgb_images])
        np.testing.assert_allclose(rgb_result, expected_rgb, rtol=1e-5)

        expected_pos = sample_proprio_data[:, :3] @ R.T  # Equivalent to (R @ pos.T).T
        np.testing.assert_allclose(proprio_result[:, :3], expected_pos, rtol=1e-5)

@pytest.mark.integration
class TestRealHydraConfigIntegration:
    """Integration tests with real Hydra configs (no mocking)."""

    def test_real_color_augmentation_pipeline(self, sample_rgb_images):
        """Test with real ColorAugmentationPipeline config."""
        config = OmegaConf.create({
            '_target_': 'albumentations.Compose',
            'transforms': [
                {'_target_': 'albumentations.ColorJitter', 'brightness': 0.3, 'contrast': 0.4, 'saturation': 0.5, 'hue': 0.1, 'p': 0.5},
                {'_target_': 'albumentations.RandomBrightnessContrast', 'brightness_limit': 0.4, 'contrast_limit': 0.4, 'p': 0.6},
            ]
        })

        # This should not raise and should create a callable pipeline
        pipeline = AugmentationPipeline(
            color_augmentation=config,
            train=True
        )

        assert pipeline.photometric_transform is not None
        assert callable(pipeline.photometric_transform)

        # Apply augmentations
        result = pipeline.apply_rgb_augmentations(sample_rgb_images, angle=0)
        assert result.shape == sample_rgb_images.shape

    def test_real_spatial_augmentation_pipeline(self, sample_rgb_images):
        """Test with real SpatialAugmentationPipeline config."""
        config = OmegaConf.create({
            '_target_': 'albumentations.Compose',
            'transforms': [
                {'_target_': 'albumentations.GaussianBlur', 'blur_limit': (3, 7), 'p': 0.5},
                {'_target_': 'albumentations.CoarseDropout', 'max_holes': 8, 'max_height': 8, 'max_width': 8, 'p': 0.3},
            ]
        })

        pipeline = AugmentationPipeline(
            spatial_augmentation=config,
            train=True
        )

        assert pipeline.spatial_transform is not None
        assert callable(pipeline.spatial_transform)

        # Apply augmentations
        result = pipeline.apply_rgb_augmentations(sample_rgb_images, angle=0)
        assert result.shape == sample_rgb_images.shape

    def test_real_both_augmentation_pipelines(self, sample_rgb_images):
        """Test with both color and spatial real configs."""
        color_config = OmegaConf.create({
            '_target_': 'albumentations.Compose',
            'transforms': [
                {'_target_': 'albumentations.ColorJitter', 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1, 'p': 1.0},
            ]
        })

        spatial_config = OmegaConf.create({
            '_target_': 'albumentations.Compose',
            'transforms': [
                {'_target_': 'albumentations.GaussianBlur', 'blur_limit': (3, 5), 'p': 1.0},
            ]
        })

        pipeline = AugmentationPipeline(
            color_augmentation=color_config,
            spatial_augmentation=spatial_config,
            train=True
        )

        assert callable(pipeline.photometric_transform)
        assert callable(pipeline.spatial_transform)

        # Apply augmentations - should work without errors
        result = pipeline.apply_rgb_augmentations(sample_rgb_images, angle=0)
        assert result.shape == sample_rgb_images.shape
        # Result should be different due to augmentation
        assert not np.allclose(result, sample_rgb_images)
