"""Tests for versatil.data.processing.transform_builder module."""

import logging
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import zarr
import zarr.storage

from versatil.data.constants import (
    CLIP_RGB_MEAN,
    ActionDiscretizerType,
    ActionTokenIdMappingType,
    BinningStrategy,
    Cameras,
    ImageNormalizationType,
    KinematicsNormalizationType,
)
from versatil.data.metadata import (
    CameraMetadata,
    ObservationMetadata,
    OnTheFlyActionMetadata,
    PositionObservationMetadata,
    PrecomputedActionMetadata,
)
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.processing.transform_builder import (
    TransformBuilder,
    _build_action_discretizer,
    _build_token_id_mapping,
)
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization.action_discretizer import BinnedActionDiscretizer
from versatil.data.tokenization.action_token_id_mapping import (
    IdentityActionTokenIdMapping,
)


def _numpy_to_zarr_array(array: np.ndarray) -> zarr.Array:
    """Wrap a numpy array into an in-memory zarr array."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store=store, mode="w")
    return root.create_array("arr", data=array, chunks=array.shape)


@pytest.fixture
def mock_replay_buffer() -> Callable[..., MagicMock]:
    """Factory for mock ReplayBuffer.

    When use_zarr=True, stored arrays are wrapped as zarr.Array (disk
    backend). When False, raw numpy arrays are returned (preloaded /
    numpy backend).
    """

    def factory(
        data: dict[str, np.ndarray] = None,
        n_steps: int = 100,
        use_zarr: bool = False,
    ) -> MagicMock:
        buffer = MagicMock()
        buffer.n_steps = n_steps
        raw_data = data or {}

        if use_zarr:
            stored_data = {
                key: _numpy_to_zarr_array(value) for key, value in raw_data.items()
            }
        else:
            stored_data = raw_data

        def getitem(mock_self, key):
            if key in stored_data:
                return stored_data[key]
            raise KeyError(key)

        def contains(mock_self, key):
            return key in stored_data

        buffer.__getitem__ = getitem
        buffer.__contains__ = contains
        return buffer

    return factory


@pytest.fixture
def transform_builder_factory(
    action_space_factory: Callable[..., ActionSpace],
    observation_space_factory: Callable[..., ObservationSpace],
    mock_replay_buffer: Callable[..., MagicMock],
) -> Callable[..., TransformBuilder]:
    """Factory for creating TransformBuilder instances with mocked dependencies."""

    def factory(
        actions_metadata: dict = None,
        observations_metadata: dict = None,
        replay_buffer_data: dict[str, np.ndarray] = None,
        n_steps: int = 100,
        episode_ends: np.ndarray = None,
        kinematics_norm_type: str = KinematicsNormalizationType.MIN_MAX.value,
        image_norm_type: str = ImageNormalizationType.ZERO_TO_ONE.value,
        depth_norm_type: str = ImageNormalizationType.ZERO_TO_ONE.value,
        prediction_horizon: int = 4,
        denoise_actions: bool = False,
        tokenization_config: MagicMock = None,
        use_zarr: bool = False,
        action_sample_size: int = 2048,
        episode_selection_mask: np.ndarray | None = None,
    ) -> TransformBuilder:
        action_space = action_space_factory(
            actions_metadata=actions_metadata or {},
            denoise_actions=denoise_actions,
        )
        observation_space = observation_space_factory(
            observations_metadata=observations_metadata or {},
        )
        buffer = mock_replay_buffer(
            data=replay_buffer_data or {},
            n_steps=n_steps,
            use_zarr=use_zarr,
        )
        mock_action_processor = MagicMock()
        mock_action_processor.action_space = action_space
        mock_action_processor.denoise_actions = denoise_actions

        if episode_ends is None:
            episode_ends = np.array([n_steps])

        return TransformBuilder(
            replay_buffer=buffer,
            action_processor=mock_action_processor,
            prediction_horizon=prediction_horizon,
            observation_space=observation_space,
            episode_ends=episode_ends,
            kinematics_norm_type=kinematics_norm_type,
            image_norm_type=image_norm_type,
            depth_norm_type=depth_norm_type,
            tokenization_config=tokenization_config,
            action_sample_size=action_sample_size,
            episode_selection_mask=episode_selection_mask,
        )

    return factory


class TestTransformBuilderInitialization:
    @pytest.mark.parametrize(
        "kinematics_norm_type", [member.value for member in KinematicsNormalizationType]
    )
    def test_stores_kinematics_norm_type(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        kinematics_norm_type: str,
    ):
        builder = transform_builder_factory(
            kinematics_norm_type=kinematics_norm_type,
        )
        assert builder.kinematics_norm_type == kinematics_norm_type

    @pytest.mark.parametrize(
        "image_norm_type", [member.value for member in ImageNormalizationType]
    )
    def test_stores_image_norm_type(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        image_norm_type: str,
    ):
        builder = transform_builder_factory(image_norm_type=image_norm_type)
        assert builder.image_norm_type == image_norm_type

    @pytest.mark.parametrize("prediction_horizon", [4, 8, 16])
    def test_stores_prediction_horizon(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        prediction_horizon: int,
    ):
        builder = transform_builder_factory(prediction_horizon=prediction_horizon)
        assert builder.prediction_horizon == prediction_horizon

    @pytest.mark.parametrize("action_sample_size", [0, 128, 2048])
    def test_stores_action_sample_size(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        action_sample_size: int,
    ):
        builder = transform_builder_factory(action_sample_size=action_sample_size)
        assert builder.action_sample_size == action_sample_size


class TestApplyWinsorization:
    @pytest.mark.parametrize(
        "quantiles",
        [(0.01, 0.99), (0.05, 0.95), (0.1, 0.9)],
    )
    def test_clips_outliers_to_quantile_bounds(
        self, rng: np.random.Generator, quantiles: tuple[float, float]
    ):
        data = rng.standard_normal((100, 3)).astype(np.float32)
        # Insert extreme outliers
        data[0] = [100.0, 100.0, 100.0]
        data[1] = [-100.0, -100.0, -100.0]

        result = TransformBuilder._apply_winsorization(
            data_dict={"position": data},
            quantiles=quantiles,
        )

        assert result["position"].max() < 100.0
        assert result["position"].min() > -100.0

    def test_preserves_data_within_quantile_bounds(self, rng: np.random.Generator):
        data = rng.uniform(0.4, 0.6, (50, 2)).astype(np.float32)

        result = TransformBuilder._apply_winsorization(
            data_dict={"position": data},
            quantiles=(0.0, 1.0),
        )

        # With quantiles (0.0, 1.0), no clipping should occur
        np.testing.assert_array_equal(result["position"], data)

    def test_logs_when_values_clipped(
        self,
        rng: np.random.Generator,
        caplog: pytest.LogCaptureFixture,
    ):
        data = rng.standard_normal((100, 1)).astype(np.float32)
        data[0] = 1000.0

        with caplog.at_level(logging.INFO):
            TransformBuilder._apply_winsorization(
                data_dict={"position": data},
                quantiles=(0.01, 0.99),
            )

        assert "Winsorized" in caplog.text
        assert "clipped" in caplog.text


class TestComputeProprioceptiveDenosingThresholds:
    def test_computes_thresholds_for_on_the_fly_position_actions(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        rng: np.random.Generator,
    ):
        position_source = position_observation_metadata_factory(dimension=3)
        position_data = rng.standard_normal((100, 3)).astype(np.float32)

        builder = transform_builder_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_source,
                ),
            },
            replay_buffer_data={"position": position_data},
            denoise_actions=True,
        )

        builder.compute_proprioceptive_denoising_thresholds()

        builder.action_processor.compute_denoising_threshold.assert_called_once()
        builder.action_processor.log_movement_distribution.assert_called_once()


class TestCreateActionChunksForTokenizer:
    @pytest.mark.parametrize(
        "prediction_horizon, episode_ends, action_dim, expected_chunks",
        [
            (4, np.array([10, 20]), 3, 12),
            (2, np.array([10, 20]), 5, 16),
            (8, np.array([15]), 2, 7),
        ],
    )
    def test_creates_correct_shape_chunks(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        rng: np.random.Generator,
        prediction_horizon: int,
        episode_ends: np.ndarray,
        action_dim: int,
        expected_chunks: int,
    ):
        builder = transform_builder_factory(
            prediction_horizon=prediction_horizon,
            episode_ends=episode_ends,
        )
        total_valid_actions = sum(
            (episode_ends[i] - (episode_ends[i - 1] if i > 0 else 0) - 1)
            for i in range(len(episode_ends))
        )
        action_data = {
            "position": rng.standard_normal((total_valid_actions, action_dim)).astype(
                np.float32
            ),
        }

        chunks = builder._create_action_chunks_for_tokenizer(
            action_dict=action_data,
        )

        assert chunks.shape == (expected_chunks, prediction_horizon, action_dim)

    def test_skips_episodes_shorter_than_prediction_horizon(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        rng: np.random.Generator,
    ):
        # First episode: 3 steps → 2 valid actions (< pred_horizon=4), skip
        # Second episode: 10 steps → 9 valid actions
        builder = transform_builder_factory(
            prediction_horizon=4,
            episode_ends=np.array([3, 13]),
        )

        action_data = {
            "position": rng.standard_normal((11, 2)).astype(np.float32),
        }

        chunks = builder._create_action_chunks_for_tokenizer(
            action_dict=action_data,
        )

        # Only second episode: 9 actions, pred_horizon=4 → 6 chunks
        assert chunks.shape == (6, 4, 2)

    def test_concatenates_multiple_action_keys_in_metadata_order(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        rng: np.random.Generator,
    ):
        builder = transform_builder_factory(
            prediction_horizon=2,
            episode_ends=np.array([5]),
        )

        action_data = {
            "position": rng.standard_normal((4, 3)).astype(np.float32),
            "gripper": rng.standard_normal((4, 1)).astype(np.float32),
        }

        chunks = builder._create_action_chunks_for_tokenizer(
            action_dict=action_data,
        )

        # Sorted: "gripper" (1) + "position" (3) = 4 dims
        assert chunks.shape[2] == 4


class TestSetupImageNormalizers:
    def test_sets_up_rgb_normalizer_for_non_depth_camera(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
    ):
        builder = transform_builder_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value,
                ),
            },
            replay_buffer_data={
                Cameras.LEFT.value: np.zeros((10, 4, 4, 3), dtype=np.uint8),
            },
        )
        normalizer = MagicMock()

        builder._setup_image_normalizers(
            normalizer=normalizer,
            device=None,
            winsorize_depth=False,
        )

        normalizer.__setitem__.assert_called_once()
        call_key = normalizer.__setitem__.call_args[0][0]
        assert call_key == Cameras.LEFT.value

    @pytest.mark.parametrize(
        ("image_norm_type", "mean"),
        [
            (ImageNormalizationType.CLIP.value, torch.tensor(CLIP_RGB_MEAN)),
        ],
    )
    def test_sets_up_pretrained_rgb_normalizers(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        image_norm_type: str,
        mean: torch.Tensor,
    ):
        builder = transform_builder_factory(image_norm_type=image_norm_type)
        normalizer = LinearNormalizer()

        builder._setup_rgb_normalizer(
            normalizer=normalizer,
            cam=Cameras.LEFT.value,
            device=None,
        )

        normalized = normalizer.normalize({Cameras.LEFT.value: mean})[
            Cameras.LEFT.value
        ]
        torch.testing.assert_close(
            normalized,
            torch.zeros_like(mean),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_sets_up_depth_normalizer_for_depth_camera(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        rng: np.random.Generator,
    ):
        depth_data = rng.uniform(0.5, 5.0, (10, 4, 4)).astype(np.float32)
        builder = transform_builder_factory(
            observations_metadata={
                Cameras.DEPTH.value: camera_metadata_factory(
                    camera_key=Cameras.DEPTH.value,
                    channels=1,
                ),
            },
            replay_buffer_data={Cameras.DEPTH.value: depth_data},
        )
        normalizer = MagicMock()

        builder._setup_image_normalizers(
            normalizer=normalizer,
            device=None,
            winsorize_depth=False,
        )

        normalizer.__setitem__.assert_called_once()
        call_key = normalizer.__setitem__.call_args[0][0]
        assert call_key == Cameras.DEPTH.value


class TestComputeDepthStatsStreaming:
    def _make_depth_builder(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        depth_data: np.ndarray,
        use_zarr: bool = False,
    ) -> TransformBuilder:
        return transform_builder_factory(
            observations_metadata={
                Cameras.DEPTH.value: camera_metadata_factory(
                    camera_key=Cameras.DEPTH.value,
                    channels=1,
                ),
            },
            replay_buffer_data={Cameras.DEPTH.value: depth_data},
            use_zarr=use_zarr,
        )

    @pytest.mark.parametrize("use_zarr", [False, True], ids=["numpy", "zarr"])
    def test_computes_correct_stats_for_uniform_data(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        use_zarr: bool,
    ):
        depth_data = np.ones((10, 4, 4), dtype=np.float32) * 3.0
        builder = self._make_depth_builder(
            transform_builder_factory,
            camera_metadata_factory,
            depth_data,
            use_zarr=use_zarr,
        )

        stats = builder._compute_depth_stats_streaming(
            camera_key=Cameras.DEPTH.value,
            winsorize=False,
        )

        assert stats["min"] == pytest.approx(3.0)
        assert stats["max"] == pytest.approx(3.0)
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["std"] == pytest.approx(0.0)

    @pytest.mark.parametrize("use_zarr", [False, True], ids=["numpy", "zarr"])
    def test_computes_correct_stats_across_multiple_chunks(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        use_zarr: bool,
    ):
        # 4 frames of 2x2 pixels: values 1..16
        # chunk_size=2 → two chunks, exercises Welford merge
        depth_data = np.arange(1, 17, dtype=np.float32).reshape(4, 2, 2)
        builder = self._make_depth_builder(
            transform_builder_factory,
            camera_metadata_factory,
            depth_data,
            use_zarr=use_zarr,
        )

        stats = builder._compute_depth_stats_streaming(
            camera_key=Cameras.DEPTH.value,
            winsorize=False,
            chunk_size=2,
        )

        # Values 1..16: min=1, max=16, mean=8.5, std=sqrt(Var) where Var = E[X²]-E[X]²
        # E[X] = 8.5, E[X²] = (1²+2²+...+16²)/16 = 1496/16 = 93.5, Var = 93.5 - 72.25 = 21.25
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(16.0)
        assert stats["mean"] == pytest.approx(8.5, abs=1e-4)
        assert stats["std"] == pytest.approx(np.sqrt(21.25), abs=1e-3)

    @pytest.mark.parametrize("use_zarr", [False, True], ids=["numpy", "zarr"])
    def test_winsorization_clips_outliers_in_stats(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        use_zarr: bool,
    ):
        # 100 pixels of value 1.0, with two outliers
        depth_data = np.ones((25, 2, 2), dtype=np.float32)
        depth_data[0, 0, 0] = 100.0
        depth_data[0, 0, 1] = -50.0

        builder = self._make_depth_builder(
            transform_builder_factory,
            camera_metadata_factory,
            depth_data,
            use_zarr=use_zarr,
        )

        stats = builder._compute_depth_stats_streaming(
            camera_key=Cameras.DEPTH.value,
            winsorize=True,
        )

        # Outliers clipped → min and max should be close to 1.0
        assert stats["min"] > -50.0
        assert stats["max"] < 100.0

    @pytest.mark.parametrize("use_zarr", [False, True], ids=["numpy", "zarr"])
    def test_returns_nan_for_empty_array(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        use_zarr: bool,
    ):
        depth_data = np.empty((0, 4, 4), dtype=np.float32)
        builder = self._make_depth_builder(
            transform_builder_factory,
            camera_metadata_factory,
            depth_data,
            use_zarr=use_zarr,
        )

        stats = builder._compute_depth_stats_streaming(
            camera_key=Cameras.DEPTH.value,
            winsorize=False,
        )

        assert np.isnan(stats["min"])
        assert np.isnan(stats["max"])

    @pytest.mark.parametrize("use_zarr", [False, True], ids=["numpy", "zarr"])
    @pytest.mark.parametrize("chunk_size", [2, 3, 7, 15])
    def test_chunked_processing_produces_same_result_as_single_chunk(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        rng: np.random.Generator,
        chunk_size: int,
        use_zarr: bool,
    ):
        depth_data = rng.uniform(0.5, 5.0, (30, 4, 4)).astype(np.float32)
        builder = self._make_depth_builder(
            transform_builder_factory,
            camera_metadata_factory,
            depth_data,
            use_zarr=use_zarr,
        )

        # Large chunk (single pass)
        stats_single = builder._compute_depth_stats_streaming(
            camera_key=Cameras.DEPTH.value,
            winsorize=False,
            chunk_size=10000,
        )
        # Small chunks (multiple passes through Welford)
        stats_chunked = builder._compute_depth_stats_streaming(
            camera_key=Cameras.DEPTH.value,
            winsorize=False,
            chunk_size=chunk_size,
        )

        assert stats_single["min"] == pytest.approx(stats_chunked["min"], abs=1e-5)
        assert stats_single["max"] == pytest.approx(stats_chunked["max"], abs=1e-5)
        assert stats_single["mean"] == pytest.approx(stats_chunked["mean"], abs=1e-4)
        assert stats_single["std"] == pytest.approx(stats_chunked["std"], abs=1e-4)

    def test_winsorized_stats_reproducible_across_global_rng_state(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        rng: np.random.Generator,
    ):
        # Large enough that the quantile reservoir must subsample pixels.
        depth_data = rng.uniform(0.5, 5.0, (300, 20, 20)).astype(np.float32)

        def compute_stats() -> dict[str, float]:
            builder = self._make_depth_builder(
                transform_builder_factory,
                camera_metadata_factory,
                depth_data,
            )
            return builder._compute_depth_stats_streaming(
                camera_key=Cameras.DEPTH.value,
                winsorize=True,
            )

        np.random.seed(0)
        first = compute_stats()
        np.random.seed(1)
        np.random.random(999)
        second = compute_stats()

        assert first == second


class TestLogCameraStatsSampled:
    def test_logs_stats_for_rgb_camera(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        caplog: pytest.LogCaptureFixture,
    ):
        rgb_data = np.ones((10, 4, 4, 3), dtype=np.uint8) * 128
        builder = transform_builder_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value,
                ),
            },
            replay_buffer_data={Cameras.LEFT.value: rgb_data},
        )
        normalizer = MagicMock()
        mock_single = MagicMock()
        mock_single.normalize.return_value = torch.zeros(5, 4, 4, 3)
        normalizer.__getitem__ = lambda self, key: mock_single

        with caplog.at_level(logging.INFO):
            builder._log_camera_stats_sampled(
                camera_key=Cameras.LEFT.value,
                normalizer=normalizer,
            )

        assert "Camera left stats" in caplog.text
        assert "after normalization" in caplog.text

    def test_handles_empty_camera_array(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        caplog: pytest.LogCaptureFixture,
    ):
        empty_data = np.empty((0, 4, 4, 3), dtype=np.uint8)
        builder = transform_builder_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value,
                ),
            },
            replay_buffer_data={Cameras.LEFT.value: empty_data},
        )
        normalizer = MagicMock()

        with caplog.at_level(logging.INFO):
            builder._log_camera_stats_sampled(
                camera_key=Cameras.LEFT.value,
                normalizer=normalizer,
            )

        assert "empty array" in caplog.text


class TestLogNormalizedProprioStats:
    def test_logs_before_and_after_normalization_stats(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        caplog: pytest.LogCaptureFixture,
        rng: np.random.Generator,
    ):
        builder = transform_builder_factory()

        normalizer = MagicMock()
        normalizer.params_dict = MagicMock()
        normalizer.params_dict.keys.return_value = ["position"]
        mock_single = MagicMock()
        mock_single.get_input_stats.return_value = {
            "min": torch.tensor([0.0]),
            "max": torch.tensor([1.0]),
            "mean": torch.tensor([0.5]),
            "std": torch.tensor([0.3]),
        }
        mock_single.normalize.return_value = torch.tensor([0.0])
        normalizer.__getitem__ = lambda self, key: mock_single

        proprio_data = {"position": rng.standard_normal((50, 1)).astype(np.float32)}

        with caplog.at_level(logging.INFO):
            builder._log_normalized_proprio_stats(
                normalizer=normalizer,
                proprio_data=proprio_data,
            )

        assert "before normalization" in caplog.text
        assert "after normalization" in caplog.text


class TestCreateNormalizer:
    def test_fits_normalizer_with_observation_and_action_data(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        rng: np.random.Generator,
    ):
        position_source = position_observation_metadata_factory(dimension=3)
        position_data = rng.standard_normal((100, 3)).astype(np.float32)

        builder = transform_builder_factory(
            observations_metadata={"position": position_source},
            replay_buffer_data={"position": position_data},
        )

        action_meta = {
            "position": on_the_fly_action_metadata_factory(
                source_metadata=position_source,
            )
        }
        action_data = {"position": rng.standard_normal((99, 3)).astype(np.float32)}

        mock_normalizer = MagicMock()
        with (
            patch(
                "versatil.data.processing.transform_builder.LinearNormalizer",
                return_value=mock_normalizer,
            ),
            patch.object(builder, "_setup_image_normalizers") as mock_setup_images,
            patch.object(builder, "_log_normalized_proprio_stats") as mock_log_stats,
        ):
            normalizer = builder._create_normalizer(
                action_data=action_data,
                action_meta=action_meta,
            )

        mock_normalizer.fit.assert_called_once()
        mock_setup_images.assert_called_once()
        mock_log_stats.assert_called_once()
        assert normalizer is mock_normalizer

    def test_skips_camera_keys_in_kinematics_fit_data(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        rng: np.random.Generator,
    ):
        position_data = rng.standard_normal((100, 3)).astype(np.float32)

        builder = transform_builder_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value,
                ),
                "position": position_observation_metadata_factory(dimension=3),
            },
            replay_buffer_data={"position": position_data},
        )

        mock_normalizer = MagicMock()
        with (
            patch(
                "versatil.data.processing.transform_builder.LinearNormalizer",
                return_value=mock_normalizer,
            ),
            patch.object(builder, "_setup_image_normalizers"),
            patch.object(builder, "_log_normalized_proprio_stats"),
        ):
            builder._create_normalizer(action_data={}, action_meta={})

        fit_call_data = mock_normalizer.fit.call_args[1]["data"]
        assert "position" in fit_call_data
        assert Cameras.LEFT.value not in fit_call_data

    def test_passes_action_sample_size_only_for_action_keys(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        rng: np.random.Generator,
    ):
        position_source = position_observation_metadata_factory(dimension=3)
        position_data = rng.standard_normal((100, 3)).astype(np.float32)

        builder = transform_builder_factory(
            observations_metadata={"position": position_source},
            replay_buffer_data={"position": position_data},
            action_sample_size=256,
        )

        action_meta = {
            "position_action": on_the_fly_action_metadata_factory(
                source_metadata=position_source,
            )
        }
        action_data = {
            "position_action": rng.standard_normal((99, 3)).astype(np.float32)
        }

        mock_normalizer = MagicMock()
        with (
            patch(
                "versatil.data.processing.transform_builder.LinearNormalizer",
                return_value=mock_normalizer,
            ),
            patch.object(builder, "_setup_image_normalizers"),
            patch.object(builder, "_log_normalized_proprio_stats"),
        ):
            builder._create_normalizer(action_data=action_data, action_meta=action_meta)

        passed_sample_size = mock_normalizer.fit.call_args[1]["sample_size"]
        assert passed_sample_size == {"position_action": 256}

    def test_action_sample_size_zero_disables_sample_storage(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        rng: np.random.Generator,
    ):
        position_source = position_observation_metadata_factory(dimension=3)
        builder = transform_builder_factory(
            observations_metadata={"position": position_source},
            replay_buffer_data={
                "position": rng.standard_normal((100, 3)).astype(np.float32)
            },
            action_sample_size=0,
        )
        action_meta = {
            "position_action": on_the_fly_action_metadata_factory(
                source_metadata=position_source,
            )
        }
        action_data = {
            "position_action": rng.standard_normal((99, 3)).astype(np.float32)
        }
        mock_normalizer = MagicMock()
        with (
            patch(
                "versatil.data.processing.transform_builder.LinearNormalizer",
                return_value=mock_normalizer,
            ),
            patch.object(builder, "_setup_image_normalizers"),
            patch.object(builder, "_log_normalized_proprio_stats"),
        ):
            builder._create_normalizer(action_data=action_data, action_meta=action_meta)
        assert mock_normalizer.fit.call_args[1]["sample_size"] == 0

    def test_raises_for_non_numerical_observation_needing_normalization(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
    ):
        non_numerical = ObservationMetadata(
            raw_data_column_keys=["label"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        # Override after construction to trigger the error path
        non_numerical.needs_normalization = True

        builder = transform_builder_factory(
            observations_metadata={"label": non_numerical},
            replay_buffer_data={"label": np.array([["a"], ["b"]])},
        )

        with pytest.raises(ValueError, match="Cannot normalize non-numerical"):
            builder._create_normalizer(action_data={}, action_meta={})

    def test_skips_observations_not_needing_normalization(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        rng: np.random.Generator,
    ):
        no_norm_position = position_observation_metadata_factory(
            dimension=3,
            needs_normalization=False,
        )
        builder = transform_builder_factory(
            observations_metadata={"position": no_norm_position},
            replay_buffer_data={
                "position": rng.standard_normal((100, 3)).astype(np.float32)
            },
        )
        mock_normalizer = MagicMock()
        with (
            patch(
                "versatil.data.processing.transform_builder.LinearNormalizer",
                return_value=mock_normalizer,
            ),
            patch.object(builder, "_setup_image_normalizers"),
            patch.object(builder, "_log_normalized_proprio_stats"),
        ):
            builder._create_normalizer(action_data={}, action_meta={})

        fit_call_data = mock_normalizer.fit.call_args[1]["data"]
        assert len(fit_call_data) == 0


class TestCreateNormalizerAndTokenizer:
    def test_returns_normalizer_and_none_tokenizer_when_no_config(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        rng: np.random.Generator,
    ):
        position_source = position_observation_metadata_factory(dimension=3)
        position_data = rng.standard_normal((100, 3)).astype(np.float32)

        builder = transform_builder_factory(
            observations_metadata={"position": position_source},
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_source,
                ),
            },
            replay_buffer_data={"position": position_data},
            n_steps=100,
            episode_ends=np.array([100]),
            tokenization_config=None,
        )
        # Mock action processor to return valid data
        builder.action_processor.compute_sample_actions.return_value = (
            {"position": rng.standard_normal((99, 3)).astype(np.float32)},
            {
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_source,
                )
            },
        )

        normalizer, tokenizer = builder.create_normalizer_and_tokenizer()

        assert isinstance(normalizer, LinearNormalizer)
        assert tokenizer is None

    def test_masks_cross_episode_transitions_in_action_data(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        rng: np.random.Generator,
    ):
        position_source = position_observation_metadata_factory(dimension=1)
        # Two episodes of 5 steps each
        position_data = rng.standard_normal((10, 1)).astype(np.float32)

        builder = transform_builder_factory(
            observations_metadata={"position": position_source},
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_source,
                ),
            },
            replay_buffer_data={"position": position_data},
            n_steps=10,
            episode_ends=np.array([5, 10]),
        )
        # 9 actions (n_steps - 1), cross-episode at index 4 should be masked
        all_actions = rng.standard_normal((9, 1)).astype(np.float32)
        all_actions[4] = 999.0  # Cross-episode transition — should be masked

        builder.action_processor.compute_sample_actions.return_value = (
            {"position": all_actions},
            {
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_source,
                )
            },
        )

        normalizer, _ = builder.create_normalizer_and_tokenizer()

        # The normalizer should be fitted without the 999.0 outlier
        # If masking works, the fitted range won't include 999.0
        stats = normalizer["position"].get_input_stats()
        assert stats["max"].item() < 900.0

    def test_precomputed_action_episode_final_rows_included_in_stats(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        rng: np.random.Generator,
    ):
        action_metadata = precomputed_action_metadata_factory(
            storage_dimension=1,
            prediction_dimension=1,
        )
        precomputed_actions = rng.uniform(-0.5, 0.5, (10, 1)).astype(np.float32)
        # The first episode's final row holds the dataset maximum. It is a
        # valid precomputed training target and must be covered by the stats.
        precomputed_actions[4] = 999.0

        builder = transform_builder_factory(
            actions_metadata={"action": action_metadata},
            replay_buffer_data={"action": precomputed_actions},
            n_steps=10,
            episode_ends=np.array([5, 10]),
        )
        builder.action_processor.compute_sample_actions.return_value = (
            {"action": precomputed_actions[:9]},
            {"action": action_metadata},
        )

        normalizer, _ = builder.create_normalizer_and_tokenizer()

        # Winsorization clips the extreme, but the fitted max must still be
        # far above the sub-unit bulk of the data.
        stats = normalizer["action"].get_input_stats()
        assert stats["max"].item() > 900.0

    def test_unselected_episodes_are_excluded_from_fitted_statistics(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        rng: np.random.Generator,
    ):
        position_source = position_observation_metadata_factory(dimension=1)
        position_data = rng.standard_normal((10, 1)).astype(np.float32)

        builder = transform_builder_factory(
            observations_metadata={"position": position_source},
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_source,
                ),
            },
            replay_buffer_data={"position": position_data},
            n_steps=10,
            episode_ends=np.array([5, 10]),
            # Second episode is the validation split: its actions must not
            # leak into the fitted normalizer statistics.
            episode_selection_mask=np.array([True, False]),
        )
        all_actions = rng.standard_normal((9, 1)).astype(np.float32)
        all_actions[6] = 999.0  # Inside the unselected (validation) episode

        builder.action_processor.compute_sample_actions.return_value = (
            {"position": all_actions},
            {
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_source,
                )
            },
        )

        normalizer, _ = builder.create_normalizer_and_tokenizer()

        stats = normalizer["position"].get_input_stats()
        assert stats["max"].item() < 900.0


class TestCreateTokenizer:
    def test_raises_when_observation_tokenizer_config_missing(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
    ):
        mock_config = MagicMock()
        mock_config.tokenize_observations = True
        mock_config.observation_tokenizer = None
        mock_config.tokenize_actions = False

        builder = transform_builder_factory(tokenization_config=mock_config)

        with pytest.raises(
            ValueError, match="observation_tokenizer config must be provided"
        ):
            builder._create_tokenizer(
                normalizer=MagicMock(),
                action_data={},
                action_meta={},
            )

    def test_raises_when_action_tokenizer_config_missing(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
    ):
        mock_config = MagicMock()
        mock_config.tokenize_observations = False
        mock_config.tokenize_actions = True
        mock_config.action_tokenizer = None

        builder = transform_builder_factory(tokenization_config=mock_config)

        with pytest.raises(
            ValueError, match="action_tokenizer config must be provided"
        ):
            builder._create_tokenizer(
                normalizer=MagicMock(),
                action_data={},
                action_meta={},
            )

    def test_creates_observation_tokenizer_when_configured(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
    ):
        mock_obs_instance = MagicMock()
        mock_obs_instance._is_fitted = True

        mock_config = MagicMock()
        mock_config.tokenize_observations = True
        mock_config.tokenize_actions = False
        mock_config.observation_tokenizer.bin_continuous_data = False

        builder = transform_builder_factory(tokenization_config=mock_config)

        with (
            patch(
                "versatil.data.processing.transform_builder.ObservationTokenizer",
                return_value=mock_obs_instance,
            ) as mock_obs_class,
            patch(
                "versatil.data.processing.transform_builder.Tokenizer"
            ) as mock_tokenizer_class,
        ):
            builder._create_tokenizer(
                normalizer=MagicMock(), action_data={}, action_meta={}
            )

        mock_obs_class.assert_called_once()
        mock_tokenizer_class.assert_called_once_with(
            observation_tokenizer=mock_obs_instance,
            action_tokenizer=None,
        )

    def test_creates_action_tokenizer_with_pretrained_fast(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
    ):
        mock_action_instance = MagicMock()
        mock_action_instance._is_fitted = True

        mock_config = MagicMock()
        mock_config.tokenize_observations = False
        mock_config.tokenize_actions = True
        mock_config.action_tokenizer.action_discretizer.type = "fast"
        mock_config.action_tokenizer.action_discretizer.use_pretrained = True
        mock_config.action_tokenizer.action_discretizer.tokenizer_model = (
            "physical-intelligence/fast"
        )
        mock_config.action_tokenizer.token_id_mapping.type = "identity"
        mock_config.action_tokenizer.max_token_len = 64

        builder = transform_builder_factory(tokenization_config=mock_config)

        with (
            patch(
                "versatil.data.processing.transform_builder.ActionTokenizer",
                return_value=mock_action_instance,
            ) as mock_action_class,
            patch(
                "versatil.data.processing.transform_builder.Tokenizer"
            ) as mock_tokenizer_class,
        ):
            builder._create_tokenizer(
                normalizer=MagicMock(), action_data={}, action_meta={}
            )

        mock_action_class.assert_called_once()
        mock_action_instance.fit.assert_not_called()
        mock_tokenizer_class.assert_called_once_with(
            observation_tokenizer=None,
            action_tokenizer=mock_action_instance,
        )

    def test_pretrained_fast_discretizer_records_chunk_shape(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        rng: np.random.Generator,
    ):
        mock_action_instance = MagicMock()
        mock_action_instance._is_fitted = True

        mock_config = MagicMock()
        mock_config.tokenize_observations = False
        mock_config.tokenize_actions = True
        mock_config.action_tokenizer.action_discretizer.type = (
            ActionDiscretizerType.FAST.value
        )
        mock_config.action_tokenizer.action_discretizer.use_pretrained = True
        mock_config.action_tokenizer.action_discretizer.tokenizer_model = (
            "physical-intelligence/fast"
        )
        mock_config.action_tokenizer.token_id_mapping.type = "identity"
        mock_config.action_tokenizer.max_token_len = 64

        builder = transform_builder_factory(
            tokenization_config=mock_config,
            prediction_horizon=4,
        )
        action_meta = {"action": precomputed_action_metadata_factory()}
        action_data = {"action": rng.standard_normal((9, 3)).astype(np.float32)}

        with (
            patch("versatil.data.tokenization.action_discretizer.load_fast_processor"),
            patch(
                "versatil.data.processing.transform_builder.ActionTokenizer",
                return_value=mock_action_instance,
            ) as mock_action_class,
            patch("versatil.data.processing.transform_builder.Tokenizer"),
        ):
            builder._create_tokenizer(
                normalizer=MagicMock(), action_data=action_data, action_meta=action_meta
            )

        action_discretizer = mock_action_class.call_args.kwargs["action_discretizer"]
        assert action_discretizer.time_horizon == 4
        assert action_discretizer.action_dim == 3

    def test_creates_action_tokenizer_with_uniform_binned_discretizer(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
    ):
        mock_action_instance = MagicMock()
        mock_action_instance._is_fitted = True

        mock_config = MagicMock()
        mock_config.tokenize_observations = False
        mock_config.tokenize_actions = True
        mock_config.action_tokenizer.action_discretizer.type = (
            ActionDiscretizerType.BINNED.value
        )
        mock_config.action_tokenizer.action_discretizer.binning_strategy = (
            BinningStrategy.UNIFORM.value
        )
        mock_config.action_tokenizer.action_discretizer.num_bins = 256
        mock_config.action_tokenizer.action_discretizer.min_value = -1.0
        mock_config.action_tokenizer.action_discretizer.max_value = 1.0
        mock_config.action_tokenizer.token_id_mapping.type = "identity"
        mock_config.action_tokenizer.max_token_len = 64

        builder = transform_builder_factory(tokenization_config=mock_config)

        with (
            patch(
                "versatil.data.processing.transform_builder.ActionTokenizer",
                return_value=mock_action_instance,
            ) as mock_action_class,
            patch("versatil.data.processing.transform_builder.Tokenizer"),
        ):
            builder._create_tokenizer(
                normalizer=MagicMock(), action_data={}, action_meta={}
            )

        action_discretizer = mock_action_class.call_args.kwargs["action_discretizer"]
        assert isinstance(action_discretizer, BinnedActionDiscretizer)
        assert action_discretizer.token_count == 256
        assert (
            action_discretizer.binner.binning_strategy == BinningStrategy.UNIFORM.value
        )
        assert action_discretizer.binner.min_value == -1.0
        assert action_discretizer.binner.max_value == 1.0

    def test_fits_action_tokenizer_when_not_pretrained(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        rng: np.random.Generator,
    ):
        mock_action_instance = MagicMock()
        mock_action_instance._is_fitted = False

        mock_config = MagicMock()
        mock_config.tokenize_observations = False
        mock_config.tokenize_actions = True
        mock_config.action_tokenizer.action_discretizer.type = "fast"
        mock_config.action_tokenizer.action_discretizer.use_pretrained = False
        mock_config.action_tokenizer.action_discretizer.tokenizer_model = (
            "physical-intelligence/fast"
        )
        mock_config.action_tokenizer.token_id_mapping.type = "identity"
        mock_config.action_tokenizer.max_token_len = 64

        action_metadata = on_the_fly_action_metadata_factory()
        builder = transform_builder_factory(
            tokenization_config=mock_config,
            prediction_horizon=2,
            episode_ends=np.array([10]),
        )

        action_data = {"position": rng.standard_normal((9, 3)).astype(np.float32)}
        action_meta = {"position": action_metadata}

        with (
            patch(
                "versatil.data.processing.transform_builder.ActionTokenizer",
                return_value=mock_action_instance,
            ),
            patch("versatil.data.processing.transform_builder.Tokenizer"),
        ):
            builder._create_tokenizer(
                normalizer=MagicMock(), action_data=action_data, action_meta=action_meta
            )

        mock_action_instance.fit.assert_called_once()

    def test_observation_tokenizer_passthrough_when_no_data_to_bin(
        self,
        transform_builder_factory: Callable[..., TransformBuilder],
        caplog: pytest.LogCaptureFixture,
    ):
        mock_obs_instance = MagicMock()
        mock_obs_instance._is_fitted = False

        mock_config = MagicMock()
        mock_config.tokenize_observations = True
        mock_config.tokenize_actions = False
        mock_config.observation_tokenizer.bin_continuous_data = True

        builder = transform_builder_factory(tokenization_config=mock_config)

        with (
            patch(
                "versatil.data.processing.transform_builder.ObservationTokenizer",
                return_value=mock_obs_instance,
            ),
            patch("versatil.data.processing.transform_builder.Tokenizer"),
            caplog.at_level(logging.WARNING),
        ):
            builder._create_tokenizer(
                normalizer=MagicMock(), action_data={}, action_meta={}
            )

        assert "pass-through" in caplog.text
        mock_obs_instance.fit.assert_called_with({})


class TestTokenizerComponentBuilders:
    def test_unsupported_discretizer_type_raises(self):
        config = MagicMock()
        config.type = "unsupported"
        with pytest.raises(ValueError, match="Unsupported action discretizer type"):
            _build_action_discretizer(
                config, device="cpu", time_horizon=4, action_dim=2
            )

    def test_identity_token_id_mapping(self):
        config = MagicMock()
        config.type = ActionTokenIdMappingType.IDENTITY.value
        mapping = _build_token_id_mapping(config)
        assert isinstance(mapping, IdentityActionTokenIdMapping)

    def test_language_vocabulary_mapping_requires_model(self):
        config = MagicMock()
        config.type = ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value
        config.language_tokenizer_model = None
        with pytest.raises(ValueError, match="language_tokenizer_model"):
            _build_token_id_mapping(config)

    def test_unsupported_token_id_mapping_raises(self):
        config = MagicMock()
        config.type = "unsupported"
        with pytest.raises(ValueError, match="Unsupported action token-id mapping"):
            _build_token_id_mapping(config)
