"""Tests for versatil.data.sample_builder module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.constants import (
    ActionComputationMethod,
    Cameras,
    SampleKey,
)
from versatil.data.metadata import (
    ActionMetadata,
    CameraMetadata,
    ObservationMetadata,
    OnTheFlyActionMetadata,
    PositionActionMetadata,
    PositionObservationMetadata,
)
from versatil.data.sample_builder import SampleBuilder
from versatil.data.task import ActionSpace, ObservationSpace


@pytest.fixture
def sample_builder_factory(
    action_space_factory: Callable[..., ActionSpace],
    observation_space_factory: Callable[..., ObservationSpace],
) -> Callable[..., SampleBuilder]:
    """Factory for creating SampleBuilder instances with mocked dependencies."""

    def factory(
        actions_metadata: dict = None,
        observations_metadata: dict = None,
        obs_horizon: int = 2,
        pred_horizon: int = 4,
        action_backward_shift: int = 0,
        normalizer: MagicMock = None,
        tokenizer: MagicMock = None,
    ) -> SampleBuilder:
        action_space = action_space_factory(actions_metadata=actions_metadata or {})
        observation_space = observation_space_factory(
            observations_metadata=observations_metadata or {},
        )
        mock_augmentation = MagicMock()

        def _mock_process(images: np.ndarray, camera_key: str) -> torch.Tensor:
            is_depth = camera_key == Cameras.DEPTH.value
            if is_depth:
                if images.ndim == 3:
                    return torch.from_numpy(images.astype(np.float32)[:, None])
                return torch.from_numpy(np.moveaxis(images.astype(np.float32), -1, 1))
            return torch.from_numpy(
                np.moveaxis(images.astype(np.float32) / 255.0, -1, 1)
            )

        mock_augmentation.process.side_effect = _mock_process

        mock_action_processor = MagicMock()

        return SampleBuilder(
            action_space=action_space,
            observation_space=observation_space,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_backward_shift=action_backward_shift,
            image_processor=mock_augmentation,
            action_processor=mock_action_processor,
            tokenizer=tokenizer,
            normalizer=normalizer,
        )

    return factory


class TestSampleBuilderInitialization:
    @pytest.mark.parametrize(
        "obs_horizon,pred_horizon,action_backward_shift",
        [
            (1, 4, 0),
            (3, 8, 1),
            (5, 16, 2),
        ],
    )
    def test_stores_horizon_parameters(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        obs_horizon: int,
        pred_horizon: int,
        action_backward_shift: int,
    ):
        builder = sample_builder_factory(
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_backward_shift=action_backward_shift,
        )

        assert builder.obs_horizon == obs_horizon
        assert builder.pred_horizon == pred_horizon
        assert builder.action_backward_shift == action_backward_shift

    def test_normalizer_and_tokenizer_stored(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
    ):
        mock_normalizer = MagicMock()
        mock_tokenizer = MagicMock()
        builder = sample_builder_factory(
            normalizer=mock_normalizer,
            tokenizer=mock_tokenizer,
        )

        assert builder.normalizer is mock_normalizer
        assert builder.tokenizer is mock_tokenizer


class TestGetSampleImages:
    def test_processes_rgb_image_with_augmentation_and_channel_reorder(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        rng: np.random.Generator,
    ):
        builder = sample_builder_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value,
                ),
            },
            obs_horizon=2,
        )
        # (T, H, W, C) uint8 images — 4 timesteps, take obs_horizon=2 starting at shift=0
        padded_images = rng.integers(
            0,
            256,
            (4, 8, 8, 3),
            dtype=np.uint8,
        )
        padded_data = {Cameras.LEFT.value: padded_images}

        result = builder._get_sample_images(padded_data=padded_data)

        assert Cameras.LEFT.value in result
        image_tensor = result[Cameras.LEFT.value]
        # Should be (obs_horizon=2, C=3, H=8, W=8) float32 in [0, 1]
        assert image_tensor.shape == (2, 3, 8, 8)
        assert image_tensor.dtype == torch.float32
        assert image_tensor.max() <= 1.0
        assert image_tensor.min() >= 0.0

    def test_processes_depth_image_with_channel_dimension_added(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        rng: np.random.Generator,
    ):
        builder = sample_builder_factory(
            observations_metadata={
                Cameras.DEPTH.value: camera_metadata_factory(
                    camera_key=Cameras.DEPTH.value,
                    channels=1,
                ),
            },
            obs_horizon=2,
        )
        # (T, H, W) depth images — no channel dimension
        padded_depth = rng.random((4, 8, 8)).astype(np.float32)
        padded_data = {Cameras.DEPTH.value: padded_depth}

        result = builder._get_sample_images(padded_data=padded_data)

        depth_tensor = result[Cameras.DEPTH.value]
        # Should have channel dimension added: (obs_horizon=2, 1, H=8, W=8)
        assert depth_tensor.shape == (2, 1, 8, 8)

    def test_applies_action_backward_shift_to_image_slicing(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
    ):
        builder = sample_builder_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value,
                ),
            },
            obs_horizon=2,
            action_backward_shift=1,
        )
        # Timestep 0 has all zeros, timesteps 1-3 have distinct values
        padded_images = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        padded_images[1:] = 128

        padded_data = {Cameras.LEFT.value: padded_images}
        result = builder._get_sample_images(padded_data=padded_data)

        # With shift=1, should take indices [1:3], skipping the zero-filled timestep 0
        assert result[Cameras.LEFT.value][0].sum() > 0

    def test_depth_image_with_explicit_channel_dimension(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        rng: np.random.Generator,
    ):
        builder = sample_builder_factory(
            observations_metadata={
                Cameras.DEPTH.value: camera_metadata_factory(
                    camera_key=Cameras.DEPTH.value,
                    channels=1,
                ),
            },
            obs_horizon=2,
        )
        # (T, H, W, 1) — depth with explicit channel dimension
        padded_depth = rng.random((4, 8, 8, 1)).astype(np.float32)
        padded_data = {Cameras.DEPTH.value: padded_depth}

        result = builder._get_sample_images(padded_data=padded_data)

        # Shape should remain (obs_horizon=2, 1, H=8, W=8) — no extra unsqueeze
        assert result[Cameras.DEPTH.value].shape == (2, 1, 8, 8)

    def test_no_cameras_returns_empty_dict(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
    ):
        builder = sample_builder_factory(observations_metadata={})

        result = builder._get_sample_images(padded_data={})

        assert result == {}

    def test_calls_rgb_image_processor(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        rng: np.random.Generator,
    ):
        builder = sample_builder_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value,
                ),
            },
            obs_horizon=2,
        )
        padded_data = {
            Cameras.LEFT.value: rng.integers(0, 256, (4, 8, 8, 3), dtype=np.uint8),
        }

        builder._get_sample_images(padded_data=padded_data)

        builder.image_processor.process.assert_called_once()

    def test_calls_depth_image_processor(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        rng: np.random.Generator,
    ):
        builder = sample_builder_factory(
            observations_metadata={
                Cameras.DEPTH.value: camera_metadata_factory(
                    camera_key=Cameras.DEPTH.value,
                    channels=1,
                ),
            },
            obs_horizon=2,
        )
        padded_data = {
            Cameras.DEPTH.value: rng.random((4, 8, 8)).astype(np.float32),
        }

        builder._get_sample_images(padded_data=padded_data)

        builder.image_processor.process.assert_called_once()


class TestSliceObservationTensor:
    @pytest.mark.parametrize(
        "dtype,expected_torch_dtype",
        [
            ("float32", torch.float32),
            ("float64", torch.float32),
            ("int32", torch.int64),
            ("int64", torch.int64),
            ("bool", torch.int64),
        ],
    )
    def test_converts_to_correct_torch_dtype(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        dtype: str,
        expected_torch_dtype: torch.dtype,
    ):
        builder = sample_builder_factory(obs_horizon=2)
        # Use mock to test dtype dispatch without metadata validation constraints
        metadata = MagicMock()
        metadata.dtype = dtype
        padded_data = {"value": np.array([[1], [2], [3], [4]], dtype=np.float32)}

        result = builder._slice_observation_tensor(
            key="value",
            padded_data=padded_data,
            metadata=metadata,
        )

        assert result.dtype == expected_torch_dtype

    def test_string_dtype_returns_list(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
    ):
        builder = sample_builder_factory(obs_horizon=2)
        metadata = ObservationMetadata(
            raw_data_column_keys=["label"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        padded_data = {"label": np.array([["a"], ["b"], ["c"], ["d"]])}

        result = builder._slice_observation_tensor(
            key="label",
            padded_data=padded_data,
            metadata=metadata,
        )

        assert isinstance(result, list)
        assert len(result) == 2

    def test_unsupported_dtype_raises(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
    ):
        builder = sample_builder_factory(obs_horizon=2)
        # Use a mock to bypass BaseMetadata validation and reach the else branch
        metadata = MagicMock()
        metadata.dtype = "complex128"
        padded_data = {"value": np.array([[1], [2], [3], [4]])}

        with pytest.raises(ValueError, match="Unsupported custom observation dtype"):
            builder._slice_observation_tensor(
                key="value",
                padded_data=padded_data,
                metadata=metadata,
            )

    def test_slices_with_correct_obs_horizon_and_shift(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
    ):
        builder = sample_builder_factory(
            obs_horizon=2,
            action_backward_shift=1,
        )
        metadata = ObservationMetadata(
            raw_data_column_keys=["value"],
            dimension=1,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
        )
        # With shift=1, obs_horizon=2: should take indices [1:3]
        padded_data = {"value": np.array([[10], [20], [30], [40]], dtype=np.float32)}

        result = builder._slice_observation_tensor(
            key="value",
            padded_data=padded_data,
            metadata=metadata,
        )

        torch.testing.assert_close(result, torch.tensor([[20.0], [30.0]]))


class TestSliceActionData:
    @pytest.mark.parametrize(
        "dtype,expected_torch_dtype",
        [
            ("float32", torch.float32),
            ("int32", torch.int64),
            ("bool", torch.int64),
        ],
    )
    def test_converts_to_correct_torch_dtype(
        self,
        dtype: str,
        expected_torch_dtype: torch.dtype,
    ):
        metadata = MagicMock()
        metadata.dtype = dtype
        action_data = {"action": np.array([[1], [2], [3]], dtype=np.float32)}

        result = SampleBuilder._slice_action_data(
            key="action",
            action_data=action_data,
            metadata=metadata,
        )

        assert result.dtype == expected_torch_dtype

    def test_string_dtype_returns_list(self):
        metadata = ActionMetadata(
            prediction_dimension=1,
            is_numerical=False,
            needs_normalization=False,
            dtype="str",
            is_precomputed=True,
            requires_prediction_head=False,
        )
        action_data = {"label": np.array([["open"], ["close"]])}

        result = SampleBuilder._slice_action_data(
            key="label",
            action_data=action_data,
            metadata=metadata,
        )

        assert isinstance(result, list)

    def test_unsupported_dtype_raises(self):
        metadata = MagicMock()
        metadata.dtype = "complex128"

        with pytest.raises(ValueError, match="Unsupported custom action dtype"):
            SampleBuilder._slice_action_data(
                key="action",
                action_data={"action": np.array([[1]])},
                metadata=metadata,
            )


class TestComputeActionPaddingMask:
    @pytest.mark.parametrize(
        "action_type, expected_mask",
        [
            ("precomputed", [True, False, False, False]),
            ("on_the_fly_delta", [True, False, False, True]),
            ("on_the_fly_absolute", [False, False, False, True]),
            ("mixed_precomputed_absolute", [True, False, False, True]),
            ("mixed_precomputed_delta", [True, False, False, True]),
        ],
    )
    def test_padding_mask_respects_action_type_data_dependency(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        action_type: str,
        expected_mask: list[bool],
    ):
        if action_type == "precomputed":
            actions_metadata = {
                "position": position_action_metadata_factory(prediction_dimension=3),
            }
        elif action_type == "on_the_fly_delta":
            actions_metadata = {
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_observation_metadata_factory(),
                    computation_method=ActionComputationMethod.DELTA.value,
                ),
            }
        elif action_type == "on_the_fly_absolute":
            actions_metadata = {
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_observation_metadata_factory(),
                    computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
                ),
            }
        elif action_type == "mixed_precomputed_absolute":
            actions_metadata = {
                "phase": position_action_metadata_factory(prediction_dimension=1),
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_observation_metadata_factory(),
                    computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
                ),
            }
        else:
            actions_metadata = {
                "phase": position_action_metadata_factory(prediction_dimension=1),
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_observation_metadata_factory(),
                    computation_method=ActionComputationMethod.DELTA.value,
                ),
            }
        builder = sample_builder_factory(
            actions_metadata=actions_metadata,
            obs_horizon=1,
            pred_horizon=4,
        )
        # sampler_indices[start_idx] = (buffer_start, buffer_end, sample_start, sample_end)
        # action_slice_start = 0, action_positions = [0, 1, 2, 3], valid range [1, 4)
        sampler_indices = np.array([[0, 6, 1, 4]])

        mask = builder._compute_action_padding_mask(
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        assert mask.dtype == torch.bool
        assert mask.shape == (4,)
        torch.testing.assert_close(mask, torch.tensor(expected_mask, dtype=torch.bool))

    def test_mask_shape_matches_prediction_horizon(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        builder = sample_builder_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(),
            },
            obs_horizon=1,
            pred_horizon=8,
        )
        sampler_indices = np.array([[0, 20, 0, 20]])

        mask = builder._compute_action_padding_mask(
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        assert mask.shape == (8,)


class TestGetActionSliceStart:
    def test_returns_obs_horizon_minus_one(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
    ):
        builder = sample_builder_factory(obs_horizon=3)

        assert builder._get_action_slice_start() == 2


class TestNormalizeAndTokenizeSample:
    def test_applies_normalizer_when_present(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
    ):
        mock_normalizer = MagicMock()
        builder = sample_builder_factory(normalizer=mock_normalizer)

        sample = {
            SampleKey.OBSERVATION.value: {"position": torch.tensor([1.0])},
            SampleKey.ACTION.value: {"position": torch.tensor([2.0])},
        }

        # normalize_sample is a module-level function; the builder delegates to it
        # We just verify it doesn't crash and returns a dict
        result = builder.normalize_and_tokenize_sample(sample=sample)

        assert isinstance(result, dict)

    def test_skips_normalization_when_normalizer_is_none(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
    ):
        builder = sample_builder_factory(normalizer=None, tokenizer=None)
        original_tensor = torch.tensor([1.0, 2.0, 3.0])

        sample = {
            SampleKey.OBSERVATION.value: {"position": original_tensor},
            SampleKey.ACTION.value: {},
        }

        result = builder.normalize_and_tokenize_sample(sample=sample)

        torch.testing.assert_close(
            result[SampleKey.OBSERVATION.value]["position"],
            original_tensor,
        )

    def test_applies_tokenizer_when_present(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.observation_tokenizer = None
        mock_tokenizer.action_tokenizer = None
        builder = sample_builder_factory(tokenizer=mock_tokenizer)

        sample = {
            SampleKey.OBSERVATION.value: {},
            SampleKey.ACTION.value: {},
        }

        result = builder.normalize_and_tokenize_sample(sample=sample)

        assert isinstance(result, dict)


class TestBuildSample:
    def test_assembles_observations_actions_and_padding_mask(
        self,
        sample_builder_factory: Callable[..., SampleBuilder],
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        rng: np.random.Generator,
    ):
        position_metadata = on_the_fly_action_metadata_factory()

        builder = sample_builder_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value,
                ),
                "position": position_observation_metadata_factory(dimension=3),
            },
            actions_metadata={"position": position_metadata},
            obs_horizon=2,
            pred_horizon=3,
        )

        padded_data = {
            Cameras.LEFT.value: rng.integers(0, 256, (8, 4, 4, 3), dtype=np.uint8),
            "position": rng.standard_normal((8, 3)).astype(np.float32),
        }
        action_data = {
            "position": rng.standard_normal((3, 3)).astype(np.float32),
        }
        action_meta = {"position": position_metadata}
        sampler_indices = np.array([[0, 8, 0, 8]])

        result = builder.build_sample(
            padded_data=padded_data,
            action_data=action_data,
            action_meta=action_meta,
            start_idx=0,
            sampler_indices=sampler_indices,
        )

        assert SampleKey.OBSERVATION.value in result
        assert SampleKey.ACTION.value in result
        assert Cameras.LEFT.value in result[SampleKey.OBSERVATION.value]
        assert "position" in result[SampleKey.OBSERVATION.value]
        assert "position" in result[SampleKey.ACTION.value]
        assert SampleKey.IS_PAD_ACTION.value in result[SampleKey.ACTION.value]
