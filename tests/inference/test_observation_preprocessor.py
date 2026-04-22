"""Tests for versatil.inference.observation_preprocessor module."""

import logging
from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tso_robotics_sockets import CompressionType
from versatil_constants.shared import ObsKey
from versatil_constants.tso import TSOProprioKey

from versatil.data.constants import Cameras
from versatil.data.metadata import CameraMetadata
from versatil.inference.observation_preprocessor import ObservationPreprocessor


@pytest.fixture
def preprocessor_factory() -> Callable[..., ObservationPreprocessor]:
    def factory(
        camera_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        has_language: bool = False,
        image_height: int = 8,
        image_width: int = 8,
        compression_type: str = CompressionType.RAW.value,
        rotate_images: bool = False,
        depth_clamp_range: tuple[float, float] | None = None,
    ) -> ObservationPreprocessor:
        if camera_keys is None:
            camera_keys = [Cameras.LEFT.value]
        if state_keys is None:
            state_keys = []
        camera_metadata = {}
        for key in camera_keys:
            channels = 1 if key == Cameras.DEPTH.value else 3
            camera_metadata[key] = CameraMetadata(
                camera_key=key,
                dtype="uint8",
                channels=channels,
                image_height=image_height,
                image_width=image_width,
            )
        return ObservationPreprocessor(
            camera_keys=camera_keys,
            state_keys=state_keys,
            has_language=has_language,
            camera_metadata=camera_metadata,
            compression_type=compression_type,
            rotate_images=rotate_images,
            depth_clamp_range=depth_clamp_range,
        )

    return factory


@pytest.fixture
def rgb_image_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    def factory(
        height: int = 16,
        width: int = 16,
        channels: int = 3,
    ) -> np.ndarray:
        return rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)

    return factory


@pytest.fixture
def depth_image_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    def factory(
        height: int = 16,
        width: int = 16,
        max_depth: float = 1000.0,
    ) -> np.ndarray:
        return (rng.random((height, width)) * max_depth).astype(np.float32)

    return factory


@pytest.fixture
def single_environment_response_factory(
    rgb_image_factory: Callable[..., np.ndarray],
) -> Callable[..., dict]:
    def factory(
        camera_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        state_values: dict[str, list[float]] | None = None,
        language: str | None = None,
        image_height: int = 16,
        image_width: int = 16,
    ) -> dict:
        if camera_keys is None:
            camera_keys = [Cameras.LEFT.value]
        if state_keys is None:
            state_keys = []
        response = {}
        for camera_key in camera_keys:
            response[camera_key] = rgb_image_factory(
                height=image_height, width=image_width
            )
        if state_values is not None:
            for key, value in state_values.items():
                response[key] = value
        else:
            for key in state_keys:
                response[key] = [0.1, 0.2, 0.3]
        if language is not None:
            response[ObsKey.LANGUAGE.value] = language
        return response

    return factory


@pytest.fixture
def multi_environment_response_factory(
    rgb_image_factory: Callable[..., np.ndarray],
) -> Callable[..., dict]:
    def factory(
        camera_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        environment_count: int = 2,
        language: str | None = None,
        image_height: int = 16,
        image_width: int = 16,
    ) -> dict:
        if camera_keys is None:
            camera_keys = [Cameras.LEFT.value]
        if state_keys is None:
            state_keys = []
        response = {}
        for camera_key in camera_keys:
            response[camera_key] = {}
            for environment_index in range(environment_count):
                response[camera_key][str(environment_index)] = rgb_image_factory(
                    height=image_height, width=image_width
                )
        for key in state_keys:
            response[key] = {}
            for environment_index in range(environment_count):
                response[key][str(environment_index)] = [0.1, 0.2, 0.3]
        if language is not None:
            response[ObsKey.LANGUAGE.value] = {}
            for environment_index in range(environment_count):
                response[ObsKey.LANGUAGE.value][str(environment_index)] = language
        return response

    return factory


@pytest.mark.unit
class TestObservationPreprocessorInitialization:
    @pytest.mark.parametrize("rotate_images", [True, False])
    @pytest.mark.parametrize("has_language", [True, False])
    @pytest.mark.parametrize(
        "compression_type",
        [CompressionType.RAW.value, CompressionType.PNG.value],
    )
    def test_stores_configuration(
        self,
        preprocessor_factory,
        rotate_images,
        has_language,
        compression_type,
    ):
        preprocessor = preprocessor_factory(
            rotate_images=rotate_images,
            has_language=has_language,
            compression_type=compression_type,
        )
        assert preprocessor.rotate_images == rotate_images
        assert preprocessor.has_language == has_language
        assert preprocessor.compression_type == compression_type

    def test_detects_depth_in_camera_keys(self, preprocessor_factory):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )
        assert Cameras.DEPTH.value in preprocessor.camera_keys
        assert Cameras.DEPTH.value not in preprocessor.rgb_camera_keys

    def test_no_depth_when_absent(self, preprocessor_factory):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
        )
        assert Cameras.DEPTH.value not in preprocessor.camera_keys
        assert preprocessor.rgb_camera_keys == [
            Cameras.LEFT.value,
            Cameras.RIGHT.value,
        ]


@pytest.mark.unit
class TestParseResponse:
    def test_single_environment_detected_when_camera_value_is_array(
        self,
        preprocessor_factory,
        rgb_image_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
        )
        image = rgb_image_factory()
        response = {Cameras.LEFT.value: image}

        with patch.object(
            preprocessor, "_parse_single_environment", return_value={0: {}}
        ) as mock_single:
            preprocessor.parse_response(response=response)
            mock_single.assert_called_once_with(response=response)

    def test_multi_environment_detected_when_camera_value_is_dict(
        self,
        preprocessor_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
        )
        response = {Cameras.LEFT.value: {"0": "some_data", "1": "other_data"}}

        with patch.object(
            preprocessor,
            "_parse_multi_environment",
            return_value={0: {}, 1: {}},
        ) as mock_multi:
            preprocessor.parse_response(response=response)
            mock_multi.assert_called_once_with(response=response)

    def test_no_camera_keys_falls_back_to_single_environment(
        self,
        preprocessor_factory,
    ):
        preprocessor = preprocessor_factory(camera_keys=[])
        response = {"some_key": "some_value"}

        with patch.object(
            preprocessor, "_parse_single_environment", return_value={0: {}}
        ) as mock_single:
            preprocessor.parse_response(response=response)
            mock_single.assert_called_once_with(response=response)


@pytest.mark.unit
class TestParseSingleEnvironment:
    def test_decompresses_camera_images(
        self,
        preprocessor_factory,
        rgb_image_factory,
    ):
        image = rgb_image_factory()
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            compression_type=CompressionType.RAW.value,
        )

        with patch(
            "versatil.inference.observation_preprocessor.decompress_array",
            return_value=image,
        ) as mock_decompress:
            response = {Cameras.LEFT.value: b"compressed_data"}
            result = preprocessor._parse_single_environment(response=response)

            mock_decompress.assert_called_once_with(
                b"compressed_data", method=CompressionType.RAW.value
            )
            np.testing.assert_array_equal(result[0][Cameras.LEFT.value], image)

    def test_decompress_called_with_correct_compression_type(
        self,
        preprocessor_factory,
        rgb_image_factory,
    ):
        image = rgb_image_factory()
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            compression_type=CompressionType.PNG.value,
        )

        with patch(
            "versatil.inference.observation_preprocessor.decompress_array",
            return_value=image,
        ) as mock_decompress:
            response = {Cameras.LEFT.value: b"png_data"}
            preprocessor._parse_single_environment(response=response)

            mock_decompress.assert_called_once_with(
                b"png_data", method=CompressionType.PNG.value
            )

    def test_rotation_flips_both_axes(
        self,
        preprocessor_factory,
    ):
        # Create a 4x4 image with known content for verifiable rotation
        image = np.array(
            [
                [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
                [[5, 0, 0], [6, 0, 0], [7, 0, 0], [8, 0, 0]],
                [[9, 0, 0], [10, 0, 0], [11, 0, 0], [12, 0, 0]],
                [[13, 0, 0], [14, 0, 0], [15, 0, 0], [16, 0, 0]],
            ],
            dtype=np.uint8,
        )
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            rotate_images=True,
        )

        with patch(
            "versatil.inference.observation_preprocessor.decompress_array",
            return_value=image.copy(),
        ):
            response = {Cameras.LEFT.value: b"data"}
            result = preprocessor._parse_single_environment(response=response)

            rotated = result[0][Cameras.LEFT.value]
            # Flipping both axes: [0,0] becomes [-1,-1] of original
            assert rotated[0, 0, 0] == 16
            assert rotated[-1, -1, 0] == 1

    def test_no_rotation_preserves_original(
        self,
        preprocessor_factory,
    ):
        image = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=np.uint8,
        )
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            rotate_images=False,
        )

        with patch(
            "versatil.inference.observation_preprocessor.decompress_array",
            return_value=image.copy(),
        ):
            response = {Cameras.LEFT.value: b"data"}
            result = preprocessor._parse_single_environment(response=response)

            np.testing.assert_array_equal(result[0][Cameras.LEFT.value], image)

    def test_state_keys_cast_to_float32(
        self,
        preprocessor_factory,
    ):
        proprio_key = TSOProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        preprocessor = preprocessor_factory(
            camera_keys=[],
            state_keys=[proprio_key],
        )
        response = {proprio_key: [1, 2, 3]}

        result = preprocessor._parse_single_environment(response=response)

        assert result[0][proprio_key].dtype == np.float32
        np.testing.assert_allclose(result[0][proprio_key], [1.0, 2.0, 3.0])

    def test_language_included_when_enabled_and_present(
        self,
        preprocessor_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[],
            has_language=True,
        )
        instruction = "pick up the red block"
        response = {ObsKey.LANGUAGE.value: instruction}

        result = preprocessor._parse_single_environment(response=response)

        assert result[0][ObsKey.LANGUAGE.value] == instruction

    def test_language_omitted_when_disabled(
        self,
        preprocessor_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[],
            has_language=False,
        )
        response = {ObsKey.LANGUAGE.value: "pick up the block"}

        result = preprocessor._parse_single_environment(response=response)

        assert ObsKey.LANGUAGE.value not in result[0]

    def test_result_keyed_by_environment_index_zero(
        self,
        preprocessor_factory,
    ):
        preprocessor = preprocessor_factory(camera_keys=[])
        response = {}

        result = preprocessor._parse_single_environment(response=response)

        assert list(result.keys()) == [0]


@pytest.mark.unit
class TestParseMultiEnvironment:
    def test_parses_multiple_environments(
        self,
        preprocessor_factory,
        rgb_image_factory,
    ):
        image_0 = rgb_image_factory(height=4, width=4)
        image_1 = rgb_image_factory(height=4, width=4)
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
        )

        images = [image_0, image_1]
        call_count = [0]

        def mock_decompress(data, method):
            index = call_count[0]
            call_count[0] += 1
            return images[index]

        with patch(
            "versatil.inference.observation_preprocessor.decompress_array",
            side_effect=mock_decompress,
        ):
            response = {
                Cameras.LEFT.value: {"0": b"data_0", "1": b"data_1"},
            }
            result = preprocessor._parse_multi_environment(response=response)

            assert set(result.keys()) == {0, 1}
            np.testing.assert_array_equal(result[0][Cameras.LEFT.value], image_0)
            np.testing.assert_array_equal(result[1][Cameras.LEFT.value], image_1)

    def test_rotation_applied_per_environment(
        self,
        preprocessor_factory,
    ):
        image = np.array(
            [[[1, 0, 0], [2, 0, 0]], [[3, 0, 0], [4, 0, 0]]],
            dtype=np.uint8,
        )
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            rotate_images=True,
        )

        with patch(
            "versatil.inference.observation_preprocessor.decompress_array",
            return_value=image.copy(),
        ):
            response = {
                Cameras.LEFT.value: {"0": b"data_0"},
            }
            result = preprocessor._parse_multi_environment(response=response)

            rotated = result[0][Cameras.LEFT.value]
            assert rotated[0, 0, 0] == 4
            assert rotated[-1, -1, 0] == 1

    def test_state_data_per_environment(
        self,
        preprocessor_factory,
    ):
        proprio_key = TSOProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            state_keys=[proprio_key],
        )
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        with patch(
            "versatil.inference.observation_preprocessor.decompress_array",
            return_value=image,
        ):
            response = {
                Cameras.LEFT.value: {"0": b"data", "1": b"data"},
                proprio_key: {"0": [1.0, 2.0], "1": [3.0, 4.0]},
            }
            result = preprocessor._parse_multi_environment(response=response)

            np.testing.assert_allclose(result[0][proprio_key], [1.0, 2.0])
            np.testing.assert_allclose(result[1][proprio_key], [3.0, 4.0])

    def test_language_per_environment(
        self,
        preprocessor_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            has_language=True,
        )
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        with patch(
            "versatil.inference.observation_preprocessor.decompress_array",
            return_value=image,
        ):
            response = {
                Cameras.LEFT.value: {"0": b"data", "1": b"data"},
                ObsKey.LANGUAGE.value: {
                    "0": "pick up block",
                    "1": "place on table",
                },
            }
            result = preprocessor._parse_multi_environment(response=response)

            assert result[0][ObsKey.LANGUAGE.value] == "pick up block"
            assert result[1][ObsKey.LANGUAGE.value] == "place on table"


@pytest.mark.unit
class TestTransformCameraObservations:
    def test_empty_camera_keys_returns_empty_dict(
        self,
        preprocessor_factory,
    ):
        preprocessor = preprocessor_factory(camera_keys=[])

        result = preprocessor.transform_camera_observations(recent_observations={})

        assert result == {}

    def test_rgb_output_shape(
        self,
        preprocessor_factory,
        rgb_image_factory,
    ):
        target_height = 8
        target_width = 8
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            image_height=target_height,
            image_width=target_width,
        )
        images = [
            rgb_image_factory(height=16, width=16),
            rgb_image_factory(height=16, width=16),
        ]
        recent_observations = {Cameras.LEFT.value: images}

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        # Shape: (observation_horizon, C, H, W)
        assert result[Cameras.LEFT.value].shape == (2, 3, target_height, target_width)

    def test_rgb_normalized_to_zero_one_range(
        self,
        preprocessor_factory,
        rgb_image_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            image_height=8,
            image_width=8,
        )
        image = rgb_image_factory(height=16, width=16)
        recent_observations = {Cameras.LEFT.value: [image]}

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        tensor = result[Cameras.LEFT.value]
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_depth_not_divided_by_255(
        self,
        preprocessor_factory,
        depth_image_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.DEPTH.value],
            image_height=8,
            image_width=8,
        )
        # Depth image with values > 1.0 (e.g., millimeter range)
        depth_image = depth_image_factory(height=16, width=16, max_depth=500.0)
        recent_observations = {Cameras.DEPTH.value: [depth_image]}

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        depth_tensor = result[Cameras.DEPTH.value]
        # Depth is treated as mask: not normalized by 255, just float conversion
        # Values should preserve the original scale (not be in [0, 1])
        assert depth_tensor.max() > 1.0

    def test_depth_clamping_applied(
        self,
        preprocessor_factory,
        depth_image_factory,
    ):
        depth_min = 100.0
        depth_max = 300.0
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.DEPTH.value],
            image_height=8,
            image_width=8,
            depth_clamp_range=(depth_min, depth_max),
        )
        # Depth with values outside clamp range
        depth_image = depth_image_factory(height=16, width=16, max_depth=500.0)
        recent_observations = {Cameras.DEPTH.value: [depth_image]}

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        depth_tensor = result[Cameras.DEPTH.value]
        assert depth_tensor.min() >= depth_min
        assert depth_tensor.max() <= depth_max

    def test_depth_no_clamping_when_range_is_none(
        self,
        preprocessor_factory,
        depth_image_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.DEPTH.value],
            image_height=8,
            image_width=8,
            depth_clamp_range=None,
        )
        depth_image = depth_image_factory(height=16, width=16, max_depth=500.0)
        recent_observations = {Cameras.DEPTH.value: [depth_image]}

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        depth_tensor = result[Cameras.DEPTH.value]
        # Without clamping, values can exceed any arbitrary range
        assert depth_tensor.max() > 1.0

    def test_depth_output_has_channel_dimension(
        self,
        preprocessor_factory,
        depth_image_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.DEPTH.value],
            image_height=8,
            image_width=8,
        )
        depth_image = depth_image_factory(height=16, width=16)
        recent_observations = {Cameras.DEPTH.value: [depth_image]}

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        # (T, C=1, H, W)
        assert result[Cameras.DEPTH.value].shape == (1, 1, 8, 8)

    def test_multiple_rgb_cameras_all_transformed(
        self,
        preprocessor_factory,
        rgb_image_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
            image_height=8,
            image_width=8,
        )
        recent_observations = {
            Cameras.LEFT.value: [rgb_image_factory(height=16, width=16)],
            Cameras.RIGHT.value: [rgb_image_factory(height=16, width=16)],
        }

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        assert Cameras.LEFT.value in result
        assert Cameras.RIGHT.value in result
        assert result[Cameras.LEFT.value].shape == (1, 3, 8, 8)
        assert result[Cameras.RIGHT.value].shape == (1, 3, 8, 8)

    def test_rgb_and_depth_together(
        self,
        preprocessor_factory,
        rgb_image_factory,
        depth_image_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            image_height=8,
            image_width=8,
            depth_clamp_range=(0.0, 200.0),
        )
        recent_observations = {
            Cameras.LEFT.value: [rgb_image_factory(height=16, width=16)],
            Cameras.DEPTH.value: [
                depth_image_factory(height=16, width=16, max_depth=500.0)
            ],
        }

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        # RGB: normalized to [0, 1]
        rgb_tensor = result[Cameras.LEFT.value]
        assert rgb_tensor.min() >= 0.0
        assert rgb_tensor.max() <= 1.0

        # Depth: clamped but not normalized to [0, 1]
        depth_tensor = result[Cameras.DEPTH.value]
        assert depth_tensor.min() >= 0.0
        assert depth_tensor.max() <= 200.0

    def test_multiple_timesteps_stacked(
        self,
        preprocessor_factory,
        rgb_image_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            image_height=8,
            image_width=8,
        )
        timestep_count = 3
        images = [rgb_image_factory(height=16, width=16) for _ in range(timestep_count)]
        recent_observations = {Cameras.LEFT.value: images}

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        assert result[Cameras.LEFT.value].shape[0] == timestep_count

    def test_depth_only_with_multiple_timesteps(
        self,
        preprocessor_factory,
        depth_image_factory,
    ):
        target_height = 8
        target_width = 8
        timestep_count = 3
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.DEPTH.value],
            image_height=target_height,
            image_width=target_width,
        )
        depth_images = [
            depth_image_factory(height=16, width=16, max_depth=500.0)
            for _ in range(timestep_count)
        ]
        recent_observations = {Cameras.DEPTH.value: depth_images}

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        # (T, C=1, H, W)
        assert result[Cameras.DEPTH.value].shape == (
            timestep_count,
            1,
            target_height,
            target_width,
        )
        # No RGB keys should be present
        assert len(result) == 1

    def test_missing_camera_key_raises(
        self,
        preprocessor_factory,
    ):
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
        )
        with pytest.raises(
            ValueError,
            match=f"Missing camera key '{Cameras.LEFT.value}' in the server observation data.",
        ):
            preprocessor.transform_camera_observations(recent_observations={})

    def test_resize_applied_to_images(
        self,
        preprocessor_factory,
        rgb_image_factory,
    ):
        target_height = 4
        target_width = 4
        preprocessor = preprocessor_factory(
            camera_keys=[Cameras.LEFT.value],
            image_height=target_height,
            image_width=target_width,
        )
        # Input is 16x16, should be resized to 4x4
        image = rgb_image_factory(height=16, width=16)
        recent_observations = {Cameras.LEFT.value: [image]}

        result = preprocessor.transform_camera_observations(
            recent_observations=recent_observations
        )

        assert result[Cameras.LEFT.value].shape[2] == target_height
        assert result[Cameras.LEFT.value].shape[3] == target_width


@pytest.mark.unit
class TestNormalizeImageTensor:
    def test_uint8_divided_by_255(self):
        image = torch.tensor([[[0, 128, 255]]], dtype=torch.uint8)

        result = ObservationPreprocessor._normalize_image_tensor(image=image)

        assert result.dtype == torch.float32
        torch.testing.assert_close(
            result,
            torch.tensor([[[0.0, 128.0 / 255.0, 1.0]]]),
        )

    def test_float_above_one_divided_by_255_with_warning(self, caplog):
        image = torch.tensor([[[0.0, 128.0, 255.0]]])

        with caplog.at_level(logging.WARNING):
            result = ObservationPreprocessor._normalize_image_tensor(image=image)

        torch.testing.assert_close(
            result,
            torch.tensor([[[0.0, 128.0 / 255.0, 1.0]]]),
        )
        assert "max" in caplog.text
        assert "dividing by 255" in caplog.text

    def test_float_in_zero_one_range_passthrough_with_warning(self, caplog):
        image = torch.tensor([[[0.0, 0.5, 1.0]]])

        with caplog.at_level(logging.WARNING):
            result = ObservationPreprocessor._normalize_image_tensor(image=image)

        torch.testing.assert_close(result, image)
        assert "already in [0, 1] range" in caplog.text

    def test_all_zero_float_image_passthrough(self, caplog):
        image = torch.zeros(1, 3, 4, 4)

        with caplog.at_level(logging.WARNING):
            result = ObservationPreprocessor._normalize_image_tensor(image=image)

        torch.testing.assert_close(result, image)
