"""Tests for versatil.data.preprocessing.create_zarr_arrays module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import zarr

from versatil.data.preprocessing.codecs import WebPCodec
from versatil.data.preprocessing.create_zarr_arrays import (
    create_zarr_arrays,
    create_zarr_replay_buffer,
    is_uint8_image_spec,
)


class TestIsUint8ImageSpec:
    @pytest.mark.parametrize(
        "shape, dtype, expected",
        [
            ((0, 64, 64, 3), "uint8", True),
            ((0, 7), "float32", False),
            ((0, 64, 64, 3), "float32", False),
            ((0,), "str", False),
            ((0, 7), "uint8", False),
        ],
        ids=[
            "4d_uint8_image",
            "2d_float32_numerical",
            "4d_float32_not_uint8",
            "1d_string",
            "2d_uint8_not_image",
        ],
    )
    def test_returns_expected_for_various_specs(
        self,
        spec_factory: Callable[..., dict],
        shape: tuple,
        dtype: str,
        expected: bool,
    ):
        spec = spec_factory(shape=shape, dtype=dtype)

        assert is_uint8_image_spec(spec) is expected


class TestCreateZarrArrays:
    def test_image_array_uses_webp_serializer_with_single_frame_chunks(
        self,
        mock_schema_factory: Callable[..., MagicMock],
        spec_factory: Callable[..., dict],
    ):
        image_spec = spec_factory(
            shape=(0, 64, 64, 3),
            chunks=(16, 64, 64, 3),
            dtype="uint8",
        )
        schema = mock_schema_factory(specs={"left": image_spec})
        data_group = MagicMock()
        image_codec = WebPCodec(level=99)
        numeric_compressor = MagicMock()

        create_zarr_arrays(
            data_group=data_group,
            schema=schema,
            image_codec=image_codec,
            numeric_compressor=numeric_compressor,
        )

        data_group.create_array.assert_called_once_with(
            name="left",
            shape=(0, 64, 64, 3),
            chunks=(1, 64, 64, 3),
            dtype=np.uint8,
            serializer=image_codec,
            compressors=None,
        )

    def test_numerical_array_with_compressor_uses_blosc(
        self,
        mock_schema_factory: Callable[..., MagicMock],
        spec_factory: Callable[..., dict],
    ):
        numerical_spec = spec_factory(
            shape=(0, 7),
            chunks=(256, 7),
            dtype="float32",
            needs_compressor=True,
        )
        schema = mock_schema_factory(specs={"proprio": numerical_spec})
        data_group = MagicMock()
        numeric_compressor = MagicMock()

        create_zarr_arrays(
            data_group=data_group,
            schema=schema,
            image_codec=MagicMock(),
            numeric_compressor=numeric_compressor,
        )

        data_group.create_array.assert_called_once_with(
            name="proprio",
            shape=(0, 7),
            chunks=(256, 7),
            dtype=np.float32,
            compressors=[numeric_compressor],
        )

    def test_string_dtype_resolves_to_python_str_with_no_compressor(
        self,
        mock_schema_factory: Callable[..., MagicMock],
        spec_factory: Callable[..., dict],
    ):
        string_spec = spec_factory(
            shape=(0,),
            chunks=(100,),
            dtype="str",
            needs_compressor=False,
        )
        schema = mock_schema_factory(specs={"language": string_spec})
        data_group = MagicMock()

        create_zarr_arrays(
            data_group=data_group,
            schema=schema,
            image_codec=MagicMock(),
            numeric_compressor=MagicMock(),
        )

        data_group.create_array.assert_called_once_with(
            name="language",
            shape=(0,),
            chunks=(100,),
            dtype=str,
            compressors=None,
        )

    def test_non_image_non_string_without_compressor_passes_none(
        self,
        mock_schema_factory: Callable[..., MagicMock],
        spec_factory: Callable[..., dict],
    ):
        spec = spec_factory(
            shape=(0, 3),
            chunks=(100, 3),
            dtype="float32",
            needs_compressor=False,
        )
        schema = mock_schema_factory(specs={"metadata_field": spec})
        data_group = MagicMock()

        create_zarr_arrays(
            data_group=data_group,
            schema=schema,
            image_codec=MagicMock(),
            numeric_compressor=MagicMock(),
        )

        data_group.create_array.assert_called_once_with(
            name="metadata_field",
            shape=(0, 3),
            chunks=(100, 3),
            dtype=np.float32,
            compressors=None,
        )

    def test_mixed_specs_create_one_array_per_key(
        self,
        mock_schema_factory: Callable[..., MagicMock],
        spec_factory: Callable[..., dict],
    ):
        schema = mock_schema_factory(
            specs={
                "left": spec_factory(
                    shape=(0, 64, 64, 3),
                    chunks=(16, 64, 64, 3),
                    dtype="uint8",
                ),
                "proprio": spec_factory(
                    shape=(0, 7),
                    chunks=(256, 7),
                    dtype="float32",
                ),
                "language": spec_factory(
                    shape=(0,),
                    chunks=(100,),
                    dtype="str",
                    needs_compressor=False,
                ),
            }
        )
        data_group = MagicMock()

        create_zarr_arrays(
            data_group=data_group,
            schema=schema,
            image_codec=MagicMock(),
            numeric_compressor=MagicMock(),
        )

        assert data_group.create_array.call_count == 3


class TestCreateZarrReplayBuffer:
    @pytest.fixture
    def position_specs(self, spec_factory: Callable[..., dict]) -> dict:
        return {
            "position": spec_factory(
                shape=(0, 3),
                chunks=(256, 3),
                dtype="float32",
                needs_compressor=True,
            ),
        }

    def test_creates_data_and_meta_groups(
        self,
        tmp_path,
        mock_schema_factory: Callable[..., MagicMock],
        position_specs: dict,
        rng: np.random.Generator,
    ):
        zarr_path = str(tmp_path / "test.zarr")
        schema = mock_schema_factory(specs=position_specs, zarr_path=zarr_path)

        create_zarr_replay_buffer(
            schema=schema,
            episodes=[{"position": rng.standard_normal((5, 3)).astype(np.float32)}],
            total_episodes=1,
        )

        root = zarr.open_group(zarr_path, mode="r")
        assert "data" in root
        assert "meta" in root

    def test_episode_ends_accumulate_correctly(
        self,
        tmp_path,
        mock_schema_factory: Callable[..., MagicMock],
        position_specs: dict,
        rng: np.random.Generator,
    ):
        zarr_path = str(tmp_path / "test.zarr")
        schema = mock_schema_factory(specs=position_specs, zarr_path=zarr_path)

        create_zarr_replay_buffer(
            schema=schema,
            episodes=[
                {"position": rng.standard_normal((5, 3)).astype(np.float32)},
                {"position": rng.standard_normal((8, 3)).astype(np.float32)},
                {"position": rng.standard_normal((3, 3)).astype(np.float32)},
            ],
            total_episodes=3,
        )

        root = zarr.open_group(zarr_path, mode="r")
        np.testing.assert_array_equal(root["meta"]["episode_ends"][:], [5, 13, 16])

    def test_data_appended_to_correct_array(
        self,
        tmp_path,
        mock_schema_factory: Callable[..., MagicMock],
        position_specs: dict,
        rng: np.random.Generator,
    ):
        zarr_path = str(tmp_path / "test.zarr")
        schema = mock_schema_factory(specs=position_specs, zarr_path=zarr_path)
        episode_data = rng.standard_normal((5, 3)).astype(np.float32)

        create_zarr_replay_buffer(
            schema=schema,
            episodes=[{"position": episode_data}],
            total_episodes=1,
        )

        root = zarr.open_group(zarr_path, mode="r")
        np.testing.assert_array_almost_equal(root["data"]["position"][:], episode_data)

    def test_progress_logged_at_multiples_of_fifty(
        self,
        tmp_path,
        mock_schema_factory: Callable[..., MagicMock],
        position_specs: dict,
        rng: np.random.Generator,
    ):
        zarr_path = str(tmp_path / "test.zarr")
        schema = mock_schema_factory(specs=position_specs, zarr_path=zarr_path)
        episodes = [
            {"position": rng.standard_normal((2, 3)).astype(np.float32)}
            for _ in range(51)
        ]

        with patch(
            "versatil.data.preprocessing.create_zarr_arrays.logging"
        ) as mock_logging:
            create_zarr_replay_buffer(
                schema=schema,
                episodes=episodes,
                total_episodes=51,
            )

            progress_calls = [
                c
                for c in mock_logging.info.call_args_list
                if "Processing episode" in str(c)
            ]
            # Episodes at index 0 and 50 are multiples of 50
            assert len(progress_calls) == 2

    def test_no_total_episodes_skips_progress_logging(
        self,
        tmp_path,
        mock_schema_factory: Callable[..., MagicMock],
        position_specs: dict,
        rng: np.random.Generator,
    ):
        zarr_path = str(tmp_path / "test.zarr")
        schema = mock_schema_factory(specs=position_specs, zarr_path=zarr_path)

        with patch(
            "versatil.data.preprocessing.create_zarr_arrays.logging"
        ) as mock_logging:
            create_zarr_replay_buffer(
                schema=schema,
                episodes=[{"position": rng.standard_normal((5, 3)).astype(np.float32)}],
                total_episodes=None,
            )

            progress_calls = [
                c
                for c in mock_logging.info.call_args_list
                if "Processing episode" in str(c)
            ]
            assert len(progress_calls) == 0

    def test_empty_episodes_produces_empty_episode_ends(
        self,
        tmp_path,
        mock_schema_factory: Callable[..., MagicMock],
        position_specs: dict,
    ):
        zarr_path = str(tmp_path / "test.zarr")
        schema = mock_schema_factory(specs=position_specs, zarr_path=zarr_path)

        create_zarr_replay_buffer(
            schema=schema,
            episodes=[],
            total_episodes=0,
        )

        root = zarr.open_group(zarr_path, mode="r")
        assert root["meta"]["episode_ends"].shape == (0,)

    def test_initial_log_includes_zarr_path_and_schema_name(
        self,
        tmp_path,
        mock_schema_factory: Callable[..., MagicMock],
        position_specs: dict,
        rng: np.random.Generator,
    ):
        zarr_path = str(tmp_path / "test.zarr")
        schema = mock_schema_factory(specs=position_specs, zarr_path=zarr_path)

        with patch(
            "versatil.data.preprocessing.create_zarr_arrays.logging"
        ) as mock_logging:
            create_zarr_replay_buffer(
                schema=schema,
                episodes=[{"position": rng.standard_normal((5, 3)).astype(np.float32)}],
                total_episodes=1,
            )

            initial_calls = [
                c
                for c in mock_logging.info.call_args_list
                if "Creating Zarr dataset" in str(c)
            ]
            assert len(initial_calls) == 1
            assert zarr_path in str(initial_calls[0])
            assert "MockSchema" in str(initial_calls[0])

    def test_completion_log_includes_episode_and_step_counts(
        self,
        tmp_path,
        mock_schema_factory: Callable[..., MagicMock],
        position_specs: dict,
        rng: np.random.Generator,
    ):
        zarr_path = str(tmp_path / "test.zarr")
        schema = mock_schema_factory(specs=position_specs, zarr_path=zarr_path)

        with patch(
            "versatil.data.preprocessing.create_zarr_arrays.logging"
        ) as mock_logging:
            create_zarr_replay_buffer(
                schema=schema,
                episodes=[
                    {"position": rng.standard_normal((5, 3)).astype(np.float32)},
                    {"position": rng.standard_normal((3, 3)).astype(np.float32)},
                ],
                total_episodes=2,
            )

            final_calls = [
                c
                for c in mock_logging.info.call_args_list
                if "2 episodes" in str(c) and "8 total steps" in str(c)
            ]
            assert len(final_calls) == 1
