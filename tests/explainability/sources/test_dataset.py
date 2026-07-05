"""Tests for versatil.explainability.sources.dataset module."""

import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.data.constants import Cameras, SampleKey
from versatil.data.dataloader import _ensure_zarr_exists
from versatil.data.raw.schemas.base import DatasetSchema
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.explainability.constants import (
    ExplanationDatasetSplit,
    ExplanationSourceType,
)
from versatil.explainability.sources.dataset import DatasetExplanationSource
from versatil.explainability.sources.schema_paths import OFFLINE_DATASET_ZARR_NAME
from versatil.explainability.sources.typedefs import ExplanationBatch


@pytest.fixture
def episodic_dataset_factory() -> Callable[..., MagicMock]:
    def factory(length: int = 6) -> MagicMock:
        dataset = MagicMock()
        dataset.__len__.return_value = length
        dataset.__getitem__.side_effect = _build_dataset_sample
        return dataset

    return factory


def _build_dataset_sample(index: int) -> dict[str, dict[str, torch.Tensor]]:
    return {
        SampleKey.OBSERVATION.value: {
            Cameras.AGENTVIEW.value: torch.full(
                (1, 3, 4, 4),
                fill_value=float(index),
            )
        },
        SampleKey.ACTION.value: {
            SampleKey.IS_PAD_ACTION.value: torch.zeros(2, dtype=torch.bool)
        },
    }


@pytest.fixture
def source_config_factory() -> Callable[..., MagicMock]:
    def factory(
        split_val_ratio: float = 0.25,
        schema: MagicMock | None = None,
    ) -> MagicMock:
        if schema is None:
            schema = MagicMock()
            schema.zarr_path = "/tmp/fake.zarr"
        dataloader_config = DataLoaderConfig()
        dataloader_config.val_ratio = split_val_ratio

        observation_space = MagicMock()
        observation_space.cameras = {Cameras.AGENTVIEW.value: MagicMock()}

        task = MagicMock()
        task.dataset_schema = schema
        task.dataloader = dataloader_config
        task.prediction_horizon = 2
        task.observation_horizon = 1
        task.action_space = MagicMock()
        task.observation_space = observation_space

        experiment = MagicMock()
        experiment.seed = 123

        config = MagicMock()
        config.task = task
        config.experiment = experiment
        return config

    return factory


@pytest.fixture
def policy_mock() -> MagicMock:
    policy = MagicMock()
    policy.normalizer = MagicMock()
    policy.tokenizer = MagicMock()
    return policy


class TestDatasetExplanationSource:
    def test_builds_all_split_dataset_without_augmentation(
        self,
        source_config_factory: Callable[..., MagicMock],
        policy_mock: MagicMock,
        episodic_dataset_factory: Callable[..., MagicMock],
    ) -> None:
        dataset = episodic_dataset_factory()
        config = source_config_factory(split_val_ratio=0.3)

        with (
            patch("versatil.explainability.sources.dataset._ensure_zarr_exists"),
            patch(
                "versatil.explainability.sources.dataset.EpisodicDataset",
                return_value=dataset,
            ) as mock_dataset_class,
        ):
            DatasetExplanationSource(
                config=config,
                policy=policy_mock,
                split=ExplanationDatasetSplit.ALL.value,
                batch_size=2,
                sample_stride=2,
                max_samples=3,
            )

        mock_dataset_class.assert_called_once()
        call_kwargs = mock_dataset_class.call_args.kwargs
        assert call_kwargs["train"] is True
        assert call_kwargs["augment_images"] is False
        assert call_kwargs["dataloader_config"].val_ratio == 0.0
        dataset.set_normalizer.assert_called_once_with(
            normalizer=policy_mock.normalizer
        )
        dataset.set_tokenizer.assert_called_once_with(tokenizer=policy_mock.tokenizer)

    def test_data_path_override_converts_raw_data_into_output_zarr(
        self,
        source_config_factory: Callable[..., MagicMock],
        policy_mock: MagicMock,
        episodic_dataset_factory: Callable[..., MagicMock],
        csv_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        dataset = episodic_dataset_factory()
        raw_path = tmp_path / "inference_raw"
        raw_path.mkdir()
        cache_directory = tmp_path / "explain_output"
        schema = csv_schema_factory()
        config = source_config_factory(schema=schema)

        with (
            patch(
                "versatil.explainability.sources.dataset._ensure_zarr_exists"
            ) as mock_ensure_zarr,
            patch(
                "versatil.explainability.sources.dataset.EpisodicDataset",
                return_value=dataset,
            ) as mock_dataset_class,
        ):
            DatasetExplanationSource(
                config=config,
                policy=policy_mock,
                split=ExplanationDatasetSplit.TRAIN.value,
                batch_size=1,
                sample_stride=1,
                max_samples=None,
                data_path_override=str(raw_path),
                zarr_cache_directory=cache_directory,
            )

        resolved_schema = mock_ensure_zarr.call_args.kwargs["schema"]
        expected_zarr_path = str(cache_directory / "offline_dataset.zarr")
        assert resolved_schema is not schema
        assert resolved_schema.dataset_folders == [str(raw_path)]
        assert resolved_schema.zarr_path == expected_zarr_path
        assert schema.dataset_folders == ["/tmp/training_raw"]
        assert mock_dataset_class.call_args.kwargs["zarr_path"] == expected_zarr_path

    def test_data_path_override_defaults_zarr_cache_to_override_parent(
        self,
        source_config_factory: Callable[..., MagicMock],
        policy_mock: MagicMock,
        episodic_dataset_factory: Callable[..., MagicMock],
        csv_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        dataset = episodic_dataset_factory()
        training_directory = tmp_path / "training"
        training_directory.mkdir()
        override_parent = tmp_path / "inference"
        override_parent.mkdir()
        raw_path = override_parent / "raw"
        raw_path.mkdir()
        schema = csv_schema_factory(
            dataset_folders=[str(training_directory / "raw")],
            zarr_path=str(training_directory / "training.zarr"),
        )
        config = source_config_factory(schema=schema)

        with (
            patch(
                "versatil.explainability.sources.dataset._ensure_zarr_exists"
            ) as mock_ensure_zarr,
            patch(
                "versatil.explainability.sources.dataset.EpisodicDataset",
                return_value=dataset,
            ) as mock_dataset_class,
        ):
            DatasetExplanationSource(
                config=config,
                policy=policy_mock,
                split=ExplanationDatasetSplit.TRAIN.value,
                batch_size=1,
                sample_stride=1,
                max_samples=None,
                data_path_override=str(raw_path),
            )

        expected_zarr_path = str(override_parent / "offline_dataset.zarr")
        resolved_schema = mock_ensure_zarr.call_args.kwargs["schema"]
        assert resolved_schema.zarr_path == expected_zarr_path
        assert resolved_schema.dataset_folders == [str(raw_path)]
        assert mock_dataset_class.call_args.kwargs["zarr_path"] == expected_zarr_path

    def test_data_path_override_uses_existing_zarr_directly(
        self,
        source_config_factory: Callable[..., MagicMock],
        policy_mock: MagicMock,
        episodic_dataset_factory: Callable[..., MagicMock],
        csv_schema_factory: Callable[..., MagicMock],
        tmp_path: Path,
    ) -> None:
        dataset = episodic_dataset_factory()
        zarr_path = tmp_path / "inference.zarr"
        zarr_path.mkdir()
        schema = csv_schema_factory()
        config = source_config_factory(schema=schema)

        with (
            patch(
                "versatil.explainability.sources.dataset._ensure_zarr_exists"
            ) as mock_ensure_zarr,
            patch(
                "versatil.explainability.sources.dataset.EpisodicDataset",
                return_value=dataset,
            ) as mock_dataset_class,
        ):
            DatasetExplanationSource(
                config=config,
                policy=policy_mock,
                split=ExplanationDatasetSplit.TRAIN.value,
                batch_size=1,
                sample_stride=1,
                max_samples=None,
                data_path_override=str(zarr_path),
                zarr_cache_directory=tmp_path / "explain_output",
            )

        resolved_schema = mock_ensure_zarr.call_args.kwargs["schema"]
        assert resolved_schema is not schema
        assert resolved_schema.dataset_folders == ["/tmp/training_raw"]
        assert resolved_schema.zarr_path == str(zarr_path)
        assert mock_dataset_class.call_args.kwargs["zarr_path"] == str(zarr_path)

    def test_yields_strided_batches_with_metadata(
        self,
        source_config_factory: Callable[..., MagicMock],
        policy_mock: MagicMock,
        episodic_dataset_factory: Callable[..., MagicMock],
    ) -> None:
        dataset = episodic_dataset_factory(length=6)
        config = source_config_factory()

        with (
            patch("versatil.explainability.sources.dataset._ensure_zarr_exists"),
            patch(
                "versatil.explainability.sources.dataset.EpisodicDataset",
                return_value=dataset,
            ),
        ):
            source = DatasetExplanationSource(
                config=config,
                policy=policy_mock,
                split=ExplanationDatasetSplit.TRAIN.value,
                batch_size=2,
                sample_stride=2,
                max_samples=3,
            )
            batches = list(source)

        assert len(batches) == 2
        assert batches[0].metadata["source"] == ExplanationSourceType.DATASET.value
        assert batches[0].metadata["sample_indices"] == [0, 2]
        assert batches[1].metadata["sample_indices"] == [4]
        assert batches[0].preprocess_observation is False
        assert batches[0].actions is not None
        assert SampleKey.IS_PAD_ACTION.value in batches[0].actions
        assert Cameras.AGENTVIEW.value in batches[0].display_observation
        assert batches[0].observation[Cameras.AGENTVIEW.value].shape == (2, 1, 3, 4, 4)

    @pytest.mark.parametrize(
        "split, batch_size, sample_stride, max_samples, error_message",
        [
            (
                "bad",
                1,
                1,
                None,
                "split must be one of ['train', 'val', 'all']. Got: bad",
            ),
            ("train", 0, 1, None, "batch_size must be positive. Got: 0"),
            ("train", 1, 0, None, "sample_stride must be positive. Got: 0"),
            (
                "train",
                1,
                1,
                0,
                "max_samples must be positive when set. Got: 0",
            ),
        ],
    )
    def test_validates_configuration(
        self,
        source_config_factory: Callable[..., MagicMock],
        policy_mock: MagicMock,
        split: str,
        batch_size: int,
        sample_stride: int,
        max_samples: int | None,
        error_message: str,
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(error_message)):
            DatasetExplanationSource(
                config=source_config_factory(),
                policy=policy_mock,
                split=split,
                batch_size=batch_size,
                sample_stride=sample_stride,
                max_samples=max_samples,
            )


@pytest.mark.integration
class TestDatasetExplanationSourceIntegration:
    @pytest.mark.parametrize(
        "schema_name",
        ["tso_csv", "libero_hdf5", "lerobot", "synthetic"],
    )
    def test_no_data_path_override_samples_training_data_for_true_schema(
        self,
        schema_name: str,
        explanation_schema_case_factory: Callable[
            ...,
            tuple[
                DatasetSchema,
                str | list[str],
                ActionSpace,
                ObservationSpace,
                list[str],
                str,
            ],
        ],
        explanation_source_config_factory: Callable[..., MagicMock],
        explanation_policy_mock: MagicMock,
    ) -> None:
        (
            schema,
            _data_path_override,
            action_space,
            observation_space,
            camera_keys,
            action_key,
        ) = explanation_schema_case_factory(schema_name=schema_name)

        source = _build_source(
            schema=schema,
            action_space=action_space,
            observation_space=observation_space,
            explanation_source_config_factory=explanation_source_config_factory,
            explanation_policy_mock=explanation_policy_mock,
            data_path_override=None,
        )
        batch = next(iter(source))

        expected_zarr_path = Path(schema.zarr_path)
        _assert_batch_samples_source(
            batch=batch,
            expected_zarr_path=expected_zarr_path,
            expected_sample_indices=[0],
            camera_keys=camera_keys,
            action_key=action_key,
        )
        _assert_first_sample_contains_action(source=source, action_key=action_key)

    @pytest.mark.parametrize(
        "schema_name",
        ["tso_csv", "libero_hdf5", "lerobot"],
    )
    def test_raw_data_path_override_creates_offline_zarr_for_true_schema(
        self,
        schema_name: str,
        explanation_schema_case_factory: Callable[
            ...,
            tuple[
                DatasetSchema,
                str | list[str],
                ActionSpace,
                ObservationSpace,
                list[str],
                str,
            ],
        ],
        explanation_source_config_factory: Callable[..., MagicMock],
        explanation_policy_mock: MagicMock,
    ) -> None:
        (
            schema,
            data_path_override,
            action_space,
            observation_space,
            camera_keys,
            action_key,
        ) = explanation_schema_case_factory(schema_name=schema_name)

        source = _build_source(
            schema=schema,
            action_space=action_space,
            observation_space=observation_space,
            explanation_source_config_factory=explanation_source_config_factory,
            explanation_policy_mock=explanation_policy_mock,
            data_path_override=data_path_override,
        )
        batch = next(iter(source))

        if not isinstance(data_path_override, str):
            raise TypeError("Single-path raw override test expected a string override.")
        expected_zarr_path = Path(data_path_override).parent / OFFLINE_DATASET_ZARR_NAME
        _assert_batch_samples_source(
            batch=batch,
            expected_zarr_path=expected_zarr_path,
            expected_sample_indices=[0],
            camera_keys=camera_keys,
            action_key=action_key,
        )
        _assert_first_sample_contains_action(source=source, action_key=action_key)

    @pytest.mark.parametrize(
        "schema_name",
        ["tso_csv", "libero_hdf5"],
    )
    def test_raw_data_path_override_accepts_multiple_paths_for_true_schema(
        self,
        schema_name: str,
        explanation_schema_case_factory: Callable[
            ...,
            tuple[
                DatasetSchema,
                str | list[str],
                ActionSpace,
                ObservationSpace,
                list[str],
                str,
            ],
        ],
        explanation_source_config_factory: Callable[..., MagicMock],
        explanation_policy_mock: MagicMock,
    ) -> None:
        (
            schema,
            data_path_override,
            action_space,
            observation_space,
            camera_keys,
            action_key,
        ) = explanation_schema_case_factory(
            schema_name=schema_name,
            raw_path_count=2,
        )

        source = _build_source(
            schema=schema,
            action_space=action_space,
            observation_space=observation_space,
            explanation_source_config_factory=explanation_source_config_factory,
            explanation_policy_mock=explanation_policy_mock,
            data_path_override=data_path_override,
            batch_size=2,
            max_samples=2,
        )
        batch = next(iter(source))

        if not isinstance(data_path_override, list):
            raise TypeError("Multi-path raw override test expected a list override.")
        expected_zarr_path = (
            Path(data_path_override[0]).parent / OFFLINE_DATASET_ZARR_NAME
        )
        _assert_batch_samples_source(
            batch=batch,
            expected_zarr_path=expected_zarr_path,
            expected_sample_indices=[0, 1],
            camera_keys=camera_keys,
            action_key=action_key,
        )
        _assert_first_sample_contains_action(source=source, action_key=action_key)
        assert source.dataset.replay_buffer.n_episodes == 2

    @pytest.mark.parametrize(
        "schema_name",
        ["tso_csv", "libero_hdf5", "lerobot", "synthetic"],
    )
    def test_zarr_data_path_override_samples_existing_zarr_for_true_schema(
        self,
        schema_name: str,
        explanation_schema_case_factory: Callable[
            ...,
            tuple[
                DatasetSchema,
                str | list[str],
                ActionSpace,
                ObservationSpace,
                list[str],
                str,
            ],
        ],
        explanation_source_config_factory: Callable[..., MagicMock],
        explanation_policy_mock: MagicMock,
    ) -> None:
        (
            schema,
            _data_path_override,
            action_space,
            observation_space,
            camera_keys,
            action_key,
        ) = explanation_schema_case_factory(schema_name=schema_name)
        _ensure_zarr_exists(schema=schema)

        source = _build_source(
            schema=schema,
            action_space=action_space,
            observation_space=observation_space,
            explanation_source_config_factory=explanation_source_config_factory,
            explanation_policy_mock=explanation_policy_mock,
            data_path_override=schema.zarr_path,
        )
        batch = next(iter(source))

        expected_zarr_path = Path(schema.zarr_path)
        _assert_batch_samples_source(
            batch=batch,
            expected_zarr_path=expected_zarr_path,
            expected_sample_indices=[0],
            camera_keys=camera_keys,
            action_key=action_key,
        )
        _assert_first_sample_contains_action(source=source, action_key=action_key)

    def test_raw_data_path_override_rejects_synthetic_schema(
        self,
        explanation_schema_case_factory: Callable[
            ...,
            tuple[
                DatasetSchema,
                str | list[str],
                ActionSpace,
                ObservationSpace,
                list[str],
                str,
            ],
        ],
        explanation_source_config_factory: Callable[..., MagicMock],
        explanation_policy_mock: MagicMock,
    ) -> None:
        (
            schema,
            data_path_override,
            action_space,
            observation_space,
            _camera_keys,
            _action_key,
        ) = explanation_schema_case_factory(schema_name="synthetic")
        expected_message = (
            "data_path_override cannot point to raw files for SyntheticSchema. "
            "Pass an existing .zarr path instead."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            _build_source(
                schema=schema,
                action_space=action_space,
                observation_space=observation_space,
                explanation_source_config_factory=explanation_source_config_factory,
                explanation_policy_mock=explanation_policy_mock,
                data_path_override=data_path_override,
            )


def _build_source(
    schema: DatasetSchema,
    action_space: ActionSpace,
    observation_space: ObservationSpace,
    explanation_source_config_factory: Callable[..., MagicMock],
    explanation_policy_mock: MagicMock,
    data_path_override: str | list[str] | None,
    batch_size: int = 1,
    sample_stride: int = 1,
    max_samples: int = 1,
) -> DatasetExplanationSource:
    config = explanation_source_config_factory(
        schema=schema,
        action_space=action_space,
        observation_space=observation_space,
    )
    return DatasetExplanationSource(
        config=config,
        policy=explanation_policy_mock,
        split=ExplanationDatasetSplit.ALL.value,
        batch_size=batch_size,
        sample_stride=sample_stride,
        max_samples=max_samples,
        data_path_override=data_path_override,
    )


def _assert_batch_samples_source(
    batch: ExplanationBatch,
    expected_zarr_path: Path,
    expected_sample_indices: list[int],
    camera_keys: list[str],
    action_key: str,
) -> None:
    assert batch.metadata["source"] == ExplanationSourceType.DATASET.value
    assert batch.metadata["split"] == ExplanationDatasetSplit.ALL.value
    assert batch.metadata["sample_indices"] == expected_sample_indices
    assert Path(batch.metadata["zarr_path"]) == expected_zarr_path
    assert expected_zarr_path.exists()
    assert batch.preprocess_observation is False
    assert batch.actions is not None
    assert action_key in batch.actions

    for camera_key in camera_keys:
        observation = batch.observation[camera_key]
        display_observation = batch.display_observation[camera_key]
        assert observation.shape == display_observation.shape
        assert display_observation.ndim == 5
        assert display_observation.shape[0] == len(expected_sample_indices)
        assert display_observation.shape[1] == 1
        assert display_observation.shape[2] == 3
        assert isinstance(display_observation, torch.Tensor)


def _assert_first_sample_contains_action(
    source: DatasetExplanationSource,
    action_key: str,
) -> None:
    sample = source.dataset[0]
    actions = sample[SampleKey.ACTION.value]
    assert action_key in actions
    assert actions[action_key].shape[-1] > 0
