"""Dataset-based explanation source."""

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import default_collate

from versatil.data.constants import SampleKey
from versatil.data.dataloader import _ensure_zarr_exists
from versatil.data.episodic_dataset import EpisodicDataset
from versatil.explainability.constants import (
    ExplanationDatasetSplit,
    ExplanationSourceType,
)
from versatil.explainability.sources.schema_paths import (
    resolve_dataset_schema_for_explanation,
)
from versatil.explainability.sources.typedefs import (
    DatasetRunnerConfig,
    ExplanationBatch,
)
from versatil.explainability.typedefs import ObservationBatch
from versatil.models.policy import Policy


class DatasetExplanationSource:
    """Yield explanation batches from versatil schema-based zarr data.

    Note:
        Missing zarr data is created from raw data described by the
        dataset schema. Override inputs must use the same schema type as the
        checkpoint configuration.
    """

    def __init__(
        self,
        config: DatasetRunnerConfig,
        policy: Policy,
        split: str,
        batch_size: int,
        sample_stride: int,
        max_samples: int | None,
        data_path_override: str | list[str] | None = None,
        zarr_cache_directory: Path | None = None,
    ) -> None:
        """Initialize the dataset-backed source.

        Args:
            config: Instantiated training configuration loaded from the
                checkpoint.
            policy: Loaded policy whose normalizer should be reused by the
                dataset sample builder.
            split: Dataset split to explain: ``train``, ``val``, or ``all``.
            batch_size: Number of sampled windows per explanation batch.
            sample_stride: Keep every Nth sample in deterministic dataset order.
            max_samples: Optional cap on the number of sampled windows.
            data_path_override: Optional offline input location to explain
                instead of the training data path stored in the checkpoint task config.
                - ``None`` uses the training data as-is.
                - A single ``.zarr``path is used directly
                - A non-zarr path has to be raw data in the same dataset schema as the checkpoint.
                - A list is only for raw schemas that already accept multiple inputs, such as CSV
                  folders or HDF5 files.
            zarr_cache_directory: Directory used to store zarr data created from
                raw ``data_path_override`` inputs. When ``None`` and
                ``data_path_override`` is set, the zarr is written beside the
                override path. When both are ``None``, the checkpoint schema
                zarr directory is used.

        Raises:
            ValueError: If ``split``, ``batch_size``, ``sample_stride``, or
                ``max_samples`` is invalid.
        """
        valid_splits = [member.value for member in ExplanationDatasetSplit]
        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}. Got: {split}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive. Got: {batch_size}")
        if sample_stride <= 0:
            raise ValueError(f"sample_stride must be positive. Got: {sample_stride}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError(
                f"max_samples must be positive when set. Got: {max_samples}"
            )

        self.config = config
        self.policy = policy
        self.split = split
        self.batch_size = batch_size
        self.sample_stride = sample_stride
        self.max_samples = max_samples
        self.data_path_override = data_path_override
        self.zarr_cache_directory = self._resolve_zarr_cache_directory(
            data_path_override=data_path_override,
            zarr_cache_directory=zarr_cache_directory,
        )
        self.dataset_schema = resolve_dataset_schema_for_explanation(
            schema=self.config.task.dataset_schema,
            data_path_override=data_path_override,
            zarr_cache_directory=self.zarr_cache_directory,
        )
        self.dataset = self._build_dataset()

    def _resolve_zarr_cache_directory(
        self,
        data_path_override: str | list[str] | None,
        zarr_cache_directory: Path | None,
    ) -> Path:
        """Resolve where raw offline data overrides are converted to zarr.

        Args:
            data_path_override: Optional raw or zarr input path for the offline
                explanation run.
            zarr_cache_directory: Optional explicit cache directory from the
                explainability runner.

        Returns:
            Directory path used by the schema path resolver.
        """
        if zarr_cache_directory is not None:
            return zarr_cache_directory
        if data_path_override is not None:
            override_parent = self._resolve_data_path_override_parent(
                data_path_override=data_path_override
            )
            if override_parent is not None:
                return override_parent
        return Path(self.config.task.dataset_schema.zarr_path).expanduser().parent

    @staticmethod
    def _resolve_data_path_override_parent(
        data_path_override: str | list[str],
    ) -> Path | None:
        """Return the parent directory of the first configured override path."""
        if isinstance(data_path_override, str):
            raw_paths = [data_path_override]
        else:
            raw_paths = data_path_override
        for raw_path in raw_paths:
            if raw_path:
                return Path(raw_path).expanduser().parent
        return None

    def _build_dataset(self) -> EpisodicDataset:
        """Build an evaluation-mode dataset for the requested split.

        Returns:
            Dataset with checkpoint normalizer/tokenizer attached.

        Note:
            The ``all`` split is implemented by setting ``val_ratio=0`` on a
            copied dataloader config and using the training side of the
            splitter. This preserves ``total_ratio`` and all other sampling
            settings while selecting every eligible episode in the configured
            subset.
        """
        dataloader_config = deepcopy(self.config.task.dataloader)
        if self.split == ExplanationDatasetSplit.ALL.value:
            dataloader_config.val_ratio = 0.0
        train_split = self.split != ExplanationDatasetSplit.VAL.value
        _ensure_zarr_exists(
            schema=self.dataset_schema,
            preload_in_memory=dataloader_config.preload_data_in_memory,
        )
        dataset = EpisodicDataset(
            zarr_path=self.dataset_schema.zarr_path,
            pred_horizon=self.config.task.prediction_horizon,
            obs_horizon=self.config.task.observation_horizon,
            dataloader_config=dataloader_config,
            train=train_split,
            seed=self.config.experiment.seed,
            action_space=self.config.task.action_space,
            observation_space=self.config.task.observation_space,
            augment_images=False,
        )
        dataset.set_normalizer(normalizer=self.policy.normalizer)
        dataset.set_tokenizer(tokenizer=self.policy.tokenizer)
        return dataset

    def __iter__(self) -> Iterator[ExplanationBatch]:
        """Yield deterministic explanation batches.

        Returns:
            Iterator of model-ready batches. Dataset samples are already
            normalized/tokenized, so ``preprocess_observation`` is ``False``.
        """
        selected_indices = list(range(0, len(self.dataset), self.sample_stride))
        if self.max_samples is not None:
            selected_indices = selected_indices[: self.max_samples]

        for batch_start in range(0, len(selected_indices), self.batch_size):
            sample_indices = selected_indices[
                batch_start : batch_start + self.batch_size
            ]
            samples = [self.dataset[index] for index in sample_indices]
            sample_batch = default_collate(samples)
            observation = sample_batch[SampleKey.OBSERVATION.value]
            actions = sample_batch.get(SampleKey.ACTION.value)
            yield ExplanationBatch(
                observation=observation,
                actions=actions,
                display_observation=self._extract_display_observation(
                    observation=observation
                ),
                metadata={
                    "source": ExplanationSourceType.DATASET.value,
                    "split": self.split,
                    "sample_indices": sample_indices,
                    "zarr_path": self.dataset_schema.zarr_path,
                },
                preprocess_observation=False,
            )

    def _extract_display_observation(
        self,
        observation: ObservationBatch,
    ) -> dict[str, torch.Tensor]:
        """Extract displayable camera tensors from a dataset observation batch.

        Args:
            observation: Batched observation dictionary from ``EpisodicDataset``.

        Returns:
            Camera tensors keyed by observation name.
        """
        display_observation = {}
        for camera_key in self.config.task.observation_space.cameras:
            value = observation.get(camera_key)
            if isinstance(value, torch.Tensor):
                display_observation[camera_key] = value.detach().cpu()
        return display_observation
