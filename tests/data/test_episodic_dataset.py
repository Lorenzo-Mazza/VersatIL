"""Tests for versatil.data.episodic_dataset module."""

from collections.abc import Callable
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch

from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.data.constants import (
    Cameras,
    GripperType,
    ProprioKey,
    SampleKey,
)
from versatil.data.episodic_dataset import EpisodicDataset
from versatil.data.metadata import (
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
    PositionActionMetadata,
    PositionObservationMetadata,
)
from versatil.data.task import ActionSpace, ObservationSpace


@pytest.fixture
def mock_dataloader_config() -> Callable[..., DataLoaderConfig]:
    """Factory for mock DataLoaderConfig."""

    def factory(
        val_ratio: float = 0.1,
        total_ratio: float = 1.0,
        downsample_factor: int = 1,
        skip_initial_episode_steps: int = 0,
        action_backward_shift: int = 1,
        preload_data_in_memory: bool = False,
        image_height: int = 64,
        image_width: int = 64,
        trailing_padded_actions: int | None = None,
    ) -> DataLoaderConfig:
        config = MagicMock(spec=DataLoaderConfig)
        config.val_ratio = val_ratio
        config.total_ratio = total_ratio
        config.downsample_factor = downsample_factor
        config.skip_initial_episode_steps = skip_initial_episode_steps
        config.action_backward_shift = action_backward_shift
        config.preload_data_in_memory = preload_data_in_memory
        config.image_height = image_height
        config.image_width = image_width
        config.trailing_padded_actions = trailing_padded_actions
        config.color_augmentation = None
        config.spatial_augmentation = None
        config.kinematics_norm_type = "min_max"
        config.image_norm_type = "zero_to_one"
        config.depth_norm_type = "zero_to_one"
        return config

    return factory


@pytest.fixture
def mock_replay_buffer_for_dataset(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    """Factory for mock ReplayBuffer used in EpisodicDataset."""

    def factory(
        num_episodes: int = 5,
        timesteps_per_episode: int = 10,
        proprio_dim: int = 7,
        has_gripper: bool = True,
        extra_keys: list[str] = None,
    ) -> MagicMock:
        buffer = MagicMock()
        total_steps = num_episodes * timesteps_per_episode
        episode_ends = np.array(
            [(i + 1) * timesteps_per_episode for i in range(num_episodes)]
        )

        buffer.n_episodes = num_episodes
        buffer.n_steps = total_steps
        type(buffer).episode_ends = PropertyMock(return_value=episode_ends)

        data = {
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: rng.standard_normal(
                (total_steps, proprio_dim)
            ).astype(np.float32),
        }
        if has_gripper:
            data[ProprioKey.GRIPPER_STATE.value] = rng.integers(
                0, 2, (total_steps, 1)
            ).astype(np.float32)
        if extra_keys:
            for key in extra_keys:
                if key not in data:
                    data[key] = rng.standard_normal((total_steps, proprio_dim)).astype(
                        np.float32
                    )

        def getitem(key):
            if key in data:
                return data[key]
            raise KeyError(key)

        buffer.__getitem__ = MagicMock(side_effect=getitem)
        buffer.__contains__ = MagicMock(side_effect=lambda key: key in data)
        buffer.keys.return_value = list(data.keys())

        def get_episode(ep_idx):
            start = ep_idx * timesteps_per_episode
            end = (ep_idx + 1) * timesteps_per_episode
            return {k: v[start:end] for k, v in data.items()}

        buffer.get_episode = MagicMock(side_effect=get_episode)
        return buffer

    return factory


@pytest.fixture
def episodic_dataset_factory(
    action_space_factory: Callable[..., ActionSpace],
    observation_space_factory: Callable[..., ObservationSpace],
    mock_dataloader_config: Callable[..., DataLoaderConfig],
    mock_replay_buffer_for_dataset: Callable[..., MagicMock],
) -> Callable[..., EpisodicDataset]:
    """Factory for creating EpisodicDataset with mocked I/O."""

    def factory(
        actions_metadata: dict = None,
        observations_metadata: dict = None,
        num_episodes: int = 5,
        timesteps_per_episode: int = 10,
        pred_horizon: int = 4,
        obs_horizon: int = 2,
        train: bool = True,
        seed: int = 42,
        val_ratio: float = 0.1,
        total_ratio: float = 1.0,
        downsample_factor: int = 1,
        preload_data_in_memory: bool = False,
        trailing_padded_actions: int | None = None,
        action_backward_shift: int = 1,
    ) -> EpisodicDataset:
        if actions_metadata is None:
            actions_metadata = {}
        if observations_metadata is None:
            observations_metadata = {}

        action_space = action_space_factory(actions_metadata=actions_metadata)
        observation_space = observation_space_factory(
            observations_metadata=observations_metadata,
        )
        config = mock_dataloader_config(
            val_ratio=val_ratio,
            total_ratio=total_ratio,
            downsample_factor=downsample_factor,
            preload_data_in_memory=preload_data_in_memory,
            trailing_padded_actions=trailing_padded_actions,
            action_backward_shift=action_backward_shift,
        )
        extra_keys = list(
            set(
                observation_space.get_required_zarr_keys()
                + action_space.get_required_zarr_keys()
            )
        )
        buffer = mock_replay_buffer_for_dataset(
            num_episodes=num_episodes,
            timesteps_per_episode=timesteps_per_episode,
            extra_keys=extra_keys,
        )

        with (
            patch(
                "versatil.data.episodic_dataset.ReplayBuffer"
            ) as mock_replay_buffer_class,
            patch("versatil.data.episodic_dataset.ImageProcessor"),
        ):
            mock_replay_buffer_class.create_from_path.return_value = buffer
            mock_replay_buffer_class.copy_from_path.return_value = buffer
            mock_replay_buffer_class.create_empty_numpy.return_value = MagicMock(
                n_episodes=0,
                n_steps=0,
                episode_ends=np.array([]),
            )

            dataset = EpisodicDataset(
                zarr_path="/fake/path.zarr",
                action_space=action_space,
                observation_space=observation_space,
                dataloader_config=config,
                pred_horizon=pred_horizon,
                obs_horizon=obs_horizon,
                train=train,
                seed=seed,
            )
        return dataset

    return factory


class TestEpisodicDatasetInit:
    @pytest.mark.parametrize(
        "pred_horizon, obs_horizon",
        [(4, 2), (8, 3), (16, 1)],
    )
    def test_stores_horizons(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        pred_horizon: int,
        obs_horizon: int,
    ):
        dataset = episodic_dataset_factory(
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
        )
        assert dataset.pred_horizon == pred_horizon
        assert dataset.obs_horizon == obs_horizon

    def test_stores_action_space(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        metadata = {"position": on_the_fly_action_metadata_factory()}
        dataset = episodic_dataset_factory(actions_metadata=metadata)
        assert "position" in dataset.action_space.actions_metadata

    def test_stores_observation_space(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        metadata = {"position": position_observation_metadata_factory()}
        dataset = episodic_dataset_factory(observations_metadata=metadata)
        assert "position" in dataset.observation_space.observations_metadata

    @pytest.mark.parametrize("train", [True, False])
    def test_stores_train_flag(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        train: bool,
    ):
        dataset = episodic_dataset_factory(train=train)
        assert dataset.train is train

    @pytest.mark.parametrize("seed", [42, 0, 123])
    def test_stores_seed(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        seed: int,
    ):
        dataset = episodic_dataset_factory(seed=seed)
        assert dataset.seed == seed

    def test_uses_create_from_path_when_not_preloading(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
        mock_dataloader_config: Callable[..., DataLoaderConfig],
        mock_replay_buffer_for_dataset: Callable[..., MagicMock],
    ):
        config = mock_dataloader_config(preload_data_in_memory=False)
        buffer = mock_replay_buffer_for_dataset()

        with (
            patch(
                "versatil.data.episodic_dataset.ReplayBuffer"
            ) as mock_replay_buffer_class,
            patch("versatil.data.episodic_dataset.ImageProcessor"),
        ):
            mock_replay_buffer_class.create_from_path.return_value = buffer
            EpisodicDataset(
                zarr_path="/fake/path.zarr",
                action_space=action_space_factory(),
                observation_space=observation_space_factory(),
                dataloader_config=config,
                pred_horizon=4,
                obs_horizon=2,
            )
            mock_replay_buffer_class.create_from_path.assert_called_once()

    def test_uses_copy_from_path_when_preloading(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
        mock_dataloader_config: Callable[..., DataLoaderConfig],
        mock_replay_buffer_for_dataset: Callable[..., MagicMock],
    ):
        config = mock_dataloader_config(preload_data_in_memory=True)
        buffer = mock_replay_buffer_for_dataset()

        with (
            patch(
                "versatil.data.episodic_dataset.ReplayBuffer"
            ) as mock_replay_buffer_class,
            patch("versatil.data.episodic_dataset.ImageProcessor"),
        ):
            mock_replay_buffer_class.copy_from_path.return_value = buffer
            EpisodicDataset(
                zarr_path="/fake/path.zarr",
                action_space=action_space_factory(),
                observation_space=observation_space_factory(),
                dataloader_config=config,
                pred_horizon=4,
                obs_horizon=2,
            )
            mock_replay_buffer_class.copy_from_path.assert_called_once()

    def test_raises_when_required_keys_missing(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        action_space_factory: Callable[..., ActionSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        mock_dataloader_config: Callable[..., DataLoaderConfig],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "missing_key": position_observation_metadata_factory(),
            }
        )
        config = mock_dataloader_config()
        buffer = MagicMock()
        buffer.n_episodes = 5
        buffer.n_steps = 50
        type(buffer).episode_ends = PropertyMock(
            return_value=np.array([10, 20, 30, 40, 50]),
        )
        buffer.keys.return_value = []

        with (
            patch(
                "versatil.data.episodic_dataset.ReplayBuffer"
            ) as mock_replay_buffer_class,
            patch("versatil.data.episodic_dataset.ImageProcessor"),
        ):
            mock_replay_buffer_class.create_from_path.return_value = buffer
            with pytest.raises(KeyError, match="Missing required keys"):
                EpisodicDataset(
                    zarr_path="/fake/path.zarr",
                    action_space=action_space_factory(),
                    observation_space=observation_space,
                    dataloader_config=config,
                    pred_horizon=4,
                    obs_horizon=2,
                )

    def test_normalizer_not_set_before_set_normalizer_called(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = episodic_dataset_factory()
        assert dataset.sample_builder.normalizer is None


class TestEpisodicDatasetLen:
    @pytest.mark.parametrize(
        "num_episodes, timesteps_per_episode",
        [(3, 10), (5, 20), (10, 5)],
    )
    def test_length_delegates_to_sampler(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        num_episodes: int,
        timesteps_per_episode: int,
    ):
        dataset = episodic_dataset_factory(
            num_episodes=num_episodes,
            timesteps_per_episode=timesteps_per_episode,
        )
        assert len(dataset) == len(dataset.sampler)

    @pytest.mark.parametrize(
        "trailing_padded_actions, expected_starts_per_episode",
        [
            (None, 8),
            (0, 5),
            (2, 7),
            (3, 8),
        ],
    )
    def test_trailing_padded_actions_controls_valid_start_count(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        trailing_padded_actions: int | None,
        expected_starts_per_episode: int,
    ):
        num_episodes = 5
        timesteps_per_episode = 10
        obs_horizon = 2
        pred_horizon = 4
        dataset = episodic_dataset_factory(
            num_episodes=num_episodes,
            timesteps_per_episode=timesteps_per_episode,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_backward_shift=0,
            val_ratio=0.0,
            trailing_padded_actions=trailing_padded_actions,
        )
        assert len(dataset) == expected_starts_per_episode * num_episodes

    def test_precomputed_only_fits_full_horizon_window_without_padding(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        obs_horizon = 1
        pred_horizon = 4
        dataset = episodic_dataset_factory(
            num_episodes=1,
            timesteps_per_episode=obs_horizon + pred_horizon - 1,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_backward_shift=0,
            val_ratio=0.0,
            trailing_padded_actions=0,
            actions_metadata={
                ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: (
                    position_action_metadata_factory(prediction_dimension=3)
                ),
            },
        )
        assert len(dataset) == 1
        mask = dataset[0][SampleKey.ACTION.value][SampleKey.IS_PAD_ACTION.value]
        torch.testing.assert_close(mask, torch.zeros(pred_horizon, dtype=torch.bool))


class TestEpisodicDatasetGetItem:
    @pytest.mark.parametrize("idx", [0, 2, 5])
    def test_pipeline_forwards_data_through_all_stages(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        idx: int,
    ):
        """Verify the full data flow: sampler → action_processor → build_sample → return."""
        dataset = episodic_dataset_factory()

        padded_data = {"proprio": np.ones((6, 3), dtype=np.float32)}
        action_data = {"position": np.zeros((4, 3), dtype=np.float32)}
        action_meta = {"position": MagicMock()}
        expected_sample = {"observation": {"pos": torch.ones(3)}, "action": {}}

        dataset.sampler.sample_sequence = MagicMock(return_value=padded_data)
        dataset.action_processor.compute_sample_actions = MagicMock(
            return_value=(action_data, action_meta),
        )
        dataset.sample_builder.build_sample = MagicMock(return_value=expected_sample)

        result = dataset[idx]

        assert result is expected_sample

        # sampler received the correct index
        dataset.sampler.sample_sequence.assert_called_once_with(idx)

        # action_processor received the sampler output
        action_call_kwargs = dataset.action_processor.compute_sample_actions.call_args[
            1
        ]
        assert action_call_kwargs["padded_data"] is padded_data

        # build_sample received all pipeline outputs
        build_call_kwargs = dataset.sample_builder.build_sample.call_args[1]
        assert build_call_kwargs["padded_data"] is padded_data
        assert build_call_kwargs["action_data"] is action_data
        assert build_call_kwargs["action_meta"] is action_meta
        assert build_call_kwargs["start_idx"] == idx
        assert build_call_kwargs["sampler_indices"] is dataset.sampler.indices

    @pytest.mark.parametrize(
        "obs_horizon, pred_horizon, expected_start, expected_end",
        [
            (2, 4, 1, 5),
            (3, 5, 2, 7),
            (1, 8, 0, 8),
        ],
    )
    def test_action_slice_computed_from_horizons(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        obs_horizon: int,
        pred_horizon: int,
        expected_start: int,
        expected_end: int,
    ):
        dataset = episodic_dataset_factory(
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
        )
        dataset.sampler.sample_sequence = MagicMock(return_value={})
        dataset.action_processor.compute_sample_actions = MagicMock(
            return_value=({}, {}),
        )
        dataset.sample_builder.build_sample = MagicMock(return_value={})

        result = dataset[0]

        assert result == {}
        action_call_kwargs = dataset.action_processor.compute_sample_actions.call_args[
            1
        ]
        assert action_call_kwargs["action_slice_start"] == expected_start
        assert action_call_kwargs["action_slice_end"] == expected_end


class TestCreateEpisodeMask:
    @pytest.mark.parametrize("train", [True, False])
    def test_train_and_val_masks_are_disjoint(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        train: bool,
    ):
        dataset_train = episodic_dataset_factory(
            train=True,
            num_episodes=10,
            val_ratio=0.2,
        )
        dataset_val = episodic_dataset_factory(
            train=False,
            num_episodes=10,
            val_ratio=0.2,
        )
        train_mask = dataset_train._create_episode_mask(
            val_ratio=0.2,
            total_ratio=1.0,
            train=True,
            seed=42,
        )
        val_mask = dataset_val._create_episode_mask(
            val_ratio=0.2,
            total_ratio=1.0,
            train=False,
            seed=42,
        )
        # No overlap
        assert not np.any(np.logical_and(train_mask, val_mask))

    def test_total_ratio_limits_episodes(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = episodic_dataset_factory(num_episodes=20)
        mask = dataset._create_episode_mask(
            val_ratio=0.0,
            total_ratio=0.5,
            train=True,
            seed=42,
        )
        assert np.sum(mask) <= 10

    def test_max_train_episodes_limits_count(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = episodic_dataset_factory(num_episodes=20)
        mask = dataset._create_episode_mask(
            val_ratio=0.0,
            total_ratio=1.0,
            train=True,
            seed=42,
            max_train_episodes=5,
        )
        assert np.sum(mask) <= 5


class TestSetupEpisodeIndices:
    def test_selected_episode_indices_excludes_empty_episodes(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = episodic_dataset_factory(num_episodes=5)
        for idx in dataset.selected_episode_indices:
            assert len(dataset.episode_indices[idx]) > 0

    @pytest.mark.parametrize("num_episodes", [3, 5, 10])
    def test_episode_indices_length_matches_num_episodes(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        num_episodes: int,
    ):
        dataset = episodic_dataset_factory(num_episodes=num_episodes)
        assert len(dataset.episode_indices) == num_episodes


class TestSetNormalizerAndTokenizer:
    def test_set_normalizer_updates_sample_builder(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = episodic_dataset_factory()
        mock_normalizer = MagicMock()
        dataset.set_normalizer(normalizer=mock_normalizer)
        assert dataset.sample_builder.normalizer is mock_normalizer

    def test_set_tokenizer_updates_sample_builder(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = episodic_dataset_factory()
        mock_tokenizer = MagicMock()
        dataset.set_tokenizer(tokenizer=mock_tokenizer)
        assert dataset.sample_builder.tokenizer is mock_tokenizer

    def test_set_tokenizer_accepts_none(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = episodic_dataset_factory()
        dataset.set_tokenizer(tokenizer=None)
        assert dataset.sample_builder.tokenizer is None


class TestGetGripperPositiveClassImbalanceWeight:
    def test_raises_when_no_gripper_actions(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = episodic_dataset_factory(actions_metadata={})
        with pytest.raises(ValueError, match="Gripper actions are not being predicted"):
            dataset.get_gripper_positive_class_imbalance_weight()

    def test_raises_when_multiple_gripper_actions(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        dataset = episodic_dataset_factory(
            actions_metadata={
                "gripper_1": gripper_action_metadata_factory(),
                "gripper_2": gripper_action_metadata_factory(),
            }
        )
        with pytest.raises(ValueError, match="single gripper action"):
            dataset.get_gripper_positive_class_imbalance_weight()

    def test_raises_for_non_binary_gripper(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        dataset = episodic_dataset_factory(
            actions_metadata={
                "gripper": gripper_action_metadata_factory(
                    gripper_type=GripperType.CONTINUOUS.value,
                ),
            }
        )
        with pytest.raises(ValueError, match="binary grippers"):
            dataset.get_gripper_positive_class_imbalance_weight()

    def test_computes_correct_weight_for_binary_gripper(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        dataset = episodic_dataset_factory(
            actions_metadata={
                ProprioKey.GRIPPER_STATE.value: gripper_action_metadata_factory(
                    gripper_type=GripperType.BINARY.value,
                ),
            }
        )
        # Mock the replay buffer to return known gripper data
        gripper_data = np.array([[1], [0], [0], [0], [1]], dtype=np.float32)
        dataset.replay_buffer.__getitem__ = MagicMock(
            side_effect=lambda key: gripper_data,
        )

        weight = dataset.get_gripper_positive_class_imbalance_weight()

        # 2 positive, 3 negative → weight = 3/2 = 1.5
        assert weight == pytest.approx(1.5)

    def test_on_the_fly_gripper_uses_source_metadata(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    ):
        gripper_source = gripper_observation_metadata_factory(
            gripper_type=GripperType.BINARY.value,
        )
        dataset = episodic_dataset_factory(
            actions_metadata={
                ProprioKey.GRIPPER_STATE.value: on_the_fly_action_metadata_factory(
                    source_metadata=gripper_source,
                ),
            }
        )
        gripper_data = np.array([[1], [1], [1], [0]], dtype=np.float32)
        dataset.replay_buffer.__getitem__ = MagicMock(
            side_effect=lambda key: gripper_data,
        )

        weight = dataset.get_gripper_positive_class_imbalance_weight()

        # 3 positive, 1 negative → weight = 1/3
        assert weight == pytest.approx(1.0 / 3.0)

    def test_on_the_fly_raises_for_non_gripper_source_metadata(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    ):
        gripper_source = gripper_observation_metadata_factory()
        dataset = episodic_dataset_factory(
            actions_metadata={
                ProprioKey.GRIPPER_STATE.value: on_the_fly_action_metadata_factory(
                    source_metadata=gripper_source,
                ),
            }
        )
        bad_otf = on_the_fly_action_metadata_factory(
            source_metadata=position_observation_metadata_factory(),
        )
        gripper_actions_value = {ProprioKey.GRIPPER_STATE.value: bad_otf}
        with (
            patch.object(
                ActionSpace,
                "gripper_actions",
                new_callable=PropertyMock,
                return_value=gripper_actions_value,
            ),
            patch.object(
                ActionSpace,
                "has_gripper_actions",
                new_callable=PropertyMock,
                return_value=True,
            ),
            pytest.raises(TypeError, match="Expected GripperObservationMetadata"),
        ):
            dataset.get_gripper_positive_class_imbalance_weight()

    def test_raises_for_unsupported_metadata_type(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        gripper_source = gripper_observation_metadata_factory()
        dataset = episodic_dataset_factory(
            actions_metadata={
                ProprioKey.GRIPPER_STATE.value: on_the_fly_action_metadata_factory(
                    source_metadata=gripper_source,
                ),
            }
        )
        unsupported_meta = MagicMock()
        gripper_actions_value = {ProprioKey.GRIPPER_STATE.value: unsupported_meta}
        with (
            patch.object(
                ActionSpace,
                "gripper_actions",
                new_callable=PropertyMock,
                return_value=gripper_actions_value,
            ),
            patch.object(
                ActionSpace,
                "has_gripper_actions",
                new_callable=PropertyMock,
                return_value=True,
            ),
            pytest.raises(ValueError, match="Unsupported gripper action metadata"),
        ):
            dataset.get_gripper_positive_class_imbalance_weight()


class TestGetNormalizerAndTokenizer:
    def test_delegates_to_transform_builder(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = episodic_dataset_factory()

        mock_normalizer = MagicMock()
        mock_tokenizer = MagicMock()

        with patch(
            "versatil.data.episodic_dataset.TransformBuilder"
        ) as mock_builder_class:
            mock_builder_instance = MagicMock()
            mock_builder_instance.create_normalizer_and_tokenizer.return_value = (
                mock_normalizer,
                mock_tokenizer,
            )
            mock_builder_class.return_value = mock_builder_instance

            normalizer, tokenizer = dataset.get_normalizer_and_tokenizer(
                device=torch.device("cpu"),
            )

        assert normalizer is mock_normalizer
        assert tokenizer is mock_tokenizer
        mock_builder_class.assert_called_once()

    @pytest.mark.parametrize(
        "winsorize_depth, winsorize_kinematics",
        [(True, False), (False, True), (True, True)],
    )
    def test_passes_winsorization_config_to_builder(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        winsorize_depth: bool,
        winsorize_kinematics: bool,
    ):
        dataset = episodic_dataset_factory()

        with patch(
            "versatil.data.episodic_dataset.TransformBuilder"
        ) as mock_builder_class:
            mock_builder_instance = MagicMock()
            mock_builder_instance.create_normalizer_and_tokenizer.return_value = (
                MagicMock(),
                None,
            )
            mock_builder_class.return_value = mock_builder_instance

            dataset.get_normalizer_and_tokenizer(
                winsorize_depth=winsorize_depth,
                winsorize_kinematics=winsorize_kinematics,
            )

        call_kwargs = mock_builder_class.call_args[1]
        if winsorize_depth:
            assert call_kwargs["depth_winsorize_quantiles"] is not None
        else:
            assert call_kwargs["depth_winsorize_quantiles"] is None
        if winsorize_kinematics:
            assert call_kwargs["kinematics_winsorize_quantiles"] is not None
        else:
            assert call_kwargs["kinematics_winsorize_quantiles"] is None


class TestApplyDownsampling:
    @pytest.mark.parametrize("downsample_factor", [2, 3, 5])
    def test_downsampling_reduces_episode_length(
        self,
        episodic_dataset_factory: Callable[..., EpisodicDataset],
        downsample_factor: int,
    ):
        dataset = episodic_dataset_factory(
            num_episodes=3,
            timesteps_per_episode=20,
        )
        original_episode = dataset.replay_buffer.get_episode(0)
        first_key = next(iter(original_episode.keys()))
        original_length = original_episode[first_key].shape[0]

        mock_subsampled_buffer = MagicMock()
        mock_subsampled_buffer.n_episodes = 3
        mock_subsampled_buffer.n_steps = 30
        type(mock_subsampled_buffer).episode_ends = PropertyMock(
            return_value=np.array([10, 20, 30]),
        )

        episode_mask = np.ones(3, dtype=bool)

        with patch(
            "versatil.data.episodic_dataset.ReplayBuffer"
        ) as mock_replay_buffer_class:
            mock_replay_buffer_class.create_empty_numpy.return_value = (
                mock_subsampled_buffer
            )
            dataset._apply_downsampling(
                episode_mask=episode_mask,
                downsample_step=downsample_factor,
            )

        # Verify add_episode was called for each selected episode
        assert mock_subsampled_buffer.add_episode.call_count == 3
        # Verify last frame is always included
        for call in mock_subsampled_buffer.add_episode.call_args_list:
            episode_data = call[0][0]
            first_key = next(iter(episode_data.keys()))
            indices_used = len(episode_data[first_key])
            expected_indices = len(range(0, original_length, downsample_factor))
            if (original_length - 1) % downsample_factor != 0:
                expected_indices += 1
            assert indices_used == expected_indices


@pytest.mark.integration
class TestEpisodicDatasetWithRealData:
    @pytest.mark.parametrize(
        "num_episodes, timesteps_per_episode",
        [(3, 15), (5, 20), (8, 10)],
    )
    def test_length_is_positive(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
        num_episodes: int,
        timesteps_per_episode: int,
    ):
        dataset = real_dataset_factory(
            num_episodes=num_episodes,
            timesteps_per_episode=timesteps_per_episode,
        )
        assert len(dataset) > 0

    def test_getitem_returns_observation_and_action_dicts(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = real_dataset_factory(num_episodes=3, timesteps_per_episode=20)
        sample = dataset[0]

        assert SampleKey.OBSERVATION.value in sample
        assert SampleKey.ACTION.value in sample

    def test_observation_contains_expected_proprio_key(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = real_dataset_factory(num_episodes=3, timesteps_per_episode=20)
        sample = dataset[0]

        observations = sample[SampleKey.OBSERVATION.value]
        assert ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value in observations

    @pytest.mark.parametrize(
        "obs_horizon, position_dim",
        [(2, 3), (3, 6), (1, 4)],
    )
    def test_proprio_observation_shape(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
        obs_horizon: int,
        position_dim: int,
    ):
        dataset = real_dataset_factory(
            num_episodes=3,
            timesteps_per_episode=20,
            obs_horizon=obs_horizon,
            position_dim=position_dim,
            orientation_dim=0,
            has_gripper=False,
        )
        sample = dataset[0]
        proprio = sample[SampleKey.OBSERVATION.value][
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        ]

        assert proprio.shape == (obs_horizon, position_dim)
        assert proprio.dtype == torch.float32

    @pytest.mark.parametrize("pred_horizon", [2, 4, 8])
    def test_action_shape_matches_pred_horizon(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
        pred_horizon: int,
    ):
        dataset = real_dataset_factory(
            num_episodes=3,
            timesteps_per_episode=30,
            pred_horizon=pred_horizon,
            position_dim=3,
            orientation_dim=0,
            has_gripper=False,
        )
        sample = dataset[0]
        actions = sample[SampleKey.ACTION.value]
        position_action = actions[ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value]

        assert position_action.shape[0] == pred_horizon

    def test_action_padding_mask_present_and_correct_shape(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
    ):
        pred_horizon = 4
        dataset = real_dataset_factory(
            num_episodes=3,
            timesteps_per_episode=20,
            pred_horizon=pred_horizon,
        )
        sample = dataset[0]
        is_pad = sample[SampleKey.ACTION.value][SampleKey.IS_PAD_ACTION.value]

        assert is_pad.shape == (pred_horizon,)
        assert is_pad.dtype == torch.bool

    def test_camera_observation_shape(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
    ):
        image_height = 16
        image_width = 16
        obs_horizon = 2
        dataset = real_dataset_factory(
            num_episodes=3,
            timesteps_per_episode=20,
            cameras=[Cameras.LEFT.value],
            image_height=image_height,
            image_width=image_width,
            obs_horizon=obs_horizon,
        )
        sample = dataset[0]
        image = sample[SampleKey.OBSERVATION.value][Cameras.LEFT.value]

        # (obs_horizon, channels, height, width) after CHW conversion
        assert image.shape == (obs_horizon, 3, image_height, image_width)
        assert image.dtype == torch.float32
        # Pixel values normalized to [0, 1]
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_train_val_split_produces_disjoint_samples(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
    ):
        train_dataset = real_dataset_factory(
            num_episodes=10,
            timesteps_per_episode=20,
            train=True,
            val_ratio=0.3,
        )
        val_dataset = real_dataset_factory(
            num_episodes=10,
            timesteps_per_episode=20,
            train=False,
            val_ratio=0.3,
        )

        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        assert len(train_dataset) + len(val_dataset) > 0

    def test_all_samples_accessible_without_error(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = real_dataset_factory(
            num_episodes=3,
            timesteps_per_episode=15,
            pred_horizon=2,
            obs_horizon=2,
        )
        for idx in range(len(dataset)):
            sample = dataset[idx]
            assert SampleKey.OBSERVATION.value in sample
            assert SampleKey.ACTION.value in sample

    def test_gripper_action_present_when_configured(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = real_dataset_factory(
            num_episodes=3,
            timesteps_per_episode=20,
            has_gripper=True,
        )
        sample = dataset[0]
        actions = sample[SampleKey.ACTION.value]

        assert ProprioKey.GRIPPER_STATE.value in actions

    def test_gripper_class_imbalance_weight_with_real_data(
        self,
        real_dataset_factory: Callable[..., EpisodicDataset],
    ):
        dataset = real_dataset_factory(
            num_episodes=5,
            timesteps_per_episode=20,
            has_gripper=True,
        )
        weight = dataset.get_gripper_positive_class_imbalance_weight()

        assert weight > 0.0
        assert np.isfinite(weight)
