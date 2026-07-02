import logging

import numpy as np
import torch

from versatil.common.tensor_ops import tensor_to_str
from versatil.configs.data.tokenizer import (
    ActionDiscretizerConfig,
    ActionTokenIdMappingConfig,
    TokenizationConfig,
)
from versatil.data.constants import (
    ActionDiscretizerType,
    ActionTokenIdMappingType,
)
from versatil.data.metadata import (
    ActionMetadata,
    OnTheFlyActionMetadata,
    PositionObservationMetadata,
)
from versatil.data.normalization.image_normalizer import (
    get_depth_image_normalizer,
    get_rgb_image_normalizer,
)
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.preprocessing.replay_buffer import ReplayBuffer
from versatil.data.processing.action_processor import ActionProcessor
from versatil.data.task import ObservationSpace
from versatil.data.tokenization.action_discretizer import (
    BinnedActionDiscretizer,
    FastActionDiscretizer,
)
from versatil.data.tokenization.action_token_id_mapping import (
    IdentityActionTokenIdMapping,
    LanguageVocabularyActionTokenIdMapping,
)
from versatil.data.tokenization.action_tokenizer import ActionTokenizer
from versatil.data.tokenization.observation_tokenizer import ObservationTokenizer
from versatil.data.tokenization.tokenizer import Tokenizer


def _build_action_discretizer(
    config: ActionDiscretizerConfig,
    device: torch.device | None,
) -> FastActionDiscretizer | BinnedActionDiscretizer:
    """Instantiate the configured action discretizer."""
    match config.type:
        case ActionDiscretizerType.FAST.value:
            return FastActionDiscretizer(
                use_pretrained=config.use_pretrained,
                tokenizer_model=config.tokenizer_model,
            )
        case ActionDiscretizerType.BINNED.value:
            return BinnedActionDiscretizer(
                num_bins=config.num_bins,
                device=device,
                binning_strategy=config.binning_strategy,
                min_value=config.min_value,
                max_value=config.max_value,
            )
        case unsupported_type:
            raise ValueError(f"Unsupported action discretizer type: {unsupported_type}")


def _build_token_id_mapping(
    config: ActionTokenIdMappingConfig,
) -> IdentityActionTokenIdMapping | LanguageVocabularyActionTokenIdMapping:
    """Instantiate the configured action token-id mapping."""
    if config.type == ActionTokenIdMappingType.IDENTITY.value:
        return IdentityActionTokenIdMapping()
    if config.type == ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value:
        if config.language_tokenizer_model is None:
            raise ValueError(
                "language_tokenizer_model must be provided for language-vocabulary "
                "action token-id mapping"
            )
        return LanguageVocabularyActionTokenIdMapping(
            language_tokenizer_model=config.language_tokenizer_model,
            num_special_tokens_to_skip=config.num_special_tokens_to_skip,
        )
    raise ValueError(f"Unsupported action token-id mapping type: {config.type}")


class TransformBuilder:
    """Builder for creating and configuring data normalizers and tokenizers."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        action_processor: ActionProcessor,
        prediction_horizon: int,
        observation_space: ObservationSpace,
        episode_ends: np.ndarray,
        kinematics_norm_type: str,
        image_norm_type: str,
        depth_norm_type: str,
        depth_winsorize_quantiles: tuple[float, float] | None = (0.01, 0.99),
        kinematics_winsorize_quantiles: tuple[float, float] | None = (0.01, 0.99),
        tokenization_config: TokenizationConfig | None = None,
        clamp_kinematics_range: bool = True,
        min_kinematics_std: float = 2e-2,
        min_kinematics_range: float = 4e-2,
        action_sample_size: int = 2048,
        episode_selection_mask: np.ndarray | None = None,
    ):
        """Initialize transform builder.

        Args:
            replay_buffer: Data source
            action_processor: For computing actions and applying denoising
            prediction_horizon: Prediction horizon for action chunking.
            observation_space: The observation space configuration
            episode_ends: Episode boundaries
            kinematics_norm_type: Normalization type for kinematics
            image_norm_type: Normalization type for RGB images
            depth_norm_type: Normalization type for depth images
            depth_winsorize_quantiles: Quantiles for depth winsorization (lower, upper).
            kinematics_winsorize_quantiles: Quantiles for kinematics winsorization.
            tokenization_config: Tokenization configuration. If None, no tokenizer created.
            clamp_kinematics_range: Whether to clamp std/range to minimum values.
            min_kinematics_std: Minimum std for Gaussian mode when clamp_kinematics_range=True.
            min_kinematics_range: Minimum range for MinMax mode when clamp_kinematics_range=True.
            action_sample_size: Number of action rows to stash alongside each action
                key's normalizer for downstream data-aware initialization (e.g.
                mixture-density head k-means++). Set to 0 to disable. Memory cost
                per action key is ``action_sample_size * action_dim * bytes_per_element``
                (four bytes for float32, eight for float64).
            episode_selection_mask: Optional boolean mask over episodes.
                Statistics, tokenizers, and denoising thresholds are fitted
                only on selected episodes (the training split), keeping
                validation data out of the fitted transforms.
        """
        self.replay_buffer = replay_buffer
        self.action_processor = action_processor
        self.observation_space = observation_space
        self.episode_ends = episode_ends
        self.episode_selection_mask = episode_selection_mask
        self._selected_step_mask = self._build_selected_step_mask()
        self.kinematics_norm_type = kinematics_norm_type
        self.image_norm_type = image_norm_type
        self.depth_norm_type = depth_norm_type
        self.depth_winsorize_quantiles = depth_winsorize_quantiles
        self.kinematics_winsorize_quantiles = kinematics_winsorize_quantiles
        self.tokenization_config = tokenization_config
        self.prediction_horizon = prediction_horizon
        self.clamp_kinematics_range = clamp_kinematics_range
        self.min_kinematics_std = min_kinematics_std
        self.min_kinematics_range = min_kinematics_range
        self.action_sample_size = action_sample_size

    def _build_selected_step_mask(self) -> np.ndarray | None:
        """Return a per-step mask of selected episodes, or None for all."""
        if self.episode_selection_mask is None or bool(
            np.all(self.episode_selection_mask)
        ):
            return None
        step_mask = np.zeros(int(self.episode_ends[-1]), dtype=bool)
        episode_start = 0
        for episode_index, episode_end in enumerate(self.episode_ends):
            if self.episode_selection_mask[episode_index]:
                step_mask[episode_start:episode_end] = True
            episode_start = int(episode_end)
        return step_mask

    def _is_episode_selected(self, episode_index: int) -> bool:
        """Return whether an episode contributes to fitted statistics."""
        if self.episode_selection_mask is None:
            return True
        return bool(self.episode_selection_mask[episode_index])

    def _select_step_rows(self, array: np.ndarray) -> np.ndarray:
        """Filter an (n_steps, ...) array down to selected-episode rows."""
        if self._selected_step_mask is None:
            return array
        return array[self._selected_step_mask]

    def _select_episodes_contiguous(
        self, array: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return selected-episode rows with remapped episode boundaries."""
        if self._selected_step_mask is None:
            return array, self.episode_ends
        selected_rows = array[self._selected_step_mask]
        selected_lengths = []
        episode_start = 0
        for episode_index, episode_end in enumerate(self.episode_ends):
            if self._is_episode_selected(episode_index=episode_index):
                selected_lengths.append(int(episode_end) - episode_start)
            episode_start = int(episode_end)
        remapped_ends = np.cumsum(np.asarray(selected_lengths, dtype=np.int64))
        return selected_rows, remapped_ends

    def create_normalizer_and_tokenizer(
        self,
        device: torch.device | None = None,
    ) -> tuple[LinearNormalizer, Tokenizer | None]:
        """Create and fit normalizer and optionally tokenizer to data.
        Pipeline: Raw data → Winsorize → Normalize → Tokenize

        Args:
            device: Target device for tensors

        Returns:
            Tuple of (normalizer, tokenizer) where tokenizer is None if not configured
        """
        self.compute_proprioceptive_denoising_thresholds()
        action_keys = self.action_processor.action_space.get_required_zarr_keys()
        action_source_data = {
            key: self.replay_buffer[key][:]
            for key in action_keys
            if key in self.replay_buffer
        }
        action_data, action_meta = self.action_processor.compute_sample_actions(
            padded_data=action_source_data,
            action_slice_start=0,
            action_slice_end=self.replay_buffer.n_steps - 1,
        )
        cross_indices = self.episode_ends[:-1] - 1
        valid_mask = np.ones(len(next(iter(action_data.values()))), dtype=bool)
        valid_mask[cross_indices] = False
        if self._selected_step_mask is not None:
            valid_mask &= self._selected_step_mask[: len(valid_mask)]
        valid_action_data = {key: data[valid_mask] for key, data in action_data.items()}

        normalizer = self._create_normalizer(
            action_data=valid_action_data,
            action_meta=action_meta,
            device=device,
        )
        tokenizer = None
        if self.tokenization_config and (
            self.tokenization_config.tokenize_observations
            or self.tokenization_config.tokenize_actions
        ):
            tokenizer = self._create_tokenizer(
                normalizer=normalizer,
                action_data=valid_action_data,
                action_meta=action_meta,
                device=device,
            )
        return normalizer, tokenizer

    def compute_proprioceptive_denoising_thresholds(
        self,
    ) -> None:
        """Compute denoising thresholds for proprioceptive observations."""
        for key, meta in self.action_processor.action_space.actions_metadata.items():
            if isinstance(meta, OnTheFlyActionMetadata):
                source_meta = meta.source_metadata
                if isinstance(source_meta, PositionObservationMetadata):
                    obs_data, selected_episode_ends = self._select_episodes_contiguous(
                        array=self.replay_buffer[key][:]
                    )
                    self.action_processor.compute_denoising_threshold(
                        obs_data=obs_data,
                        key=key,
                        meta=source_meta,
                        episode_ends=selected_episode_ends,
                    )
        self.action_processor.log_movement_distribution()

    def _create_normalizer(
        self,
        action_data: dict[str, np.ndarray],
        action_meta: dict[str, ActionMetadata],
        device: torch.device | None = None,
        winsorize_depth: bool = True,
    ) -> LinearNormalizer:
        """Create and fit normalizer for this dataset.

        Args:
            action_data: Action data for fitting
            action_meta: Action metadata for fitting
            device: Target device for tensors
            winsorize_depth: Apply winsorization to depth values

        Returns:
            Fitted LinearNormalizer instance

        Note: additionally computes denoising thresholds for action processor
        """
        normalizer = LinearNormalizer()
        data_to_normalize = {}
        camera_keys = set(self.observation_space.cameras.keys())
        for key, meta in self.observation_space.observations_metadata.items():
            if key in camera_keys:
                continue
            if meta.needs_normalization:
                if not meta.is_numerical:
                    raise ValueError(
                        f"Cannot normalize non-numerical observation key: {key}"
                    )
                data_to_normalize[key] = self._select_step_rows(
                    array=self.replay_buffer[key][:]
                )

        action_normalization_keys: set[str] = set()
        for key, meta in action_meta.items():
            if meta.needs_normalization:
                data_to_normalize[key] = action_data[key]
                action_normalization_keys.add(key)
        if self.kinematics_winsorize_quantiles:
            data_to_normalize = self._apply_winsorization(
                data_to_normalize, self.kinematics_winsorize_quantiles
            )
        sample_sizes = (
            dict.fromkeys(action_normalization_keys, self.action_sample_size)
            if self.action_sample_size > 0
            else 0
        )
        normalizer.fit(
            data=data_to_normalize,
            last_n_dims=1,
            mode=self.kinematics_norm_type,
            device=device,
            range_eps=1e-10,
            clamp_range=self.clamp_kinematics_range,
            min_std=self.min_kinematics_std,
            min_range=self.min_kinematics_range,
            sample_size=sample_sizes,
        )
        self._setup_image_normalizers(normalizer, device, winsorize_depth)
        self._log_normalized_proprio_stats(normalizer, data_to_normalize)
        return normalizer

    def _setup_image_normalizers(
        self,
        normalizer: LinearNormalizer,
        device: torch.device | None,
        winsorize_depth: bool,
    ) -> None:
        """Setup normalizers for all cameras.

        Args:
            normalizer: Normalizer to configure
            device: Target device
            winsorize_depth: Apply winsorization to depth
        """
        for camera_key, camera_metadata in self.observation_space.cameras.items():
            if camera_metadata.is_depth:
                depth_stats = self._compute_depth_stats_streaming(
                    camera_key, winsorize_depth
                )
                self._setup_depth_normalizer(
                    normalizer, camera_key, depth_stats, device
                )
            else:
                self._setup_rgb_normalizer(normalizer, camera_key, device)
            self._log_camera_stats_sampled(camera_key, normalizer)

    def _compute_depth_stats_streaming(
        self,
        camera_key: str,
        winsorize: bool,
        chunk_size: int = 1000,
    ) -> dict[str, float]:
        """Compute depth statistics using streaming to avoid loading entire array.

        Uses vectorized per-chunk operations and parallel Welford for exact mean/variance.
        Winsorization (if enabled) uses fast uniform random subsampling for quantiles.

        Args:
            camera_key: Key for depth camera in replay buffer
            winsorize: Whether to apply winsorization
            chunk_size: Number of frames to process at a time

        Returns:
            Dictionary with min, max, mean, std statistics (on clipped data if winsorized)
        """
        depth_array = self.replay_buffer[camera_key]
        n_frames = depth_array.shape[0]
        selected_frame_indices = (
            np.flatnonzero(self._selected_step_mask[:n_frames])
            if self._selected_step_mask is not None
            else None
        )
        total_pixels = depth_array.size
        p_lower, p_upper = None, None
        if winsorize and self.depth_winsorize_quantiles:
            reservoir_size = min(100_000, total_pixels)
            dtype = depth_array.dtype
            if total_pixels == 0:
                reservoir = np.empty(0, dtype=dtype)
            elif selected_frame_indices is not None:
                # Quantiles must come from selected episodes only: sample a
                # subset of selected frames and pool their pixels.
                sampled_frame_count = min(200, len(selected_frame_indices))
                sampled_frames = np.sort(
                    np.random.choice(
                        selected_frame_indices, sampled_frame_count, replace=False
                    )
                )
                pooled = depth_array[sampled_frames].ravel()
                if pooled.size > reservoir_size:
                    pooled = np.random.choice(pooled, reservoir_size, replace=False)
                reservoir = pooled
            elif total_pixels <= reservoir_size:
                # Small array - load all and ravel
                reservoir = depth_array[:].ravel()
            else:
                # Large array - sample using multi-dimensional indexing
                flat_indices = np.random.choice(
                    total_pixels, reservoir_size, replace=False
                )
                multi_indices = np.unravel_index(flat_indices, depth_array.shape)
                reservoir = depth_array[multi_indices]
            if reservoir.size > 0:
                lower_q, upper_q = self.depth_winsorize_quantiles
                p_lower, p_upper = np.quantile(reservoir, [lower_q, upper_q])
                logging.info(
                    f"Depth winsorization bounds [{lower_q}, {upper_q}]: "
                    f"lower={p_lower:.4f}, upper={p_upper:.4f}"
                )
        global_min = np.inf
        global_max = -np.inf
        global_count = 0
        global_mean = 0.0
        global_m2 = 0.0
        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            chunk = depth_array[start:end]
            if self._selected_step_mask is not None:
                chunk = chunk[self._selected_step_mask[start:end]]
            if chunk.size == 0:
                continue
            if p_lower is not None:
                chunk = np.clip(chunk, p_lower, p_upper)
            flat = chunk.ravel()
            n = flat.size
            if n == 0:
                continue
            chunk_mean = flat.mean()
            chunk_var = flat.var(ddof=0)
            chunk_m2 = chunk_var * n
            global_min = min(global_min, flat.min())
            global_max = max(global_max, flat.max())
            if global_count == 0:
                global_mean = chunk_mean
                global_m2 = chunk_m2
                global_count = n
            else:
                delta = chunk_mean - global_mean
                new_count = global_count + n
                global_mean += delta * n / new_count
                global_m2 += chunk_m2 + delta**2 * global_count * n / new_count
                global_count = new_count
        if global_count == 0:
            return {"min": float("nan"), "max": float("nan"), "mean": 0.0, "std": 0.0}
        std = np.sqrt(global_m2 / global_count)
        logging.info(
            f"Depth stats (streaming) - min: {global_min:.4f}, max: {global_max:.4f}, "
            f"mean: {global_mean:.4f}, std: {std:.4f}"
        )
        return {
            "min": float(global_min),
            "max": float(global_max),
            "mean": global_mean,
            "std": std,
        }

    def _setup_depth_normalizer(
        self,
        normalizer: LinearNormalizer,
        cam: str,
        depth_stats: dict[str, float],
        device: torch.device | None,
    ) -> None:
        """Setup depth image normalizer from pre-computed stats.

        Args:
            normalizer: Normalizer to configure
            cam: Camera name
            depth_stats: Pre-computed statistics dict with min, max, mean, std
            device: Target device
        """
        normalizer[cam] = get_depth_image_normalizer(
            input_min=depth_stats["min"],
            input_max=depth_stats["max"],
            input_mean=depth_stats["mean"],
            input_std=depth_stats["std"],
            norm_type=self.depth_norm_type,
            device=device,
        )

    def _setup_rgb_normalizer(
        self, normalizer: LinearNormalizer, cam: str, device: torch.device | None
    ) -> None:
        """Setup RGB image normalizer.

        Args:
            normalizer: Normalizer to configure
            cam: Camera name
            device: Target device
        """
        normalizer[cam] = get_rgb_image_normalizer(
            norm_type=self.image_norm_type, device=device
        )

    def _create_tokenizer(
        self,
        normalizer: LinearNormalizer,
        action_data: dict[str, np.ndarray],
        action_meta: dict[str, ActionMetadata],
        device: torch.device | None = None,
    ) -> Tokenizer:
        """Create tokenizer fitted on normalized (and winsorized) data.

        Args:
            normalizer: Already-fitted normalizer to use for normalizing data
            device: Target device

        Returns:
            Fitted tokenizer with observation and/or action tokenizers
        """
        observation_tokenizer = None
        action_tokenizer = None

        if self.tokenization_config.tokenize_observations:
            obs_config = self.tokenization_config.observation_tokenizer
            if obs_config is None:
                raise ValueError(
                    "observation_tokenizer config must be provided when tokenize_observations=True"
                )

            observation_tokenizer = ObservationTokenizer(
                tokenizer_model=obs_config.tokenizer_model,
                observation_keys=obs_config.observation_keys,
                bin_continuous_data=obs_config.bin_continuous_data,
                num_bins=obs_config.num_bins,
                max_token_len=obs_config.max_token_len,
                device=device,
                raw_text=obs_config.raw_text,
                prompt_template=obs_config.prompt_template,
                padding_strategy=obs_config.padding_strategy,
                trust_remote_code=obs_config.trust_remote_code,
            )
            if obs_config.bin_continuous_data:
                data_to_bin = {}
                camera_keys = set(self.observation_space.cameras.keys())
                for key, meta in self.observation_space.observations_metadata.items():
                    if key in camera_keys:
                        continue
                    if not meta.is_numerical:
                        continue
                    obs_data = self._select_step_rows(array=self.replay_buffer[key][:])
                    if meta.needs_normalization:
                        obs_data = normalizer[key].normalize(obs_data)
                        obs_data = (
                            obs_data.detach().cpu().numpy()
                            if isinstance(obs_data, torch.Tensor)
                            else obs_data
                        )
                    data_to_bin[key] = obs_data

                if len(data_to_bin.values()) > 0:
                    observation_tokenizer.fit(data_to_bin)

            if not observation_tokenizer._is_fitted:
                logging.warning(
                    "No observation data was used for observation binning."
                    " Observation binning will be a pass-through."
                )
                observation_tokenizer.fit({})  # Pass-through

        if self.tokenization_config.tokenize_actions:
            action_config = self.tokenization_config.action_tokenizer
            if action_config is None:
                raise ValueError(
                    "action_tokenizer config must be provided when tokenize_actions=True"
                )

            action_tokenizer = ActionTokenizer(
                action_discretizer=_build_action_discretizer(
                    action_config.action_discretizer,
                    device=device,
                ),
                token_id_mapping=_build_token_id_mapping(
                    action_config.token_id_mapping
                ),
                max_token_len=action_config.max_token_len,
                device=device,
            )
            if not action_tokenizer._is_fitted:
                actions_to_tokenize = {}
                for key, meta in action_meta.items():
                    # Match the encode-time filter in normalize_actions: a
                    # numerical key without a prediction head is excluded from
                    # the tokenized tensor, so fitting on it would misalign
                    # every per-dimension bin.
                    if not (meta.is_numerical and meta.requires_prediction_head):
                        continue
                    if meta.needs_normalization:
                        action = normalizer[key].normalize(action_data[key])
                        actions_to_tokenize[key] = (
                            action.detach().cpu().numpy()
                            if isinstance(action, torch.Tensor)
                            else action
                        )
                    else:
                        actions_to_tokenize[key] = action_data[key]
                action_chunks = self._create_action_chunks_for_tokenizer(
                    action_dict=actions_to_tokenize,
                )
                action_tokenizer.fit(action_chunks)
        return Tokenizer(
            observation_tokenizer=observation_tokenizer,
            action_tokenizer=action_tokenizer,
        )

    def _create_action_chunks_for_tokenizer(
        self,
        action_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Create action chunks respecting episode boundaries.

        Note: We need to create chunks here for fitting the tokenizer, since the tokenizer needs to see the full action
          chunks dataset, but we don't store pre-chunked actions in memory.

        Args:
            action_dict: Dictionary of action arrays (already filtered, no cross-episode entries)

        Returns:
            Action chunks of shape (N_chunks, prediction_horizon, total_D)
        """
        action_components = []
        for key in sorted(action_dict.keys()):
            action_components.append(action_dict[key])
        all_actions = np.concatenate(action_components, axis=-1)
        # Compute episode lengths (each episode loses 1 action for on-the-fly computation)
        episode_lengths = []
        for i in range(len(self.episode_ends)):
            if i == 0:
                episode_lengths.append(self.episode_ends[i] - 1)
            else:
                episode_lengths.append(
                    self.episode_ends[i] - self.episode_ends[i - 1] - 1
                )
        chunks = []
        episode_start = 0
        for episode_index, length in enumerate(episode_lengths):
            if not self._is_episode_selected(episode_index=episode_index):
                # Unselected episodes contributed no rows to the filtered
                # action arrays, so the running offset must not advance.
                continue
            episode_actions = all_actions[episode_start : episode_start + length]
            if length >= self.prediction_horizon:
                for i in range(length - self.prediction_horizon + 1):
                    chunks.append(episode_actions[i : i + self.prediction_horizon])
            episode_start += length

        return np.stack(chunks, axis=0)

    @staticmethod
    def _apply_winsorization(
        data_dict: dict[str, np.ndarray],
        quantiles: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        """Apply winsorization to clip outliers to specified quantiles.

        Args:
            data_dict: Dictionary of data arrays to winsorize
            quantiles: (lower, upper) quantiles, e.g., (0.01, 0.99)

        Returns:
            Dictionary with winsorized arrays
        """
        winsorized = {}
        lower_q, upper_q = quantiles
        for key, data in data_dict.items():
            lower_bound = np.quantile(data, lower_q, axis=0)
            upper_bound = np.quantile(data, upper_q, axis=0)
            winsorized_data = np.clip(data, lower_bound, upper_bound)
            winsorized[key] = winsorized_data
            n_clipped = np.sum(data != winsorized_data)
            if n_clipped > 0:
                logging.info(
                    f"Winsorized {key} to [{lower_q:.3f}, {upper_q:.3f}] quantiles - "
                    f"clipped {n_clipped}/{data.size} values "
                    f"({100 * n_clipped / data.size:.2f}%)"
                )
        return winsorized

    def _log_camera_stats_sampled(
        self,
        camera_key: str,
        normalizer: LinearNormalizer,
        n_samples: int = 100,
    ) -> None:
        """Log camera statistics using a small sample to avoid loading full array.

        Args:
            camera_key: Camera name
            normalizer: Configured normalizer
            n_samples: Number of frames to sample for logging
        """
        cam_array = self.replay_buffer[camera_key]
        n_frames = cam_array.shape[0]
        if n_frames == 0:
            logging.info(f"Camera {camera_key}: empty array")
            return
        sample_indices = np.random.choice(
            n_frames, size=min(n_samples, n_frames), replace=False
        )
        sample = cam_array[sample_indices]
        logging.info(
            f"Camera {camera_key} stats (sampled {len(sample_indices)} frames) - "
            f"min: {sample.min():.4f}, max: {sample.max():.4f}, "
            f"mean: {sample.mean():.4f}, std: {sample.std():.4f}"
        )
        camera_metadata = self.observation_space.cameras[camera_key]
        if not camera_metadata.is_depth:
            sample = sample.astype(np.float32) / 255.0
        sample_normalized = normalizer[camera_key].normalize(sample)
        logging.info(
            f"Camera {camera_key} after normalization (sampled) - "
            f"mean: {tensor_to_str(sample_normalized.mean())}, "
            f"std: {tensor_to_str(sample_normalized.std())}, "
            f"min: {tensor_to_str(sample_normalized.min())}, "
            f"max: {tensor_to_str(sample_normalized.max())}"
        )

    def _log_normalized_proprio_stats(
        self, normalizer: LinearNormalizer, proprio_data: dict[str, np.ndarray]
    ) -> None:
        """Log proprioceptive statistics before and after normalization.

        Args:
            normalizer: Configured normalizer
            proprio_data: Proprioceptive data used for fitting
        """
        camera_keys = list(self.observation_space.cameras.keys())
        for key in normalizer.params_dict.keys():
            if key in camera_keys:
                continue
            pre_norm = normalizer[key].get_input_stats()
            logging.info(
                f"{key} stats before normalization - "
                f"min: {tensor_to_str(pre_norm['min'])}, max: {tensor_to_str(pre_norm['max'])}, "
                f"mean: {tensor_to_str(pre_norm['mean'])}, std: {tensor_to_str(pre_norm['std'])}"
            )
            after_norm = normalizer[key].normalize(proprio_data[key])
            logging.info(
                f"{key} stats after normalization - "
                f"mean: {tensor_to_str(after_norm.mean())}, "
                f"std: {tensor_to_str(after_norm.std())},"
                f"min:  {tensor_to_str(after_norm.min())}, "
                f"max:  {tensor_to_str(after_norm.max())}"
            )
