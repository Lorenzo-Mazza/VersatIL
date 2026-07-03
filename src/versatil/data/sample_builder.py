"""Sample construction for episodic dataset.

Builds the final training/validation samples by:
- Processing images
- Adding proprioceptive data
- Adding actions
- Computing padding masks
- Normalizing (and optionally tokenizing) observations and actions
"""

import numpy as np
import torch

from versatil.data.constants import (
    SampleKey,
)
from versatil.data.metadata import ActionMetadata, ObservationMetadata
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.processing.action_processor import ActionProcessor
from versatil.data.processing.image_processor import ImageProcessor
from versatil.data.processing.transform import normalize_sample, tokenize_sample
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import Tokenizer


class SampleBuilder:
    """Builds training samples from raw episode data."""

    def __init__(
        self,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        obs_horizon: int,
        pred_horizon: int,
        action_backward_shift: int,
        image_processor: ImageProcessor,
        action_processor: ActionProcessor,
        tokenizer: Tokenizer | None = None,
        normalizer: LinearNormalizer | None = None,
    ):
        """
        Args:
            action_space: Action space of the experiment
            observation_space: Observation space of the experiment
            obs_horizon: Observation history length
            pred_horizon: Action prediction horizon
            action_backward_shift: Backward shift to give to action timesteps (if actions have latency)
            image_processor: Handles augmentations
            action_processor: Processes actions
            tokenizer: Unified tokenizer for observations and actions
            normalizer: Normalizer for observations and actions
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_backward_shift = action_backward_shift
        self.image_processor = image_processor
        self.action_processor = action_processor
        self.tokenizer = tokenizer
        self.normalizer = normalizer

    def build_sample(
        self,
        padded_data: dict[str, np.ndarray],
        action_data: dict[str, np.ndarray],
        action_meta: dict[str, ActionMetadata],
        start_idx: int,
        sampler_indices: np.ndarray,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Build a complete training sample.

        Args:
            padded_data: Dictionary of padded episode data
            action_data: Dictionary of computed actions
            action_meta: Dictionary of action metadata
            start_idx: Starting index in sampler
            sampler_indices: Array of sampler indices

        Note:
            Padded data layout: [historical buffer | observation window | future]
                                    [0:k]             [k:k+H]            [k+H:end]
            where k=action_backward_shift, H=obs_horizon, end= prediction_horizon+k+H

        Returns:
            Dictionary containing observation and action dictionaries. Each sub-dictionary maps keys to tensors.
        """
        sample: dict[str, dict[str, torch.Tensor]] = {
            SampleKey.OBSERVATION.value: {},
            SampleKey.ACTION.value: {},
        }
        image_dict = self._get_sample_images(padded_data=padded_data)
        sample[SampleKey.OBSERVATION.value].update(image_dict)
        for key, metadata in self.observation_space.observations_metadata.items():
            if isinstance(metadata, ObservationMetadata):  # Excludes cameras
                sample[SampleKey.OBSERVATION.value].update(
                    {
                        key: self._slice_observation_tensor(
                            key=key, metadata=metadata, padded_data=padded_data
                        )
                    }
                )
        for key, _data in action_data.items():
            metadata = action_meta[key]
            sample[SampleKey.ACTION.value].update(
                {
                    key: self._slice_action_data(
                        key=key, metadata=metadata, action_data=action_data
                    )
                }
            )

        sample[SampleKey.ACTION.value][SampleKey.IS_PAD_ACTION.value] = (
            self._compute_action_padding_mask(
                start_idx=start_idx, sampler_indices=sampler_indices
            )
        )
        sample = self.normalize_and_tokenize_sample(sample=sample)
        return sample

    def normalize_and_tokenize_sample(
        self,
        sample: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Normalize and tokenize a pre-built sample.

        Args:
            sample: Pre-built sample with observation and action dictionaries.

        Returns:
            Normalized and tokenized sample.
        """
        if self.normalizer is not None:
            sample = normalize_sample(
                sample=sample,
                normalizer=self.normalizer,
                observation_space=self.observation_space,
                action_space=self.action_space,
            )
        if self.tokenizer is not None:
            sample = tokenize_sample(
                sample=sample, tokenizer=self.tokenizer, action_space=self.action_space
            )
        return sample

    def _get_sample_images(
        self,
        padded_data: dict[str, np.ndarray],
    ) -> dict[str, torch.Tensor]:
        """Process images and return them as a dictionary of tensors."""
        image_dict = {}
        self.image_processor.begin_sample()
        for cam in self.observation_space.cameras:
            # the sampler now fetches action_backward_shift extra prior observations at the start of padded_data, effectively offsetting the entire sequence
            img = padded_data[cam][
                self.action_backward_shift : self.action_backward_shift
                + self.obs_horizon
            ]
            image_dict[cam] = self.image_processor.process(images=img, camera_key=cam)
        return image_dict

    def _slice_observation_tensor(
        self,
        key: str,
        padded_data: dict[str, np.ndarray],
        metadata: ObservationMetadata,
    ) -> torch.Tensor | list[list[str]]:
        """Slice `key` from observation data, and return it as tensor."""
        observation_data = padded_data[key][
            self.action_backward_shift : self.action_backward_shift + self.obs_horizon
        ]
        if metadata.dtype == "str":
            # TODO: get rid of squeeze when tokenizer todo is addressed
            return observation_data.squeeze(axis=-1).tolist()
        elif "float" in metadata.dtype:
            return torch.from_numpy(observation_data).float()
        elif "int" in metadata.dtype or "bool" in metadata.dtype:
            return torch.from_numpy(observation_data).long()
        else:
            raise ValueError(f"Unsupported custom observation dtype: {metadata.dtype}")

    @staticmethod
    def _slice_action_data(
        key: str,
        action_data: dict[str, np.ndarray],
        metadata: ActionMetadata,
    ) -> torch.Tensor | list[list[str]]:
        """Slice `key` from action data and return it as tensor."""
        action_array = action_data[key]
        if metadata.dtype == "str":
            return action_array.tolist()
        elif "float" in metadata.dtype:
            return torch.from_numpy(action_array).float()
        elif "int" in metadata.dtype or "bool" in metadata.dtype:
            return torch.from_numpy(action_array).long()
        else:
            raise ValueError(f"Unsupported custom action dtype: {metadata.dtype}")

    def _compute_action_padding_mask(
        self,
        start_idx: int,
        sampler_indices: np.ndarray,
    ) -> torch.Tensor:
        """Add action padding mask indicating which timesteps are padded."""
        action_slice_start = self._get_action_slice_start()
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = sampler_indices[start_idx]

        action_positions = np.arange(self.pred_horizon) + action_slice_start

        # A timestep is padded if any action component at that timestep needs a
        # position outside the valid window. Precomputed actions are read from
        # zarr at the current position; on-the-fly actions read the next
        # position, and deltas additionally read the current one. Mixed action
        # spaces take the union of these requirements.
        needs_current_position = (
            self.action_space.has_precomputed_actions
            or self.action_space.has_delta_actions
        )
        needs_next_position = self.action_space.has_on_the_fly_actions

        is_pad = np.zeros(self.pred_horizon, dtype=bool)
        if needs_current_position:
            is_pad |= np.logical_or(
                action_positions < sample_start_idx,
                action_positions >= sample_end_idx,
            )
        if needs_next_position:
            is_pad |= np.logical_or(
                action_positions + 1 < sample_start_idx,
                action_positions + 1 >= sample_end_idx,
            )
        return torch.from_numpy(is_pad).bool()

    def _get_action_slice_start(self) -> int:
        """Get the starting index for action slice."""
        return self.obs_horizon - 1
