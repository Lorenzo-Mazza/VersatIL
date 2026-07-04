import logging

import torch

from versatil.data.constants import (
    BinaryGripperRange,
    GripperType,
    ProprioceptiveType,
    SampleKey,
)
from versatil.data.metadata import (
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
)
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import (
    ActionTokenizer,
    ObservationTokenizer,
    Tokenizer,
)


def normalize_sample(
    sample: dict[str, dict[str, torch.Tensor]],
    normalizer: LinearNormalizer,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
) -> dict[str, dict[str, torch.Tensor]]:
    """Normalize and tokenize a pre-built sample.

    Args:
        sample: Pre-built sample with observation and action dictionaries.
        normalizer: Normalizer to use for normalization.
        observation_space: Observation space configuration.
        action_space: Action space configuration.

    Returns:
        Normalized and tokenized sample.
    """
    sample_copy = sample.copy()  # Avoid modifying in-place
    observation = sample_copy[SampleKey.OBSERVATION.value]
    sample_copy[SampleKey.OBSERVATION.value] = normalize_observation(
        observation=observation,
        normalizer=normalizer,
        observation_space=observation_space,
    )
    actions = sample_copy[SampleKey.ACTION.value]
    sample_copy[SampleKey.ACTION.value] = normalize_actions(
        actions=actions, normalizer=normalizer, action_space=action_space
    )
    return sample_copy


def normalize_observation(
    observation: dict[str, torch.Tensor],
    normalizer: LinearNormalizer,
    observation_space: ObservationSpace,
) -> dict[str, torch.Tensor]:
    """Normalize observations.

    Args:
        observation: Observation dictionary.
        normalizer: Normalizer to use for normalization.
        observation_space: Observation space configuration.

    Returns:
        Normalized observation dictionary.
    """
    normalized_observation = observation.copy()
    for key in observation_space.observations_metadata:
        if key in normalizer.params_dict.keys():
            normalized_observation[key] = normalizer[key].normalize(observation[key])
    return normalized_observation


def normalize_actions(
    actions: dict[str, torch.Tensor],
    normalizer: LinearNormalizer,
    action_space: ActionSpace,
) -> dict[str, torch.Tensor]:
    """Normalize actions.

    Args:
        actions: Action dictionary.
        normalizer: Normalizer to use for normalization.
        action_space: Action space configuration.

    Returns:
        Normalized action dictionary.
    """
    normalized_actions = actions.copy()
    for key in action_space.actions_metadata:
        if key in normalizer.params_dict.keys():
            normalized_actions[key] = normalizer[key].normalize(actions[key])
    return normalized_actions


def unnormalize_actions(
    normalized_actions: dict[str, torch.Tensor],
    normalizer: LinearNormalizer,
    action_space: ActionSpace,
) -> dict[str, torch.Tensor]:
    """Unnormalize actions using the normalizer with per-key statistics."""
    actions = normalized_actions.copy()
    for key in action_space.actions_metadata:
        if key in normalizer.params_dict.keys():
            actions[key] = normalizer[key].unnormalize(normalized_actions[key])
    return actions


def tokenize_sample(
    sample: dict[str, dict[str, torch.Tensor]],
    tokenizer: Tokenizer,
    action_space: ActionSpace,
) -> dict[str, dict[str, torch.Tensor]]:
    """Tokenize observations and actions according to the action space configuration."""
    if tokenizer.observation_tokenizer is not None:
        observation = sample[SampleKey.OBSERVATION.value]
        sample[SampleKey.OBSERVATION.value] = tokenize_observation(
            observation=observation, obs_tokenizer=tokenizer.observation_tokenizer
        )
    if tokenizer.action_tokenizer is not None:
        actions = sample[SampleKey.ACTION.value]
        sample[SampleKey.ACTION.value] = tokenize_actions(
            actions=actions,
            action_space=action_space,
            action_tokenizer=tokenizer.action_tokenizer,
        )
    return sample


def tokenize_observation(
    observation: dict[str, torch.Tensor], obs_tokenizer: ObservationTokenizer
) -> dict[str, torch.Tensor]:
    """Tokenize observations."""
    obs_copy = observation.copy()
    obs_to_tokenize = {}
    for key in obs_tokenizer.observation_keys:
        if key in obs_copy:
            obs_to_tokenize[key] = obs_copy[key]
        else:
            raise KeyError(
                f"Observation key '{key}' not found in sample for tokenization."
            )
    tokenized = obs_tokenizer.tokenize(obs_to_tokenize)
    obs_copy[SampleKey.TOKENIZED_OBSERVATIONS.value] = tokenized[
        SampleKey.TOKENIZED_OBSERVATIONS.value
    ]
    obs_copy[SampleKey.IS_PAD_OBSERVATION.value] = tokenized[
        SampleKey.IS_PAD_OBSERVATION.value
    ]
    return obs_copy


def tokenize_actions(
    actions: dict[str, torch.Tensor],
    action_tokenizer: ActionTokenizer,
    action_space: ActionSpace,
) -> dict[str, torch.Tensor]:
    """Tokenize actions."""
    actions_to_tokenize = actions.copy()
    action_components = []
    for key in sorted(action_space.actions_metadata.keys()):
        meta = action_space.actions_metadata[key]
        if meta.is_numerical and meta.requires_prediction_head:
            action_tensor = actions_to_tokenize[key]
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(-1)
            action_components.append(action_tensor)

    # Concatenate along last dimension: (pred_horizon, action_dim)
    action_tensor = torch.cat(action_components, dim=-1)
    is_pad_mask = actions_to_tokenize.get(SampleKey.IS_PAD_ACTION.value, None)
    tokenized = action_tokenizer.encode(action_tensor, is_pad_mask=is_pad_mask)
    actions_to_tokenize[SampleKey.TOKENIZED_ACTIONS.value] = tokenized[
        SampleKey.TOKENIZED_ACTIONS.value
    ]
    actions_to_tokenize[SampleKey.IS_PAD_ACTION.value] = tokenized[
        SampleKey.IS_PAD_ACTION.value
    ]  # Replace padding mask with tokenizer mask
    return actions_to_tokenize


def detokenize_actions(
    action_tokens: torch.Tensor,
    action_tokenizer: ActionTokenizer,
    action_space: ActionSpace,
) -> dict[str, torch.Tensor]:
    """Detokenize actions."""
    action_tokens_cpu = action_tokens.cpu()
    # Squeeze if needed: (B, pred_horizon, 1) -> (B, pred_horizon)
    if action_tokens_cpu.ndim == 3 and action_tokens_cpu.shape[-1] == 1:
        action_tokens_cpu = action_tokens_cpu.squeeze(-1)
    batch_size, pred_horizon = action_tokens_cpu.shape[:2]
    detokenized_actions = []
    for chunk in range(batch_size):
        actions_np = action_tokenizer.decode(action_tokens_cpu[chunk])
        detokenized_actions.append(torch.from_numpy(actions_np).float())
    actions = torch.stack(detokenized_actions, dim=0)  # (B, pred_horizon, action_dim)
    action_dict = {}
    current_idx = 0
    for key in sorted(action_space.actions_metadata.keys()):
        meta = action_space.actions_metadata[key]
        if meta.is_numerical and meta.requires_prediction_head:
            dim = meta.prediction_dimension
            action_dict[key] = actions[..., current_idx : current_idx + dim]
            current_idx += dim
            if meta.action_type == ProprioceptiveType.GRIPPER.value:
                if isinstance(meta, GripperActionMetadata):
                    gripper_meta = meta
                elif isinstance(meta, OnTheFlyActionMetadata):
                    if not isinstance(meta.source_metadata, GripperObservationMetadata):
                        raise TypeError(
                            f"Expected GripperObservationMetadata, got {type(meta.source_metadata)}"
                        )
                    gripper_meta = meta.source_metadata
                else:
                    raise TypeError(
                        f"Unexpected metadata type for gripper action: {type(meta)}"
                    )
                if gripper_meta.gripper_type == GripperType.BINARY.value:
                    if (
                        gripper_meta.binary_gripper_range
                        == BinaryGripperRange.MINUS_ONE_ONE.value
                    ):
                        action_dict[key] = torch.where(
                            action_dict[key] > 0.0, 1, -1
                        ).long()
                    else:
                        if (
                            gripper_meta.binary_gripper_range
                            != BinaryGripperRange.ZERO_ONE.value
                        ):
                            logging.warning(
                                "Gripper type is binary but range is not set. "
                                "Assuming {0,1}."
                            )
                        action_dict[key] = (action_dict[key] > 0.5).long()

    return action_dict
