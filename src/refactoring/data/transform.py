import torch

from refactoring.data.constants import OBSERVATION_KEY, LANGUAGE_KEY, GRIPPER_STATE_OBS_KEY, GripperType, ACTION_KEY, POSITION_ACTION_KEY, \
    ORIENTATION_ACTION_KEY, GRIPPER_ACTION_KEY, TOKENIZED_OBSERVATIONS_KEY, IS_PAD_OBSERVATION_KEY, IS_PAD_ACTION_KEY, PHASE_LABEL_KEY, TOKENIZED_ACTIONS_KEY
from refactoring.data.normalization.normalizer import LinearNormalizer
from refactoring.data.task import ObservationSpace, ActionSpace
from refactoring.data.tokenization import ActionTokenizer, Tokenizer, ObservationTokenizer


def normalize_sample(
        sample: dict[str, dict[str, torch.Tensor]],
        normalizer: LinearNormalizer,
        observation_space: ObservationSpace,
        action_space:ActionSpace
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
        sample_copy = sample.copy() # Avoid modifying in-place
        observation = sample_copy[OBSERVATION_KEY]
        sample_copy[OBSERVATION_KEY] = normalize_observation(observation=observation,
                                                             normalizer=normalizer, observation_space=observation_space)
        actions = sample_copy[ACTION_KEY]
        sample_copy[ACTION_KEY] = normalize_actions(actions=actions, normalizer=normalizer, action_space=action_space)
        return sample_copy


def normalize_observation(
        observation: dict[str, torch.Tensor],
        normalizer: LinearNormalizer,
        observation_space: ObservationSpace
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
        if observation_space.use_language:
            # Language observations are not normalized
            del normalized_observation[LANGUAGE_KEY]
        if observation_space.use_gripper_state and observation_space.gripper_type == GripperType.BINARY.value:
            # Binary gripper actions are not normalized
            del normalized_observation[GRIPPER_STATE_OBS_KEY]
        normalized_observation = normalizer.normalize(normalized_observation)  # type: ignore[assignment]
        # Restore non-normalized keys
        if observation_space.use_language:
            normalized_observation[LANGUAGE_KEY] = observation[LANGUAGE_KEY]
        if observation_space.use_gripper_state and observation_space.gripper_type == GripperType.BINARY.value:
            normalized_observation[GRIPPER_STATE_OBS_KEY] = observation[GRIPPER_STATE_OBS_KEY]
        return normalized_observation


def normalize_actions(
        actions: dict[str, torch.Tensor],
        normalizer: LinearNormalizer,
        action_space: ActionSpace
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
    if action_space.has_position:
        normalized_actions[POSITION_ACTION_KEY] = normalizer[POSITION_ACTION_KEY].normalize(actions[POSITION_ACTION_KEY])
    if action_space.has_orientation:
        normalized_actions[ORIENTATION_ACTION_KEY] = normalizer[ORIENTATION_ACTION_KEY].normalize(actions[ORIENTATION_ACTION_KEY])
    if action_space.gripper_type == GripperType.CONTINUOUS.value:
        normalized_actions[GRIPPER_ACTION_KEY] = normalizer[GRIPPER_ACTION_KEY].normalize(actions[GRIPPER_ACTION_KEY])
    if action_space.custom_action_dims:
        for custom_key in action_space.custom_action_dims:
            normalized_actions[custom_key] = normalizer[custom_key].normalize(actions[custom_key])
    return normalized_actions


def unnormalize_actions(
        normalized_actions: dict[str, torch.Tensor],
        normalizer: LinearNormalizer,
        action_space: ActionSpace
) -> dict[str, torch.Tensor]:
    actions = normalized_actions.copy()
    if action_space.has_position:
        actions[POSITION_ACTION_KEY] = normalizer[POSITION_ACTION_KEY].unnormalize(normalized_actions[POSITION_ACTION_KEY])
    if action_space.has_orientation:
        actions[ORIENTATION_ACTION_KEY] = normalizer[ORIENTATION_ACTION_KEY].unnormalize(normalized_actions[ORIENTATION_ACTION_KEY])
    if action_space.gripper_type == GripperType.CONTINUOUS.value:
        actions[GRIPPER_ACTION_KEY] = normalizer[GRIPPER_ACTION_KEY].unnormalize(normalized_actions[GRIPPER_ACTION_KEY])
    if action_space.custom_action_dims:
        for custom_key in action_space.custom_action_dims:
            actions[custom_key] = normalizer[custom_key].unnormalize(normalized_actions[custom_key])
    # If gripper is binary, no unnormalization is applied. Also, no unnormalization is applied to padding mask and phase labels if present.
    return actions


def tokenize_sample(
        sample: dict[str, dict[str, torch.Tensor]],
        tokenizer: Tokenizer,
) -> dict[str, dict[str, torch.Tensor]]:
    """Tokenize observations and actins according to the action space configuration."""
    if tokenizer.observation_tokenizer is not None:
        observation = sample[OBSERVATION_KEY]
        sample[OBSERVATION_KEY] = tokenize_observation(observation=observation, obs_tokenizer=tokenizer.observation_tokenizer)
    if tokenizer.action_tokenizer is not None:
        actions = sample[ACTION_KEY]
        sample[ACTION_KEY] = tokenize_actions(actions=actions, action_tokenizer=tokenizer.action_tokenizer)
    return sample


def tokenize_observation(
        observation: dict[str, torch.Tensor],
        obs_tokenizer: ObservationTokenizer
) -> dict[str, torch.Tensor]:
    """Tokenize observations."""
    obs_copy = observation.copy()
    obs_to_tokenize = {}
    for key in obs_tokenizer.observation_keys:
        if key in obs_copy:
            obs_to_tokenize[key] = obs_copy[key]
        else:
            raise KeyError(f"Observation key '{key}' not found in sample for tokenization.")
    tokenized = obs_tokenizer.tokenize(obs_to_tokenize)
    obs_copy[TOKENIZED_OBSERVATIONS_KEY] = tokenized[TOKENIZED_OBSERVATIONS_KEY]
    obs_copy[IS_PAD_OBSERVATION_KEY] = tokenized[IS_PAD_OBSERVATION_KEY]
    return obs_copy

def tokenize_actions(
        actions: dict[str, torch.Tensor],
        action_tokenizer: ActionTokenizer,
) -> dict[str, torch.Tensor]:
    """Tokenize actions."""
    actions_to_tokenize = actions.copy()
    action_components = []
    for key in actions_to_tokenize.keys():
        if key not in [IS_PAD_ACTION_KEY, PHASE_LABEL_KEY]:
            action_tensor = actions_to_tokenize[key]
            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(-1)
            action_components.append(action_tensor)
    # Concatenate along last dimension: (pred_horizon, action_dim)
    action_tensor = torch.cat(action_components, dim=-1)
    is_pad_mask = actions_to_tokenize.get(IS_PAD_ACTION_KEY, None)
    tokenized = action_tokenizer.encode(action_tensor, is_pad_mask=is_pad_mask)
    actions_to_tokenize[TOKENIZED_ACTIONS_KEY] = tokenized[TOKENIZED_ACTIONS_KEY]
    actions_to_tokenize[IS_PAD_ACTION_KEY] = tokenized[IS_PAD_ACTION_KEY] # Replace padding mask with tokenizer mask
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
    if action_space.has_position:
        pos_dim = action_space.position_dim
        action_dict[POSITION_ACTION_KEY] = actions[..., current_idx:current_idx + pos_dim]
        current_idx += pos_dim

    if action_space.has_orientation:
        ori_dim = action_space.orientation_dim
        action_dict[ORIENTATION_ACTION_KEY] = actions[..., current_idx:current_idx + ori_dim]
        current_idx += ori_dim

    if action_space.has_gripper:
        gripper_dim = action_space.gripper_dim
        if action_space.gripper_type == GripperType.BINARY.value:
            # Binary gripper: round to 0 or 1
            action_dict[GRIPPER_ACTION_KEY] = torch.round(actions[..., current_idx:current_idx + gripper_dim]).long()
        else:
            action_dict[GRIPPER_ACTION_KEY] = actions[..., current_idx:current_idx + gripper_dim]
        current_idx += gripper_dim
    return action_dict
