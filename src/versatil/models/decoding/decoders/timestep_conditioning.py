"""Shared validation helpers for timestep-conditioned action decoders."""

from collections.abc import Mapping

import torch

from versatil.models.decoding.action_heads.base import BaseActionHead
from versatil.models.decoding.constants import DecoderOutputKey


def validate_noisy_action_tensors(
    actions: dict[str, torch.Tensor],
    action_heads: Mapping[str, BaseActionHead],
    prediction_horizon: int,
    decoder_name: str,
) -> tuple[int, torch.device]:
    """Validate noisy action tensors before concatenating them.

    Args:
        actions: Dictionary of noisy actions provided by the decoding algorithm.
        action_heads: Decoder action heads keyed by action name.
        prediction_horizon: Expected action chunk length.
        decoder_name: Name used in error messages.

    Returns:
        Batch size and device shared by all action tensors.

    Raises:
        ValueError: If keys, ranks, horizon, dimensions, batch size, or devices are inconsistent.
    """
    expected_action_keys = sorted(action_heads.keys())
    if not expected_action_keys:
        raise ValueError(f"{decoder_name} requires at least one action head.")

    actual_action_keys = sorted(actions.keys())
    if actual_action_keys != expected_action_keys:
        raise ValueError(
            f"{decoder_name} expected action keys "
            f"{expected_action_keys}, got {actual_action_keys}."
        )

    first_action_key = ""
    batch_size = 0
    action_device = torch.device("cpu")
    for action_key in expected_action_keys:
        action = actions[action_key]
        if action.ndim != 3:
            raise ValueError(
                f"Action '{action_key}' must have shape "
                f"(B, prediction_horizon, action_dim), got {action.shape}."
            )
        if action.shape[1] != prediction_horizon:
            raise ValueError(
                f"Action '{action_key}' must have prediction horizon "
                f"{prediction_horizon}, got {action.shape[1]}."
            )
        expected_dimension = action_heads[action_key].output_dim
        if action.shape[2] != expected_dimension:
            raise ValueError(
                f"Action '{action_key}' must have last dimension "
                f"{expected_dimension}, got {action.shape[2]}."
            )
        if first_action_key == "":
            first_action_key = action_key
            batch_size = action.shape[0]
            action_device = action.device
            continue
        if action.shape[0] != batch_size:
            raise ValueError(
                "All action tensors must have the same batch size, "
                f"got {action.shape[0]} for '{action_key}' and "
                f"{batch_size} for '{first_action_key}'."
            )
        if action.device != action_device:
            raise ValueError(
                "All action tensors must be on the same device, "
                f"got {action.device} for '{action_key}' and "
                f"{action_device} for '{first_action_key}'."
            )

    return batch_size, action_device


def extract_timestep_conditioning(
    features: dict[str, torch.Tensor],
    batch_size: int,
    action_device: torch.device,
) -> torch.Tensor:
    """Extract and validate timestep conditioning without mutating features.

    Args:
        features: Decoder feature dictionary.
        batch_size: Batch size inferred from action tensors.
        action_device: Device shared by action tensors.

    Returns:
        Timestep tensor with shape ``(B,)``.

    Raises:
        ValueError: If the timestep key is missing or has an invalid shape, batch size, or device.
    """
    if DecoderOutputKey.TIMESTEP.value not in features:
        raise ValueError(
            f"Missing '{DecoderOutputKey.TIMESTEP.value}' in features dict. "
            "The algorithm should inject timesteps into features."
        )
    timesteps = features[DecoderOutputKey.TIMESTEP.value]
    if timesteps.ndim == 2 and timesteps.shape[-1] == 1:
        timesteps = timesteps.squeeze(-1)
    if timesteps.ndim != 1:
        raise ValueError(
            f"'{DecoderOutputKey.TIMESTEP.value}' must have shape "
            f"(B,) or (B, 1), got {timesteps.shape}."
        )
    if timesteps.shape[0] != batch_size:
        raise ValueError(
            f"'{DecoderOutputKey.TIMESTEP.value}' batch size must match "
            f"actions batch size {batch_size}, got {timesteps.shape[0]}."
        )
    if timesteps.device != action_device:
        raise ValueError(
            f"'{DecoderOutputKey.TIMESTEP.value}' must be on the same device "
            f"as actions, got {timesteps.device} and {action_device}."
        )
    return timesteps


def filter_timestep_feature(
    features: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return observation features without timestep conditioning.

    Args:
        features: Decoder feature dictionary.

    Returns:
        New feature dictionary without the timestep key.
    """
    return {
        key: value
        for key, value in features.items()
        if key != DecoderOutputKey.TIMESTEP.value
    }
