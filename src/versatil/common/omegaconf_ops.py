"""OmegaConf utility operations."""

from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import OmegaConf


def resolve_dict_keys(d: Mapping[Any, Any]) -> dict[Any, Any]:
    """Resolve any OmegaConf interpolations in dictionary keys recursively.

    OmegaConf doesn't resolve interpolations in dict keys by default.
    This function resolves keys like '${cameras:AGENTVIEW}' to 'agentview_rgb'.

    Args:
        d: Dictionary with potentially unresolved interpolation keys.

    Returns:
        New dictionary with resolved keys.
    """
    resolved = {}
    for key, value in d.items():
        if isinstance(key, str) and key.startswith("${") and key.endswith("}"):
            temp_cfg = OmegaConf.create({"_key": key})
            OmegaConf.resolve(temp_cfg)
            resolved_key = OmegaConf.select(temp_cfg, "_key")
        else:
            resolved_key = key
        resolved_value = (
            resolve_dict_keys(value) if isinstance(value, Mapping) else value
        )
        resolved[resolved_key] = resolved_value
    return resolved


def make_config_yaml_safe(value: Any) -> Any:
    """Convert resolved config values unsupported by OmegaConf into YAML-safe values."""
    if isinstance(value, torch.dtype):
        # Serialize as the resolver interpolation so reloading the saved
        # config reconstructs the real dtype
        return "${torch_dtype:" + str(value).removeprefix("torch.") + "}"
    if isinstance(value, dict):
        return {
            make_config_yaml_safe(key): make_config_yaml_safe(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [make_config_yaml_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(make_config_yaml_safe(item) for item in value)
    return value
