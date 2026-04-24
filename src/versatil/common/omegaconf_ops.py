"""OmegaConf utility operations."""

from collections.abc import Mapping
from typing import Any

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
