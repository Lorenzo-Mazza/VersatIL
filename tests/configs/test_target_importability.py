"""Tests for versatil.configs ConfigStore target importability."""

import pytest
from hydra.core.config_store import ConfigStore
from hydra.utils import get_object
from omegaconf import OmegaConf

from versatil.configs import register_configs


def _collect_targets(node: dict | list) -> set[str]:
    targets: set[str] = set()
    if isinstance(node, dict):
        target = node.get("_target_")
        if isinstance(target, str) and target != "???" and "${" not in target:
            targets.add(target)
        for value in node.values():
            targets |= _collect_targets(value)
    elif isinstance(node, list):
        for value in node:
            targets |= _collect_targets(value)
    return targets


def _registered_targets() -> list[str]:
    register_configs()
    targets: set[str] = set()
    stack: list[dict] = [ConfigStore.instance().repo]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
            continue
        config_node = getattr(current, "node", None)
        if config_node is None:
            continue
        container = OmegaConf.to_container(
            config_node, resolve=False, throw_on_missing=False
        )
        targets |= _collect_targets(container)
    return sorted(targets)


@pytest.mark.unit
@pytest.mark.parametrize("target", _registered_targets())
def test_registered_target_is_importable(target: str):
    get_object(target)
