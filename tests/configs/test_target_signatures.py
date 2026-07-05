"""Tests for config-field compatibility with Hydra target signatures."""

import inspect
import pathlib
from dataclasses import fields, is_dataclass

import pytest
import yaml
from hydra.core.config_store import ConfigStore
from hydra.utils import get_object

from versatil.configs import register_configs
from versatil.configs.paths import get_hydra_configs_dir

HYDRA_SPECIAL_KEYS = {
    "_target_",
    "_partial_",
    "_recursive_",
    "_convert_",
    "_args_",
    "defaults",
}


def _is_checkable_target(target: object) -> bool:
    return isinstance(target, str) and target != "???" and "${" not in target


def _unaccepted_keys(target: str, keys: set[str]) -> list[str]:
    target_object = get_object(target)
    parameters = inspect.signature(target_object).parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return []
    return sorted(keys - set(parameters))


def _config_store_cases() -> list[tuple[str, str, set[str]]]:
    register_configs()
    cases = []
    stack: list[tuple[str, object]] = [("", ConfigStore.instance().repo)]
    while stack:
        path, current = stack.pop()
        if isinstance(current, dict):
            stack.extend((f"{path}/{key}", value) for key, value in current.items())
            continue
        node = getattr(current, "node", None)
        node_type = getattr(node, "_metadata", None)
        object_type = getattr(node_type, "object_type", None)
        if object_type is None or not is_dataclass(object_type):
            continue
        field_names = {field.name for field in fields(object_type)}
        target = next(
            (
                field.default
                for field in fields(object_type)
                if field.name == "_target_"
            ),
            None,
        )
        if not _is_checkable_target(target):
            continue
        cases.append((path, target, field_names - HYDRA_SPECIAL_KEYS))
    return sorted(cases)


def _yaml_cases() -> list[tuple[str, str, set[str]]]:
    configs_root = pathlib.Path(get_hydra_configs_dir())
    cases = []
    for yaml_path in sorted(configs_root.rglob("*.yaml")):
        document = yaml.safe_load(yaml_path.read_text())
        stack = [document]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                stack.extend(current.values())
                target = current.get("_target_")
                if _is_checkable_target(target):
                    keys = set(current) - HYDRA_SPECIAL_KEYS
                    cases.append(
                        (str(yaml_path.relative_to(configs_root)), target, keys)
                    )
            elif isinstance(current, list):
                stack.extend(current)
    return sorted(cases)


@pytest.mark.unit
@pytest.mark.parametrize(
    "location, target, keys",
    _config_store_cases(),
    ids=[f"{path}::{target}" for path, target, keys in _config_store_cases()],
)
def test_config_store_fields_match_target_signature(
    location: str, target: str, keys: set[str]
):
    unaccepted = _unaccepted_keys(target=target, keys=keys)
    assert unaccepted == [], (
        f"Config node '{location}' declares fields {unaccepted} that "
        f"'{target}' does not accept."
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "location, target, keys",
    _yaml_cases(),
    ids=[f"{path}::{target}" for path, target, keys in _yaml_cases()],
)
def test_yaml_keys_match_target_signature(location: str, target: str, keys: set[str]):
    unaccepted = _unaccepted_keys(target=target, keys=keys)
    assert unaccepted == [], (
        f"YAML '{location}' passes keys {unaccepted} that '{target}' does not accept."
    )


def _exported_dataclass_cases() -> list[tuple[str, str, set[str]]]:
    import versatil.configs as configs_package  # noqa: PLC0415

    cases = []
    for name, exported in vars(configs_package).items():
        if not (isinstance(exported, type) and is_dataclass(exported)):
            continue
        field_names = {field.name for field in fields(exported)}
        if "_target_" not in field_names:
            continue
        target = next(
            field.default for field in fields(exported) if field.name == "_target_"
        )
        if not _is_checkable_target(target):
            continue
        cases.append(
            (f"versatil.configs.{name}", target, field_names - HYDRA_SPECIAL_KEYS)
        )
    return sorted(cases)


@pytest.mark.unit
@pytest.mark.parametrize(
    "location, target, keys",
    _exported_dataclass_cases(),
    ids=[f"{path}::{target}" for path, target, keys in _exported_dataclass_cases()],
)
def test_exported_config_fields_match_target_signature(
    location: str, target: str, keys: set[str]
):
    unaccepted = _unaccepted_keys(target=target, keys=keys)
    assert unaccepted == [], (
        f"Config class '{location}' declares fields {unaccepted} that "
        f"'{target}' does not accept."
    )
