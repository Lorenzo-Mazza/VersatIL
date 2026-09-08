"""Tests for versatil.configs.store.quantization module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from hydra.core.config_store import ConfigStore

from versatil.configs.quantization import (
    DirectQuantizationSchemaConfig,
    QuantizationSchemaConfig,
    SmoothQuantSchemaConfig,
)
from versatil.configs.store.quantization import register


@pytest.fixture
def config_store_factory() -> Callable[[], MagicMock]:
    def factory() -> MagicMock:
        return MagicMock(spec=ConfigStore)

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, schema_config",
    [
        ("direct", DirectQuantizationSchemaConfig),
        ("smoothquant", SmoothQuantSchemaConfig),
    ],
)
def test_registers_concrete_schemas_in_the_quantization_schema_group(
    config_store_factory: Callable[[], MagicMock],
    name: str,
    schema_config: type[QuantizationSchemaConfig],
) -> None:
    config_store = config_store_factory()

    register(cs=config_store)

    config_store.store.assert_any_call(
        group="quantization/schema", name=name, node=schema_config
    )
