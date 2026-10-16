"""Tests for versatil.quantization.resolution module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from versatil.quantization.module_target import EagerQuantizationModuleTarget
from versatil.quantization.resolution import (
    resolve_quantization_targets,
    validate_quantization_targets,
    validate_target_configuration,
)

RESOLUTION_MODULE = "versatil.quantization.resolution"


class QuantizationValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        self.decoder = nn.Linear(8, 2)


@pytest.fixture
def eager_target_factory() -> Callable[[str], EagerQuantizationModuleTarget]:
    """Factory for eager targets used by shared validator tests."""

    def factory(module_path: str) -> EagerQuantizationModuleTarget:
        return EagerQuantizationModuleTarget(
            module_path=module_path,
            quantize_config=MagicMock(spec=[]),
        )

    return factory


@pytest.mark.unit
class TestQuantizationTargetResolution:
    @pytest.mark.parametrize("conversion_device", [None, torch.device("cuda:1")])
    @pytest.mark.parametrize("for_conversion", [False, True])
    @pytest.mark.parametrize("with_backend", [False, True])
    def test_delegates_numerical_and_deployment_checks(
        self,
        schema_target_factory: Callable[..., MagicMock],
        selection_policy_factory: Callable[..., MagicMock],
        deployment_backend_factory: Callable[[], MagicMock],
        for_conversion: bool,
        with_backend: bool,
        conversion_device: torch.device | None,
    ) -> None:
        target = schema_target_factory(module_path="decoder", needs_calibration=False)
        model = selection_policy_factory(modules=(("decoder.projection", 64),))
        backend = deployment_backend_factory() if with_backend else None
        validate_target_configuration(
            model=model,
            target=target,
            module_names={"decoder.projection"},
            activation_dtype=torch.bfloat16,
            deployment_backend=backend,
            for_conversion=for_conversion,
            conversion_device=conversion_device,
        )
        target.schema.validate_configuration.assert_called_once_with(
            model=model,
            module_names={"decoder.projection"},
            label="decoder",
            activation_dtype=torch.bfloat16,
            for_conversion=for_conversion,
            conversion_device=conversion_device,
        )
        if backend is not None:
            backend.validate_eager_target.assert_called_once_with(
                model=model,
                target=target,
                module_names={"decoder.projection"},
                for_conversion=for_conversion,
                conversion_device=conversion_device,
            )

    @pytest.mark.parametrize("auto_filter", [False, True])
    @pytest.mark.parametrize("conversion_device", [None, torch.device("cuda:1")])
    def test_resolves_selection_and_preparation_conversion_configs_once(
        self,
        schema_target_factory: Callable[..., MagicMock],
        selection_policy_factory: Callable[..., MagicMock],
        deployment_backend_factory: Callable[[], MagicMock],
        auto_filter: bool,
        conversion_device: torch.device | None,
    ) -> None:
        target = schema_target_factory(
            module_path="decoder", needs_calibration=False, needs_preparation=True
        )
        reason = "Weight row width 48 requires divisibility by group_size 32"
        target.select_modules.return_value = (
            ["decoder.projection"],
            {"decoder.head": reason},
        )
        model = selection_policy_factory(modules=(("decoder.projection", 64),))
        backend = deployment_backend_factory()
        with (
            patch(
                f"{RESOLUTION_MODULE}.validate_quantization_targets"
            ) as validate_paths,
            patch(
                f"{RESOLUTION_MODULE}.validate_target_configuration"
            ) as validate_configuration,
            patch(f"{RESOLUTION_MODULE}.logger.info") as log,
        ):
            prepared = resolve_quantization_targets(
                model=model,
                targets=[target],
                is_qat=True,
                auto_filter_incompatible_linears=auto_filter,
                deployment_backend=backend,
                activation_dtype=torch.bfloat16,
                for_conversion=False,
                conversion_device=conversion_device,
            )
        validate_paths.assert_called_once_with(model=model, targets=[target])
        target.select_modules.assert_called_once_with(
            model=model, auto_filter_incompatible=auto_filter
        )
        validate_configuration.assert_called_once_with(
            model=model,
            target=target,
            module_names={"decoder.projection"},
            activation_dtype=torch.bfloat16,
            deployment_backend=backend,
            for_conversion=False,
            conversion_device=conversion_device,
        )
        target.schema.preparation_config.assert_called_once_with(is_qat=True)
        target.schema.conversion_config.assert_called_once_with(is_qat=True)
        assert prepared[0].module_names == {"decoder.projection"}
        assert prepared[0].skipped == {"decoder.head": reason}
        assert prepared[0].preparation_config == (
            target.schema.preparation_config.return_value
        )
        assert (
            prepared[0].conversion_config
            == target.schema.conversion_config.return_value
        )
        log.assert_called_once_with(
            f"Skipping quantization module decoder.head: {reason}"
        )

    def test_path_validation_precedes_selection(
        self,
        schema_target_factory: Callable[..., MagicMock],
        selection_policy_factory: Callable[..., MagicMock],
    ) -> None:
        target = schema_target_factory(module_path="decoder", needs_calibration=False)
        model = selection_policy_factory(modules=(("decoder.projection", 64),))
        message = "Quantization target paths overlap."
        with (
            patch(
                f"{RESOLUTION_MODULE}.validate_quantization_targets",
                side_effect=ValueError(message),
            ),
            pytest.raises(ValueError, match=re.escape(message)),
        ):
            resolve_quantization_targets(
                model=model,
                targets=[target],
                is_qat=False,
                auto_filter_incompatible_linears=True,
            )
        target.select_modules.assert_not_called()
        target.schema.validate_configuration.assert_not_called()


@pytest.mark.integration
@pytest.mark.parametrize(
    "module_path, expectation",
    [
        ("", does_not_raise()),
        ("encoder", does_not_raise()),
        ("encoder.0", does_not_raise()),
        (
            "missing",
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Quantization target 'missing' not found in model. "
                    "Available top-level modules: ['encoder', 'decoder']."
                ),
            ),
        ),
    ],
)
def test_validate_quantization_targets_checks_module_paths(
    eager_target_factory: Callable[[str], EagerQuantizationModuleTarget],
    module_path: str,
    expectation: AbstractContextManager,
) -> None:
    model = QuantizationValidationModel()
    targets = [eager_target_factory(module_path)]

    with expectation:
        validate_quantization_targets(model=model, targets=targets)


@pytest.mark.integration
@pytest.mark.parametrize(
    "module_paths, expectation",
    [
        (["encoder", "decoder"], does_not_raise()),
        (
            ["encoder", "encoder.0"],
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Quantization targets overlap: 'encoder' and 'encoder.0'."
                ),
            ),
        ),
        (
            ["", "decoder"],
            pytest.raises(
                ValueError,
                match=re.escape("Quantization targets overlap: '' and 'decoder'."),
            ),
        ),
    ],
)
def test_validate_quantization_targets_rejects_overlapping_paths(
    eager_target_factory: Callable[[str], EagerQuantizationModuleTarget],
    module_paths: list[str],
    expectation: AbstractContextManager,
) -> None:
    model = QuantizationValidationModel()
    targets = [eager_target_factory(module_path) for module_path in module_paths]

    with expectation:
        validate_quantization_targets(model=model, targets=targets)
