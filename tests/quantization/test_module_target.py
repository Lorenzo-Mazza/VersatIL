"""Tests for versatil.quantization.module_target module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from torch import nn

from versatil.quantization.module_target import (
    EagerQuantizationModuleTarget,
    PT2EQuantizationModuleTarget,
    QuantizationModuleTarget,
)
from versatil.quantization.schemas.base import QuantizationSchema

MODULE_TARGET_MODULE = "versatil.quantization.module_target"


@pytest.fixture
def target_configuration_factory() -> Callable[
    ..., tuple[MagicMock | None, MagicMock | None]
]:
    def factory(
        with_config: bool, with_schema: bool
    ) -> tuple[MagicMock | None, MagicMock | None]:
        config = MagicMock(spec=[]) if with_config else None
        schema = MagicMock(spec=QuantizationSchema) if with_schema else None
        if schema is not None:
            schema.base_config = MagicMock(spec=[])
        return config, schema

    return factory


@pytest.fixture
def eager_selection_target_factory() -> Callable[..., EagerQuantizationModuleTarget]:
    def factory(
        module_path: str, group_size: int | None
    ) -> EagerQuantizationModuleTarget:
        schema = MagicMock(spec=QuantizationSchema)
        schema.base_config = MagicMock(spec=[])
        schema.base_config.version = 2
        schema.parameters = {"alpha": "0.75"}
        schema.needs_calibration = True
        schema.weight_group_size = group_size
        return EagerQuantizationModuleTarget(module_path=module_path, schema=schema)

    return factory


@pytest.fixture
def layer_selection_model_factory() -> Callable[..., MagicMock]:
    def factory(
        layers: tuple[tuple[str, int], ...],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> MagicMock:
        modules = {"activation": MagicMock(spec=nn.ReLU)}
        for name, in_features in layers:
            layer = MagicMock(spec=nn.Linear)
            layer.in_features = in_features
            layer.out_features = 16
            layer.weight = MagicMock(spec=torch.Tensor)
            layer.weight.device = torch.device(device)
            layer.weight.dtype = dtype
            modules[name] = layer
        model = MagicMock(spec=nn.Module)
        model.named_modules.return_value = list(modules.items())
        model.get_submodule.side_effect = modules.__getitem__
        return model

    return factory


@pytest.mark.unit
@pytest.mark.parametrize("with_config, with_schema", [(False, False), (True, True)])
def test_eager_target_requires_exactly_one_configuration_form(
    target_configuration_factory: Callable,
    with_config: bool,
    with_schema: bool,
) -> None:
    config, schema = target_configuration_factory(
        with_config=with_config, with_schema=with_schema
    )
    with pytest.raises(
        ValueError,
        match=re.escape("Specify exactly one of quantize_config or schema."),
    ):
        EagerQuantizationModuleTarget(
            module_path="decoder", quantize_config=config, schema=schema
        )


@pytest.mark.unit
def test_schema_target_exposes_base_config_for_layer_compatibility(
    target_configuration_factory: Callable,
) -> None:
    _, schema = target_configuration_factory(with_config=False, with_schema=True)
    target = EagerQuantizationModuleTarget(module_path="decoder", schema=schema)
    assert target.quantize_config is schema.base_config


@pytest.mark.unit
def test_quantize_config_constructs_direct_schema_with_the_base_configuration(
    target_configuration_factory: Callable,
) -> None:
    config, _ = target_configuration_factory(with_config=True, with_schema=False)
    with patch(
        "versatil.quantization.module_target.DirectQuantizationSchema"
    ) as direct_schema:
        target = EagerQuantizationModuleTarget(
            module_path="decoder", quantize_config=config
        )

    direct_schema.assert_called_once_with(base_config=config)
    assert target.schema is direct_schema.return_value


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_path, module_name, expected",
    [
        ("", "encoder.0", True),
        ("decoder", "decoder", True),
        ("decoder", "decoder.head", True),
        ("decoder", "encoder.head", False),
        ("decoder", "decoder_head", False),
    ],
)
def test_contains_module_matches_exact_path_and_children(
    module_path: str,
    module_name: str,
    expected: bool,
) -> None:
    target = QuantizationModuleTarget(module_path=module_path)

    assert target.contains_module(module_name=module_name) is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_path, expected",
    [
        ("", "(root)"),
        ("decoder", "decoder"),
    ],
)
def test_label_returns_root_name_for_empty_path(
    module_path: str,
    expected: str,
) -> None:
    target = QuantizationModuleTarget(module_path=module_path)

    assert target.label == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "left_path, right_path, expected",
    [
        ("", "decoder", True),
        ("encoder", "encoder", True),
        ("encoder", "encoder.backbone", True),
        ("encoder.backbone", "encoder", True),
        ("encoder", "decoder", False),
        ("encoder", "encoder_head", False),
    ],
)
def test_overlaps_detects_root_same_and_nested_targets(
    left_path: str,
    right_path: str,
    expected: bool,
) -> None:
    left = QuantizationModuleTarget(module_path=left_path)
    right = QuantizationModuleTarget(module_path=right_path)

    assert left.overlaps(other=right) is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "is_dynamic, expected",
    [
        (True, False),
        (False, True),
    ],
)
def test_pt2e_target_needs_calibration_reflects_backend_dynamic_flag(
    mock_pt2e_backend_factory: Callable[..., MagicMock],
    is_dynamic: bool,
    expected: bool,
) -> None:
    target = PT2EQuantizationModuleTarget(
        module_path="",
        pt2e_backend=mock_pt2e_backend_factory(is_dynamic=is_dynamic),
    )

    assert target.needs_calibration is expected


@pytest.mark.unit
class TestEagerLayerSelection:
    @pytest.mark.parametrize(
        "module_path, expected",
        [
            (
                "",
                [
                    "encoder.projection",
                    "encoder.head",
                    "encoder_other.projection",
                    "decoder.head",
                ],
            ),
            ("encoder", ["encoder.projection", "encoder.head"]),
            ("decoder.head", ["decoder.head"]),
        ],
    )
    def test_scope_selects_linears_and_preserves_sibling_boundaries(
        self,
        eager_selection_target_factory: Callable[..., EagerQuantizationModuleTarget],
        layer_selection_model_factory: Callable[..., MagicMock],
        module_path: str,
        expected: list[str],
    ) -> None:
        target = eager_selection_target_factory(
            module_path=module_path, group_size=None
        )
        model = layer_selection_model_factory(
            layers=(
                ("encoder.projection", 64),
                ("encoder.head", 16),
                ("encoder_other.projection", 32),
                ("decoder.head", 16),
            )
        )

        selected, skipped = target.select_linear_modules(
            model=model, auto_filter_incompatible_linears=True
        )

        assert selected == expected
        assert skipped == {}
        model.named_modules.assert_called_once_with()
        model.get_submodule.assert_not_called()

    @pytest.mark.parametrize("auto_filter", [False, True])
    def test_group_mismatch_is_skipped_or_rejected(
        self,
        eager_selection_target_factory: Callable[..., EagerQuantizationModuleTarget],
        layer_selection_model_factory: Callable[..., MagicMock],
        auto_filter: bool,
    ) -> None:
        target = eager_selection_target_factory(module_path="encoder", group_size=32)
        model = layer_selection_model_factory(
            layers=(("encoder.projection", 64), ("encoder.head", 48))
        )
        reason = "in_features 48 is not divisible by group_size 32"
        expectation = (
            does_not_raise()
            if auto_filter
            else pytest.raises(
                ValueError, match=re.escape(f"Module 'encoder.head': {reason}.")
            )
        )

        with expectation:
            selected, skipped = target.select_linear_modules(
                model=model, auto_filter_incompatible_linears=auto_filter
            )

        if auto_filter:
            assert selected == ["encoder.projection"]
            assert skipped == {"encoder.head": reason}
        model.named_modules.assert_called_once_with()

    @pytest.mark.parametrize("group_size", [0, -32])
    def test_nonpositive_group_size_fails_before_model_traversal(
        self,
        eager_selection_target_factory: Callable[..., EagerQuantizationModuleTarget],
        layer_selection_model_factory: Callable[..., MagicMock],
        group_size: int,
    ) -> None:
        target = eager_selection_target_factory(module_path="", group_size=group_size)
        model = layer_selection_model_factory(layers=(("projection", 64),))

        with pytest.raises(
            ValueError,
            match=re.escape(f"Target '(root)' has invalid group_size {group_size}."),
        ):
            target.select_linear_modules(
                model=model, auto_filter_incompatible_linears=True
            )

        model.named_modules.assert_not_called()

    @pytest.mark.parametrize(
        "layers, group_size, expected_skipped",
        [
            ((), None, {}),
            ((("decoder.head", 16),), None, {}),
            (
                (("encoder.head", 8),),
                32,
                {"encoder.head": "in_features 8 is not divisible by group_size 32"},
            ),
        ],
    )
    def test_empty_selection_identifies_excluded_layers(
        self,
        eager_selection_target_factory: Callable[..., EagerQuantizationModuleTarget],
        layer_selection_model_factory: Callable[..., MagicMock],
        layers: tuple[tuple[str, int], ...],
        group_size: int | None,
        expected_skipped: dict[str, str],
    ) -> None:
        target = eager_selection_target_factory(
            module_path="encoder", group_size=group_size
        )
        model = layer_selection_model_factory(layers=layers)

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Target 'encoder' selects zero eligible nn.Linear modules; skipped modules: {expected_skipped}."
            ),
        ):
            target.select_linear_modules(
                model=model, auto_filter_incompatible_linears=True
            )

        model.named_modules.assert_called_once_with()


@pytest.mark.unit
@pytest.mark.parametrize("dataclass_config", [False, True])
def test_metadata_captures_resolved_weights_and_schema_settings_without_reselection(
    eager_selection_target_factory: Callable[..., EagerQuantizationModuleTarget],
    layer_selection_model_factory: Callable[..., MagicMock],
    dataclass_config: bool,
) -> None:
    target = eager_selection_target_factory(module_path="encoder", group_size=32)
    model = layer_selection_model_factory(
        layers=(("encoder.projection", 64), ("encoder.head", 32)),
        device="cuda:1",
        dtype=torch.bfloat16,
    )
    skipped = {"encoder.excluded": "in_features 48 is not divisible by group_size 32"}
    with (
        patch(
            f"{MODULE_TARGET_MODULE}.is_dataclass", return_value=dataclass_config
        ) as is_dataclass,
        patch(
            f"{MODULE_TARGET_MODULE}.fields",
            return_value=[SimpleNamespace(name="version")],
        ) as fields,
    ):
        metadata = target.build_metadata(
            model=model,
            module_names={"encoder.projection", "encoder.head"},
            skipped=skipped,
        )

    is_dataclass.assert_called_once_with(target.quantize_config)
    if dataclass_config:
        fields.assert_called_once_with(target.quantize_config)
    else:
        fields.assert_not_called()
    model.named_modules.assert_not_called()
    assert model.get_submodule.call_args_list == [
        call("encoder.head"),
        call("encoder.projection"),
    ]
    assert metadata.module_path == "encoder"
    config_type = type(target.quantize_config)
    assert (
        metadata.base_config == f"{config_type.__module__}.{config_type.__qualname__}"
    )
    assert metadata.base_config_parameters == (
        {"version": "2"} if dataclass_config else {}
    )
    schema_type = type(target.schema)
    assert metadata.schema == f"{schema_type.__module__}.{schema_type.__qualname__}"
    assert metadata.schema_parameters == {"alpha": "0.75"}
    assert metadata.requires_calibration is True
    assert metadata.group_size == 32
    assert [layer.name for layer in metadata.selected] == [
        "encoder.head",
        "encoder.projection",
    ]
    assert [layer.in_features for layer in metadata.selected] == [32, 64]
    assert [layer.out_features for layer in metadata.selected] == [16, 16]
    assert [layer.device for layer in metadata.selected] == ["cuda:1", "cuda:1"]
    assert [layer.dtype for layer in metadata.selected] == [
        "torch.bfloat16",
        "torch.bfloat16",
    ]
    assert metadata.skipped == skipped
    skipped.clear()
    assert metadata.skipped == {
        "encoder.excluded": "in_features 48 is not divisible by group_size 32"
    }
    assert metadata.weight_representations == {}
