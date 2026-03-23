"""Tests for versatil.post_training_compression.export module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.export import (
    _export_with_dynamic_batch,
    build_example_inputs,
    export_policy,
)

EXPORT_MODULE = "versatil.post_training_compression.export"


@pytest.fixture
def mock_encoder_factory() -> Callable[..., MagicMock]:
    """Factory for mock encoders with input specification."""

    def factory(
        keys: list[str],
        shape: tuple[int, ...],
    ) -> MagicMock:
        encoder = MagicMock()
        spec = MagicMock()
        spec.keys = keys
        spec.shape = shape
        encoder.input_specification = spec
        return encoder

    return factory


@pytest.fixture
def flat_input_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Factory for flat input tensor tuples."""

    def factory(
        batch_size: int = 2,
        feature_dimension: int = 8,
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(
                rng.standard_normal((batch_size, feature_dimension)).astype(np.float32)
            ),
        )

    return factory


@pytest.fixture
def simple_exportable_model(
    rng: np.random.Generator,
) -> nn.Module:
    """A minimal model that can be exported with torch.export."""
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            data = rng.standard_normal(parameter.shape).astype(np.float32)
            parameter.copy_(torch.from_numpy(data))
    model.eval()
    return model


@pytest.mark.unit
class TestBuildExampleInputs:
    @pytest.mark.parametrize(
        "encoder_specs, conditional_specs, expected_keys",
        [
            (
                [("rgb", ["left"], (3, 32, 32))],
                [],
                {"left"},
            ),
            (
                [("rgb", ["left", "right"], (3, 32, 32))],
                [],
                {"left", "right"},
            ),
            (
                [("rgb", ["left"], (3, 32, 32))],
                [("depth", ["depth"], (1, 32, 32))],
                {"left", "depth"},
            ),
        ],
        ids=["single_key", "multi_key_encoder", "with_conditional"],
    )
    def test_collects_shapes_from_all_encoders(
        self,
        mock_encoder_factory,
        encoder_specs,
        conditional_specs,
        expected_keys,
    ):
        encoders = [
            mock_encoder_factory(keys=keys, shape=shape)
            for _, keys, shape in encoder_specs
        ]
        conditionals = [
            mock_encoder_factory(keys=keys, shape=shape)
            for _, keys, shape in conditional_specs
        ]

        policy = MagicMock()
        policy.encoding_pipeline.encoders.values.return_value = encoders
        policy.encoding_pipeline.conditional_encoders.values.return_value = conditionals

        exportable = MagicMock(spec=ExportablePolicy)
        exportable.get_example_inputs.return_value = tuple(
            torch.zeros(2, *shape)
            for _, keys, shape in encoder_specs + conditional_specs
            for _ in keys
        )

        result = build_example_inputs(policy=policy, exportable=exportable)

        call_shapes = exportable.get_example_inputs.call_args[1]["observation_shapes"]
        assert set(call_shapes.keys()) == expected_keys
        assert len(result) == len(expected_keys)


@pytest.mark.unit
class TestExportWithDynamicBatch:
    def test_exported_model_produces_valid_output(
        self, simple_exportable_model, flat_input_factory
    ):
        example_inputs = flat_input_factory()

        exported = _export_with_dynamic_batch(
            model=simple_exportable_model,
            example_inputs=example_inputs,
        )

        with torch.no_grad():
            result = exported.module()(*example_inputs)
        assert result.shape == (2, 2)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_dynamic_batch_accepts_different_sizes(
        self,
        simple_exportable_model,
        flat_input_factory,
        batch_size,
    ):
        example_inputs = flat_input_factory()

        exported = _export_with_dynamic_batch(
            model=simple_exportable_model,
            example_inputs=example_inputs,
        ).module()

        different_batch = flat_input_factory(batch_size=batch_size)
        with torch.no_grad():
            result = exported(*different_batch)
        assert result.shape == (batch_size, 2)

    def test_dynamic_shapes_key_wraps_in_dict(
        self, simple_exportable_model, flat_input_factory
    ):
        class NamedArgsModel(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, *observation_tensors: torch.Tensor) -> torch.Tensor:
                return self.inner(observation_tensors[0])

        model = NamedArgsModel(inner=simple_exportable_model)
        model.eval()
        example_inputs = flat_input_factory()

        exported = _export_with_dynamic_batch(
            model=model,
            example_inputs=example_inputs,
            dynamic_shapes_key="observation_tensors",
        ).module()

        different_batch = flat_input_factory(batch_size=3)
        with torch.no_grad():
            result = exported(*different_batch)
        assert result.shape == (3, 2)

    def test_multiple_inputs_all_get_dynamic_batch(self, rng: np.random.Generator):
        class TwoInputModel(nn.Module):
            def forward(
                self, image: torch.Tensor, features: torch.Tensor
            ) -> torch.Tensor:
                return image.sum(dim=-1, keepdim=True) + features

        model = TwoInputModel()
        model.eval()
        example_inputs = (
            torch.from_numpy(rng.standard_normal((2, 3)).astype(np.float32)),
            torch.from_numpy(rng.standard_normal((2, 1)).astype(np.float32)),
        )

        exported = _export_with_dynamic_batch(
            model=model,
            example_inputs=example_inputs,
        ).module()

        with torch.no_grad():
            result = exported(
                torch.from_numpy(rng.standard_normal((5, 3)).astype(np.float32)),
                torch.from_numpy(rng.standard_normal((5, 1)).astype(np.float32)),
            )
        assert result.shape == (5, 1)


@pytest.mark.unit
class TestExportPolicy:
    def test_eager_forward_runs_before_export(self):
        call_order = []

        exportable = MagicMock(spec=ExportablePolicy)
        exportable.side_effect = lambda *args, **kwargs: call_order.append("eager")

        mock_exported_program = MagicMock()
        mock_exported_program.module.return_value = MagicMock()

        def mock_torch_export(*args, **kwargs):
            call_order.append("export")
            return mock_exported_program

        with patch(
            f"{EXPORT_MODULE}.torch.export.export",
            side_effect=mock_torch_export,
        ):
            export_policy(
                exportable=exportable,
                example_inputs=(torch.zeros(2, 3),),
            )

        assert call_order == ["eager", "export"]

    def test_returns_module_and_passes_correct_kwargs(self):
        exportable = MagicMock(spec=ExportablePolicy)
        mock_graph_module = MagicMock()
        mock_exported_program = MagicMock()
        mock_exported_program.module.return_value = mock_graph_module

        with patch(
            f"{EXPORT_MODULE}.torch.export.export",
            return_value=mock_exported_program,
        ) as mock_export:
            result = export_policy(
                exportable=exportable,
                example_inputs=(torch.zeros(2, 3),),
            )

        assert result is mock_graph_module
        call_kwargs = mock_export.call_args[1]
        assert call_kwargs["strict"] is False
        assert "observation_tensors" in call_kwargs["dynamic_shapes"]
