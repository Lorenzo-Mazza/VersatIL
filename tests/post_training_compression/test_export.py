"""Tests for versatil.post_training_compression.export module."""

import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from torch import nn

from versatil.data.constants import SampleKey
from versatil.data.metadata import CameraMetadata
from versatil.data.task import ObservationSpace
from versatil.data.tokenization.observation_tokenizer import ObservationTokenizer
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.exportable.base import ExportablePolicy
from versatil.post_training_compression.export import (
    _export_with_dynamic_batch,
    build_example_inputs,
    export_policy,
    move_exported_model,
)
from versatil.training.constants import PrecisionType

EXPORT_MODULE = "versatil.post_training_compression.export"


@pytest.fixture
def observation_space_factory() -> Callable[..., MagicMock]:
    """Factory for mock ObservationSpace with camera and numerical-state metadata."""

    def factory(
        cameras: dict[str, CameraMetadata] | None = None,
        numerical_observations: dict[str, MagicMock] | None = None,
    ) -> MagicMock:
        obs_space = MagicMock(spec=ObservationSpace)
        obs_space.cameras = cameras or {}
        obs_space.numerical_observations = numerical_observations or {}
        return obs_space

    return factory


@pytest.fixture
def tokenizer_factory() -> Callable[..., MagicMock]:
    """Factory for mock Tokenizer with observation tokenizer."""

    def factory(
        max_token_len: int = 64,
    ) -> MagicMock:
        obs_tokenizer = MagicMock(spec=ObservationTokenizer)
        obs_tokenizer.max_token_len = max_token_len
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = obs_tokenizer
        return tokenizer

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


@pytest.fixture
def device_allocating_model_factory() -> Callable[..., nn.Module]:
    class DeviceAllocatingModel(nn.Module):
        def __init__(self, device: torch.device) -> None:
            super().__init__()
            self.register_buffer("offset", torch.full((8,), 0.25, device=device))

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            positions = torch.arange(
                observations.shape[-1], device=observations.device
            )  # (8,)
            return observations + positions + self.offset  # (B, 8)

    def factory(device: torch.device) -> nn.Module:
        return DeviceAllocatingModel(device=device)

    return factory


@pytest.mark.integration
@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    "source_device, target_device", [("cuda", "cpu"), ("cpu", "cuda")]
)
def test_graph_transfer_retargets_tensor_factories_and_survives_reload(
    device_allocating_model_factory: Callable[..., nn.Module],
    flat_input_factory: Callable[..., tuple[torch.Tensor, ...]],
    source_device: str,
    target_device: str,
    tmp_path: Path,
) -> None:
    model = device_allocating_model_factory(device=torch.device(source_device))
    source_inputs = tuple(
        tensor.to(device=source_device) for tensor in flat_input_factory(batch_size=2)
    )  # each (2, 8)
    exported = _export_with_dynamic_batch(
        model=model, example_inputs=source_inputs
    ).module()
    moved = move_exported_model(
        model=exported, example_inputs=source_inputs, device=torch.device(target_device)
    )
    target_inputs = tuple(
        tensor.to(device=target_device) for tensor in source_inputs
    )  # each (2, 8)
    artifact = tmp_path / "moved.pt2"
    torch.export.save(
        _export_with_dynamic_batch(model=moved, example_inputs=target_inputs), artifact
    )
    reloaded = torch.export.load(artifact).module()
    for batch_size in (1, 4):
        observations = flat_input_factory(batch_size=batch_size)[0]  # (B, 8)
        with torch.no_grad():
            expected = model(observations.to(device=source_device)).to(
                device=target_device
            )  # (B, 8)
            actual = reloaded(observations.to(device=target_device))  # (B, 8)
        torch.testing.assert_close(actual, expected)
        assert actual.device.type == target_device


@pytest.mark.unit
class TestBuildExampleInputs:
    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:1")])
    def test_camera_shapes_from_metadata_and_config(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        device: torch.device,
    ) -> None:
        cameras = {
            "left": CameraMetadata(
                camera_key="agentview_rgb",
                dtype="float32",
                channels=3,
                image_height=48,
                image_width=64,
            ),
        }
        obs_space = observation_space_factory(cameras=cameras)
        exportable = MagicMock(spec=ExportablePolicy)
        exportable.get_example_inputs.return_value = (torch.zeros(2, 3, 48, 64),)
        build_example_inputs(
            exportable=exportable,
            observation_space=obs_space,
            observation_horizon=1,
            device=device,
        )

        call_shapes = exportable.get_example_inputs.call_args[1]["observation_shapes"]
        assert call_shapes["left"] == (1, 3, 48, 64)
        assert exportable.get_example_inputs.call_args.kwargs["device"] == device

    def test_numerical_state_shapes_from_metadata(
        self,
        observation_space_factory,
    ):
        state_meta = MagicMock()
        state_meta.dimension = 7
        state_meta.dtype = "float32"
        obs_space = observation_space_factory(
            numerical_observations={"proprio_robot_frame": state_meta},
        )

        exportable = MagicMock(spec=ExportablePolicy)
        exportable.get_example_inputs.return_value = (torch.zeros(2, 7),)

        build_example_inputs(
            exportable=exportable,
            observation_space=obs_space,
            observation_horizon=1,
        )

        call_shapes = exportable.get_example_inputs.call_args[1]["observation_shapes"]
        assert call_shapes["proprio_robot_frame"] == (1, 7)

    def test_tokenized_shapes_from_tokenizer(
        self,
        observation_space_factory,
        tokenizer_factory,
    ):
        tokenizer = tokenizer_factory(max_token_len=128)

        exportable = MagicMock(spec=ExportablePolicy)
        exportable.get_example_inputs.return_value = (
            torch.zeros(2, 128),
            torch.zeros(2, 128),
        )

        build_example_inputs(
            exportable=exportable,
            observation_space=observation_space_factory(),
            observation_horizon=1,
            tokenizer=tokenizer,
        )

        call_kwargs = exportable.get_example_inputs.call_args[1]
        shapes = call_kwargs["observation_shapes"]
        dtypes = call_kwargs["observation_dtypes"]
        assert shapes[SampleKey.TOKENIZED_OBSERVATIONS.value] == (1, 128)
        assert shapes[SampleKey.IS_PAD_OBSERVATION.value] == (1, 128)
        assert dtypes[SampleKey.TOKENIZED_OBSERVATIONS.value] == torch.long
        assert dtypes[SampleKey.IS_PAD_OBSERVATION.value] == torch.bool

    def test_no_tokenizer_omits_language_keys(
        self,
        observation_space_factory,
    ):
        cameras = {
            "left": CameraMetadata(
                camera_key="agentview_rgb",
                dtype="float32",
                channels=3,
                image_height=48,
                image_width=64,
            ),
        }
        obs_space = observation_space_factory(cameras=cameras)

        exportable = MagicMock(spec=ExportablePolicy)
        exportable.get_example_inputs.return_value = (torch.zeros(2, 3, 32, 32),)

        build_example_inputs(
            exportable=exportable,
            observation_space=obs_space,
            observation_horizon=1,
            tokenizer=None,
        )

        call_shapes = exportable.get_example_inputs.call_args[1]["observation_shapes"]
        assert SampleKey.TOKENIZED_OBSERVATIONS.value not in call_shapes
        assert SampleKey.IS_PAD_OBSERVATION.value not in call_shapes

    def test_mixed_cameras_and_tokenizer(
        self,
        observation_space_factory,
        tokenizer_factory,
    ):
        cameras = {
            "left": CameraMetadata(
                camera_key="agentview_rgb",
                dtype="float32",
                channels=3,
                image_height=32,
                image_width=32,
            ),
            "right": CameraMetadata(
                camera_key="eye_in_hand_rgb",
                dtype="float32",
                channels=3,
                image_height=32,
                image_width=32,
            ),
        }
        obs_space = observation_space_factory(cameras=cameras)
        tokenizer = tokenizer_factory(max_token_len=64)

        exportable = MagicMock(spec=ExportablePolicy)
        exportable.get_example_inputs.return_value = tuple(
            torch.zeros(2, 8) for _ in range(4)
        )

        build_example_inputs(
            exportable=exportable,
            observation_space=obs_space,
            observation_horizon=1,
            tokenizer=tokenizer,
        )

        call_shapes = exportable.get_example_inputs.call_args[1]["observation_shapes"]
        assert set(call_shapes.keys()) == {
            "left",
            "right",
            SampleKey.TOKENIZED_OBSERVATIONS.value,
            SampleKey.IS_PAD_OBSERVATION.value,
        }


@pytest.mark.integration
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

    @pytest.mark.parametrize(
        "max_batch_size, runtime_batch_size",
        [
            (2, 2),
            (8, 8),
        ],
    )
    def test_bounded_dynamic_batch_accepts_sizes_within_bound(
        self,
        simple_exportable_model: nn.Module,
        flat_input_factory: Callable[..., tuple[torch.Tensor, ...]],
        max_batch_size: int,
        runtime_batch_size: int,
    ) -> None:
        example_inputs = flat_input_factory()

        exported = _export_with_dynamic_batch(
            model=simple_exportable_model,
            example_inputs=example_inputs,
            max_batch_size=max_batch_size,
        ).module()

        runtime_inputs = flat_input_factory(batch_size=runtime_batch_size)
        with torch.no_grad():
            result = exported(*runtime_inputs)
        assert result.shape == (runtime_batch_size, 2)

    @pytest.mark.parametrize(
        "max_batch_size, message",
        [
            (0, "max_batch_size must be >= 1."),
            (1, "max_batch_size 1 is smaller than example batch size 2."),
        ],
    )
    def test_bounded_dynamic_batch_rejects_invalid_bound(
        self,
        simple_exportable_model: nn.Module,
        flat_input_factory: Callable[..., tuple[torch.Tensor, ...]],
        max_batch_size: int,
        message: str,
    ) -> None:
        example_inputs = flat_input_factory()

        with pytest.raises(ValueError, match=re.escape(message)):
            _export_with_dynamic_batch(
                model=simple_exportable_model,
                example_inputs=example_inputs,
                max_batch_size=max_batch_size,
            )

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
@pytest.mark.parametrize("autocast_precision", [None, PrecisionType.BF16_MIXED])
def test_export_materializes_layers_then_records_autocast_as_explicit_casts(
    flat_input_factory: Callable[..., tuple[torch.Tensor, ...]],
    autocast_precision: PrecisionType | None,
) -> None:
    exportable = MagicMock(spec=ExportablePolicy)
    exportable.autocast_precision = autocast_precision
    inputs = flat_input_factory(batch_size=2, feature_dimension=8)
    program = MagicMock(spec=torch.export.ExportedProgram)
    decomposed = program.run_decompositions.return_value
    selected_program = decomposed if autocast_precision is not None else program
    execution = MagicMock()
    execution.attach_mock(exportable, "materialize")
    execution.attach_mock(program.run_decompositions, "decompose")
    execution.attach_mock(selected_program.module, "module")
    with patch(
        f"{EXPORT_MODULE}._export_with_dynamic_batch", return_value=program
    ) as export:
        execution.attach_mock(export, "export")
        result = export_policy(exportable=exportable, example_inputs=inputs)

    expected = [
        call.materialize(*inputs),
        call.export(
            model=exportable,
            example_inputs=inputs,
            dynamic_shapes_key="observation_tensors",
        ),
    ]
    if autocast_precision is not None:
        expected.append(call.decompose({}))
    expected.append(call.module())
    assert execution.mock_calls == expected
    assert result is selected_program.module.return_value
