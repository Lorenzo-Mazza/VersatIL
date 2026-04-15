"""Tests for versatil.post_training_compression.export module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from versatil.data.constants import SampleKey
from versatil.data.metadata import CameraMetadata
from versatil.data.task import ObservationSpace
from versatil.data.tokenization.observation_tokenizer import ObservationTokenizer
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.export import (
    _export_with_dynamic_batch,
    build_example_inputs,
    export_policy,
)

EXPORT_MODULE = "versatil.post_training_compression.export"


@pytest.fixture
def observation_space_factory() -> Callable[..., MagicMock]:
    """Factory for mock ObservationSpace with camera and proprio metadata."""

    def factory(
        cameras: dict[str, CameraMetadata] | None = None,
        proprioceptive_observations: dict[str, MagicMock] | None = None,
    ) -> MagicMock:
        obs_space = MagicMock(spec=ObservationSpace)
        obs_space.cameras = cameras or {}
        obs_space.proprioceptive_observations = proprioceptive_observations or {}
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


@pytest.mark.unit
class TestBuildExampleInputs:
    def test_camera_shapes_from_metadata_and_config(
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
        exportable.get_example_inputs.return_value = (torch.zeros(2, 3, 48, 64),)
        build_example_inputs(
            exportable=exportable,
            observation_space=obs_space,
            observation_horizon=1,
        )

        call_shapes = exportable.get_example_inputs.call_args[1]["observation_shapes"]
        assert call_shapes["left"] == (1, 3, 48, 64)

    def test_proprioceptive_shapes_from_metadata(
        self,
        observation_space_factory,
    ):
        proprio = MagicMock()
        proprio.dimension = 7
        obs_space = observation_space_factory(
            proprioceptive_observations={"proprio_robot_frame": proprio},
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
