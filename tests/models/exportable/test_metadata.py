"""Tests for versatil.models.exportable.metadata module."""

import json
import re
from collections.abc import Callable, Iterator
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from versatil.models.exportable.metadata import (
    NoiseInput,
    PolicyExportMetadata,
    PredictionOutput,
    SamplingInput,
)


@pytest.fixture
def export_metadata_factory() -> Callable[..., PolicyExportMetadata]:
    def factory(
        output: PredictionOutput,
        noise_specs: tuple[tuple[str, tuple[int, ...]], ...],
        version: int = 1,
    ) -> PolicyExportMetadata:
        return PolicyExportMetadata(
            output=output,
            noise_inputs=tuple(
                NoiseInput(name=name, shape=shape) for name, shape in noise_specs
            ),
            version=version,
        )

    return factory


@pytest.fixture
def inference_observations_factory() -> Callable[..., tuple[torch.Tensor, ...]]:
    def factory(
        batch_size: int, dtypes: tuple[torch.dtype, ...], scalar: bool = False
    ) -> tuple[torch.Tensor, ...]:
        if scalar:
            return tuple(torch.tensor(1, dtype=dtype) for dtype in dtypes)
        return tuple(
            torch.zeros(batch_size, 1, 3, dtype=dtype)  # (batch, 1, observation_dim)
            for dtype in dtypes
        )

    return factory


@pytest.fixture
def sampled_noise_factory() -> Callable[..., tuple[torch.Tensor, ...]]:
    def factory(
        batch_size: int, shapes: tuple[tuple[int, ...], ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.full(
                size=(batch_size, *shape), fill_value=index + 1, dtype=dtype
            )  # (batch, *noise_shape)
            for index, shape in enumerate(shapes)
        )

    return factory


@pytest.fixture
def noise_sampler_factory() -> Iterator[Callable[..., MagicMock]]:
    with patch("versatil.models.exportable.metadata.torch.randn") as sampler:

        def factory(outputs: tuple[torch.Tensor, ...]) -> MagicMock:
            sampler.side_effect = outputs
            return sampler

        yield factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, shape, error_message",
    [
        (SamplingInput.INITIAL_NOISE.value, (2, 3), None),
        (SamplingInput.STEP_NOISE.value, (4, 2, 3), None),
        ("", (2, 3), "Noise input names must be nonempty strings, got ''."),
        (
            SamplingInput.INITIAL_NOISE.value,
            (),
            "Noise input 'initial_noise' requires a nonempty shape of positive "
            "integers, got ().",
        ),
        (
            SamplingInput.INITIAL_NOISE.value,
            (0, 3),
            "Noise input 'initial_noise' requires a nonempty shape of positive "
            "integers, got (0, 3).",
        ),
        (
            SamplingInput.STEP_NOISE.value,
            (-1, 2, 3),
            "Noise input 'step_noise' requires a nonempty shape of positive "
            "integers, got (-1, 2, 3).",
        ),
    ],
)
def test_noise_input_validates_name_and_dimensions(
    name: str, shape: tuple[int, ...], error_message: str | None
) -> None:
    expectation = (
        does_not_raise()
        if error_message is None
        else pytest.raises(ValueError, match=re.escape(error_message))
    )
    with expectation:
        noise = NoiseInput(name=name, shape=shape)
        assert noise.name == name
        assert noise.shape == shape


@pytest.mark.unit
@pytest.mark.parametrize(
    "metadata",
    [{}, {"version": 1}, {"output": "actions", "noise_inputs": []}],
)
def test_legacy_metadata_uses_normalized_action_output_without_noise(
    metadata: dict[str, Any],
) -> None:
    export_metadata = PolicyExportMetadata.from_dict(metadata=metadata)

    assert export_metadata.output == PredictionOutput.ACTIONS
    assert export_metadata.noise_inputs == ()
    assert export_metadata.version == 1


@pytest.mark.unit
@pytest.mark.parametrize("output", list(PredictionOutput))
@pytest.mark.parametrize(
    "noise_specs",
    [
        (),
        (
            (SamplingInput.INITIAL_NOISE.value, (2, 3)),
            (SamplingInput.STEP_NOISE.value, (4, 2, 3)),
        ),
    ],
)
def test_serialization_preserves_output_and_ordered_noise_shapes(
    export_metadata_factory: Callable[..., PolicyExportMetadata],
    output: PredictionOutput,
    noise_specs: tuple[tuple[str, tuple[int, ...]], ...],
) -> None:
    export_metadata = export_metadata_factory(output=output, noise_specs=noise_specs)

    metadata = json.loads(json.dumps(export_metadata.to_dict()))
    restored = PolicyExportMetadata.from_dict(metadata=metadata)

    assert metadata == {
        "output": output.value,
        "noise_inputs": [
            {"name": name, "shape": list(shape)} for name, shape in noise_specs
        ],
        "version": 1,
    }
    assert restored == export_metadata


@pytest.mark.unit
@pytest.mark.parametrize("version", [0, 2])
def test_metadata_rejects_unsupported_format_versions(version: int) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(f"Unsupported policy export metadata version {version}."),
    ):
        PolicyExportMetadata.from_dict(metadata={"version": version})


@pytest.mark.unit
def test_metadata_rejects_unknown_prediction_output() -> None:
    with pytest.raises(
        ValueError, match=re.escape("'invalid' is not a valid PredictionOutput")
    ):
        PolicyExportMetadata.from_dict(metadata={"output": "invalid"})


@pytest.mark.unit
def test_metadata_rejects_duplicate_noise_input_names() -> None:
    name = SamplingInput.INITIAL_NOISE.value
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Policy export noise input names must be unique, got {(name, name)}."
        ),
    ):
        PolicyExportMetadata.from_dict(
            metadata={
                "noise_inputs": [
                    {"name": name, "shape": [2, 3]},
                    {"name": name, "shape": [4, 2, 3]},
                ]
            }
        )


@pytest.mark.unit
@pytest.mark.parametrize("dtypes", [(), (torch.long, torch.bool), (torch.float32,)])
def test_prepare_inputs_preserves_observations_without_noise(
    export_metadata_factory: Callable[..., PolicyExportMetadata],
    inference_observations_factory: Callable[..., tuple[torch.Tensor, ...]],
    noise_sampler_factory: Callable[..., MagicMock],
    dtypes: tuple[torch.dtype, ...],
) -> None:
    export_metadata = export_metadata_factory(
        output=PredictionOutput.ACTION_TOKENS, noise_specs=()
    )
    observations = inference_observations_factory(batch_size=2, dtypes=dtypes)
    sampler = noise_sampler_factory(outputs=())

    prepared = export_metadata.prepare_inputs(observations=observations)

    torch.testing.assert_close(prepared, observations)
    sampler.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prepare_inputs_appends_noise_using_first_floating_observation(
    export_metadata_factory: Callable[..., PolicyExportMetadata],
    inference_observations_factory: Callable[..., tuple[torch.Tensor, ...]],
    sampled_noise_factory: Callable[..., tuple[torch.Tensor, ...]],
    noise_sampler_factory: Callable[..., MagicMock],
    dtype: torch.dtype,
) -> None:
    shapes = ((2, 3), (4, 2, 3))
    export_metadata = export_metadata_factory(
        output=PredictionOutput.ACTIONS,
        noise_specs=(
            (SamplingInput.INITIAL_NOISE.value, shapes[0]),
            (SamplingInput.STEP_NOISE.value, shapes[1]),
        ),
    )
    observations = inference_observations_factory(
        batch_size=2, dtypes=(torch.long, dtype, torch.float64)
    )
    noises = sampled_noise_factory(batch_size=2, shapes=shapes, dtype=dtype)
    sampler = noise_sampler_factory(outputs=noises)

    prepared = export_metadata.prepare_inputs(observations=observations)

    torch.testing.assert_close(prepared, observations + noises)
    assert sampler.call_args_list == [
        call((2, *shape), device=observations[1].device, dtype=dtype)
        for shape in shapes
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtypes", [(torch.long,), (torch.long, torch.bool), (torch.bool, torch.long)]
)
def test_prepare_inputs_uses_float32_noise_for_integer_and_boolean_observations(
    export_metadata_factory: Callable[..., PolicyExportMetadata],
    inference_observations_factory: Callable[..., tuple[torch.Tensor, ...]],
    sampled_noise_factory: Callable[..., tuple[torch.Tensor, ...]],
    noise_sampler_factory: Callable[..., MagicMock],
    dtypes: tuple[torch.dtype, ...],
) -> None:
    noise_shape = (2, 3)
    export_metadata = export_metadata_factory(
        output=PredictionOutput.ACTIONS,
        noise_specs=((SamplingInput.INITIAL_NOISE.value, noise_shape),),
    )
    observations = inference_observations_factory(batch_size=2, dtypes=dtypes)
    noises = sampled_noise_factory(
        batch_size=2, shapes=(noise_shape,), dtype=torch.float32
    )
    sampler = noise_sampler_factory(outputs=noises)

    prepared = export_metadata.prepare_inputs(observations=observations)

    torch.testing.assert_close(prepared, observations + noises)
    sampler.assert_called_once_with(
        (2, *noise_shape), device=observations[0].device, dtype=torch.float32
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtypes, scalar, error_message",
    [
        (
            (),
            False,
            "Policy export noise inputs require at least one observation tensor.",
        ),
        (
            (torch.long,),
            True,
            "Policy export noise inputs require an observation "
            "tensor with a batch dimension.",
        ),
        (
            (torch.float32,),
            True,
            "Policy export noise inputs require an observation "
            "tensor with a batch dimension.",
        ),
    ],
)
def test_prepare_inputs_rejects_observations_that_cannot_define_noise_tensors(
    export_metadata_factory: Callable[..., PolicyExportMetadata],
    inference_observations_factory: Callable[..., tuple[torch.Tensor, ...]],
    noise_sampler_factory: Callable[..., MagicMock],
    dtypes: tuple[torch.dtype, ...],
    scalar: bool,
    error_message: str,
) -> None:
    export_metadata = export_metadata_factory(
        output=PredictionOutput.ACTIONS,
        noise_specs=((SamplingInput.INITIAL_NOISE.value, (2, 3)),),
    )
    observations = inference_observations_factory(
        batch_size=2, dtypes=dtypes, scalar=scalar
    )
    sampler = noise_sampler_factory(outputs=())

    with pytest.raises(ValueError, match=re.escape(error_message)):
        export_metadata.prepare_inputs(observations=observations)

    sampler.assert_not_called()
