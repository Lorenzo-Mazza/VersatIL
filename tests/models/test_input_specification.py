"""Tests for versatil.models.input_specification module."""

import re
from collections.abc import Callable

import pytest

from versatil.data.constants import CameraModality
from versatil.models.input_specification import InputSpecification


@pytest.fixture
def input_specification_factory() -> Callable[..., InputSpecification]:
    def factory(
        keys: str | list[str],
        required: list[str],
        exactly_one_camera_modality: list[CameraModality],
        required_camera_modalities: list[CameraModality],
        conditioning_key: str | None,
        conditioning_required: list[str],
        conditioning_one_of_groups: list[list[str]],
        requires_tokenized: bool,
    ) -> InputSpecification:
        return InputSpecification(
            keys=keys,
            required=required,
            exactly_one_camera_modality=exactly_one_camera_modality,
            required_camera_modalities=required_camera_modalities,
            conditioning_key=conditioning_key,
            conditioning_required=conditioning_required,
            conditioning_one_of_groups=conditioning_one_of_groups,
            requires_tokenized=requires_tokenized,
        )

    return factory


@pytest.mark.unit
@pytest.mark.parametrize("requires_tokenized", [True, False])
@pytest.mark.parametrize(
    "keys, expected_keys",
    [
        ("left", ["left"]),
        (["left", "depth"], ["left", "depth"]),
    ],
)
def test_stores_configuration(
    input_specification_factory: Callable[..., InputSpecification],
    keys: str | list[str],
    expected_keys: list[str],
    requires_tokenized: bool,
) -> None:
    input_specification = input_specification_factory(
        keys=keys,
        required=["left"],
        exactly_one_camera_modality=[CameraModality.RGB],
        required_camera_modalities=[CameraModality.DEPTH],
        conditioning_key="rgb_embedding",
        conditioning_required=["rgb_embedding"],
        conditioning_one_of_groups=[["rgb_embedding", "depth_embedding"]],
        requires_tokenized=requires_tokenized,
    )

    assert input_specification.keys == expected_keys
    assert input_specification.required == ["left"]
    assert input_specification.exactly_one_camera_modality == [CameraModality.RGB]
    assert input_specification.required_camera_modalities == [CameraModality.DEPTH]
    assert input_specification.conditioning_key == "rgb_embedding"
    assert input_specification.conditioning_required == ["rgb_embedding"]
    assert input_specification.conditioning_one_of_groups == [
        ["rgb_embedding", "depth_embedding"]
    ]
    assert input_specification.requires_tokenized is requires_tokenized


@pytest.mark.unit
def test_validate_does_not_inspect_observation_metadata(
    input_specification_factory: Callable[..., InputSpecification],
) -> None:
    input_specification = input_specification_factory(
        keys=["tokenized_observations"],
        required=[],
        exactly_one_camera_modality=[CameraModality.RGB],
        required_camera_modalities=[CameraModality.DEPTH],
        conditioning_key=None,
        conditioning_required=[],
        conditioning_one_of_groups=[],
        requires_tokenized=True,
    )

    input_specification.validate()


@pytest.mark.unit
def test_validate_rejects_missing_required_inputs(
    input_specification_factory: Callable[..., InputSpecification],
) -> None:
    input_specification = input_specification_factory(
        keys=["left"],
        required=["left", "right"],
        exactly_one_camera_modality=[],
        required_camera_modalities=[],
        conditioning_key=None,
        conditioning_required=[],
        conditioning_one_of_groups=[],
        requires_tokenized=False,
    )

    with pytest.raises(
        ValueError, match=re.escape("Missing required inputs: {'right'}")
    ):
        input_specification.validate()


@pytest.mark.unit
def test_validate_rejects_missing_required_conditioning(
    input_specification_factory: Callable[..., InputSpecification],
) -> None:
    input_specification = input_specification_factory(
        keys=["left"],
        required=[],
        exactly_one_camera_modality=[],
        required_camera_modalities=[],
        conditioning_key="rgb_embedding",
        conditioning_required=["missing_key"],
        conditioning_one_of_groups=[],
        requires_tokenized=False,
    )

    with pytest.raises(
        ValueError, match=re.escape("Missing required conditioning: {'missing_key'}")
    ):
        input_specification.validate()


@pytest.mark.unit
def test_validate_rejects_conditioning_key_outside_one_of_group(
    input_specification_factory: Callable[..., InputSpecification],
) -> None:
    input_specification = input_specification_factory(
        keys=["left"],
        required=[],
        exactly_one_camera_modality=[],
        required_camera_modalities=[],
        conditioning_key="other_key",
        conditioning_required=[],
        conditioning_one_of_groups=[["rgb_embedding", "depth_embedding"]],
        requires_tokenized=False,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Exactly one from ['rgb_embedding', 'depth_embedding'] "
            "required for conditioning"
        ),
    ):
        input_specification.validate()


@pytest.mark.parametrize(
    (
        "exactly_one_camera_modality",
        "required_camera_modalities",
        "expected_message",
    ),
    [
        (
            [CameraModality.DEPTH, CameraModality.DEPTH],
            [],
            "Camera modality constraint 'exactly_one_camera_modality' contains "
            "duplicate modalities: ['depth']",
        ),
        (
            [],
            [CameraModality.RGB, CameraModality.RGB],
            "Camera modality constraint 'required_camera_modalities' contains "
            "duplicate modalities: ['rgb']",
        ),
    ],
)
@pytest.mark.unit
def test_validate_rejects_duplicate_camera_modality_constraints(
    input_specification_factory: Callable[..., InputSpecification],
    exactly_one_camera_modality: list[CameraModality],
    required_camera_modalities: list[CameraModality],
    expected_message: str,
) -> None:
    input_specification = input_specification_factory(
        keys=["left"],
        required=[],
        exactly_one_camera_modality=exactly_one_camera_modality,
        required_camera_modalities=required_camera_modalities,
        conditioning_key=None,
        conditioning_required=[],
        conditioning_one_of_groups=[],
        requires_tokenized=False,
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        input_specification.validate()
