"""Tests for versatil.models.decoding.decoders.vlm module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.decoders.vlm import VLMBackboneDecoderMixin
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)
from versatil.models.input_specification import InputSpecification


class ConcreteVLMBackboneDecoder(VLMBackboneDecoderMixin):
    def __init__(self, vlm_backbone: GenerativeVLM) -> None:
        self.vlm_backbone = vlm_backbone


@pytest.fixture
def vlm_backbone_factory() -> Callable[..., MagicMock]:
    def factory(input_keys: list[str]) -> MagicMock:
        vlm_backbone = MagicMock(spec=GenerativeVLM)
        vlm_backbone.input_specification = InputSpecification(keys=input_keys)
        return vlm_backbone

    return factory


@pytest.fixture
def vlm_decoder_factory(
    vlm_backbone_factory: Callable[..., MagicMock],
) -> Callable[..., ConcreteVLMBackboneDecoder]:
    def factory(input_keys: list[str]) -> ConcreteVLMBackboneDecoder:
        return ConcreteVLMBackboneDecoder(
            vlm_backbone=vlm_backbone_factory(input_keys=input_keys)
        )

    return factory


@pytest.mark.unit
def test_vlm_decoder_input_keys_deduplicate_configured_and_backbone_keys(
    vlm_backbone_factory: Callable[..., MagicMock],
) -> None:
    vlm_backbone = vlm_backbone_factory(
        input_keys=["left_rgb", "tokenized_observations", "is_pad_observation"]
    )

    keys = ConcreteVLMBackboneDecoder._vlm_decoder_input_keys(
        input_keys=["encoded_proprio", "left_rgb"],
        vlm_backbone=vlm_backbone,
    )

    assert keys == [
        "encoded_proprio",
        "left_rgb",
        "tokenized_observations",
        "is_pad_observation",
    ]


@pytest.mark.unit
def test_validate_no_extra_input_keys_accepts_empty_input_keys() -> None:
    ConcreteVLMBackboneDecoder._validate_no_extra_input_keys(
        decoder_name="AutoregressiveVLADecoder",
        input_keys=[],
    )


@pytest.mark.unit
def test_validate_no_extra_input_keys_rejects_extra_input_keys() -> None:
    input_keys = ["encoded_proprio"]
    expected_message = (
        "AutoregressiveVLADecoder builds its prefix from vlm_backbone inputs. "
        f"Set input_keys to an empty list, got {input_keys}."
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        ConcreteVLMBackboneDecoder._validate_no_extra_input_keys(
            decoder_name="AutoregressiveVLADecoder",
            input_keys=input_keys,
        )


@pytest.mark.unit
def test_build_vlm_prefix_delegates_to_backbone(
    vlm_decoder_factory: Callable[..., ConcreteVLMBackboneDecoder],
) -> None:
    decoder = vlm_decoder_factory(input_keys=["left_rgb"])
    features = {"left_rgb": torch.ones(2, 3, 8, 8)}
    prefix_tokens = torch.ones(2, 5, 4)
    prefix_mask = torch.tensor(
        [
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]
    )
    decoder.vlm_backbone.build_prefix.return_value = prefix_tokens, prefix_mask

    output_tokens, output_mask = decoder._build_vlm_prefix(features=features)

    decoder.vlm_backbone.build_prefix.assert_called_once_with(inputs=features)
    torch.testing.assert_close(output_tokens, prefix_tokens)
    torch.testing.assert_close(output_mask, prefix_mask)
