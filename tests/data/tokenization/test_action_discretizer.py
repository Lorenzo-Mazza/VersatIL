"""Tests for versatil.data.tokenization.action_discretizer module."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from transformers.processing_utils import ProcessorMixin

from versatil.data.tokenization.action_discretizer import FastActionDiscretizer


@pytest.fixture
def fast_assets_factory() -> Callable[..., FastActionDiscretizer]:
    def factory(use_pretrained: bool, has_processor: bool) -> FastActionDiscretizer:
        discretizer = FastActionDiscretizer.__new__(FastActionDiscretizer)
        discretizer.use_pretrained = use_pretrained
        discretizer.processor = (
            MagicMock(spec=ProcessorMixin) if has_processor else None
        )
        return discretizer

    return factory


@pytest.mark.unit
@pytest.mark.parametrize("use_pretrained", [False, True])
@pytest.mark.parametrize("has_processor", [False, True])
def test_saves_processor_code_before_serializing_fast_assets(
    fast_assets_factory: Callable[..., FastActionDiscretizer],
    use_pretrained: bool,
    has_processor: bool,
) -> None:
    discretizer = fast_assets_factory(
        use_pretrained=use_pretrained, has_processor=has_processor
    )
    destination = Path("/artifact/tokenizer")

    discretizer.save_pretrained(path=destination)

    if has_processor:
        assert discretizer.processor.mock_calls == [
            call.register_for_auto_class(),
            call.save_pretrained(str(destination / "fast_processor")),
        ]
