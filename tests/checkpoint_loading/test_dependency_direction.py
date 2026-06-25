"""Tests for checkpoint-loading dependency direction."""

from pathlib import Path

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_path",
    [
        "src/versatil/checkpoint_loading/base.py",
        "src/versatil/checkpoint_loading/compressed_policy.py",
        "src/versatil/checkpoint_loading/float_policy.py",
        "src/versatil/checkpoint_loading/qat_policy.py",
    ],
)
def test_checkpoint_loading_does_not_import_inference_runtime(module_path: str) -> None:
    source = Path(module_path).read_text()

    assert "versatil.inference" not in source
