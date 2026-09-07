"""Integration tests for VersatIL's supported TorchAO dependencies."""

import subprocess
import sys

import pytest
from torchao.quantization import Int4WeightOnlyConfig
from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs


@pytest.mark.integration
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
def test_int4_qat_preserves_the_requested_ptq_group_size(group_size: int) -> None:
    base_config = Int4WeightOnlyConfig(group_size=group_size, version=2)

    _, weight_config = _infer_fake_quantize_configs(base_config)

    assert weight_config.group_size == group_size


@pytest.mark.integration
def test_pt2e_imports_without_versatil_compatibility_patches() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; import torchao.quantization.pt2e; "
            "import torchao.quantization.pt2e.quantizer.quantizer; "
            "print('versatil' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    assert result.stdout.strip() == "False"
