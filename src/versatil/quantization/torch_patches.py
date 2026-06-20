"""Compatibility patches for torchao version-specific issues."""

import importlib
import importlib.util
import sys
from pathlib import Path

_PT2E_MODULE_PATCHES: dict[str, dict[str, str]] = {
    "torchao.quantization.pt2e": {
        'ObserverOrFakeQuantize.__module__ = "torchao.quantization.pt2e"': "pass",
        'ObserverOrFakeQuantizeConstructor.__module__ = "torchao.quantization.pt2e"': "pass",
    },
    "torchao.quantization.pt2e.quantizer.quantizer": {
        'EdgeOrNode.__module__ = "torchao.quantization.pt2e.quantizer.quantizer"': "pass",
    },
}

_QAT_INT4_GROUP_SIZE_ORIGINAL = """weight_config = Int4WeightFakeQuantizeConfig(
                group_size=128,
                activation_dtype=torch.bfloat16,
            )"""
_QAT_INT4_GROUP_SIZE_REPLACEMENT = """weight_config = Int4WeightFakeQuantizeConfig(
                group_size=base_config.group_size,
                activation_dtype=torch.bfloat16,
            )"""


def patch_pt2e_python314() -> None:
    """Patch torchao pt2e files for Python 3.14+ compatibility.

    Python 3.14 makes Union objects immutable. torchao still assigns
    ``__module__`` to Union aliases at PT2E import time, which crashes
    on 3.14+.

    Patches the installed .py files on disk by replacing the
    crashing lines with ``pass``. Idempotent and skips files that
    are already patched.

    Must be called BEFORE importing torchao.quantization.pt2e.

    See:
        https://github.com/pytorch/ao/issues/3619
        https://github.com/pytorch/ao/pull/3657
    """
    if sys.version_info < (3, 14):
        return
    spec = importlib.util.find_spec("torchao")
    if spec is None or not spec.submodule_search_locations:
        return

    pt2e_dir = Path(spec.submodule_search_locations[0]) / "quantization" / "pt2e"
    if not pt2e_dir.exists():
        return

    patched_marker = "# versatil-pt2e-patched"
    patched_any = False

    for module_name, replacements in _PT2E_MODULE_PATCHES.items():
        relative_path = module_name.replace("torchao.quantization.pt2e", "").lstrip(".")
        if not relative_path:
            file_path = pt2e_dir / "__init__.py"
        else:
            file_path = pt2e_dir / relative_path.replace(".", "/")
            if file_path.is_dir():
                file_path = file_path / "__init__.py"
            else:
                file_path = file_path.with_suffix(".py")

        if not file_path.exists():
            continue

        source = file_path.read_text()
        if patched_marker in source:
            continue

        patched = source
        for original, replacement in replacements.items():
            patched = patched.replace(original, replacement)

        if patched != source:
            file_path.write_text(patched + f"\n{patched_marker}\n")
            patched_any = True

    if patched_any:
        importlib.invalidate_caches()


def patch_qat_int4_group_size() -> None:
    """Patch torchao QAT int4 fake-quant group size propagation.

    torchao 0.17 hardcodes ``Int4WeightFakeQuantizeConfig(group_size=128)``
    when preparing QAT from ``Int4WeightOnlyConfig(version=2)``. That ignores
    ``Int4WeightOnlyConfig.group_size`` and crashes training forwards for
    linear layers whose input dimension is compatible with the user-selected
    group size but not 128.

    Patches the installed torchao file on disk so the fake-quant config uses
    ``base_config.group_size``. Idempotent and skips clean torchao wheels that
    already contain the upstream fix.

    Must be called before importing ``torchao.quantization.qat``.

    See:
        https://github.com/pytorch/ao/issues/3572
        https://github.com/pytorch/ao/pull/4518
    """
    spec = importlib.util.find_spec("torchao")
    if spec is None or not spec.submodule_search_locations:
        return

    package_path = Path(spec.submodule_search_locations[0])
    file_path = package_path / "quantization" / "qat" / "fake_quantize_config.py"
    if not file_path.exists():
        return

    patched_marker = "# versatil-qat-int4-group-size-patched"
    source = file_path.read_text()
    if _QAT_INT4_GROUP_SIZE_REPLACEMENT in source:
        return
    if patched_marker in source:
        return

    patched = source.replace(
        _QAT_INT4_GROUP_SIZE_ORIGINAL,
        _QAT_INT4_GROUP_SIZE_REPLACEMENT,
    )
    if patched == source:
        return

    file_path.write_text(patched + f"\n{patched_marker}\n")
    importlib.invalidate_caches()


patch_pt2e_python314()
patch_qat_int4_group_size()
