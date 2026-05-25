"""Compatibility patch for torchao PT2E imports on Python 3.14."""

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


patch_pt2e_python314()
