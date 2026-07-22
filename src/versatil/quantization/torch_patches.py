"""In-memory compatibility patches for torchao version-specific issues.

The patches rewrite torchao module source as it is imported, through a
meta-path finder registered by ``register_torchao_patches``. Nothing is
written to site-packages: other processes and projects sharing the
environment see pristine torchao, read-only installs work, and
uninstalling versatil leaves no trace.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys
from dataclasses import dataclass
from types import CodeType


@dataclass(frozen=True)
class SourcePatch:
    """One source-fragment replacement applied at module load time.

    Attributes:
        original: Source fragment the installed torchao version contains.
        replacement: Fragment compiled in its place.
        required: Whether the module must contain either fragment. Required
            patches raise on unknown torchao source because their absence
            means silently wrong behavior; optional patches disable crashing
            statements, so their absence means upstream already removed them.
    """

    original: str
    replacement: str
    required: bool


_PT2E_MODULE_PATCHES: dict[str, list[SourcePatch]] = {
    # Python 3.14 makes Union objects immutable; torchao still assigns
    # __module__ to Union aliases at import time, which crashes on 3.14+.
    # https://github.com/pytorch/ao/issues/3619
    # https://github.com/pytorch/ao/pull/3657
    "torchao.quantization.pt2e": [
        SourcePatch(
            original=(
                'ObserverOrFakeQuantize.__module__ = "torchao.quantization.pt2e"'
            ),
            replacement="pass",
            required=False,
        ),
        SourcePatch(
            original=(
                "ObserverOrFakeQuantizeConstructor.__module__"
                ' = "torchao.quantization.pt2e"'
            ),
            replacement="pass",
            required=False,
        ),
    ],
    "torchao.quantization.pt2e.quantizer.quantizer": [
        SourcePatch(
            original=(
                "EdgeOrNode.__module__"
                ' = "torchao.quantization.pt2e.quantizer.quantizer"'
            ),
            replacement="pass",
            required=False,
        ),
    ],
}

# torchao 0.17 hardcodes Int4WeightFakeQuantizeConfig(group_size=128) when
# preparing QAT from Int4WeightOnlyConfig(version=2), ignoring the
# user-selected group size.
# https://github.com/pytorch/ao/issues/3572
# https://github.com/pytorch/ao/pull/4518
_QAT_MODULE_PATCHES: dict[str, list[SourcePatch]] = {
    "torchao.quantization.qat.fake_quantize_config": [
        SourcePatch(
            original="""weight_config = Int4WeightFakeQuantizeConfig(
                group_size=128,
                activation_dtype=torch.bfloat16,
            )""",
            replacement="""weight_config = Int4WeightFakeQuantizeConfig(
                group_size=base_config.group_size,
                activation_dtype=torch.bfloat16,
            )""",
            required=True,
        ),
    ],
}


# torch's darwin arm64 build ships no mkldnn ops; torchao's x86 inductor
# passes call torch.ops.mkldnn._is_mkldnn_acl_supported() at import time,
# which raises AttributeError on macOS. The guard keeps upstream's intent:
# the fusion registrations are skipped wherever mkldnn is unavailable.
_INDUCTOR_MODULE_PATCHES: dict[str, list[SourcePatch]] = {
    "torchao.quantization.pt2e.inductor_passes.x86": [
        SourcePatch(
            original="if not torch.ops.mkldnn._is_mkldnn_acl_supported():",
            replacement=(
                'if hasattr(torch.ops.mkldnn, "_is_mkldnn_acl_supported") '
                "and not torch.ops.mkldnn._is_mkldnn_acl_supported():"
            ),
            required=False,
        ),
    ],
}


class PatchingSourceLoader(importlib.machinery.SourceFileLoader):
    """Source loader applying string replacements before compilation."""

    def __init__(self, fullname: str, path: str, patches: list[SourcePatch]) -> None:
        """Initialize the loader.

        Args:
            fullname: Fully qualified module name.
            path: Filesystem path of the module source.
            patches: Replacements to apply to the module source.
        """
        super().__init__(fullname, path)
        self._patches = patches

    def get_code(self, fullname: str) -> CodeType:
        """Compile the patched source, bypassing the bytecode cache.

        Reading the cache could load unpatched bytecode; writing it would
        leak patched bytecode to imports that bypass the hook.

        Args:
            fullname: Fully qualified module name.
        """
        data = self.get_data(self.path)
        return self.source_to_code(data, self.path)

    def source_to_code(self, data: bytes, path: str) -> CodeType:
        """Apply the replacements and compile.

        Args:
            data: Raw module source bytes.
            path: Filesystem path of the module source.

        Raises:
            RuntimeError: If the source contains neither the original nor the
                patched fragment, meaning the installed torchao version is not
                the one these patches target.
        """
        source = importlib.util.decode_source(data)
        for patch in self._patches:
            if patch.original in source:
                source = source.replace(patch.original, patch.replacement)
                continue
            if patch.required and patch.replacement not in source:
                raise RuntimeError(
                    f"torchao module '{self.name}' does not match the version "
                    "the versatil compatibility patches target. Install the "
                    "torchao version pinned in pyproject.toml or update "
                    "versatil.quantization.torch_patches."
                )
        code: CodeType = super().source_to_code(source.encode(), path)
        return code


class TorchaoPatchFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder routing patched torchao modules through the loader."""

    def __init__(self, module_patches: dict[str, list[SourcePatch]]) -> None:
        """Initialize the finder.

        Args:
            module_patches: Mapping from module name to its source patches.
        """
        self._module_patches = module_patches

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None = None,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Return a patched spec for targeted modules.

        Args:
            fullname: Fully qualified module name being imported.
            path: Parent package search path from the import system.
            target: Existing module for reloads, unused.
        """
        patches = self._module_patches.get(fullname)
        if patches is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None:
            return None
        loader = PatchingSourceLoader(
            fullname=fullname, path=spec.origin, patches=patches
        )
        return importlib.util.spec_from_file_location(
            fullname,
            spec.origin,
            loader=loader,
            submodule_search_locations=spec.submodule_search_locations,
        )


def register_torchao_patches() -> None:
    """Install the import hook applying the torchao compatibility patches.

    Idempotent. Must run before ``torchao.quantization`` is imported;
    targeted modules that are already imported cannot be patched in place
    and are reported with a warning.
    """
    if any(isinstance(finder, TorchaoPatchFinder) for finder in sys.meta_path):
        return
    module_patches = dict(_QAT_MODULE_PATCHES)
    if sys.version_info >= (3, 14):
        module_patches.update(_PT2E_MODULE_PATCHES)
    if sys.platform == "darwin":
        module_patches.update(_INDUCTOR_MODULE_PATCHES)
    already_imported = [name for name in module_patches if name in sys.modules]
    if already_imported:
        logging.warning(
            f"torchao modules {already_imported} were imported before versatil "
            "registered its compatibility patches; the patches do not apply to "
            "this process."
        )
    sys.meta_path.insert(0, TorchaoPatchFinder(module_patches=module_patches))
