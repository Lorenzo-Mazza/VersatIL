"""Tests for versatil.quantization.torch_patches module."""

import importlib
import importlib.util
from collections.abc import Callable, Iterator
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from versatil.quantization.torch_patches import (
    _PT2E_MODULE_PATCHES,
    _QAT_INT4_GROUP_SIZE_ORIGINAL,
    _QAT_INT4_GROUP_SIZE_REPLACEMENT,
    patch_pt2e_python314,
    patch_qat_int4_group_size,
)


@pytest.fixture
def python_314_version() -> Iterator[None]:
    """Simulate Python 3.14 so the patch gate opens regardless of host version."""
    with patch(
        "versatil.quantization.torch_patches.sys.version_info",
        (3, 14, 0),
    ):
        yield


@pytest.fixture
def torchao_package_factory(tmp_path: Path) -> Callable[[], Path]:
    """Factory for fake torchao package trees."""

    def factory() -> Path:
        package_path = tmp_path / "torchao"
        pt2e_path = package_path / "quantization" / "pt2e"
        quantizer_path = pt2e_path / "quantizer"
        quantizer_path.mkdir(parents=True)
        (pt2e_path / "__init__.py").write_text(
            "\n".join(_PT2E_MODULE_PATCHES["torchao.quantization.pt2e"])
        )
        (quantizer_path / "quantizer.py").write_text(
            "\n".join(
                _PT2E_MODULE_PATCHES["torchao.quantization.pt2e.quantizer.quantizer"]
            )
        )
        return package_path

    return factory


@pytest.fixture
def eager_package_factory(tmp_path: Path) -> Callable[[str], Path]:
    """Factory for fake torchao package trees with QAT fake-quant source."""

    def factory(source: str = _QAT_INT4_GROUP_SIZE_ORIGINAL) -> Path:
        package_path = tmp_path / "torchao"
        qat_path = package_path / "quantization" / "qat"
        qat_path.mkdir(parents=True)
        (qat_path / "fake_quantize_config.py").write_text(source)
        return package_path

    return factory


@pytest.mark.unit
class TestPatchPT2EPython314:
    def test_skips_on_python_below_314(self) -> None:
        find_spec_mock = MagicMock(spec=importlib.util.find_spec)

        with (
            patch(
                "versatil.quantization.torch_patches.sys.version_info",
                (3, 13, 0),
            ),
            patch(
                "versatil.quantization.torch_patches.importlib.util.find_spec",
                new=find_spec_mock,
            ),
        ):
            patch_pt2e_python314()

        find_spec_mock.assert_not_called()

    def test_patch_targets_cover_all_known_crash_sites(self) -> None:
        expected_modules = {
            "torchao.quantization.pt2e",
            "torchao.quantization.pt2e.quantizer.quantizer",
        }

        assert set(_PT2E_MODULE_PATCHES.keys()) == expected_modules

    def test_replaces_known_crashing_assignments(
        self,
        python_314_version: None,
        torchao_package_factory: Callable[[], Path],
    ) -> None:
        package_path = torchao_package_factory()
        spec = MagicMock(spec=ModuleSpec)
        spec.submodule_search_locations = [str(package_path)]
        invalidate_caches_mock = MagicMock(spec=importlib.invalidate_caches)

        with (
            patch(
                "versatil.quantization.torch_patches.importlib.util.find_spec",
                return_value=spec,
            ),
            patch(
                "versatil.quantization.torch_patches.importlib.invalidate_caches",
                new=invalidate_caches_mock,
            ),
        ):
            patch_pt2e_python314()

        pt2e_source = (
            package_path / "quantization" / "pt2e" / "__init__.py"
        ).read_text()
        quantizer_source = (
            package_path / "quantization" / "pt2e" / "quantizer" / "quantizer.py"
        ).read_text()
        assert "ObserverOrFakeQuantize.__module__" not in pt2e_source
        assert "ObserverOrFakeQuantizeConstructor.__module__" not in pt2e_source
        assert "EdgeOrNode.__module__" not in quantizer_source
        assert "# versatil-pt2e-patched" in pt2e_source
        assert "# versatil-pt2e-patched" in quantizer_source
        invalidate_caches_mock.assert_called_once_with()

    def test_skips_when_torchao_is_missing(self, python_314_version: None) -> None:
        find_spec_mock = MagicMock(spec=importlib.util.find_spec, return_value=None)
        invalidate_caches_mock = MagicMock(spec=importlib.invalidate_caches)

        with (
            patch(
                "versatil.quantization.torch_patches.importlib.util.find_spec",
                new=find_spec_mock,
            ),
            patch(
                "versatil.quantization.torch_patches.importlib.invalidate_caches",
                new=invalidate_caches_mock,
            ),
        ):
            patch_pt2e_python314()

        find_spec_mock.assert_called_once_with("torchao")
        invalidate_caches_mock.assert_not_called()

    def test_skips_when_torchao_has_no_package_path(
        self, python_314_version: None
    ) -> None:
        spec = MagicMock(spec=ModuleSpec)
        spec.submodule_search_locations = None
        find_spec_mock = MagicMock(spec=importlib.util.find_spec, return_value=spec)
        invalidate_caches_mock = MagicMock(spec=importlib.invalidate_caches)

        with (
            patch(
                "versatil.quantization.torch_patches.importlib.util.find_spec",
                new=find_spec_mock,
            ),
            patch(
                "versatil.quantization.torch_patches.importlib.invalidate_caches",
                new=invalidate_caches_mock,
            ),
        ):
            patch_pt2e_python314()

        find_spec_mock.assert_called_once_with("torchao")
        invalidate_caches_mock.assert_not_called()

    def test_skips_when_pt2e_directory_is_missing(
        self, python_314_version: None, tmp_path: Path
    ) -> None:
        package_path = tmp_path / "torchao"
        package_path.mkdir()
        spec = MagicMock(spec=ModuleSpec)
        spec.submodule_search_locations = [str(package_path)]
        find_spec_mock = MagicMock(spec=importlib.util.find_spec, return_value=spec)
        invalidate_caches_mock = MagicMock(spec=importlib.invalidate_caches)

        with (
            patch(
                "versatil.quantization.torch_patches.importlib.util.find_spec",
                new=find_spec_mock,
            ),
            patch(
                "versatil.quantization.torch_patches.importlib.invalidate_caches",
                new=invalidate_caches_mock,
            ),
        ):
            patch_pt2e_python314()

        find_spec_mock.assert_called_once_with("torchao")
        invalidate_caches_mock.assert_not_called()

    def test_skips_files_that_are_already_patched(
        self,
        python_314_version: None,
        torchao_package_factory: Callable[[], Path],
    ) -> None:
        package_path = torchao_package_factory()
        pt2e_file = package_path / "quantization" / "pt2e" / "__init__.py"
        quantizer_file = (
            package_path / "quantization" / "pt2e" / "quantizer" / "quantizer.py"
        )
        pt2e_file.write_text(pt2e_file.read_text() + "\n# versatil-pt2e-patched\n")
        quantizer_file.write_text(
            quantizer_file.read_text() + "\n# versatil-pt2e-patched\n"
        )
        spec = MagicMock(spec=ModuleSpec)
        spec.submodule_search_locations = [str(package_path)]
        invalidate_caches_mock = MagicMock(spec=importlib.invalidate_caches)

        with (
            patch(
                "versatil.quantization.torch_patches.importlib.util.find_spec",
                return_value=spec,
            ),
            patch(
                "versatil.quantization.torch_patches.importlib.invalidate_caches",
                new=invalidate_caches_mock,
            ),
        ):
            patch_pt2e_python314()

        assert "ObserverOrFakeQuantize.__module__" in pt2e_file.read_text()
        assert "EdgeOrNode.__module__" in quantizer_file.read_text()
        invalidate_caches_mock.assert_not_called()


@pytest.mark.integration
class TestPatchPT2EPython314Integration:
    def test_pt2e_imports_succeed_on_current_python(self) -> None:
        quantize_module = importlib.import_module(
            "torchao.quantization.pt2e.quantize_pt2e"
        )
        quantizer_module = importlib.import_module(
            "torchao.quantization.pt2e.quantizer"
        )
        x86_module = importlib.import_module(
            "torchao.quantization.pt2e.quantizer.x86_inductor_quantizer"
        )

        assert quantize_module.prepare_pt2e.__name__ == "prepare_pt2e"
        assert quantize_module.convert_pt2e.__name__ == "convert_pt2e"
        assert quantizer_module.Quantizer.__name__ == "Quantizer"
        assert x86_module.X86InductorQuantizer.__name__ == "X86InductorQuantizer"


@pytest.mark.unit
class TestPatchQATInt4GroupSize:
    def test_replaces_hardcoded_group_size(
        self,
        eager_package_factory: Callable[[str], Path],
    ) -> None:
        package_path = eager_package_factory()
        spec = MagicMock(spec=ModuleSpec)
        spec.submodule_search_locations = [str(package_path)]
        invalidate_caches_mock = MagicMock(spec=importlib.invalidate_caches)

        with (
            patch(
                "versatil.quantization.torch_patches.importlib.util.find_spec",
                return_value=spec,
            ),
            patch(
                "versatil.quantization.torch_patches.importlib.invalidate_caches",
                new=invalidate_caches_mock,
            ),
        ):
            patch_qat_int4_group_size()

        source = (
            package_path / "quantization" / "qat" / "fake_quantize_config.py"
        ).read_text()
        assert _QAT_INT4_GROUP_SIZE_ORIGINAL not in source
        assert _QAT_INT4_GROUP_SIZE_REPLACEMENT in source
        assert "# versatil-qat-int4-group-size-patched" in source
        invalidate_caches_mock.assert_called_once_with()

    def test_skips_when_upstream_fix_exists(
        self,
        eager_package_factory: Callable[[str], Path],
    ) -> None:
        package_path = eager_package_factory(source=_QAT_INT4_GROUP_SIZE_REPLACEMENT)
        spec = MagicMock(spec=ModuleSpec)
        spec.submodule_search_locations = [str(package_path)]
        invalidate_caches_mock = MagicMock(spec=importlib.invalidate_caches)

        with (
            patch(
                "versatil.quantization.torch_patches.importlib.util.find_spec",
                return_value=spec,
            ),
            patch(
                "versatil.quantization.torch_patches.importlib.invalidate_caches",
                new=invalidate_caches_mock,
            ),
        ):
            patch_qat_int4_group_size()

        source = (
            package_path / "quantization" / "qat" / "fake_quantize_config.py"
        ).read_text()
        assert source == _QAT_INT4_GROUP_SIZE_REPLACEMENT
        invalidate_caches_mock.assert_not_called()

    def test_skips_when_torchao_is_missing(self) -> None:
        find_spec_mock = MagicMock(spec=importlib.util.find_spec, return_value=None)
        invalidate_caches_mock = MagicMock(spec=importlib.invalidate_caches)

        with (
            patch(
                "versatil.quantization.torch_patches.importlib.util.find_spec",
                new=find_spec_mock,
            ),
            patch(
                "versatil.quantization.torch_patches.importlib.invalidate_caches",
                new=invalidate_caches_mock,
            ),
        ):
            patch_qat_int4_group_size()

        find_spec_mock.assert_called_once_with("torchao")
        invalidate_caches_mock.assert_not_called()
