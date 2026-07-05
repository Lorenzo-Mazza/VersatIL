"""Tests for versatil.quantization.torch_patches module."""

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from versatil.quantization.torch_patches import (
    SourcePatch,
    TorchaoPatchFinder,
    register_torchao_patches,
)

FAKE_MODULE_NAME = "torch_patches_fake_target"


@pytest.fixture
def fake_module_factory(
    tmp_path: Path,
) -> Iterator:
    installed_finders: list[TorchaoPatchFinder] = []

    def factory(source: str, patches: list[SourcePatch]) -> Path:
        module_path = tmp_path / f"{FAKE_MODULE_NAME}.py"
        module_path.write_text(source)
        finder = TorchaoPatchFinder(module_patches={FAKE_MODULE_NAME: patches})
        sys.meta_path.insert(0, finder)
        installed_finders.append(finder)
        sys.path.insert(0, str(tmp_path))
        return module_path

    yield factory

    for finder in installed_finders:
        sys.meta_path.remove(finder)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
    sys.modules.pop(FAKE_MODULE_NAME, None)


@pytest.mark.unit
class TestPatchingImport:
    def test_replacement_applies_in_memory_only(self, fake_module_factory) -> None:
        module_path = fake_module_factory(
            source='VALUE = "original"\n',
            patches=[
                SourcePatch(
                    original='VALUE = "original"',
                    replacement='VALUE = "patched"',
                    required=True,
                ),
            ],
        )

        module = __import__(FAKE_MODULE_NAME)

        assert module.VALUE == "patched"
        assert 'VALUE = "original"' in module_path.read_text()
        assert not (module_path.parent / "__pycache__").exists()

    def test_already_fixed_source_is_accepted(self, fake_module_factory) -> None:
        fake_module_factory(
            source='VALUE = "patched"\n',
            patches=[
                SourcePatch(
                    original='VALUE = "original"',
                    replacement='VALUE = "patched"',
                    required=True,
                ),
            ],
        )

        module = __import__(FAKE_MODULE_NAME)

        assert module.VALUE == "patched"

    def test_unknown_source_raises_for_required_patch(
        self, fake_module_factory
    ) -> None:
        fake_module_factory(
            source='VALUE = "rewritten upstream"\n',
            patches=[
                SourcePatch(
                    original='VALUE = "original"',
                    replacement='VALUE = "patched"',
                    required=True,
                ),
            ],
        )

        with pytest.raises(RuntimeError, match="does not match the version"):
            __import__(FAKE_MODULE_NAME)

    def test_missing_original_is_skipped_for_optional_patch(
        self, fake_module_factory
    ) -> None:
        fake_module_factory(
            source='VALUE = "statement removed upstream"\n',
            patches=[
                SourcePatch(
                    original='VALUE = "original"',
                    replacement="pass",
                    required=False,
                ),
            ],
        )

        module = __import__(FAKE_MODULE_NAME)

        assert module.VALUE == "statement removed upstream"

    def test_common_replacement_does_not_mask_later_patches(
        self, fake_module_factory
    ) -> None:
        # Both patches replace with "pass"; applying the first must not make
        # the second look already applied.
        fake_module_factory(
            source="FIRST = 1\nSECOND = 2\n",
            patches=[
                SourcePatch(original="FIRST = 1", replacement="pass", required=True),
                SourcePatch(original="SECOND = 2", replacement="pass", required=True),
            ],
        )

        module = __import__(FAKE_MODULE_NAME)

        assert not hasattr(module, "FIRST")
        assert not hasattr(module, "SECOND")


@pytest.mark.unit
class TestRegisterTorchaoPatches:
    def test_registration_is_idempotent(self) -> None:
        register_torchao_patches()
        finder_count_before = sum(
            isinstance(finder, TorchaoPatchFinder) for finder in sys.meta_path
        )

        register_torchao_patches()

        finder_count_after = sum(
            isinstance(finder, TorchaoPatchFinder) for finder in sys.meta_path
        )
        assert finder_count_before == finder_count_after == 1

    def test_warns_when_target_module_already_imported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        installed = [
            finder for finder in sys.meta_path if isinstance(finder, TorchaoPatchFinder)
        ]
        for finder in installed:
            sys.meta_path.remove(finder)
        sys.modules.setdefault("torchao.quantization.qat.fake_quantize_config", sys)
        try:
            with caplog.at_level(logging.WARNING):
                register_torchao_patches()
            assert any(
                "imported before versatil" in record.message
                for record in caplog.records
            )
        finally:
            if sys.modules.get("torchao.quantization.qat.fake_quantize_config") is sys:
                del sys.modules["torchao.quantization.qat.fake_quantize_config"]
            for finder in list(sys.meta_path):
                if isinstance(finder, TorchaoPatchFinder):
                    sys.meta_path.remove(finder)
            for finder in installed:
                sys.meta_path.insert(0, finder)


@pytest.mark.integration
class TestTorchaoPatchesIntegration:
    def test_qat_group_size_propagates_in_memory(self) -> None:
        fake_quantize_config = pytest.importorskip(
            "torchao.quantization.qat.fake_quantize_config"
        )
        quantization = pytest.importorskip("torchao.quantization")

        base_config = quantization.Int4WeightOnlyConfig(group_size=32, version=2)
        _, weight_config = fake_quantize_config._infer_fake_quantize_configs(
            base_config
        )

        assert weight_config.group_size == 32

    @pytest.mark.skipif(
        sys.version_info < (3, 14), reason="crash only exists on Python 3.14+"
    )
    def test_pt2e_imports_on_python_314(self) -> None:
        pytest.importorskip("torchao.quantization.pt2e")
