"""Tests for versatil.quantization.quantize module."""

import re
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.quantization.backends.base import BasePT2EBackend
from versatil.quantization.quantize import (
    apply_pt2e_quantization,
    apply_quantize_api,
)
from versatil.quantization.strategies import PT2EStrategy

QUANTIZE_MODULE = "versatil.quantization.quantize"


@pytest.fixture
def mock_pt2e_backend_factory():
    """Factory for mock PT2E backends with configurable dynamics."""

    def factory(is_dynamic: bool = False) -> MagicMock:
        backend = MagicMock(spec=BasePT2EBackend)
        backend.is_dynamic = is_dynamic
        backend.create_quantizer.return_value = MagicMock()
        backend.environment_context.return_value = does_not_raise()
        return backend

    return factory


@pytest.fixture
def compression_target_factory(mock_pt2e_backend_factory):
    """Factory for CompressionTarget with PT2E quantization."""

    def factory(
        module_path: str = "",
        needs_calibration: bool = False,
    ) -> CompressionTarget:
        backend = mock_pt2e_backend_factory(is_dynamic=not needs_calibration)
        quant = MagicMock(spec=PT2EStrategy)
        quant.needs_calibration = needs_calibration
        quant.pt2e_backend = backend
        return CompressionTarget(
            module_path=module_path,
            quantization=quant,
        )

    return factory


@pytest.fixture
def quantize_api_compressor_factory():
    """Factory for CompressionTarget with quantize_() API quantization."""

    def factory(module_path: str = "") -> CompressionTarget:
        mock_quantization = MagicMock()
        mock_quantization.quantize_config = MagicMock()
        return CompressionTarget(
            module_path=module_path,
            quantization=mock_quantization,
        )

    return factory


@pytest.fixture
def pt2e_mocks():
    """Patch all external dependencies of apply_pt2e_quantization."""
    with (
        patch(f"{QUANTIZE_MODULE}.convert_pt2e") as mock_convert,
        patch(f"{QUANTIZE_MODULE}.prepare_pt2e") as mock_prepare,
        patch(f"{QUANTIZE_MODULE}.ComposableQuantizer") as mock_composer,
    ):
        mock_convert.return_value = MagicMock()
        mock_convert.return_value.graph = MagicMock()
        mock_convert.return_value.graph.__str__ = MagicMock(return_value="")
        yield {
            "convert": mock_convert,
            "prepare": mock_prepare,
            "composer": mock_composer,
        }


@pytest.mark.unit
class TestApplyPT2EQuantization:
    def test_empty_modules_returns_exported_unchanged(self):
        exported = MagicMock(spec=nn.Module)

        result = apply_pt2e_quantization(
            exported=exported,
            pt2e_modules=[],
            calibration=None,
        )

        assert result is exported

    @pytest.mark.parametrize(
        "needs_calibration, has_calibration, expectation",
        [
            (
                True,
                False,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "PT2E static quantization requires calibration data "
                        "but no CalibrationDataProvider was supplied."
                    ),
                ),
            ),
            (True, True, does_not_raise()),
            (False, False, does_not_raise()),
        ],
    )
    def test_calibration_validation(
        self,
        compression_target_factory,
        mock_calibration_provider_factory,
        pt2e_mocks,
        needs_calibration,
        has_calibration,
        expectation,
    ):
        compressor = compression_target_factory(needs_calibration=needs_calibration)
        calibration = mock_calibration_provider_factory() if has_calibration else None

        with expectation:
            apply_pt2e_quantization(
                exported=MagicMock(spec=nn.Module),
                pt2e_modules=[compressor],
                calibration=calibration,
            )

    @pytest.mark.parametrize("module_path", ["", "encoder.backbone"])
    def test_delegates_to_backend_with_correct_module_path(
        self,
        compression_target_factory,
        pt2e_mocks,
        module_path,
    ):
        compressor = compression_target_factory(module_path=module_path)

        apply_pt2e_quantization(
            exported=MagicMock(spec=nn.Module),
            pt2e_modules=[compressor],
            calibration=None,
        )

        compressor.quantization.pt2e_backend.create_quantizer.assert_called_once_with(
            module_path=module_path,
        )

    def test_multiple_modules_creates_quantizer_per_module(
        self,
        compression_target_factory,
        pt2e_mocks,
    ):
        compressor_a = compression_target_factory(module_path="encoder")
        compressor_b = compression_target_factory(module_path="decoder")

        apply_pt2e_quantization(
            exported=MagicMock(spec=nn.Module),
            pt2e_modules=[compressor_a, compressor_b],
            calibration=None,
        )

        compressor_a.quantization.pt2e_backend.create_quantizer.assert_called_once()
        compressor_b.quantization.pt2e_backend.create_quantizer.assert_called_once()

    def test_prepare_convert_chain(
        self,
        compression_target_factory,
        pt2e_mocks,
    ):
        exported = MagicMock(spec=nn.Module)
        compressor = compression_target_factory()
        mock_prepared = MagicMock()
        pt2e_mocks["prepare"].return_value = mock_prepared
        mock_converted = MagicMock()
        mock_converted.graph.__str__ = MagicMock(return_value="")
        pt2e_mocks["convert"].return_value = mock_converted

        result = apply_pt2e_quantization(
            exported=exported,
            pt2e_modules=[compressor],
            calibration=None,
        )

        pt2e_mocks["prepare"].assert_called_once()
        pt2e_mocks["convert"].assert_called_once_with(mock_prepared)
        assert result is mock_converted

    def test_calibration_batches_flow_through_prepared_model(
        self,
        compression_target_factory,
        mock_calibration_provider_factory,
        pt2e_mocks,
    ):
        compressor = compression_target_factory(needs_calibration=True)
        calibration = mock_calibration_provider_factory(num_batches=3)
        mock_prepared = MagicMock()
        pt2e_mocks["prepare"].return_value = mock_prepared

        apply_pt2e_quantization(
            exported=MagicMock(spec=nn.Module),
            pt2e_modules=[compressor],
            calibration=calibration,
        )

        assert mock_prepared.call_count == 3

    def test_enters_backend_environment_context(
        self,
        compression_target_factory,
        pt2e_mocks,
    ):
        compressor = compression_target_factory()

        apply_pt2e_quantization(
            exported=MagicMock(spec=nn.Module),
            pt2e_modules=[compressor],
            calibration=None,
        )

        compressor.quantization.pt2e_backend.environment_context.assert_called_once()


@pytest.mark.unit
class TestApplyQuantizeApi:
    @pytest.mark.parametrize("module_path", ["", "decoder"])
    @patch(f"{QUANTIZE_MODULE}.quantize_")
    def test_calls_quantize_with_correct_scoping(
        self,
        mock_quantize_,
        quantize_api_compressor_factory,
        module_path,
    ):
        compressor = quantize_api_compressor_factory(module_path=module_path)

        apply_quantize_api(
            model=MagicMock(spec=nn.Module),
            quantize_api_modules=[compressor],
        )

        mock_quantize_.assert_called_once()
        if module_path == "":
            assert "filter_fn" not in mock_quantize_.call_args.kwargs
        else:
            assert "filter_fn" in mock_quantize_.call_args.kwargs

    @patch(f"{QUANTIZE_MODULE}.quantize_")
    def test_multiple_modules_calls_quantize_per_module(
        self,
        mock_quantize_,
        quantize_api_compressor_factory,
    ):
        compressor_a = quantize_api_compressor_factory(module_path="encoder")
        compressor_b = quantize_api_compressor_factory(module_path="decoder")

        apply_quantize_api(
            model=MagicMock(spec=nn.Module),
            quantize_api_modules=[compressor_a, compressor_b],
        )

        assert mock_quantize_.call_count == 2


@pytest.mark.unit
class TestFilterFnBehavior:
    @pytest.mark.parametrize(
        "module_path, fqn, module_type, expected",
        [
            ("decoder", "decoder", nn.Linear, True),
            ("decoder", "decoder.layer1", nn.Linear, True),
            ("decoder", "encoder.backbone", nn.Linear, False),
            ("decoder", "decoder_head", nn.Linear, False),
            ("decoder", "decoder", nn.Conv2d, False),
            ("decoder", "decoder.layer1", nn.ReLU, False),
        ],
    )
    @patch(f"{QUANTIZE_MODULE}.quantize_")
    def test_filter_fn_scoping(
        self,
        mock_quantize_,
        quantize_api_compressor_factory,
        module_path,
        fqn,
        module_type,
        expected,
    ):
        compressor = quantize_api_compressor_factory(module_path=module_path)

        apply_quantize_api(
            model=MagicMock(spec=nn.Module),
            quantize_api_modules=[compressor],
        )

        filter_fn = mock_quantize_.call_args.kwargs["filter_fn"]
        # Create real module instances for isinstance checks
        module_instances = {
            nn.Linear: nn.Linear(4, 2),
            nn.Conv2d: nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU: nn.ReLU(),
        }
        result = filter_fn(module_instances[module_type], fqn)

        assert result == expected
