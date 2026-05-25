"""Quantization execution for PT2E and quantize_() APIs."""

import logging

import torch
import torch.nn as nn
from torchao.quantization import quantize_
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer.composable_quantizer import (
    ComposableQuantizer,
)

from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.constants import FXNodePattern

logger = logging.getLogger(__name__)


def apply_pt2e_quantization(
    exported: nn.Module,
    pt2e_modules: list[CompressionTarget],
    calibration: CalibrationDataProvider | None,
) -> nn.Module:
    """Apply PT2E quantization to targeted modules via ComposableQuantizer.

    Args:
        exported: The exported FX GraphModule.
        pt2e_modules: Resolved modules with PT2EStrategy configs.
        calibration: Calibration data provider (None for all-dynamic).

    Returns:
        The convert_pt2e result.
    """
    if not pt2e_modules:
        return exported
    needs_calibration = any(
        module.quantization.needs_calibration for module in pt2e_modules
    )
    if needs_calibration and calibration is None:
        raise ValueError(
            "PT2E static quantization requires calibration data "
            "but no CalibrationDataProvider was supplied."
        )
    quantizers = []
    for module in pt2e_modules:
        backend = module.quantization.pt2e_backend
        quantizers.append(backend.create_quantizer(module_path=module.module_path))
        logger.info("PT2E target: %s", module.module_path or "(root)")

    composed = ComposableQuantizer(quantizers)
    first_backend = pt2e_modules[0].quantization.pt2e_backend
    with first_backend.environment_context():
        prepared = prepare_pt2e(exported, composed)
        if calibration is not None:
            logger.info("Calibrating PT2E...")
            with torch.no_grad():
                for batch in calibration:
                    prepared(*batch)
        converted = convert_pt2e(prepared)
    logger.info(
        "PT2E done, static ops: %d",
        str(converted.graph).count(FXNodePattern.QUANTIZE_PER_TENSOR.value),
    )
    return converted


def apply_quantize_api(
    model: nn.Module,
    quantize_api_modules: list[CompressionTarget],
) -> None:
    """Apply dynamic quantize_() to targeted modules with filter_fn scoping.

    Modifies the model in-place.

    Note:
        Only `nn.Linear` layers are currently supported by torchao for this API.
        Use PT2E for broader class of layers.

    Args:
        model: The model (possibly already PT2E-converted).
        quantize_api_modules: Resolved modules with QuantizeApiStrategy
            configs.
    """
    for module in quantize_api_modules:
        module_path = module.module_path
        label = module_path or "(root)"
        logger.info("quantize_() target: %s", label)

        if module_path == "":
            quantize_(model, module.quantization.quantize_config)
        else:
            quantize_(
                model,
                module.quantization.quantize_config,
                filter_fn=lambda mod, fqn, mp=module_path: (
                    (fqn == mp or fqn.startswith(mp + "."))
                    and isinstance(mod, nn.Linear)
                ),
            )
