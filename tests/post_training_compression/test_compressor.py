"""Tests for versatil.post_training_compression.compressor module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.post_training_compression.compressor import (
    ModuleCompressor,
    PostTrainingCompressor,
)
from versatil.post_training_compression.pruning.structured import StructuredPruner
from versatil.post_training_compression.pruning.unstructured import UnstructuredPruner
from versatil.quantization.backends.x86_inductor import X86InductorBackend
from versatil.quantization.strategies import PT2EStrategy, QuantizeApiStrategy


@pytest.fixture
def policy_with_submodules() -> nn.Module:
    class Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
            self.decoder = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.backbone(x))

    return Policy()


@pytest.fixture
def pruning_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    """Factory for a model with Conv2d and Linear layers."""

    def factory(
        input_channels: int = 3,
        hidden_channels: int = 32,
        linear_features: int = 64,
        output_features: int = 4,
    ) -> nn.Module:
        model = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_channels,
                out_features=linear_features,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=linear_features,
                out_features=output_features,
            ),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                data = rng.standard_normal(parameter.shape).astype(np.float32)
                parameter.copy_(torch.from_numpy(data))
        return model

    return factory


@pytest.mark.unit
class TestModuleCompressorValidation:
    @pytest.mark.parametrize(
        "quantization, module_path, expectation",
        [
            (None, "encoder", does_not_raise()),
            (MagicMock(), "encoder", does_not_raise()),
            (
                # spec=[] prevents hasattr from matching act_quant_scale
                QuantizeApiStrategy(quantize_config=MagicMock(spec=[])),
                "encoder",
                does_not_raise(),
            ),
            (
                QuantizeApiStrategy(
                    quantize_config=MagicMock(act_quant_scale=None),
                ),
                "backbone",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Module 'backbone' uses a static activation "
                        "quantize_() config. Static quantization is only "
                        "supported via PT2E. Use PT2EStrategy or a "
                        "dynamic/weight-only config."
                    ),
                ),
            ),
            (
                QuantizeApiStrategy(
                    quantize_config=MagicMock(act_quant_scale=None),
                ),
                "",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Module '(root)' uses a static activation quantize_() config."
                    ),
                ),
            ),
        ],
        ids=[
            "none",
            "non_quantize_api",
            "dynamic_quantize_api",
            "static_quantize_api",
            "static_quantize_api_root",
        ],
    )
    def test_quantization_validation(self, quantization, module_path, expectation):
        with expectation:
            ModuleCompressor(
                module_path=module_path,
                quantization=quantization,
            )


@pytest.mark.unit
class TestPostTrainingCompressorValidate:
    @pytest.mark.parametrize(
        "module_path, expectation",
        [
            ("backbone", does_not_raise()),
            ("decoder", does_not_raise()),
            ("backbone.0", does_not_raise()),
            ("", does_not_raise()),
            (
                "nonexistent_module",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Module path 'nonexistent_module' not found in "
                        "policy. Available top-level modules: "
                        "['backbone', 'decoder']"
                    ),
                ),
            ),
        ],
    )
    def test_module_path_validation(
        self,
        policy_with_submodules,
        module_path,
        expectation,
    ):
        compressor = PostTrainingCompressor(
            checkpoint_path="/tmp/ckpt",
            modules=[ModuleCompressor(module_path=module_path)],
            preparation=MagicMock(),
        )

        with expectation:
            compressor.validate(policy=policy_with_submodules)

    def test_raises_when_mixing_pt2e_and_quantize_api(
        self,
        policy_with_submodules,
    ):
        pt2e_quant = PT2EStrategy(pt2e_backend=X86InductorBackend())
        qapi_quant = QuantizeApiStrategy(quantize_config=MagicMock(spec=[]))
        compressor = PostTrainingCompressor(
            checkpoint_path="/tmp/ckpt",
            modules=[
                ModuleCompressor(module_path="backbone", quantization=pt2e_quant),
                ModuleCompressor(module_path="decoder", quantization=qapi_quant),
            ],
            preparation=MagicMock(),
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "PT2E and quantize_() strategies cannot be combined. "
                "PT2E operates on the exported FX graph while "
                "quantize_() requires eager nn.Module submodules. "
                "Use one strategy per compression run."
            ),
        ):
            compressor.validate(policy=policy_with_submodules)

    def test_empty_modules_passes_validation(
        self,
        policy_with_submodules,
    ):
        compressor = PostTrainingCompressor(
            checkpoint_path="/tmp/ckpt",
            modules=[],
            preparation=MagicMock(),
        )

        compressor.validate(policy=policy_with_submodules)

    def test_multiple_modules_with_one_invalid_path(
        self,
        policy_with_submodules,
    ):
        compressor = PostTrainingCompressor(
            checkpoint_path="/tmp/ckpt",
            modules=[
                ModuleCompressor(module_path="backbone"),
                ModuleCompressor(module_path="nonexistent"),
            ],
            preparation=MagicMock(),
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Module path 'nonexistent' not found in policy. "
                "Available top-level modules: ['backbone', 'decoder']"
            ),
        ):
            compressor.validate(policy=policy_with_submodules)


@pytest.mark.unit
class TestResolveModules:
    def test_returns_explicit_modules_when_provided(self):
        explicit = [ModuleCompressor(module_path="backbone")]
        compressor = PostTrainingCompressor(
            checkpoint_path="/tmp/ckpt",
            modules=explicit,
            preparation=MagicMock(),
        )

        assert compressor.resolve_modules() is explicit

    def test_falls_back_to_global_settings_when_empty(self):
        preparation = MagicMock()
        pruning = [MagicMock(), MagicMock()]
        quantization = PT2EStrategy(pt2e_backend=X86InductorBackend())
        compressor = PostTrainingCompressor(
            checkpoint_path="/tmp/ckpt",
            modules=[],
            preparation=preparation,
            pruning=pruning,
            quantization=quantization,
        )

        resolved = compressor.resolve_modules()

        assert len(resolved) == 1
        assert resolved[0].module_path == ""
        assert resolved[0].preparation is preparation
        assert resolved[0].pruning is pruning
        assert resolved[0].quantization is quantization


@pytest.mark.unit
class TestSequentialPruning:
    def test_structured_then_unstructured_increases_sparsity(
        self,
        pruning_model_factory: Callable[..., nn.Module],
    ):
        model = pruning_model_factory()
        structured = StructuredPruner(amount=0.3)
        unstructured = UnstructuredPruner(amount=0.3)

        _, zeroed_after_structured = structured.prune(module=model)
        _, zeroed_after_both = unstructured.prune(module=model)

        assert zeroed_after_both > zeroed_after_structured

    def test_unstructured_then_structured_increases_sparsity(
        self,
        pruning_model_factory: Callable[..., nn.Module],
    ):
        model = pruning_model_factory()
        unstructured = UnstructuredPruner(amount=0.3)
        structured = StructuredPruner(amount=0.3)

        _, zeroed_after_unstructured = unstructured.prune(module=model)
        _, zeroed_after_both = structured.prune(module=model)

        assert zeroed_after_both > zeroed_after_unstructured

    def test_model_produces_finite_output_after_both(
        self,
        pruning_model_factory: Callable[..., nn.Module],
        rng: np.random.Generator,
    ):
        model = pruning_model_factory()
        input_data = torch.from_numpy(
            rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
        )

        StructuredPruner(amount=0.3).prune(module=model)
        UnstructuredPruner(amount=0.5).prune(module=model)

        with torch.no_grad():
            output = model(input_data)
        assert output.shape == (2, 4)
        assert output.isfinite().all()
