"""Tests for versatil.quantization.strategies module."""

import re
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    PerGroup,
)

from versatil.quantization.backends.x86_inductor import X86InductorBackend
from versatil.quantization.strategies import (
    PT2EStrategy,
    QATStrategy,
    QuantizeApiStrategy,
)

STRATEGIES_MODULE = "versatil.quantization.strategies"


class PolicyWithLinearModules(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(32, 16), nn.Linear(8, 8))


@pytest.mark.unit
class TestPT2EStrategy:
    @pytest.mark.parametrize("is_dynamic", [True, False])
    def test_needs_calibration_reflects_dynamic_flag(self, is_dynamic):
        backend = X86InductorBackend(is_dynamic=is_dynamic)
        strategy = PT2EStrategy(pt2e_backend=backend)

        assert strategy.needs_calibration == (not is_dynamic)

    def test_backend_accessible_via_property(self):
        backend = X86InductorBackend(is_dynamic=True)
        strategy = PT2EStrategy(pt2e_backend=backend)

        assert strategy.pt2e_backend.is_dynamic is True


@pytest.mark.unit
class TestQuantizeApiStrategy:
    def test_config_accessible_via_attribute(self):
        config = Int8DynamicActivationInt8WeightConfig()
        strategy = QuantizeApiStrategy(quantize_config=config)

        assert isinstance(
            strategy.quantize_config, Int8DynamicActivationInt8WeightConfig
        )


@pytest.mark.unit
class TestQATStrategy:
    def test_config_accessible_via_attribute(self):
        base_config = Int4WeightOnlyConfig(group_size=32)
        strategy = QATStrategy(
            base_config=base_config,
            module_paths=["decoder"],
            auto_filter_incompatible_linears=False,
        )

        assert strategy.base_config is base_config
        assert strategy.module_paths == ["decoder"]
        assert strategy.auto_filter_incompatible_linears is False

    def test_prepare_calls_quantize_with_qat_prepare_config(self):
        model = PolicyWithLinearModules()
        base_config = Int4WeightOnlyConfig(group_size=32)
        strategy = QATStrategy(base_config=base_config, module_paths=["decoder"])

        with patch(f"{STRATEGIES_MODULE}.quantize_") as quantize_mock:
            strategy.prepare_model(model=model)

        call_kwargs = quantize_mock.call_args.kwargs
        assert call_kwargs["model"] is model
        assert call_kwargs["config"].base_config is base_config
        assert call_kwargs["config"].step == "prepare"
        filter_fn = call_kwargs["filter_fn"]
        assert filter_fn(model.decoder[0], "decoder.0") is True
        assert filter_fn(model.encoder[0], "encoder.0") is False
        assert filter_fn(model.decoder[1], "decoder.1") is False

    def test_prepare_filters_group_incompatible_linears(self):
        model = PolicyWithLinearModules()
        base_config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
        )
        strategy = QATStrategy(base_config=base_config)

        with patch(f"{STRATEGIES_MODULE}.quantize_") as quantize_mock:
            strategy.prepare_model(model=model)

        filter_fn = quantize_mock.call_args.kwargs["filter_fn"]
        assert filter_fn(model.encoder[0], "encoder.0") is True
        assert filter_fn(model.decoder[0], "decoder.0") is True
        assert filter_fn(model.decoder[1], "decoder.1") is False

    def test_prepare_raises_when_module_path_is_missing(self):
        model = PolicyWithLinearModules()
        strategy = QATStrategy(
            base_config=Int4WeightOnlyConfig(group_size=32),
            module_paths=["missing"],
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "QAT module path 'missing' not found in model. "
                "Available top-level modules: ['encoder', 'decoder']."
            ),
        ):
            strategy.prepare_model(model=model)

    def test_prepare_raises_when_no_linear_is_eligible(self):
        model = nn.Sequential(nn.Linear(8, 8))
        strategy = QATStrategy(base_config=Int4WeightOnlyConfig(group_size=32))

        with pytest.raises(
            ValueError,
            match=re.escape(
                "QAT selected zero eligible nn.Linear modules. "
                "Skipped modules: 0: in_features 8 is not divisible by group_size 32."
            ),
        ):
            strategy.prepare_model(model=model)

    def test_convert_requires_prepare_first(self):
        strategy = QATStrategy(base_config=Int4WeightOnlyConfig(group_size=32))

        with pytest.raises(
            ValueError,
            match=re.escape("QAT convert_model() requires prepare_model() first."),
        ):
            strategy.convert_model(model=MagicMock(spec=nn.Module))

    def test_convert_calls_quantize_with_qat_convert_config(self):
        model = PolicyWithLinearModules()
        base_config = Int4WeightOnlyConfig(group_size=32)
        strategy = QATStrategy(base_config=base_config)

        with patch(f"{STRATEGIES_MODULE}.quantize_"):
            strategy.prepare_model(model=model)
        with patch(f"{STRATEGIES_MODULE}.quantize_") as quantize_mock:
            strategy.convert_model(model=model)

        call_kwargs = quantize_mock.call_args.kwargs
        assert call_kwargs["model"] is model
        assert call_kwargs["config"].base_config is base_config
        assert call_kwargs["config"].step == "convert"
        filter_fn = call_kwargs["filter_fn"]
        assert filter_fn(model.encoder[0], "encoder.0") is True
