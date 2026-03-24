"""Tests for versatil.common.module_attr_mixin module."""

import pytest
import torch
import torch.nn as nn

from versatil.common.module_attr_mixin import ModuleAttrMixin


class ConcreteModuleAttrMixin(ModuleAttrMixin):
    """Concrete subclass for testing (adds a real parameter)."""

    def __init__(self, dimension: int = 4):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension)


@pytest.fixture
def module_attr_mixin_factory():
    """Factory for creating ConcreteModuleAttrMixin instances."""

    def factory(dimension: int = 4) -> ConcreteModuleAttrMixin:
        return ConcreteModuleAttrMixin(dimension=dimension)

    return factory


@pytest.mark.unit
class TestModuleAttrMixinDevice:
    def test_reports_cpu_device(self, module_attr_mixin_factory):
        module = module_attr_mixin_factory(dimension=8)
        assert module.device.type == "cpu"

    @pytest.mark.requires_gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reports_cuda_device_after_move(self, module_attr_mixin_factory):
        module = module_attr_mixin_factory(dimension=8).cuda()
        assert module.device.type == "cuda"


@pytest.mark.unit
class TestModuleAttrMixinDtype:
    def test_reports_float32_dtype_by_default(self, module_attr_mixin_factory):
        module = module_attr_mixin_factory(dimension=8)
        assert module.dtype == torch.float32

    def test_reports_float16_after_conversion(self, module_attr_mixin_factory):
        module = module_attr_mixin_factory(dimension=8).half()
        assert module.dtype == torch.float16

    def test_reports_float64_after_conversion(self, module_attr_mixin_factory):
        module = module_attr_mixin_factory(dimension=8).double()
        assert module.dtype == torch.float64


@pytest.mark.unit
class TestModuleAttrMixinDummyVariable:
    def test_dummy_variable_is_registered_parameter(self, module_attr_mixin_factory):
        module = module_attr_mixin_factory(dimension=4)
        parameter_names = [name for name, _ in module.named_parameters()]
        assert "_dummy_variable" in parameter_names

    def test_device_accessible_without_additional_submodules(self):
        # ModuleAttrMixin alone (without subclass adding layers)
        # should still have a device via _dummy_variable
        module = ModuleAttrMixin()
        assert module.device.type == "cpu"
