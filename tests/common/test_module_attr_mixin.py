"""Tests for versatil.common.module_attr_mixin module."""

import pytest
import torch
import torch.nn as nn

from versatil.common.module_attr_mixin import ModuleAttrMixin


class ConcreteModuleAttrMixin(ModuleAttrMixin):
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
class TestModuleAttrMixinReferenceState:
    def test_has_no_private_reference_in_state_dict(self, module_attr_mixin_factory):
        module = module_attr_mixin_factory(dimension=4)
        assert "_module_attr_reference" not in module.state_dict()

    def test_device_accessible_without_additional_submodules(self):
        module = ModuleAttrMixin()
        assert module.device.type == "cpu"

    def test_dtype_accessible_without_additional_submodules(self):
        module = ModuleAttrMixin()
        assert module.dtype == torch.float32

    def test_device_assignment_moves_parameters(self, module_attr_mixin_factory):
        module = module_attr_mixin_factory(dimension=4)
        module.device = torch.device("cpu")
        assert module.linear.weight.device.type == "cpu"

    def test_dtype_assignment_casts_parameters(self, module_attr_mixin_factory):
        module = module_attr_mixin_factory(dimension=4)
        module.dtype = torch.float64
        assert module.linear.weight.dtype == torch.float64
