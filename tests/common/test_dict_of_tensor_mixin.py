"""Tests for versatil.common.dict_of_tensor_mixin module."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.common.dict_of_tensor_mixin import DictOfTensorMixin


@pytest.fixture
def params_dict_factory(rng):
    """Factory for creating ParameterDict with named parameters."""

    def factory(
        keys: list[str] | None = None,
        dimension: int = 4,
    ) -> nn.ParameterDict:
        if keys is None:
            keys = ["alpha", "beta"]
        params = nn.ParameterDict()
        for key in keys:
            data = torch.from_numpy(
                rng.standard_normal((dimension,)).astype(np.float32)
            )
            params[key] = nn.Parameter(data)
        return params

    return factory


@pytest.fixture
def dict_of_tensor_mixin_factory(params_dict_factory):
    """Factory for creating DictOfTensorMixin instances."""

    def factory(
        keys: list[str] | None = None,
        dimension: int = 4,
        params_dict: nn.ParameterDict | None = None,
    ) -> DictOfTensorMixin:
        if params_dict is None:
            params_dict = params_dict_factory(keys=keys, dimension=dimension)
        return DictOfTensorMixin(params_dict=params_dict)

    return factory


@pytest.mark.unit
class TestDictOfTensorMixinInit:
    def test_stores_provided_params_dict(self, params_dict_factory):
        params = params_dict_factory(keys=["weight", "bias"])
        mixin = DictOfTensorMixin(params_dict=params)
        assert set(mixin.params_dict.keys()) == {"weight", "bias"}

    def test_creates_empty_params_dict_when_none(self):
        mixin = DictOfTensorMixin(params_dict=None)
        assert len(mixin.params_dict) == 0

    def test_default_creates_empty_params_dict(self):
        mixin = DictOfTensorMixin()
        assert len(mixin.params_dict) == 0


@pytest.mark.unit
class TestDictOfTensorMixinDevice:
    def test_device_matches_parameter_device(self, dict_of_tensor_mixin_factory):
        mixin = dict_of_tensor_mixin_factory(keys=["weight"])
        assert mixin.device.type == "cpu"


@pytest.mark.unit
class TestDictOfTensorMixinLoadFromStateDict:
    def test_loads_flat_state_dict(self, rng):
        mixin = DictOfTensorMixin()
        values = torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        state_dict = {"params_dict.gamma": values}
        mixin._load_from_state_dict(
            state_dict=state_dict,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        assert "gamma" in mixin.params_dict
        assert torch.allclose(mixin.params_dict["gamma"], values)

    def test_loaded_params_have_gradients_disabled(self, rng):
        mixin = DictOfTensorMixin()
        values = torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        state_dict = {"params_dict.gamma": values}
        mixin._load_from_state_dict(
            state_dict=state_dict,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        assert not mixin.params_dict["gamma"].requires_grad

    def test_loads_nested_state_dict(self, rng):
        mixin = DictOfTensorMixin()
        values = torch.from_numpy(rng.standard_normal((3,)).astype(np.float32))
        state_dict = {"params_dict.outer.inner": values}
        mixin._load_from_state_dict(
            state_dict=state_dict,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        assert "outer" in mixin.params_dict
        assert "inner" in mixin.params_dict["outer"]
        assert torch.allclose(mixin.params_dict["outer"]["inner"], values)

    def test_loads_with_prefix(self, rng):
        mixin = DictOfTensorMixin()
        values = torch.from_numpy(rng.standard_normal((2,)).astype(np.float32))
        state_dict = {"model.params_dict.delta": values}
        mixin._load_from_state_dict(
            state_dict=state_dict,
            prefix="model.",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        assert "delta" in mixin.params_dict
        assert torch.allclose(mixin.params_dict["delta"], values)

    def test_loaded_values_are_cloned(self, rng):
        mixin = DictOfTensorMixin()
        values = torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        state_dict = {"params_dict.gamma": values}
        mixin._load_from_state_dict(
            state_dict=state_dict,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        # Mutate original -- loaded copy should be unaffected
        original_loaded = mixin.params_dict["gamma"].clone()
        values.fill_(999.0)
        assert torch.allclose(mixin.params_dict["gamma"], original_loaded)

    def test_loads_multiple_keys(self, rng):
        mixin = DictOfTensorMixin()
        alpha = torch.from_numpy(rng.standard_normal((3,)).astype(np.float32))
        beta = torch.from_numpy(rng.standard_normal((5,)).astype(np.float32))
        state_dict = {
            "params_dict.alpha": alpha,
            "params_dict.beta": beta,
        }
        mixin._load_from_state_dict(
            state_dict=state_dict,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        assert "alpha" in mixin.params_dict
        assert "beta" in mixin.params_dict
        assert torch.allclose(mixin.params_dict["alpha"], alpha)
        assert torch.allclose(mixin.params_dict["beta"], beta)

    def test_ignores_unrelated_state_dict_keys(self, rng):
        mixin = DictOfTensorMixin()
        values = torch.from_numpy(rng.standard_normal((4,)).astype(np.float32))
        state_dict = {
            "params_dict.gamma": values,
            "other_module.weight": torch.ones(10),
        }
        mixin._load_from_state_dict(
            state_dict=state_dict,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        assert "gamma" in mixin.params_dict
        assert len(mixin.params_dict) == 1
