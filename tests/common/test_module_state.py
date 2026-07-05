"""Tests for versatil.common.module_state module."""

import pytest
import torch
import torch.nn as nn

from versatil.common.module_state import module_side_effects_guard


@pytest.mark.unit
class TestModuleSideEffectsGuard:
    def test_batchnorm_statistics_are_restored(self) -> None:
        module = nn.BatchNorm1d(4)
        module.train()
        running_mean_before = module.running_mean.clone()

        with module_side_effects_guard(module=module):
            module(torch.randn(8, 4) + 3.0)
            assert not torch.equal(module.running_mean, running_mean_before)

        torch.testing.assert_close(module.running_mean, running_mean_before)

    def test_rng_state_is_restored(self) -> None:
        module = nn.Linear(2, 2)
        torch.manual_seed(7)
        expected = torch.randn(3)

        torch.manual_seed(7)
        with module_side_effects_guard(module=module):
            torch.randn(100)

        torch.testing.assert_close(torch.randn(3), expected)

    def test_restores_even_when_forward_raises(self) -> None:
        module = nn.BatchNorm1d(4)
        module.train()
        running_mean_before = module.running_mean.clone()

        with (
            pytest.raises(RuntimeError, match="boom"),
            module_side_effects_guard(module=module),
        ):
            module(torch.randn(8, 4) + 3.0)
            raise RuntimeError("boom")

        torch.testing.assert_close(module.running_mean, running_mean_before)

    def test_buffers_created_inside_guard_are_kept(self) -> None:
        module = nn.Linear(2, 2)

        with module_side_effects_guard(module=module):
            module.register_buffer("materialized", torch.ones(2))

        torch.testing.assert_close(module.materialized, torch.ones(2))
