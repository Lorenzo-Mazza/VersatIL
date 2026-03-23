"""Tests for versatil.post_training_compression.preparation.batchnorm module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from versatil.models.layers.normalization.frozen_batchnorm import FrozenBatchNorm2d
from versatil.post_training_compression.preparation.batchnorm import (
    _create_replacement_batchnorm,
    extract_activation,
    extract_batchnorm_parameters,
    has_batchnorm_buffers,
    is_frozen_batchnorm,
    prepare_batchnorms_for_quantization,
    replace_frozen_batchnorm,
)


@pytest.fixture
def bn_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for batchnorm-compatible input tensors."""

    def factory(
        batch_size: int = 2,
        num_features: int = 8,
        spatial_dims: tuple[int, ...] = (4, 4),
    ) -> torch.Tensor:
        shape = (batch_size, num_features, *spatial_dims)
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    return factory


@pytest.mark.unit
class TestHasBatchnormBuffers:
    @pytest.mark.parametrize(
        "module, expected",
        [
            (nn.BatchNorm1d(16), True),
            (nn.BatchNorm2d(16), True),
            (nn.BatchNorm3d(16), True),
            (nn.SyncBatchNorm(16), True),
            (FrozenBatchNorm2d(16), True),
            (nn.Linear(16, 32), False),
            (nn.Conv2d(3, 16, 3), False),
        ],
    )
    def test_detects_batchnorm_buffers_by_module_type(self, module, expected):
        assert has_batchnorm_buffers(module) is expected

    def test_custom_module_with_all_four_buffers(self):
        module = nn.Module()
        module.register_buffer("running_mean", torch.zeros(8))
        module.register_buffer("running_var", torch.ones(8))
        module.register_buffer("weight", torch.ones(8))
        module.register_buffer("bias", torch.zeros(8))
        assert has_batchnorm_buffers(module) is True

    def test_module_with_partial_buffers_returns_false(self):
        module = nn.Module()
        module.register_buffer("running_mean", torch.zeros(8))
        module.register_buffer("running_var", torch.ones(8))
        assert has_batchnorm_buffers(module) is False


@pytest.mark.unit
class TestIsFrozenBatchnorm:
    def test_versatil_frozen_batchnorm_is_frozen(self):
        assert is_frozen_batchnorm(FrozenBatchNorm2d(dimension=16)) is True

    @pytest.mark.parametrize(
        "training, track_running_stats",
        [
            (True, True),
            (False, True),
            (False, False),
        ],
    )
    def test_standard_batchnorm_is_not_frozen(self, training, track_running_stats):
        module = nn.BatchNorm2d(num_features=16)
        module.train(training)
        module.track_running_stats = track_running_stats
        assert is_frozen_batchnorm(module) is False

    def test_sync_batchnorm_is_not_frozen(self):
        module = nn.SyncBatchNorm(num_features=16)
        assert is_frozen_batchnorm(module) is False

    def test_linear_is_not_frozen(self):
        assert is_frozen_batchnorm(nn.Linear(16, 32)) is False

    def test_custom_module_with_bn_buffers_is_frozen(self):
        module = nn.Module()
        module.register_buffer("running_mean", torch.zeros(8))
        module.register_buffer("running_var", torch.ones(8))
        module.register_buffer("weight", torch.ones(8))
        module.register_buffer("bias", torch.zeros(8))
        assert is_frozen_batchnorm(module) is True


@pytest.mark.unit
class TestExtractBatchnormParameters:
    @pytest.mark.parametrize(
        "source_factory, num_features, eps",
        [
            (lambda n: nn.BatchNorm2d(num_features=n, eps=1e-3), 16, 1e-3),
            (lambda n: FrozenBatchNorm2d(dimension=n), 8, 1e-5),
        ],
        ids=["standard_bn2d", "frozen_bn"],
    )
    def test_extracts_all_parameters_with_correct_shapes(
        self,
        source_factory,
        num_features,
        eps,
    ):
        module = source_factory(num_features)
        result = extract_batchnorm_parameters(module)

        assert result is not None
        running_mean, running_var, weight, bias, extracted_eps = result
        assert running_mean.shape == (num_features,)
        assert running_var.shape == (num_features,)
        assert weight.shape == (num_features,)
        assert bias.shape == (num_features,)
        assert extracted_eps == eps

    def test_returns_none_for_module_without_buffers(self):
        assert extract_batchnorm_parameters(nn.Linear(16, 32)) is None

    def test_defaults_eps_when_attribute_missing(self):
        module = nn.Module()
        module.register_buffer("running_mean", torch.zeros(4))
        module.register_buffer("running_var", torch.ones(4))
        module.register_buffer("weight", torch.ones(4))
        module.register_buffer("bias", torch.zeros(4))

        _, _, _, _, eps = extract_batchnorm_parameters(module)

        assert eps == 1e-5


@pytest.mark.unit
class TestExtractActivation:
    @pytest.mark.parametrize("activation_class", [nn.ReLU, nn.GELU])
    def test_returns_activation_when_present(self, activation_class):
        module = nn.Module()
        module.act = activation_class()

        result = extract_activation(module)

        assert result is module.act

    def test_returns_none_when_no_act_attribute(self):
        assert extract_activation(nn.Module()) is None

    def test_returns_none_when_act_is_identity(self):
        module = nn.Module()
        module.act = nn.Identity()
        assert extract_activation(module) is None


@pytest.mark.unit
class TestCreateReplacementBatchnorm:
    @pytest.mark.parametrize(
        "source_class, input_shape",
        [
            (nn.BatchNorm1d, (2, 8)),
            (nn.BatchNorm2d, (2, 8, 4, 4)),
            (nn.BatchNorm3d, (2, 8, 4, 4, 4)),
        ],
    )
    def test_creates_correct_dimension_variant(self, rng, source_class, input_shape):
        num_features = 8
        source = source_class(num_features=num_features)
        source.eval()

        replacement = _create_replacement_batchnorm(
            batchnorm=source,
            num_features=num_features,
        )

        assert type(replacement) is source_class
        replacement.eval()
        input_data = torch.from_numpy(
            rng.standard_normal(input_shape).astype(np.float32)
        )
        assert replacement(input_data).shape == input_shape

    def test_frozen_bn_falls_back_to_batchnorm2d(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
    ):
        replacement = _create_replacement_batchnorm(
            batchnorm=frozen_batchnorm_factory(num_features=8),
            num_features=8,
        )
        assert type(replacement) is nn.BatchNorm2d

    def test_raises_for_module_without_bn_buffers(self):
        with pytest.raises(
            ValueError,
            match="Module Linear does not have the required BatchNorm buffers",
        ):
            _create_replacement_batchnorm(
                batchnorm=nn.Linear(16, 8),
                num_features=8,
            )

    def test_copies_parameters_from_source(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
    ):
        frozen = frozen_batchnorm_factory(num_features=8)
        original_mean = frozen.running_mean.clone()
        original_weight = frozen.weight.clone()

        replacement = _create_replacement_batchnorm(
            batchnorm=frozen,
            num_features=8,
        )

        assert torch.equal(replacement.running_mean, original_mean)
        assert torch.equal(replacement.weight, original_weight)

    def test_replacement_is_in_eval_mode_with_tracking_disabled(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
    ):
        replacement = _create_replacement_batchnorm(
            batchnorm=frozen_batchnorm_factory(num_features=8),
            num_features=8,
        )

        assert not replacement.training
        assert not replacement.track_running_stats


@pytest.mark.unit
class TestReplaceFrozenBatchnorm:
    def test_replaces_frozen_bn_and_returns_count(
        self,
        frozen_batchnorm_model_factory: Callable[..., nn.Module],
    ):
        model = frozen_batchnorm_model_factory(num_frozen_layers=3)
        assert replace_frozen_batchnorm(model) == 3

    def test_replacement_produces_equivalent_output(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
        bn_input_factory: Callable[..., torch.Tensor],
    ):
        num_features = 8
        frozen_bn = frozen_batchnorm_factory(num_features=num_features)
        input_data = bn_input_factory(num_features=num_features)

        frozen_bn.eval()
        expected_output = frozen_bn(input_data)

        model = nn.Module()
        model.add_module("bn", frozen_bn)
        replace_frozen_batchnorm(model)

        model.bn.eval()
        actual_output = model.bn(input_data)
        assert torch.allclose(actual_output, expected_output, atol=1e-6)

    def test_returns_zero_when_no_frozen_bn(self):
        model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())
        assert replace_frozen_batchnorm(model) == 0

    def test_handles_nested_models(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
    ):
        inner = nn.Module()
        inner.add_module("bn", frozen_batchnorm_factory(num_features=8))
        outer = nn.Module()
        outer.add_module("inner", inner)
        outer.add_module("bn", frozen_batchnorm_factory(num_features=8))

        assert replace_frozen_batchnorm(outer) == 2
        assert isinstance(outer.bn, nn.BatchNorm2d)
        assert isinstance(outer.inner.bn, nn.BatchNorm2d)

    def test_preserves_activation_from_fused_bn(self):
        frozen = FrozenBatchNorm2d(dimension=8)
        frozen.act = nn.ReLU()
        model = nn.Module()
        model.add_module("bn", frozen)

        replace_frozen_batchnorm(model)

        assert isinstance(model.bn, nn.Sequential)
        assert isinstance(model.bn[0], nn.BatchNorm2d)
        assert isinstance(model.bn[1], nn.ReLU)


@pytest.mark.unit
class TestPrepareBatchnormsForQuantization:
    def test_sets_all_standard_bn_to_eval(self):
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
        )
        model.train()

        prepare_batchnorms_for_quantization(model)

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert not module.training
                assert not module.track_running_stats

    def test_replaces_frozen_bn_and_sets_to_eval(
        self,
        frozen_batchnorm_model_factory: Callable[..., nn.Module],
    ):
        model = frozen_batchnorm_model_factory(num_frozen_layers=2)

        count = prepare_batchnorms_for_quantization(model)

        assert count == 2
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert not module.training
                assert not module.track_running_stats

    def test_mixed_model_produces_equivalent_output(
        self,
        rng: np.random.Generator,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
    ):
        num_features = 16
        model = nn.Sequential(
            nn.Conv2d(3, num_features, 3, padding=1),
            frozen_batchnorm_factory(num_features=num_features),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
        )
        model.eval()

        input_data = torch.from_numpy(
            rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
        )
        with torch.no_grad():
            output_before = model(input_data)

        prepare_batchnorms_for_quantization(model)

        with torch.no_grad():
            output_after = model(input_data)
        assert torch.allclose(output_after, output_before, atol=1e-6)

    def test_returns_zero_for_model_without_frozen_bn(self):
        model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.BatchNorm2d(16))
        assert prepare_batchnorms_for_quantization(model) == 0
