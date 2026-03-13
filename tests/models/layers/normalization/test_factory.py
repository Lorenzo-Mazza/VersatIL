"""Tests for versatil.models.layers.normalization.factory module."""
import re
from contextlib import nullcontext as does_not_raise

import pytest
from torch import nn

from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.normalization.frozen_batchnorm import FrozenBatchNorm2d
from versatil.models.layers.normalization.rms_norm import RMSNorm


class TestCreateNormalizationLayer:

    @pytest.mark.parametrize("normalization_type, expected_type", [
        (NormalizationType.LAYER_NORM.value, nn.LayerNorm),
        (NormalizationType.RMS_NORM.value, RMSNorm),
        (NormalizationType.FROZEN_BATCHNORM2D.value, FrozenBatchNorm2d),
        (NormalizationType.ADALN.value, AdaNorm),
        (NormalizationType.ADARMS.value, AdaNorm),
    ])
    def test_creates_correct_layer_type(
        self,
        normalization_type: str,
        expected_type: type,
    ):
        kwargs = {"normalization_type": normalization_type, "dimension": 64}
        if normalization_type in (
            NormalizationType.ADALN.value,
            NormalizationType.ADARMS.value,
        ):
            kwargs["condition_dim"] = 32
        layer = create_normalization_layer(**kwargs)
        assert isinstance(layer, expected_type)

    @pytest.mark.parametrize("normalization_type", [
        NormalizationType.ADALN.value,
        NormalizationType.ADARMS.value,
    ])
    def test_adaptive_norm_without_condition_dim_raises(
        self,
        normalization_type: str,
    ):
        with pytest.raises(
            ValueError,
            match=re.escape("condition_dim is required for ada_ln / ada_rms"),
        ):
            create_normalization_layer(
                normalization_type=normalization_type,
                dimension=64,
                condition_dim=None,
            )

    def test_invalid_type_raises_value_error(self):
        invalid_type = "nonexistent_norm"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unsupported normalization type: {invalid_type}. "
                f"Must be one of {[e.value for e in NormalizationType]}."
            ),
        ):
            create_normalization_layer(
                normalization_type=invalid_type,
                dimension=64,
            )

    @pytest.mark.parametrize("normalization_type, condition_dim, expectation", [
        (NormalizationType.LAYER_NORM.value, None, does_not_raise()),
        (NormalizationType.RMS_NORM.value, None, does_not_raise()),
        (NormalizationType.FROZEN_BATCHNORM2D.value, None, does_not_raise()),
        (NormalizationType.ADALN.value, 32, does_not_raise()),
        (NormalizationType.ADARMS.value, 32, does_not_raise()),
        (
            NormalizationType.ADALN.value,
            None,
            pytest.raises(
                ValueError,
                match=re.escape("condition_dim is required for ada_ln / ada_rms"),
            ),
        ),
    ])
    def test_valid_types_do_not_raise(
        self,
        normalization_type: str,
        condition_dim: int | None,
        expectation,
    ):
        with expectation:
            create_normalization_layer(
                normalization_type=normalization_type,
                dimension=64,
                condition_dim=condition_dim,
            )

    def test_adaln_wraps_layer_norm_base(self):
        layer = create_normalization_layer(
            normalization_type=NormalizationType.ADALN.value,
            dimension=64,
            condition_dim=32,
        )
        assert isinstance(layer.norm, nn.LayerNorm)

    def test_adarms_wraps_rms_norm_base(self):
        layer = create_normalization_layer(
            normalization_type=NormalizationType.ADARMS.value,
            dimension=64,
            condition_dim=32,
        )
        assert isinstance(layer.norm, RMSNorm)

    @pytest.mark.parametrize("epsilon", [1e-5, 1e-8])
    def test_epsilon_is_forwarded(self, epsilon: float):
        layer = create_normalization_layer(
            normalization_type=NormalizationType.LAYER_NORM.value,
            dimension=64,
            epsilon=epsilon,
        )
        assert layer.eps == epsilon
