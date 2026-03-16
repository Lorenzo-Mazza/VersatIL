"""Tests for versatil.models.layers.normalization.factory module."""
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch

from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer


@pytest.fixture
def norm_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for normalization input tensors."""

    def factory(
        batch_size: int = 2,
        channels: int = 64,
        spatial: bool = False,
        height: int = 4,
        width: int = 4,
        sequence_length: int = 8,
    ) -> torch.Tensor:
        if spatial:
            shape = (batch_size, channels, height, width)
        else:
            shape = (batch_size, sequence_length, channels)
        return torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32)
        )

    return factory


class TestCreateNormalizationLayer:

    @pytest.mark.parametrize(
        "normalization_type, spatial",
        [
            (NormalizationType.LAYER_NORM.value, False),
            (NormalizationType.RMS_NORM.value, False),
            (NormalizationType.FROZEN_BATCHNORM2D.value, True),
            (NormalizationType.ADALN.value, False),
            (NormalizationType.ADARMS.value, False),
        ],
    )
    def test_created_layer_produces_valid_output(
        self,
        norm_input_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        normalization_type: str,
        spatial: bool,
    ):
        dimension = 64
        condition_dim = 32
        kwargs = {
            "normalization_type": normalization_type,
            "dimension": dimension,
        }
        if normalization_type in (
            NormalizationType.ADALN.value,
            NormalizationType.ADARMS.value,
        ):
            kwargs["condition_dim"] = condition_dim
        layer = create_normalization_layer(**kwargs)
        tensor = norm_input_factory(batch_size=2, channels=dimension, spatial=spatial)
        if normalization_type in (
            NormalizationType.ADALN.value,
            NormalizationType.ADARMS.value,
        ):
            condition = condition_factory(batch_size=2, condition_dim=condition_dim)
            output = layer(tensor, condition)
        else:
            output = layer(tensor)
        assert output.shape == tensor.shape
        assert torch.all(torch.isfinite(output))

    @pytest.mark.parametrize(
        "normalization_type, condition_dim, expectation",
        [
            (
                NormalizationType.ADALN.value,
                None,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "condition_dim is required for ada_ln / ada_rms"
                    ),
                ),
            ),
            (
                NormalizationType.ADARMS.value,
                None,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "condition_dim is required for ada_ln / ada_rms"
                    ),
                ),
            ),
            (NormalizationType.ADALN.value, 32, does_not_raise()),
            (NormalizationType.LAYER_NORM.value, None, does_not_raise()),
        ],
    )
    def test_adaptive_norm_requires_condition_dim(
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

    def test_invalid_type_raises_with_supported_types_message(self):
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

    def test_adaln_base_norm_has_no_learnable_affine_parameters(self):
        layer = create_normalization_layer(
            normalization_type=NormalizationType.ADALN.value,
            dimension=64,
            condition_dim=32,
        )
        assert len(list(layer.norm.parameters())) == 0

    def test_adarms_base_norm_has_no_learnable_affine_parameters(self):
        layer = create_normalization_layer(
            normalization_type=NormalizationType.ADARMS.value,
            dimension=64,
            condition_dim=32,
        )
        assert len(list(layer.norm.parameters())) == 0

    @pytest.mark.parametrize("epsilon", [1e-5, 1e-8])
    def test_epsilon_is_forwarded_to_layer(self, epsilon: float):
        layer = create_normalization_layer(
            normalization_type=NormalizationType.LAYER_NORM.value,
            dimension=64,
            epsilon=epsilon,
        )
        assert layer.eps == epsilon

    def test_layer_norm_output_is_normalized(
        self,
        norm_input_factory: Callable[..., torch.Tensor],
    ):
        dimension = 64
        layer = create_normalization_layer(
            normalization_type=NormalizationType.LAYER_NORM.value,
            dimension=dimension,
        )
        tensor = norm_input_factory(batch_size=4, channels=dimension)
        output = layer(tensor)
        # LayerNorm produces mean ≈ 0 along last dimension
        assert torch.allclose(
            output.mean(dim=-1),
            torch.zeros_like(output.mean(dim=-1)),
            atol=1e-5,
        )

    def test_frozen_batchnorm_default_acts_like_identity(
        self,
        norm_input_factory: Callable[..., torch.Tensor],
    ):
        dimension = 64
        layer = create_normalization_layer(
            normalization_type=NormalizationType.FROZEN_BATCHNORM2D.value,
            dimension=dimension,
        )
        tensor = norm_input_factory(
            batch_size=2, channels=dimension, spatial=True,
        )
        output = layer(tensor)
        assert torch.allclose(output, tensor, atol=1e-4)