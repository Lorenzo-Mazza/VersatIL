"""Tests for versatil.models.decoding.latent.vq.residual_vq module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.models.decoding.latent.vq.residual_vq import ResidualVQ


@pytest.fixture
def mock_vq_layer_output_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

    def factory(
        batch_size: int = 4,
        input_dimension: int = 8,
        code_dim: int | None = None,
        num_codes: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if code_dim is None:
            code_dim = input_dimension
        z_q = torch.from_numpy(
            rng.standard_normal((batch_size, input_dimension)).astype(np.float32)
        )
        indices = torch.from_numpy(
            rng.integers(0, num_codes, size=(batch_size,)).astype(np.int64)
        )
        z_e_projected = torch.from_numpy(
            rng.standard_normal((batch_size, code_dim)).astype(np.float32)
        )
        z_q_code = torch.from_numpy(
            rng.standard_normal((batch_size, code_dim)).astype(np.float32)
        )
        return z_q, indices, z_e_projected, z_q_code

    return factory


class TestResidualVQInit:
    @pytest.mark.unit
    @pytest.mark.parametrize("num_layers", [1, 2, 4])
    def test_creates_correct_number_of_layers(self, num_layers: int) -> None:
        rvq = ResidualVQ(
            input_dimension=8, code_dim=8, num_codes=4, num_layers=num_layers
        )
        assert len(rvq.layers) == num_layers

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "input_dimension, code_dim, num_codes",
        [(8, 8, 4), (16, 4, 32)],
    )
    def test_stores_configuration(
        self, input_dimension: int, code_dim: int, num_codes: int
    ) -> None:
        rvq = ResidualVQ(
            input_dimension=input_dimension, code_dim=code_dim, num_codes=num_codes
        )
        assert rvq.input_dimension == input_dimension
        assert rvq.code_dim == code_dim
        assert rvq.num_codes == num_codes

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "input_dimension, code_dim, num_codes, num_layers, expected_message",
        [
            (0, 8, 4, 1, "input_dimension must be positive, got 0."),
            (8, 0, 4, 1, "code_dim must be positive, got 0."),
            (8, 8, 0, 1, "num_codes must be positive, got 0."),
            (8, 8, 4, 0, "num_layers must be positive, got 0."),
        ],
    )
    def test_rejects_invalid_configuration(
        self,
        input_dimension: int,
        code_dim: int,
        num_codes: int,
        num_layers: int,
        expected_message: str,
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ResidualVQ(
                input_dimension=input_dimension,
                code_dim=code_dim,
                num_codes=num_codes,
                num_layers=num_layers,
            )


class TestResidualVQForward:
    @pytest.mark.unit
    def test_first_layer_receives_original_input(
        self,
        z_e_factory: Callable[..., torch.Tensor],
        mock_vq_layer_output_factory: Callable,
    ) -> None:
        rvq = ResidualVQ(input_dimension=8, code_dim=8, num_codes=4, num_layers=2)
        z_e = z_e_factory(batch_size=4, dim=8)

        for layer in rvq.layers:
            layer.forward = MagicMock(
                return_value=mock_vq_layer_output_factory(
                    batch_size=4, input_dimension=8
                )
            )

        rvq(z_e)

        first_call_input = rvq.layers[0].forward.call_args[0][0]
        assert torch.allclose(first_call_input, z_e)

    @pytest.mark.unit
    def test_second_layer_receives_detached_residual(
        self,
        z_e_factory: Callable[..., torch.Tensor],
        mock_vq_layer_output_factory: Callable,
    ) -> None:
        rvq = ResidualVQ(input_dimension=8, code_dim=8, num_codes=4, num_layers=2)
        z_e = z_e_factory(batch_size=4, dim=8)

        output_0 = mock_vq_layer_output_factory(batch_size=4, input_dimension=8)
        output_1 = mock_vq_layer_output_factory(batch_size=4, input_dimension=8)
        rvq.layers[0].forward = MagicMock(return_value=output_0)
        rvq.layers[1].forward = MagicMock(return_value=output_1)

        rvq(z_e)

        second_call_input = rvq.layers[1].forward.call_args[0][0]
        expected_residual = z_e - output_0[0].detach()
        assert torch.allclose(second_call_input, expected_residual)

    @pytest.mark.unit
    def test_output_is_sum_of_layer_outputs(
        self,
        z_e_factory: Callable[..., torch.Tensor],
        mock_vq_layer_output_factory: Callable,
    ) -> None:
        rvq = ResidualVQ(input_dimension=8, code_dim=8, num_codes=4, num_layers=2)
        z_e = z_e_factory(batch_size=4, dim=8)

        output_0 = mock_vq_layer_output_factory(batch_size=4, input_dimension=8)
        output_1 = mock_vq_layer_output_factory(batch_size=4, input_dimension=8)
        rvq.layers[0].forward = MagicMock(return_value=output_0)
        rvq.layers[1].forward = MagicMock(return_value=output_1)

        z_q_total, _, _, _ = rvq(z_e)
        assert torch.allclose(z_q_total, output_0[0] + output_1[0])

    @pytest.mark.unit
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_returns_per_layer_indices_and_stacked_projections(
        self,
        z_e_factory: Callable[..., torch.Tensor],
        mock_vq_layer_output_factory: Callable,
        num_layers: int,
    ) -> None:
        rvq = ResidualVQ(
            input_dimension=8, code_dim=8, num_codes=4, num_layers=num_layers
        )
        z_e = z_e_factory(batch_size=4, dim=8)

        for layer in rvq.layers:
            layer.forward = MagicMock(
                return_value=mock_vq_layer_output_factory(
                    batch_size=4, input_dimension=8
                )
            )

        _, all_indices, z_e_per_layer, z_q_per_layer = rvq(z_e)
        assert len(all_indices) == num_layers
        # Per-layer tensors stacked along dim 0: (L, B, code_dim).
        assert z_e_per_layer.shape == (num_layers, 4, 8)
        assert z_q_per_layer.shape == (num_layers, 4, 8)

    @pytest.mark.unit
    def test_stacks_preserve_per_layer_tensors(
        self,
        z_e_factory: Callable[..., torch.Tensor],
        mock_vq_layer_output_factory: Callable,
    ) -> None:
        rvq = ResidualVQ(input_dimension=8, code_dim=8, num_codes=4, num_layers=2)
        z_e = z_e_factory(batch_size=4, dim=8)
        output_0 = mock_vq_layer_output_factory(batch_size=4, input_dimension=8)
        output_1 = mock_vq_layer_output_factory(batch_size=4, input_dimension=8)
        rvq.layers[0].forward = MagicMock(return_value=output_0)
        rvq.layers[1].forward = MagicMock(return_value=output_1)

        _, _, z_e_per_layer, z_q_per_layer = rvq(z_e)
        # output indices 2 and 3 are z_e_projected and z_q_code
        assert torch.equal(z_e_per_layer[0], output_0[2])
        assert torch.equal(z_e_per_layer[1], output_1[2])
        assert torch.equal(z_q_per_layer[0], output_0[3])
        assert torch.equal(z_q_per_layer[1], output_1[3])


class TestResidualVQForwardIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    @pytest.mark.parametrize("input_dimension, code_dim", [(8, 8), (16, 4)])
    @pytest.mark.parametrize("batch_size", [1, 10])
    def test_output_shapes(
        self,
        z_e_factory: Callable[..., torch.Tensor],
        num_layers: int,
        input_dimension: int,
        code_dim: int,
        batch_size: int,
    ) -> None:
        rvq = ResidualVQ(
            input_dimension=input_dimension,
            code_dim=code_dim,
            num_codes=4,
            num_layers=num_layers,
            kmeans_init=False,
        )
        rvq.eval()
        z_e = z_e_factory(batch_size=batch_size, dim=input_dimension)
        z_q, all_indices, z_e_per_layer, z_q_per_layer = rvq(z_e)
        assert z_q.shape == (batch_size, input_dimension)
        assert len(all_indices) == num_layers
        for indices in all_indices:
            assert indices.shape == (batch_size,)
        assert z_e_per_layer.shape == (num_layers, batch_size, code_dim)
        assert z_q_per_layer.shape == (num_layers, batch_size, code_dim)

    @pytest.mark.integration
    def test_single_layer_matches_vector_quantize(
        self, z_e_factory: Callable[..., torch.Tensor]
    ) -> None:
        rvq = ResidualVQ(
            input_dimension=8,
            code_dim=8,
            num_codes=4,
            num_layers=1,
            kmeans_init=False,
        )
        rvq.eval()
        z_e = z_e_factory(batch_size=8, dim=8)
        z_q_rvq, indices_rvq, _, _ = rvq(z_e)
        z_q_vq, indices_vq, _, _ = rvq.layers[0](z_e)
        assert torch.allclose(z_q_rvq, z_q_vq)
        assert torch.equal(indices_rvq[0], indices_vq)

    @pytest.mark.integration
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_straight_through_gradient_has_unit_scale(
        self, z_e_factory: Callable[..., torch.Tensor], num_layers: int
    ) -> None:
        rvq = ResidualVQ(
            input_dimension=8,
            code_dim=8,
            num_codes=4,
            num_layers=num_layers,
            kmeans_init=False,
        )
        rvq.eval()
        z_e = z_e_factory(batch_size=8, dim=8)
        z_e.requires_grad_(True)
        z_q, _, _, _ = rvq(z_e)
        z_q.sum().backward()
        assert torch.allclose(z_e.grad, torch.ones_like(z_e), atol=1e-6)

    @pytest.mark.integration
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_z_q_per_layer_is_detached(
        self,
        z_e_factory: Callable[..., torch.Tensor],
        num_layers: int,
    ) -> None:
        rvq = ResidualVQ(
            input_dimension=8,
            code_dim=8,
            num_codes=4,
            num_layers=num_layers,
            kmeans_init=False,
        )
        rvq.eval()
        z_e = z_e_factory(batch_size=6, dim=8)
        z_e.requires_grad_(True)
        _, _, z_e_per_layer, z_q_per_layer = rvq(z_e)
        # z_e_per_layer carries gradient back to the encoder input;
        # z_q_per_layer must not (detached codebook lookup).
        assert z_e_per_layer.requires_grad is True
        assert z_q_per_layer.requires_grad is False


class TestResidualVQDecodeFromIndices:
    @pytest.mark.integration
    @pytest.mark.parametrize("num_layers", [1, 2, 3])
    def test_decode_matches_forward(
        self,
        z_e_factory: Callable[..., torch.Tensor],
        num_layers: int,
    ) -> None:
        rvq = ResidualVQ(
            input_dimension=8,
            code_dim=8,
            num_codes=4,
            num_layers=num_layers,
            kmeans_init=False,
        )
        rvq.eval()
        z_e = z_e_factory(batch_size=8, dim=8)
        z_q_forward, all_indices, _, _ = rvq(z_e)
        z_q_decoded = rvq.decode_from_indices(all_indices)
        assert torch.allclose(z_q_forward, z_q_decoded, atol=1e-6)

    @pytest.mark.integration
    @pytest.mark.parametrize("input_dimension, code_dim", [(8, 8), (16, 4)])
    def test_output_shape(
        self,
        z_e_factory: Callable[..., torch.Tensor],
        input_dimension: int,
        code_dim: int,
    ) -> None:
        rvq = ResidualVQ(
            input_dimension=input_dimension,
            code_dim=code_dim,
            num_codes=4,
            num_layers=2,
            kmeans_init=False,
        )
        rvq.eval()
        z_e = z_e_factory(batch_size=6, dim=input_dimension)
        _, all_indices, _, _ = rvq(z_e)
        z_q_decoded = rvq.decode_from_indices(all_indices)
        assert z_q_decoded.shape == (6, input_dimension)
