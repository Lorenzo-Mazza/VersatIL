"""Tests for versatil.models.decoding.latent.vq.vector_quantize module."""

import re
from collections.abc import Callable

import pytest
import torch

from versatil.models.decoding.latent.vq.vector_quantize import VectorQuantize


@pytest.fixture
def vq_factory() -> Callable[..., VectorQuantize]:

    def factory(
        input_dimension: int = 8,
        code_dim: int = 8,
        num_codes: int = 4,
        ema_decay: float = 0.99,
        kmeans_init: bool = False,
    ) -> VectorQuantize:
        return VectorQuantize(
            input_dimension=input_dimension,
            code_dim=code_dim,
            num_codes=num_codes,
            ema_decay=ema_decay,
            kmeans_init=kmeans_init,
        )

    return factory


class TestVectorQuantizeInit:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "input_dimension, code_dim, expect_projection",
        [
            (16, 16, False),
            (32, 8, True),
            (4, 16, True),
        ],
        ids=["same_dim_no_projection", "input_larger", "input_smaller"],
    )
    def test_projection_creation(
        self,
        vq_factory: Callable[..., VectorQuantize],
        input_dimension: int,
        code_dim: int,
        expect_projection: bool,
    ) -> None:
        vq = vq_factory(input_dimension=input_dimension, code_dim=code_dim)
        has_projection = not isinstance(vq.project_in, torch.nn.Identity)
        assert has_projection == expect_projection

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "input_dimension, code_dim, num_codes, expected_message",
        [
            (0, 8, 4, "input_dimension must be positive, got 0."),
            (8, 0, 4, "code_dim must be positive, got 0."),
            (8, 8, 0, "num_codes must be positive, got 0."),
        ],
    )
    def test_rejects_invalid_configuration(
        self,
        input_dimension: int,
        code_dim: int,
        num_codes: int,
        expected_message: str,
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            VectorQuantize(
                input_dimension=input_dimension,
                code_dim=code_dim,
                num_codes=num_codes,
            )


class TestVectorQuantizeForward:
    @pytest.mark.unit
    @pytest.mark.parametrize("input_dimension, code_dim", [(8, 8), (16, 4), (4, 16)])
    @pytest.mark.parametrize("num_codes", [2, 8])
    @pytest.mark.parametrize("batch_size", [1, 12])
    def test_output_shapes(
        self,
        vq_factory: Callable[..., VectorQuantize],
        z_e_factory: Callable[..., torch.Tensor],
        input_dimension: int,
        code_dim: int,
        num_codes: int,
        batch_size: int,
    ) -> None:
        vq = vq_factory(
            input_dimension=input_dimension, code_dim=code_dim, num_codes=num_codes
        )
        vq.eval()
        z_e = z_e_factory(batch_size=batch_size, dim=input_dimension)
        z_q, indices, z_e_projected, z_q_code = vq(z_e)
        assert z_q.shape == (batch_size, input_dimension)
        assert indices.shape == (batch_size,)
        assert z_e_projected.shape == (batch_size, code_dim)
        assert z_q_code.shape == (batch_size, code_dim)

    @pytest.mark.unit
    @pytest.mark.parametrize("num_codes", [2, 4, 16])
    def test_straight_through_gradient(
        self,
        vq_factory: Callable[..., VectorQuantize],
        z_e_factory: Callable[..., torch.Tensor],
        num_codes: int,
    ) -> None:
        vq = vq_factory(input_dimension=8, code_dim=8, num_codes=num_codes)
        vq.eval()
        z_e = z_e_factory(batch_size=8, dim=8)
        z_e.requires_grad_(True)
        z_q, _, _, _ = vq(z_e)
        z_q.sum().backward()
        assert z_e.grad is not None
        assert not torch.all(z_e.grad == 0.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("num_codes", [2, 4, 16])
    def test_quantized_equal_codebook_entries_in_eval_without_projection(
        self,
        vq_factory: Callable[..., VectorQuantize],
        z_e_factory: Callable[..., torch.Tensor],
        num_codes: int,
    ) -> None:
        vq = vq_factory(input_dimension=8, code_dim=8, num_codes=num_codes)
        vq.eval()
        z_e = z_e_factory(batch_size=8, dim=8)
        z_q, indices, _, _ = vq(z_e)
        for i in range(8):
            assert torch.allclose(z_q[i], vq.codebook.embed[indices[i]], atol=1e-6)

    @pytest.mark.unit
    def test_projection_transforms_values(
        self,
        vq_factory: Callable[..., VectorQuantize],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        vq = vq_factory(input_dimension=16, code_dim=4, num_codes=4)
        vq.eval()
        z_e = z_e_factory(batch_size=4, dim=16)
        _, _, z_e_projected, _ = vq(z_e)
        # z_e_projected lives in code_dim space, not input_dimension space
        assert z_e_projected.shape == (4, 4)
        # z_q is back in input_dimension space via project_out
        z_q, _, _, _ = vq(z_e)
        assert z_q.shape == (4, 16)
        # projected values differ from raw input (projection is not identity)
        assert not torch.allclose(z_e[:, :4], z_e_projected)

    @pytest.mark.unit
    @pytest.mark.parametrize("num_codes", [2, 4, 16])
    def test_z_q_code_is_detached_codebook_entry(
        self,
        vq_factory: Callable[..., VectorQuantize],
        z_e_factory: Callable[..., torch.Tensor],
        num_codes: int,
    ) -> None:
        vq = vq_factory(input_dimension=8, code_dim=8, num_codes=num_codes)
        vq.eval()
        z_e = z_e_factory(batch_size=8, dim=8)
        z_e.requires_grad_(True)
        _, indices, _, z_q_code = vq(z_e)
        # z_q_code must be detached: no grad propagation back to z_e
        assert z_q_code.requires_grad is False
        # And must equal the raw codebook lookup for each chosen index
        for sample_index in range(8):
            assert torch.allclose(
                z_q_code[sample_index],
                vq.codebook.embed[indices[sample_index]],
                atol=1e-6,
            )

    @pytest.mark.unit
    def test_z_q_code_differs_from_z_e_projected(
        self,
        vq_factory: Callable[..., VectorQuantize],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        # With a fresh codebook (kmeans_init=False, default random init), the
        # hard-quantized codebook vector should almost surely differ from the
        # continuous pre-quantization embedding.
        vq = vq_factory(input_dimension=8, code_dim=8, num_codes=4, kmeans_init=False)
        vq.eval()
        z_e = z_e_factory(batch_size=16, dim=8)
        _, _, z_e_projected, z_q_code = vq(z_e)
        assert not torch.allclose(z_e_projected, z_q_code)
