"""Tests for versatil.models.decoding.latent.vq.euclidean_codebook module."""

import re
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch

from versatil.models.decoding.latent.vq.euclidean_codebook import EuclideanCodebook


@pytest.fixture
def codebook_factory() -> Callable[..., EuclideanCodebook]:

    def factory(
        num_codes: int = 4,
        code_dim: int = 8,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
        kmeans_init: bool = False,
    ) -> EuclideanCodebook:
        return EuclideanCodebook(
            num_codes=num_codes,
            code_dim=code_dim,
            ema_decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
            kmeans_init=kmeans_init,
        )

    return factory


class TestEuclideanCodebookInit:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_codes, code_dim",
        [(2, 4), (16, 64), (256, 32)],
    )
    def test_stores_configuration(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        num_codes: int,
        code_dim: int,
    ) -> None:
        codebook = codebook_factory(num_codes=num_codes, code_dim=code_dim)
        assert codebook.num_codes == num_codes
        assert codebook.code_dim == code_dim
        assert codebook.embed.shape == (num_codes, code_dim)
        assert codebook.cluster_size.shape == (num_codes,)
        assert codebook.embed_avg.shape == (num_codes, code_dim)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_codes, code_dim, ema_decay, dead_code_threshold, expected_message",
        [
            (0, 8, 0.99, 1.0, "num_codes must be positive, got 0."),
            (4, 0, 0.99, 1.0, "code_dim must be positive, got 0."),
            (
                4,
                8,
                1.0,
                1.0,
                "ema_decay must be in the interval [0.0, 1.0), got 1.0.",
            ),
            (
                4,
                8,
                -0.1,
                1.0,
                "ema_decay must be in the interval [0.0, 1.0), got -0.1.",
            ),
            (
                4,
                8,
                0.99,
                -1.0,
                "dead_code_threshold must be non-negative, got -1.0.",
            ),
        ],
    )
    def test_rejects_invalid_configuration(
        self,
        num_codes: int,
        code_dim: int,
        ema_decay: float,
        dead_code_threshold: float,
        expected_message: str,
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            EuclideanCodebook(
                num_codes=num_codes,
                code_dim=code_dim,
                ema_decay=ema_decay,
                dead_code_threshold=dead_code_threshold,
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "kmeans_init, expect_zeros, expect_initialized",
        [
            (True, True, False),
            (False, False, True),
        ],
        ids=["kmeans_deferred", "random_immediate"],
    )
    def test_initialization_mode(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        kmeans_init: bool,
        expect_zeros: bool,
        expect_initialized: bool,
    ) -> None:
        codebook = codebook_factory(num_codes=4, code_dim=8, kmeans_init=kmeans_init)
        assert (torch.all(codebook.embed == 0.0)) == expect_zeros
        assert codebook.initialized.item() == expect_initialized


class TestEuclideanCodebookInitializeFromData:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, num_codes",
        [(16, 4), (3, 8), (1, 2)],
        ids=["more_samples_than_codes", "fewer_samples_than_codes", "single_sample"],
    )
    def test_sets_flag_and_shape(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
        batch_size: int,
        num_codes: int,
    ) -> None:
        codebook = codebook_factory(num_codes=num_codes, code_dim=8, kmeans_init=True)
        data = z_e_factory(batch_size=batch_size, dim=8)
        codebook._initialize_from_data(data)
        assert codebook.initialized.item()
        assert codebook.embed.shape == (num_codes, 8)

    @pytest.mark.unit
    def test_embed_values_come_from_data(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        codebook = codebook_factory(num_codes=4, code_dim=8, kmeans_init=True)
        data = z_e_factory(batch_size=16, dim=8)
        codebook._initialize_from_data(data)
        for i in range(4):
            distances = torch.cdist(
                codebook.embed[i].unsqueeze(0),
                data,  # (1, D) vs (N, D) -> (1, N)
            )
            assert distances.min().item() < 1e-6


class TestEuclideanCodebookForward:
    @pytest.mark.unit
    @pytest.mark.parametrize("num_codes", [2, 4, 16])
    @pytest.mark.parametrize("code_dim", [1, 4, 32])
    @pytest.mark.parametrize("batch_size", [1, 16])
    def test_output_shapes(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
        num_codes: int,
        code_dim: int,
        batch_size: int,
    ) -> None:
        codebook = codebook_factory(num_codes=num_codes, code_dim=code_dim)
        codebook.eval()
        z_e = z_e_factory(batch_size=batch_size, dim=code_dim)
        quantized, indices = codebook(z_e)
        assert quantized.shape == (batch_size, code_dim)
        assert indices.shape == (batch_size,)
        assert indices.max().item() < num_codes

    @pytest.mark.unit
    @pytest.mark.parametrize("num_codes", [2, 4, 16])
    @pytest.mark.parametrize("code_dim", [4, 32])
    def test_quantized_are_codebook_entries(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
        num_codes: int,
        code_dim: int,
    ) -> None:
        codebook = codebook_factory(num_codes=num_codes, code_dim=code_dim)
        codebook.eval()
        z_e = z_e_factory(batch_size=8, dim=code_dim)
        quantized, indices = codebook(z_e)
        for i in range(8):
            assert torch.allclose(quantized[i], codebook.embed[indices[i]])

    @pytest.mark.unit
    @pytest.mark.parametrize("num_codes", [2, 4, 16])
    @pytest.mark.parametrize("code_dim", [4, 32])
    def test_indices_are_nearest_neighbors(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
        num_codes: int,
        code_dim: int,
    ) -> None:
        codebook = codebook_factory(num_codes=num_codes, code_dim=code_dim)
        codebook.eval()
        z_e = z_e_factory(batch_size=8, dim=code_dim)
        _, indices = codebook(z_e)
        dist = torch.cdist(z_e, codebook.embed)  # (B, K)
        expected_indices = dist.argmin(dim=-1)  # (B,)
        assert torch.equal(indices, expected_indices)

    @pytest.mark.unit
    def test_kmeans_init_triggers_on_first_forward(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        codebook = codebook_factory(num_codes=4, code_dim=8, kmeans_init=True)
        assert not codebook.initialized.item()
        codebook.train()
        z_e = z_e_factory(batch_size=16, dim=8)
        codebook(z_e)
        assert codebook.initialized.item()


class TestEuclideanCodebookEMAUpdate:
    @pytest.mark.unit
    def test_embed_changes_during_training(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        codebook = codebook_factory(num_codes=4, code_dim=8)
        codebook.train()
        embed_before = codebook.embed.clone()
        z_e = z_e_factory(batch_size=32, dim=8)
        codebook(z_e)
        assert not torch.allclose(codebook.embed, embed_before)

    @pytest.mark.unit
    def test_embed_unchanged_during_eval(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        codebook = codebook_factory(num_codes=4, code_dim=8)
        codebook.eval()
        embed_before = codebook.embed.clone()
        z_e = z_e_factory(batch_size=32, dim=8)
        codebook(z_e)
        assert torch.allclose(codebook.embed, embed_before)

    @pytest.mark.unit
    def test_training_update_all_reduces_cluster_statistics_across_distributed_ranks(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        codebook = codebook_factory(
            num_codes=4,
            code_dim=8,
            ema_decay=0.5,
            dead_code_threshold=0.0,
        )
        codebook.train()
        z_e = z_e_factory(batch_size=16, dim=8)

        with (
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.is_available",
                return_value=True,
            ),
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.is_initialized",
                return_value=True,
            ),
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.all_reduce"
            ) as all_reduce,
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.broadcast"
            ) as broadcast,
        ):
            codebook(z_e)

        assert all_reduce.call_count == 2
        assert all_reduce.call_args_list[0].args[0].shape == (4,)
        assert all_reduce.call_args_list[1].args[0].shape == (4, 8)
        assert broadcast.call_count == 0

    @pytest.mark.unit
    def test_kmeans_initialization_broadcasts_codebook_buffers_across_distributed_ranks(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        codebook = codebook_factory(num_codes=4, code_dim=8, kmeans_init=True)
        codebook.eval()
        z_e = z_e_factory(batch_size=16, dim=8)

        with (
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.is_available",
                return_value=True,
            ),
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.is_initialized",
                return_value=True,
            ),
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.broadcast"
            ) as broadcast,
        ):
            codebook(z_e)

        assert broadcast.call_count == 4

    @pytest.mark.unit
    def test_dead_code_replacement_broadcasts_codebook_buffers_across_distributed_ranks(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        codebook = codebook_factory(
            num_codes=4,
            code_dim=8,
            dead_code_threshold=10.0,
        )
        codebook.train()
        z_e = z_e_factory(batch_size=16, dim=8)

        with (
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.is_available",
                return_value=True,
            ),
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.is_initialized",
                return_value=True,
            ),
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.all_reduce"
            ),
            patch(
                "versatil.models.decoding.latent.vq.euclidean_codebook.torch_dist.broadcast"
            ) as broadcast,
        ):
            codebook(z_e)

        assert broadcast.call_count == 4


class TestEuclideanCodebookReplaceDeadCodes:
    @pytest.mark.unit
    @pytest.mark.parametrize("dead_code_threshold", [1.0, 2.0, 5.0])
    def test_dead_codes_get_replaced(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
        dead_code_threshold: float,
    ) -> None:
        codebook = codebook_factory(
            num_codes=4, code_dim=8, dead_code_threshold=dead_code_threshold
        )
        codebook.cluster_size.data.fill_(0.0)
        data = z_e_factory(batch_size=16, dim=8)
        replaced_codes = codebook._replace_dead_codes(data)
        assert replaced_codes
        assert torch.all(codebook.cluster_size == 1.0)

    @pytest.mark.unit
    def test_alive_codes_unchanged(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        codebook = codebook_factory(num_codes=4, code_dim=8, dead_code_threshold=2.0)
        codebook.cluster_size.data.fill_(10.0)
        embed_before = codebook.embed.clone()
        data = z_e_factory(batch_size=16, dim=8)
        replaced_codes = codebook._replace_dead_codes(data)
        assert not replaced_codes
        assert torch.allclose(codebook.embed, embed_before)

    @pytest.mark.unit
    def test_no_replacement_when_all_alive(
        self,
        codebook_factory: Callable[..., EuclideanCodebook],
        z_e_factory: Callable[..., torch.Tensor],
    ) -> None:
        codebook = codebook_factory(num_codes=4, code_dim=8, dead_code_threshold=1.0)
        codebook.cluster_size.data.fill_(5.0)
        embed_before = codebook.embed.clone()
        cluster_size_before = codebook.cluster_size.clone()
        data = z_e_factory(batch_size=16, dim=8)
        replaced_codes = codebook._replace_dead_codes(data)
        assert not replaced_codes
        assert torch.allclose(codebook.embed, embed_before)
        assert torch.allclose(codebook.cluster_size, cluster_size_before)
