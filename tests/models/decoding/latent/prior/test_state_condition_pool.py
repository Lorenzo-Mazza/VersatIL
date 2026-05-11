"""Tests for versatil.models.decoding.latent.prior.state_condition_pool module."""

import re

import pytest
import torch
import torch.nn.functional as F

from versatil.models.decoding.latent.prior.state_condition_pool import (
    StateConditionPool,
)


@pytest.mark.unit
class TestStateConditionPoolInit:
    def test_rejects_non_positive_embedding_dimension(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("embedding_dimension must be positive, got 0."),
        ):
            StateConditionPool(embedding_dimension=0)


@pytest.mark.unit
class TestStateConditionPoolForward:
    def test_pools_unmasked_tokens(self) -> None:
        tokens = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [100.0, 100.0, 100.0, 100.0],
                    [5.0, 6.0, 7.0, 8.0],
                ],
                [
                    [-1.0, 0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0, 10.0],
                ],
            ]
        )
        padding_mask = torch.tensor(
            [
                [False, True, False],
                [False, False, False],
            ]
        )
        expected_unscaled = torch.stack(
            [
                tokens[0, [0, 2]].mean(dim=0),
                tokens[1].mean(dim=0),
            ]
        )
        expected = F.layer_norm(expected_unscaled, normalized_shape=(4,))
        pool = StateConditionPool(embedding_dimension=4)

        result = pool(tokens=tokens, padding_mask=padding_mask)

        torch.testing.assert_close(result, expected)

    def test_without_mask_pools_all_tokens(self) -> None:
        tokens = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ]
        )
        expected = F.layer_norm(tokens.mean(dim=1), normalized_shape=(2,))
        pool = StateConditionPool(embedding_dimension=2)

        result = pool(tokens=tokens)

        torch.testing.assert_close(result, expected)

    def test_all_padded_tokens_return_zero_vector(self) -> None:
        tokens = torch.ones(2, 3, 4)
        padding_mask = torch.ones(2, 3, dtype=torch.bool)
        pool = StateConditionPool(embedding_dimension=4)

        result = pool(tokens=tokens, padding_mask=padding_mask)

        torch.testing.assert_close(result, torch.zeros(2, 4))

    def test_rejects_wrong_token_rank(self) -> None:
        pool = StateConditionPool(embedding_dimension=4)
        tokens = torch.zeros(2, 4)

        with pytest.raises(
            ValueError,
            match=re.escape("tokens must have shape (batch, sequence, embedding)"),
        ):
            pool(tokens=tokens)

    def test_rejects_wrong_embedding_dimension(self) -> None:
        pool = StateConditionPool(embedding_dimension=4)
        tokens = torch.zeros(2, 3, 5)

        with pytest.raises(
            ValueError,
            match=re.escape("tokens embedding dimension must be 4, got 5."),
        ):
            pool(tokens=tokens)

    def test_rejects_empty_sequence(self) -> None:
        pool = StateConditionPool(embedding_dimension=4)
        tokens = torch.zeros(2, 0, 4)

        with pytest.raises(
            ValueError,
            match=re.escape("StateConditionPool requires at least one token to pool."),
        ):
            pool(tokens=tokens)

    def test_rejects_wrong_padding_mask_shape(self) -> None:
        pool = StateConditionPool(embedding_dimension=4)
        tokens = torch.zeros(2, 3, 4)
        padding_mask = torch.zeros(2, 4, dtype=torch.bool)

        with pytest.raises(
            ValueError,
            match=re.escape("padding_mask must have shape"),
        ):
            pool(tokens=tokens, padding_mask=padding_mask)
