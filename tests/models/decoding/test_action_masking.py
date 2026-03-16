"""Tests for versatil.models.decoding.action_masking module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.decoding.action_masking import make_attention_mask


@pytest.fixture
def token_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for token embedding tensors."""
    def factory(
        batch_size: int = 2,
        sequence_length: int = 4,
        embedding_dim: int = 32,
    ) -> torch.Tensor:
        shape = (batch_size, sequence_length, embedding_dim)
        return torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32)
        )
    return factory


class TestMakeAttentionMask:

    def test_output_shapes(
        self,
        token_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        prefix_len = 4
        action_len = 8
        feature_tokens = token_factory(
            batch_size=batch_size,
            sequence_length=prefix_len,
        )
        action_tokens = token_factory(
            batch_size=batch_size,
            sequence_length=action_len,
        )
        full_mask, key_mask = make_attention_mask(
            action_tokens=action_tokens,
            feature_tokens=feature_tokens,
        )
        total_len = prefix_len + action_len
        assert full_mask.shape == (batch_size, 1, total_len, total_len)
        assert key_mask.shape == (batch_size, total_len)

    def test_prefix_cannot_attend_to_actions(
        self,
        token_factory: Callable[..., torch.Tensor],
    ):
        prefix_len = 4
        action_len = 6
        feature_tokens = token_factory(sequence_length=prefix_len)
        action_tokens = token_factory(sequence_length=action_len)
        full_mask, _ = make_attention_mask(
            action_tokens=action_tokens,
            feature_tokens=feature_tokens,
        )
        # full_mask[:, :, :prefix_len, prefix_len:] should all be True (masked)
        prefix_to_action = full_mask[:, :, :prefix_len, prefix_len:]
        assert prefix_to_action.all()

    def test_action_tokens_are_causal(
        self,
        token_factory: Callable[..., torch.Tensor],
    ):
        prefix_len = 4
        action_len = 4
        feature_tokens = token_factory(sequence_length=prefix_len)
        action_tokens = token_factory(sequence_length=action_len)
        full_mask, _ = make_attention_mask(
            action_tokens=action_tokens,
            feature_tokens=feature_tokens,
        )
        # Check the action-to-action block is strictly upper triangular (True above diagonal = masked future)
        action_block = full_mask[0, 0, prefix_len:, prefix_len:]
        for i in range(action_len):
            for j in range(action_len):
                if j > i:
                    assert action_block[i, j].item() is True
                else:
                    assert action_block[i, j].item() is False

    def test_prefix_tokens_attend_to_each_other(
        self,
        token_factory: Callable[..., torch.Tensor],
    ):
        prefix_len = 4
        action_len = 4
        feature_tokens = token_factory(sequence_length=prefix_len)
        action_tokens = token_factory(sequence_length=action_len)
        full_mask, _ = make_attention_mask(
            action_tokens=action_tokens,
            feature_tokens=feature_tokens,
        )
        prefix_block = full_mask[0, 0, :prefix_len, :prefix_len]
        assert not prefix_block.any()

    def test_feature_token_mask_propagated(
        self,
        token_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        prefix_len = 4
        action_len = 4
        feature_tokens = token_factory(
            batch_size=batch_size,
            sequence_length=prefix_len,
        )
        action_tokens = token_factory(
            batch_size=batch_size,
            sequence_length=action_len,
        )
        feature_mask = torch.zeros(batch_size, prefix_len, dtype=torch.bool)
        feature_mask[:, 2] = True  # Mask third feature token
        _, key_mask = make_attention_mask(
            action_tokens=action_tokens,
            feature_tokens=feature_tokens,
            feature_token_mask=feature_mask,
        )
        assert key_mask[:, 2].all()
        assert not key_mask[:, 0].any()

    @pytest.mark.parametrize("prefix_len, action_len", [
        (1, 1),
        (8, 16),
        (4, 4),
    ])
    def test_different_sequence_lengths(
        self,
        token_factory: Callable[..., torch.Tensor],
        prefix_len: int,
        action_len: int,
    ):
        feature_tokens = token_factory(sequence_length=prefix_len)
        action_tokens = token_factory(sequence_length=action_len)
        full_mask, key_mask = make_attention_mask(
            action_tokens=action_tokens,
            feature_tokens=feature_tokens,
        )
        total_len = prefix_len + action_len
        assert full_mask.shape == (2, 1, total_len, total_len)
        assert key_mask.shape == (2, total_len)
