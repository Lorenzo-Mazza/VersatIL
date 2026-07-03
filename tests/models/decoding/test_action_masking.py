"""Tests for versatil.models.decoding.action_masking module."""

import re
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
        embedding_dimension: int = 32,
    ) -> torch.Tensor:
        shape = (batch_size, sequence_length, embedding_dimension)
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

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

    def test_feature_token_mask_converted_to_bool(
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
        feature_mask = torch.zeros(batch_size, prefix_len, dtype=torch.float32)
        feature_mask[:, 1] = 1.0
        _, key_mask = make_attention_mask(
            action_tokens=action_tokens,
            feature_tokens=feature_tokens,
            feature_token_mask=feature_mask,
        )
        assert key_mask.dtype == torch.bool
        assert key_mask[:, 1].all()
        assert not key_mask[:, 0].any()

    def test_feature_token_mask_shape_mismatch_raises(
        self,
        token_factory: Callable[..., torch.Tensor],
    ):
        batch_size = 2
        prefix_len = 4
        wrong_prefix_len = 5
        feature_tokens = token_factory(
            batch_size=batch_size,
            sequence_length=prefix_len,
        )
        action_tokens = token_factory(
            batch_size=batch_size,
            sequence_length=4,
        )
        feature_mask = torch.zeros(batch_size, wrong_prefix_len, dtype=torch.bool)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"feature_token_mask must have shape {(batch_size, prefix_len)}, "
                f"got {feature_mask.shape}."
            ),
        ):
            make_attention_mask(
                action_tokens=action_tokens,
                feature_tokens=feature_tokens,
                feature_token_mask=feature_mask,
            )

    def test_bidirectional_actions_when_causal_disabled(
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
            causal_actions=False,
        )
        action_block = full_mask[0, 0, prefix_len:, prefix_len:]
        assert not action_block.any()

    def test_prefix_still_blocked_from_actions_when_bidirectional(
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
            causal_actions=False,
        )
        prefix_to_action = full_mask[:, :, :prefix_len, prefix_len:]
        assert prefix_to_action.all()

    @pytest.mark.parametrize(
        "prefix_len, action_len",
        [
            (1, 1),
            (8, 16),
            (4, 4),
        ],
    )
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

    def test_feature_tokens_must_be_three_dimensional(
        self,
        token_factory: Callable[..., torch.Tensor],
    ):
        feature_tokens = torch.zeros(2, 32)
        action_tokens = token_factory(sequence_length=4)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"feature_tokens must have shape (B, P, D), got {feature_tokens.shape}."
            ),
        ):
            make_attention_mask(
                action_tokens=action_tokens,
                feature_tokens=feature_tokens,
            )

    def test_action_tokens_must_be_three_dimensional(
        self,
        token_factory: Callable[..., torch.Tensor],
    ):
        feature_tokens = token_factory(sequence_length=4)
        action_tokens = torch.zeros(2, 32)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"action_tokens must have shape (B, A, D), got {action_tokens.shape}."
            ),
        ):
            make_attention_mask(
                action_tokens=action_tokens,
                feature_tokens=feature_tokens,
            )

    def test_token_batch_size_mismatch_raises(
        self,
        token_factory: Callable[..., torch.Tensor],
    ):
        feature_batch_size = 2
        action_batch_size = 3
        feature_tokens = token_factory(
            batch_size=feature_batch_size,
            sequence_length=4,
        )
        action_tokens = token_factory(
            batch_size=action_batch_size,
            sequence_length=4,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "feature_tokens and action_tokens must have matching batch size, "
                f"got {feature_batch_size} and {action_batch_size}."
            ),
        ):
            make_attention_mask(
                action_tokens=action_tokens,
                feature_tokens=feature_tokens,
            )

    @pytest.mark.parametrize("causal_prefix_suffix_length", [-1, 5])
    def test_causal_prefix_suffix_length_out_of_range_raises(
        self,
        token_factory: Callable[..., torch.Tensor],
        causal_prefix_suffix_length: int,
    ):
        prefix_len = 4
        feature_tokens = token_factory(sequence_length=prefix_len)
        action_tokens = token_factory(sequence_length=4)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "causal_prefix_suffix_length must be between 0 and prefix length "
                f"{prefix_len}, got {causal_prefix_suffix_length}."
            ),
        ):
            make_attention_mask(
                action_tokens=action_tokens,
                feature_tokens=feature_tokens,
                causal_prefix_suffix_length=causal_prefix_suffix_length,
            )

    def test_causal_prefix_suffix_blocks_earlier_prefix_queries(
        self,
        token_factory: Callable[..., torch.Tensor],
    ):
        prefix_len = 4
        causal_prefix_suffix_length = 1
        feature_tokens = token_factory(sequence_length=prefix_len)
        action_tokens = token_factory(sequence_length=4)
        full_mask, _ = make_attention_mask(
            action_tokens=action_tokens,
            feature_tokens=feature_tokens,
            causal_prefix_suffix_length=causal_prefix_suffix_length,
        )
        causal_start = prefix_len - causal_prefix_suffix_length
        blocked = full_mask[:, :, :causal_start, causal_start:prefix_len]
        suffix_to_prefix = full_mask[:, :, causal_start:prefix_len, :causal_start]
        assert blocked.all()
        assert not suffix_to_prefix.any()
