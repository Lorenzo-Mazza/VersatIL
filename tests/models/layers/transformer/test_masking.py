"""Tests for versatil.models.layers.transformer.masking module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.layers.transformer.masking import (
    create_full_padding_mask,
    generate_causal_mask,
)


class TestGenerateCausalMask:

    @pytest.mark.parametrize("sequence_length", [1, 4, 8])
    def test_output_shape(self, device: torch.device, sequence_length: int):
        mask = generate_causal_mask(
            sequence_length=sequence_length, device=device
        )
        assert mask.shape == (1, 1, sequence_length, sequence_length)

    def test_diagonal_is_not_masked(self, device: torch.device):
        mask = generate_causal_mask(sequence_length=4, device=device)
        diagonal = torch.diagonal(mask.squeeze(0).squeeze(0))
        assert not diagonal.any()

    def test_future_positions_are_masked(self, device: torch.device):
        mask = generate_causal_mask(sequence_length=4, device=device)
        mask_2d = mask.squeeze(0).squeeze(0)
        # Position 0 should not attend to positions 1, 2, 3
        assert mask_2d[0, 1].item() is True
        assert mask_2d[0, 2].item() is True
        assert mask_2d[0, 3].item() is True

    def test_past_positions_are_not_masked(self, device: torch.device):
        mask = generate_causal_mask(sequence_length=4, device=device)
        mask_2d = mask.squeeze(0).squeeze(0)
        # Position 3 should attend to positions 0, 1, 2
        assert mask_2d[3, 0].item() is False
        assert mask_2d[3, 1].item() is False
        assert mask_2d[3, 2].item() is False

    def test_mask_is_upper_triangular(self, device: torch.device):
        mask = generate_causal_mask(sequence_length=5, device=device)
        mask_2d = mask.squeeze(0).squeeze(0)
        expected = torch.triu(
            torch.ones(5, 5, dtype=torch.bool, device=device), diagonal=1
        )
        assert torch.equal(mask_2d, expected)

    def test_single_token_produces_no_masking(self, device: torch.device):
        mask = generate_causal_mask(sequence_length=1, device=device)
        assert not mask.any()

    def test_device_placement(self, device: torch.device):
        mask = generate_causal_mask(sequence_length=4, device=device)
        assert mask.device.type == device.type


class TestCreateFullPaddingMask:

    def test_causal_mask_without_padding_or_cache(self, device: torch.device):
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=None,
            cached_key_padding_mask=None,
            self_attention_mask=None,
            batch_size=2,
            query_length=4,
            cache_length=0,
            device=device,
        )
        assert total_mask.shape == (1, 1, 4, 4)
        assert full_key_padding_mask is None
        # Should be causal
        assert total_mask[0, 0, 0, 1].item() is True
        assert total_mask[0, 0, 1, 0].item() is False

    def test_causal_mask_with_cache_extends_key_length(self, device: torch.device):
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=None,
            cached_key_padding_mask=None,
            self_attention_mask=None,
            batch_size=2,
            query_length=1,
            cache_length=3,
            device=device,
        )
        # key_length = cache_length + query_length = 4
        assert total_mask.shape == (1, 1, 1, 4)
        # Single query at position 3 can attend to all 4 key positions
        assert not total_mask[0, 0, 0, :].any()

    def test_key_padding_mask_without_cache(self, device: torch.device):
        key_padding_mask = torch.tensor(
            [[False, False, True, True], [False, True, False, True]],
            device=device,
        )
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=key_padding_mask,
            cached_key_padding_mask=None,
            self_attention_mask=None,
            batch_size=2,
            query_length=4,
            cache_length=0,
            device=device,
        )
        assert torch.equal(full_key_padding_mask, key_padding_mask)
        # Causal + padding: padded positions should be masked for all queries
        assert total_mask[0, 0, 0, 2].item() is True
        assert total_mask[0, 0, 0, 3].item() is True

    def test_key_padding_mask_concatenated_with_cached_mask(
        self, device: torch.device
    ):
        cached_mask = torch.tensor(
            [[False, True], [True, False]], device=device
        )
        current_mask = torch.tensor(
            [[False, False], [False, True]], device=device
        )
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=current_mask,
            cached_key_padding_mask=cached_mask,
            self_attention_mask=None,
            batch_size=2,
            query_length=2,
            cache_length=2,
            device=device,
        )
        expected_full = torch.cat((cached_mask, current_mask), dim=1)
        assert torch.equal(full_key_padding_mask, expected_full)
        assert full_key_padding_mask.shape == (2, 4)

    def test_cached_mask_without_current_padding(self, device: torch.device):
        cached_mask = torch.tensor(
            [[False, True], [True, False]], device=device
        )
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=None,
            cached_key_padding_mask=cached_mask,
            self_attention_mask=None,
            batch_size=2,
            query_length=2,
            cache_length=2,
            device=device,
        )
        assert full_key_padding_mask.shape == (2, 4)
        # Current positions should not be padded
        assert full_key_padding_mask[0, 2].item() is False
        assert full_key_padding_mask[0, 3].item() is False
        # Cached positions preserved
        assert full_key_padding_mask[0, 1].item() is True

    def test_custom_self_attention_mask_replaces_causal(
        self, device: torch.device
    ):
        custom_mask = torch.ones(2, 1, 4, 4, dtype=torch.bool, device=device)
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=None,
            cached_key_padding_mask=None,
            self_attention_mask=custom_mask,
            batch_size=2,
            query_length=4,
            cache_length=0,
            device=device,
        )
        assert total_mask.shape == (2, 1, 4, 4)
        # Custom all-True mask is OR'd into the query columns, so all positions are masked
        assert total_mask.all()
        assert full_key_padding_mask is None

    def test_custom_self_attention_mask_with_cache_offset(
        self, device: torch.device
    ):
        custom_mask = torch.zeros(2, 1, 1, 1, dtype=torch.bool, device=device)
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=None,
            cached_key_padding_mask=None,
            self_attention_mask=custom_mask,
            batch_size=2,
            query_length=1,
            cache_length=3,
            device=device,
        )
        # key_length = 3 + 1 = 4
        assert total_mask.shape == (2, 1, 1, 4)
        # Cache positions should be unmasked (no padding, no causal)
        assert total_mask[0, 0, 0, 0].item() is False

    def test_custom_mask_with_padding_combined(self, device: torch.device):
        custom_mask = torch.zeros(2, 1, 3, 3, dtype=torch.bool, device=device)
        key_padding = torch.tensor(
            [[False, False, True], [True, False, False]],
            device=device,
        )
        total_mask, full_key_padding_mask = create_full_padding_mask(
            key_padding_mask=key_padding,
            cached_key_padding_mask=None,
            self_attention_mask=custom_mask,
            batch_size=2,
            query_length=3,
            cache_length=0,
            device=device,
        )
        # Padded positions should be masked across all queries
        assert total_mask[0, 0, 0, 2].item() is True
        assert total_mask[0, 0, 1, 2].item() is True
        assert total_mask[1, 0, 0, 0].item() is True
