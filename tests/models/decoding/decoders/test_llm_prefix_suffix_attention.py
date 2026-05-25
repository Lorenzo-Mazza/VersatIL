"""Tests for versatil.models.decoding.decoders.llm_prefix_suffix_attention module."""

import re

import pytest
import torch

from versatil.models.decoding.decoders.llm_prefix_suffix_attention import (
    LLMPrefixSuffixAttentionMixin,
)


class ConcretePrefixSuffixAttention(LLMPrefixSuffixAttentionMixin):
    causal_prefix: bool = False


@pytest.mark.unit
def test_all_false_or_none_returns_none_for_missing_padding_mask() -> None:
    result = ConcretePrefixSuffixAttention._all_false_or_none(mask=None)

    assert result is None


@pytest.mark.unit
def test_all_false_or_none_returns_none_for_empty_padding_mask() -> None:
    mask = torch.zeros(2, 3, dtype=torch.bool)

    result = ConcretePrefixSuffixAttention._all_false_or_none(mask=mask)

    assert result is None


@pytest.mark.unit
def test_all_false_or_none_preserves_mask_with_padding() -> None:
    mask = torch.tensor([[False, True, False]])

    result = ConcretePrefixSuffixAttention._all_false_or_none(mask=mask)

    assert result is mask


@pytest.mark.unit
def test_causal_attention_mask_returns_none_without_padding_mask() -> None:
    helper = ConcretePrefixSuffixAttention()
    tokens = torch.zeros(2, 3, 4)

    attention_mask = helper._build_causal_attention_mask(
        padding_mask=None,
        tokens=tokens,
    )

    assert attention_mask is None


@pytest.mark.unit
def test_causal_attention_mask_converts_padding_to_lm_visibility_mask() -> None:
    helper = ConcretePrefixSuffixAttention()
    tokens = torch.zeros(2, 3, 4)
    padding_mask = torch.tensor(
        [
            [False, True, False],
            [True, False, False],
        ]
    )

    attention_mask = helper._build_causal_attention_mask(
        padding_mask=padding_mask,
        tokens=tokens,
    )

    expected = torch.tensor(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=torch.long,
    )
    torch.testing.assert_close(attention_mask, expected)


@pytest.mark.unit
def test_prefix_attention_mask_blocks_prefix_queries_from_suffix_tokens() -> None:
    prefix_tokens = torch.zeros(1, 2, 4)
    suffix_tokens = torch.zeros(1, 2, 4)

    attention_mask = ConcretePrefixSuffixAttention._build_prefix_attention_mask(
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        prefix_mask=None,
        causal_suffix=True,
    )

    assert attention_mask.shape == (1, 1, 4, 4)
    assert not attention_mask[0, 0, 0, 2]
    assert not attention_mask[0, 0, 1, 3]
    assert attention_mask[0, 0, 2, 0]
    assert attention_mask[0, 0, 3, 1]
    assert attention_mask[0, 0, 2, 2]
    assert not attention_mask[0, 0, 2, 3]
    assert attention_mask[0, 0, 3, 2]
    assert attention_mask[0, 0, 3, 3]


@pytest.mark.unit
def test_prefix_attention_mask_can_make_suffix_bidirectional() -> None:
    prefix_tokens = torch.zeros(1, 1, 4)
    suffix_tokens = torch.zeros(1, 2, 4)

    attention_mask = ConcretePrefixSuffixAttention._build_prefix_attention_mask(
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        prefix_mask=None,
        causal_suffix=False,
    )

    assert attention_mask[0, 0, 1, 2]
    assert attention_mask[0, 0, 2, 1]


@pytest.mark.unit
def test_prefix_attention_mask_applies_prefix_key_padding() -> None:
    prefix_tokens = torch.zeros(1, 2, 4)
    suffix_tokens = torch.zeros(1, 1, 4)
    prefix_mask = torch.tensor([[False, True]])

    attention_mask = ConcretePrefixSuffixAttention._build_prefix_attention_mask(
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        prefix_mask=prefix_mask,
        causal_suffix=True,
    )

    assert not attention_mask[0, 0, 0, 1]
    assert not attention_mask[0, 0, 2, 1]


@pytest.mark.unit
def test_build_attention_mask_uses_causal_padding_branch_when_configured() -> None:
    helper = ConcretePrefixSuffixAttention()
    helper.causal_prefix = True
    tokens = torch.zeros(1, 3, 4)
    padding_mask = torch.tensor([[False, True, False]])

    attention_mask = helper._build_attention_mask(
        padding_mask=padding_mask,
        tokens=tokens,
        prefix_length=0,
        causal_suffix=True,
    )

    torch.testing.assert_close(
        attention_mask,
        torch.tensor([[1, 0, 1]], dtype=torch.long),
    )


@pytest.mark.unit
@pytest.mark.parametrize("prefix_length", [0, 4])
def test_build_attention_mask_rejects_invalid_prefix_length(
    prefix_length: int,
) -> None:
    helper = ConcretePrefixSuffixAttention()
    tokens = torch.zeros(1, 3, 4)
    expected_message = (
        "prefix_length must be in [1, sequence_length], got "
        f"prefix_length={prefix_length}, sequence_length=3."
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        helper._build_attention_mask(
            padding_mask=None,
            tokens=tokens,
            prefix_length=prefix_length,
            causal_suffix=True,
        )


@pytest.mark.unit
def test_build_attention_mask_drops_all_visible_prefix_mask() -> None:
    helper = ConcretePrefixSuffixAttention()
    tokens = torch.zeros(1, 3, 4)

    attention_mask = helper._build_attention_mask(
        padding_mask=None,
        tokens=tokens,
        prefix_length=tokens.shape[1],
        causal_suffix=True,
    )

    assert attention_mask is None


@pytest.mark.unit
def test_append_unmasked_tokens_returns_none_without_padding_mask() -> None:
    suffix_tokens = torch.zeros(1, 3, 4)

    full_mask = ConcretePrefixSuffixAttention._append_unmasked_tokens(
        padding_mask=None,
        tokens=suffix_tokens,
    )

    assert full_mask is None


@pytest.mark.unit
def test_append_unmasked_tokens_extends_padding_mask() -> None:
    prefix_mask = torch.tensor([[False, True]])
    suffix_tokens = torch.zeros(1, 3, 4)

    full_mask = ConcretePrefixSuffixAttention._append_unmasked_tokens(
        padding_mask=prefix_mask,
        tokens=suffix_tokens,
    )

    torch.testing.assert_close(
        full_mask,
        torch.tensor([[False, True, False, False, False]]),
    )


@pytest.mark.unit
def test_build_prefix_suffix_inputs_concatenates_tokens_and_mask() -> None:
    helper = ConcretePrefixSuffixAttention()
    prefix_tokens = torch.ones(1, 2, 4)
    suffix_tokens = torch.full((1, 1, 4), fill_value=2.0)
    prefix_mask = torch.tensor([[False, False]])

    full_tokens, attention_mask = helper._build_prefix_suffix_inputs(
        prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
        prefix_mask=prefix_mask,
        causal_suffix=True,
    )

    torch.testing.assert_close(
        full_tokens,
        torch.cat([prefix_tokens, suffix_tokens], dim=1),
    )
    assert attention_mask.shape == (1, 1, 3, 3)
