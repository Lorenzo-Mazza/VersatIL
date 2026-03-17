"""Masking utilities for causal transformers."""

import torch


def generate_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    """Generate causal attention mask.

    Args:
        sequence_length: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask (1, 1, seq_len, seq_len) as boolean tensor where True means masked position
    """
    # Create triangular matrix mask with True for future positions
    mask = torch.triu(
        torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
        diagonal=1,  # `True`s start above the main diagonal
    )
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def create_full_padding_mask(
    key_padding_mask: torch.Tensor | None,
    cached_key_padding_mask: torch.Tensor | None,
    self_attention_mask: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    cache_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Create full causal self attention mask and full key padding mask by combining cached and current masks.

    Args:
        key_padding_mask: Current key padding mask (B, query_length) with True=padding.
        cached_key_padding_mask: Cached key padding mask (B, cache_length) with True=padding.
        self_attention_mask: self-attention mask (B, 1, query_length, key_length) where True=masked.
        batch_size: Batch size.
        query_length: Current query length.
        cache_length: Cached sequence length.
        device: Device to create mask on.

    Returns:
        Tuple with:
          total_mask: Updated self-attention mask (B, 1, query_length, key_length) where True=masked.
          full_key_padding_mask: Combined key padding mask (B, key_length) with True=padding.

    Note:
        This function always applies causal masking if the self_attention_mask is None.
    """
    key_length = cache_length + query_length
    full_key_padding_mask = None
    if key_padding_mask is not None:
        if cached_key_padding_mask is None:
            full_key_padding_mask = key_padding_mask  # (B, seq_len + 0 = key_length)
        else:
            full_key_padding_mask = torch.cat(
                (cached_key_padding_mask, key_padding_mask), dim=1
            )  # (B, total_len)
    else:
        # full_key_padding_mask is None
        if cached_key_padding_mask is not None:
            full_key_padding_mask = torch.cat(
                (
                    cached_key_padding_mask,
                    torch.zeros(
                        batch_size, query_length, device=device, dtype=torch.bool
                    ),
                ),
                dim=1,
            )  # (B, total_len)

    if self_attention_mask is None:
        causal_mask = generate_causal_mask(
            key_length, device
        )  # (1, 1, key_length, key_length)
        total_mask = causal_mask[
            :, :, -query_length:, :
        ]  # (1,1,query_length,key_length)
        if full_key_padding_mask is not None:
            padding_mask = full_key_padding_mask.unsqueeze(1).unsqueeze(
                1
            )  # (B,1,1,key_length)
            # Broadcasting to (B,1,query_length,key_length)
            total_mask = total_mask | padding_mask
    else:
        # self_attention_mask (B, 1, query_length, query_length)
        # total_mask (B, 1, query_length, key_length)
        total_mask = torch.zeros(
            batch_size, 1, query_length, key_length, dtype=torch.bool, device=device
        )
        if full_key_padding_mask is not None:
            padding_mask = full_key_padding_mask.unsqueeze(1).unsqueeze(
                1
            )  # (B, 1, 1, key_length)
            # Broadcasting to (B,1, query_length, key_length)
            total_mask |= padding_mask
        # Slicing the query_length elements in total mask
        # total_mask[:, :, :, cache_length:] has shape (B,1, query_length, query_length)
        total_mask[:, :, :, cache_length:] |= self_attention_mask

    return total_mask, full_key_padding_mask
