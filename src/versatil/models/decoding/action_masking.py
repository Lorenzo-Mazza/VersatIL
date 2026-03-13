"""Attention masking for models with bidirectional prefix and causal action tokens."""
import torch


def make_attention_mask(
    action_tokens: torch.Tensor,
    feature_tokens: torch.Tensor,
    feature_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention mask with bidirectional prefix and causal actions.

    Args:
        action_tokens: Action token embeddings (B, action_token_len, emb_dim)
        feature_tokens: Feature token embeddings (B, feat_token_len, emb_dim)
        feature_token_mask: Optional feature token mask (B, feat_token_len)

    Returns:
        full_padding_mask: Attention mask (B, 1, total_len, total_len)
        full_key_padding_mask: Key padding mask (B, total_len)

    Note: True indicates masked tokens, False indicates valid tokens. Action padding is handled in loss computation.
    """
    batch_size = feature_tokens.shape[0]
    prefix_len = feature_tokens.shape[1]
    action_len = action_tokens.shape[1]
    total_len = prefix_len + action_len
    if feature_token_mask is None:
        feature_token_mask = torch.zeros(
            batch_size, prefix_len, dtype=torch.bool, device=feature_tokens.device
        )
    action_ar_pad_mask = (
        torch.triu(
            torch.ones(
                action_tokens.shape[1],
                action_tokens.shape[1],
                device=action_tokens.device,
                dtype=torch.bool,
            ),
            diagonal=0,  # `True`s start on the main diagonal, i.e. don't attend to current and future action tokens
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1, 1, action_len, action_len)
    action_ar_pad_mask = action_ar_pad_mask.expand(
        batch_size, 1, action_tokens.shape[1], action_tokens.shape[1]
    )  # (B, 1, action_len, action_len)
    full_padding_mask = torch.zeros(
        batch_size,
        1,
        total_len,
        total_len,
        dtype=torch.bool,
        device=feature_tokens.device,
    )
    full_padding_mask[:, :, prefix_len:, prefix_len:] = action_ar_pad_mask
    key_padding_mask = torch.cat(
        (
            feature_token_mask,
            torch.zeros(
                batch_size, action_len, dtype=torch.bool, device=feature_tokens.device
            ),
        ),
        dim=1,
    )  # (B, total_len)
    key_padding_mask_4d = key_padding_mask.unsqueeze(1).unsqueeze(
        2
    )  # (B, 1, 1, total_len)
    full_padding_mask = full_padding_mask | key_padding_mask_4d.expand(
        -1, -1, total_len, -1
    )  # (B, 1, total_len, total_len)
    full_padding_mask[
        :, :, :prefix_len, prefix_len:
    ] = True  # Prefix tokens cannot attend to future action tokens
    return (
        full_padding_mask,
        key_padding_mask,
    )  # (B, 1, total_len, total_len), (B, total_len)
