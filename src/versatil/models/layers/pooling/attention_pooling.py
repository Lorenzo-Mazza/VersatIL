"""Inspired from https://benjaminwarner.dev/2022/07/14/tinkering-with-attention-pooling"""

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


class AttentionPool2d(nn.Module):
    """Attention for Learned Aggregation."""

    def __init__(
        self,
        feature_dimension: int,
        bias: bool = True,
        norm: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm = norm(feature_dimension)
        self.q = nn.Linear(feature_dimension, feature_dimension, bias=bias)
        self.vk = nn.Linear(feature_dimension, feature_dimension * 2, bias=bias)
        self.proj = nn.Linear(feature_dimension, feature_dimension)

    def forward(
        self,
        features: Tensor,
        class_query: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Pool a feature sequence with a learned query.

        Args:
            features: Spatial feature map or token sequence.
            class_query: Learned query vector used to aggregate the sequence.
            padding_mask: Optional token padding mask where ``True`` means padded.

        Returns:
            Aggregated feature tensor of shape ``(batch_size, feature_dimension)``.
        """
        if features.ndim == 4:
            # Convolutional feature map: (batch, channels, height, width)
            # to (batch, height * width, channels).
            if features.shape[1] == self.norm.normalized_shape[0]:
                features = features.flatten(2).transpose(1, 2)
            elif features.shape[-1] == self.norm.normalized_shape[0]:
                features = features.permute(0, 3, 1, 2)
                features = features.flatten(2).transpose(1, 2)
            else:
                raise ValueError(
                    f"Input shape {features.shape} not compatible with AttentionPool2d "
                    f"of size {self.norm.normalized_shape[0]}"
                )

        elif features.ndim == 3:
            # Token sequence: ensure features are (batch, sequence, channels).
            if features.shape[1] == self.norm.normalized_shape[0]:
                features = features.transpose(1, 2)
            elif features.shape[-1] == self.norm.normalized_shape[0]:
                pass
            else:
                raise ValueError(
                    f"Input shape {features.shape} not compatible with AttentionPool2d "
                    f"of size {self.norm.normalized_shape[0]}"
                )

        batch_size, sequence_length, feature_dimension = features.shape
        attention_mask = None
        if padding_mask is not None:
            expected_shape = (batch_size, sequence_length)
            if padding_mask.shape != expected_shape:
                raise ValueError(
                    f"padding_mask must have shape {expected_shape}, got {padding_mask.shape}."
                )
            attention_mask = ~padding_mask.to(device=features.device, dtype=torch.bool)
            attention_mask = attention_mask.unsqueeze(1)
        features = self.norm(features)
        query = self.q(class_query.expand(batch_size, -1)).unsqueeze(1)
        key, value = (
            self.vk(features)
            .reshape(batch_size, sequence_length, 2, feature_dimension)
            .permute(2, 0, 1, 3)
            .chunk(2, 0)
        )
        key, value = key[0], value[0]
        attended_values = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
        )
        return self.proj(attended_values.squeeze(1))


class LearnedAggregation(nn.Module):
    """Learned Aggregation from https://arxiv.org/abs/2112.13692."""

    def __init__(
        self,
        feature_dimension: int,
        attention_bias: bool = True,
        feedforward_expand: int | float = 3,
        norm: Callable[[int], nn.Module] = nn.LayerNorm,
        activation_class: Callable[[], nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones(feature_dimension))
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones(feature_dimension))
        self.cls_q = nn.Parameter(torch.zeros(feature_dimension))
        self.attn = AttentionPool2d(feature_dimension, attention_bias, norm)
        self.norm = norm(feature_dimension)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dimension, int(feature_dimension * feedforward_expand)),
            activation_class(),
            nn.Linear(int(feature_dimension * feedforward_expand), feature_dimension),
        )
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)

    def forward(self, features: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Aggregate a token sequence while optionally ignoring padded tokens.

        Args:
            features: Feature tensor to aggregate.
            padding_mask: Optional token padding mask where ``True`` means padded.

        Returns:
            Aggregated feature tensor of shape ``(batch_size, feature_dimension)``.
        """
        features = self.cls_q + self.gamma_1 * self.attn(
            features=features,
            class_query=self.cls_q,
            padding_mask=padding_mask,
        )
        return features + self.gamma_2 * self.ffn(self.norm(features))

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
