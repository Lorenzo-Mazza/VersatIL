"""Inspired from https://benjaminwarner.dev/2022/07/14/tinkering-with-attention-pooling"""
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


class AttentionPool2d(nn.Module):
    "Attention for Learned Aggregation"

    def __init__(
        self,
        ni: int,
        bias: bool = True,
        norm: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm = norm(ni)
        self.q = nn.Linear(ni, ni, bias=bias)
        self.vk = nn.Linear(ni, ni * 2, bias=bias)
        self.proj = nn.Linear(ni, ni)

    def forward(self, x: Tensor, cls_q: Tensor):
        if x.ndim == 4:
            # CNN feature map B,C,H,W -> B,H*W,C
            if x.shape[1] == self.norm.normalized_shape[0]:
                x = x.flatten(2).transpose(1, 2)
            elif x.shape[-1] == self.norm.normalized_shape[0]:
                x = x.permute(0, 3, 1, 2)
                x = x.flatten(2).transpose(1, 2)
            else:
                raise ValueError(
                    f"Input shape {x.shape} not compatible with AttentionPool2d "
                    f"of size {self.norm.normalized_shape[0]}"
                )

        elif x.ndim == 3:
            # ViT: Normalize after ensuring [B, N, C] (last dim == C)
            if x.shape[1] == self.norm.normalized_shape[0]:
                x = x.transpose(1, 2)
            elif x.shape[-1] == self.norm.normalized_shape[0]:
                pass
            else:
                raise ValueError(
                    f"Input shape {x.shape} not compatible with AttentionPool2d "
                    f"of size {self.norm.normalized_shape[0]}"
                )

        B, N, C = x.shape
        x = self.norm(x)
        q = self.q(cls_q.expand(B, -1)).unsqueeze(1)  # [B,1,C]
        k, v = self.vk(x).reshape(B, N, 2, C).permute(2, 0, 1, 3).chunk(2, 0)
        k, v = k[0], v[0]  # [B,N,C]
        attended_values = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v
        )  # [B,1,C]
        return self.proj(attended_values.squeeze(1))  # Squeeze to [B,C] before proj


class LearnedAggregation(nn.Module):
    "Learned Aggregation from https://arxiv.org/abs/2112.13692"

    def __init__(
        self,
        ni: int,
        attn_bias: bool = True,
        ffn_expand: int | float = 3,
        norm: Callable[[int], nn.Module] = nn.LayerNorm,
        act_cls: Callable[[None], nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones(ni))
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones(ni))
        self.cls_q = nn.Parameter(torch.zeros(ni))
        self.attn = AttentionPool2d(ni, attn_bias, norm)
        self.norm = norm(ni)
        self.ffn = nn.Sequential(
            nn.Linear(ni, int(ni * ffn_expand)),
            act_cls(),
            nn.Linear(int(ni * ffn_expand), ni),
        )
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        x = self.cls_q + self.gamma_1 * self.attn(x, self.cls_q)
        return x + self.gamma_2 * self.ffn(self.norm(x))

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
