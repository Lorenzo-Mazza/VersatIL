"""Taken from https://benjaminwarner.dev/2022/07/14/tinkering-with-attention-pooling"""
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


class AttentionPool1d(nn.Module):
    "Attention for Learned Aggregation"
    def __init__(self,
        ni:int,
        bias:bool=True,
        norm:Callable[[int], nn.Module]=nn.LayerNorm
    ):
        super().__init__()
        self.norm = norm(ni)
        self.q = nn.Linear(ni, ni, bias=bias)
        self.vk = nn.Linear(ni, ni*2, bias=bias)
        self.proj = nn.Linear(ni, ni)

    def forward(self, x:Tensor, cls_q:Tensor):
        x = self.norm(x)
        B, N, C = x.shape
        q = self.q(cls_q.expand(B, -1)).unsqueeze(1)
        k, v = self.vk(x).reshape(B, N, 2, C).permute(2, 0, 1, 3).chunk(2, 0)
        k, v = k.squeeze(0), v.squeeze(0)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, C)
        return self.proj(x)


class LearnedAggregation(nn.Module):
    "Learned Aggregation from https://arxiv.org/abs/2112.13692"
    def __init__(self,
                 ni:int,
                 attn_bias:bool=True,
                 ffn_expand:int|float=3,
                 norm:Callable[[int], nn.Module]=nn.LayerNorm,
                 activation_function:type[nn.Module]=nn.GELU,
                 ):
        super().__init__()
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones(ni))
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones(ni))
        self.cls_q = nn.Parameter(torch.zeros(ni))
        self.attn = AttentionPool1d(ni, attn_bias, norm)
        self.norm = norm(ni)
        self.ffn = nn.Sequential(
            nn.Linear(ni, int(ni*ffn_expand)),
            activation_function(),
            nn.Linear(int(ni*ffn_expand), ni)
        )
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x:Tensor):
        x = self.cls_q + self.gamma_1 * self.attn(x, self.cls_q)
        return x + self.gamma_2 * self.ffn(self.norm(x))

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
