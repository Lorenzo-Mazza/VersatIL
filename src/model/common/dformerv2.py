"""
DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation
Code: https://github.com/VCIP-RGBD/DFormer

Author: yinbow
Email: bowenyin@mail.nankai.edu.cn

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Tuple
from collections import OrderedDict
from torch.nn.init import trunc_normal_  # noqa: E402

# ---------------------------------------------------------------------
# Minimal DropPath (stochastic depth) implementation
# ---------------------------------------------------------------------
def _drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Work with broadcastable shape (N, 1, 1, 1, ...) to drop whole samples
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if scale_by_keep and keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Stochastic depth as in timm's DropPath, but implemented locally."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        """
        input shape (b c h w)
        """
        x = x.permute(0, 2, 3, 1).contiguous()  # (b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        input (b h w c)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.SyncBatchNorm(out_dim)

    def forward(self, x):
        """
        action_embedding: B H W C
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # (b c h w)
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (b oh ow oc)
        return x


def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class GeoPriorGen(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(
            1
            - 2
            ** (
                -initial_value
                - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads
            )
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_depth_decay(self, H: int, W: int, depth_grid):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H: int, W: int):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        """
        generate 1d depth decay mask, the result is l*l
        """
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        assert mask.shape[2:] == (W, H, H)
        return mask

    def generate_1d_decay(self, l: int):
        """
        generate 1d decay mask, the result is l*l
        """
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        """
        depth_map: depth patches
        HW_tuple: (H, W)
        H * W == l
        """
        depth_map = F.interpolate(
            depth_map, size=HW_tuple, mode="bilinear", align_corners=False
        )

        if split_or_not:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)

            mask_d_h = self.generate_1d_depth_decay(
                HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1)
            )
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = (
                self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2)
                + self.weight[1] * mask_d_h
            )
            mask_w = (
                self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2)
                + self.weight[1] * mask_d_w
            )

            geo_prior = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])

            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d

            geo_prior = ((sin, cos), mask)

        return geo_prior


class Decomposed_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos, split_or_not=False):
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(
            0, 3, 1, 2, 4
        )  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(
            0, 3, 1, 2, 4
        )  # (b n h w d1)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)

        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class Full_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos, split_or_not=False):
        """
        action_embedding: (b h w c)
        rel_pos: mask: (n l l)
        """
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        assert h * w == mask.size(3)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        vr = vr.flatten(2, 3)
        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = qk_mat + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        """
        input shape: (b h w c)
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class RGBD_Block(nn.Module):
    def __init__(
        self,
        split_or_not: str,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        drop_path=0.0,
        layerscale=False,
        layer_init_values=1e-5,
        init_value=2,
        heads_range=4,
    ):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        if split_or_not:
            self.Attention = Decomposed_GSA(embed_dim, num_heads)
        else:
            self.Attention = Full_GSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        # FFN
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)
        # the function to generate the geometry prior for the current block
        self.Geo = GeoPriorGen(embed_dim, num_heads, init_value, heads_range)

        if layerscale:
            self.gamma_1 = nn.Parameter(
                layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True
            )

    def forward(self, x: torch.Tensor, x_e: torch.Tensor, split_or_not=False):
        x = x + self.cnn_pos_encode(x)
        b, h, w, d = x.size()

        geo_prior = self.Geo((h, w), x_e, split_or_not=split_or_not)
        if self.layerscale:
            x = x + self.drop_path(
                self.gamma_1
                * self.Attention(self.layer_norm1(x), geo_prior, split_or_not)
            )
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.layer_norm2(x)))
        else:
            x = x + self.drop_path(
                self.Attention(self.layer_norm1(x), geo_prior, split_or_not)
            )
            x = x + self.drop_path(self.ffn(self.layer_norm2(x)))
        return x


class BasicLayer(nn.Module):
    """
    A basic RGB-D layer in DFormerv2.
    """

    def __init__(
        self,
        embed_dim,
        out_dim,
        depth,
        num_heads,
        init_value: float,
        heads_range: float,
        ffn_dim=96.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        split_or_not=False,
        downsample: PatchMerging = None,
        use_checkpoint=False,
        layerscale=False,
        layer_init_values=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.split_or_not = split_or_not

        # build blocks
        self.blocks = nn.ModuleList(
            [
                RGBD_Block(
                    split_or_not,
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layerscale,
                    layer_init_values,
                    init_value=init_value,
                    heads_range=heads_range,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, x_e):
        b, h, w, d = x.size()
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(
                    blk, x=x, x_e=x_e, split_or_not=self.split_or_not
                )
            else:
                x = blk(x, x_e, split_or_not=self.split_or_not)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x


class dformerv2(nn.Module):
    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 8, 2],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        projection=1024,
        norm_cfg=None,
        layerscales=[False, False, False, False],
        layer_init_values=1e-6,
        norm_eval=True,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.norm_eval = norm_eval

        # patch embedding
        self.patch_embed = PatchEmbed(
            in_chans=3,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # drop path rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1]
                if (i_layer < self.num_layers - 1)
                else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                split_or_not=(i_layer != 3),
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
            )
            self.layers.append(layer)

        self.extra_norms = nn.ModuleList()
        for i in range(3):
            self.extra_norms.append(nn.LayerNorm(embed_dims[i + 1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            _state_dict = torch.load(pretrained)
            if "model" in _state_dict.keys():
                _state_dict = _state_dict["model"]
            if "state_dict" in _state_dict.keys():
                _state_dict = _state_dict["state_dict"]
            state_dict = OrderedDict()

            for k, v in _state_dict.items():
                if k.startswith("backbone."):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            print("load " + pretrained)
            self.load_state_dict(state_dict, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward(self, x, x_e):
        # rgb input
        x = self.patch_embed(x)
        # depth input
        x_e = x_e[:, 0, :, :].unsqueeze(1)

        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x, x_e)
            if i in self.out_indices:
                if i != 0:
                    x_out = self.extra_norms[i - 1](x_out)
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


def DFormerv2_S(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        heads_ranges=[4, 4, 6, 6],
        **kwargs,
    )
    return model


def DFormerv2_B(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16],
        heads_ranges=[5, 5, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )
    return model


def DFormerv2_L(pretrained=False, **kwargs):
    model = dformerv2(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        heads_ranges=[6, 6, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )
    return model


def check_dformerv2(model_ctor, image_size=(256, 256), batch_size=2, device=None):
    """
    Runs a forward + backward (eval-mode) check and validates tensor shapes.

    Args:
        model_ctor: callable returning an instance of dformerv2 (e.g., DFormerv2_S)
        image_size: (H, W) divisible by 32 is recommended
        batch_size: batch size for the dummy run
        device: torch.device or None (auto: 'cuda' if available else 'cpu')
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = image_size
    torch.manual_seed(0)

    # Instantiate model
    model = model_ctor()
    model.eval()  # important to avoid SyncBatchNorm requiring a process group
    model.to(device)

    # Figure expected channel dims per stage from the model config
    # (embed_dims is fixed in the constructor)
    embed_dims = []
    for layer in model.layers:
        embed_dims.append(layer.embed_dim)

    # Dummy inputs
    x_rgb = torch.randn(batch_size, 3, H, W, device=device, requires_grad=True)
    # Depth with at least 1 channel (the model uses only the first): (B, 1, H, W)
    x_depth = torch.randn(batch_size, 1, H, W, device=device, requires_grad=True)

    # Forward
    with torch.set_grad_enabled(True):
        outs = model(x_rgb, x_depth)  # tuple of 4 feature maps

    assert (
        isinstance(outs, tuple) and len(outs) == 4
    ), f"Expected 4 outputs, got {len(outs)}"
    # Expected spatial sizes by stage:
    # Stage0: /4 (due to PatchEmbed: two stride-2 convs)
    # Stage1: /8  (PatchMerging stride-2)
    # Stage2: /16
    # Stage3: /32
    expected_hw = [
        (H // 4, W // 4),
        (H // 8, W // 8),
        (H // 16, W // 16),
        (H // 32, W // 32),
    ]

    # Validate shapes rigorously
    for i, (y, (eh, ew), ec) in enumerate(zip(outs, expected_hw, embed_dims)):
        assert y.dim() == 4, f"Stage {i} output must be 4D (B,C,H,W), got {y.shape}"
        B, C, h, w = y.shape
        assert B == batch_size, f"Stage {i}: wrong batch {B} != {batch_size}"
        assert C == ec, f"Stage {i}: wrong channels {C} != {ec}"
        assert h == eh and w == ew, f"Stage {i}: wrong spatial {(h,w)} != {(eh,ew)}"
        print(f"Stage {i}: OK  ->  {tuple(y.shape)}")

    # Simple scalar loss over all outputs (sum of means) to test backward
    loss = sum(y.mean() for y in outs)
    loss.backward()

    # Check that some gradients flowed
    some_param_with_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert some_param_with_grad, "No gradients found — something is off."
    print("Backward pass: OK (non-zero grads detected)")

    print("All checks passed ✔️")
    return outs


# ---- Run checks for S / B / L variants ----
if __name__ == "__main__":
    # S
    print("\nChecking DFormerv2_S...")
    _ = check_dformerv2(
        lambda: dformerv2(
            out_indices=(0, 1, 2, 3),
            embed_dims=[64, 128, 256, 512],
            depths=[3, 4, 6, 3],  # trimmed for speed
            num_heads=[4, 4, 8, 16],
            heads_ranges=[4, 4, 6, 6],
            drop_path_rate=0.0,
        )
    )

    # B
    print("\nChecking DFormerv2_B...")
    _ = check_dformerv2(
        lambda: dformerv2(
            out_indices=(0, 1, 2, 3),
            depths=[2, 4, 8, 2],  # trimmed for speed
            drop_path_rate=0.0,
            num_heads=[4, 4, 8, 16],
            heads_ranges=[4, 4, 6, 6],
            embed_dims=[64, 128, 256, 512],
        )
    )

    # L
    print("\nChecking DFormerv2_L...")
    _ = check_dformerv2(
        lambda: dformerv2(
            out_indices=(0, 1, 2, 3),
            depths=[2, 4, 8, 2],  # trimmed for speed
            drop_path_rate=0.0,
            num_heads=[4, 4, 8, 16],
            heads_ranges=[4, 4, 6, 6],
            embed_dims=[64, 128, 256, 512],
        )
    )
