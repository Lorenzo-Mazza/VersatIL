import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSoftmax2d(nn.Module):
    """
    Spatial soft-argmax pooling layer.
    Given feature maps X in R^{B action_embedding C action_embedding H action_embedding W}, returns per-channel expected
    coordinates [E_x, E_y] — shape (B, 2*C).

    Args:
        normalize (bool): if True, coordinates are in [-1, 1] (default).
                          if False, coordinates are in pixel units: action_embedding in [0, W-1], y in [0, H-1].
        temperature (float): softmax temperature τ (>0); lower -> sharper.
        learnable_temperature (bool): if True, τ is a learnable parameter.
        eps (float): numerical safety for τ.
    """
    def __init__(self,
                 normalize: bool = True,
                 temperature: float = 1.0,
                 learnable_temperature: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

        tau = torch.tensor(temperature)
        self.temperature = nn.Parameter(tau, requires_grad=learnable_temperature)

        # simple cache to avoid rebuilding grids when H,W repeat
        self._grid_cache = {}

    def _get_grids(self, h: int, w: int, device, dtype):
        """
        Returns (grid_x, grid_y) each of shape (1, 1, H*W) for broadcast.
        """
        key = (h, w, device, dtype)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        if self.normalize:
            xs = torch.linspace(-1.0,  1.0, w, device=device, dtype=dtype)
            ys = torch.linspace(-1.0,  1.0, h, device=device, dtype=dtype)
        else:
            xs = torch.linspace(0.0,  w - 1.0, w, device=device, dtype=dtype)
            ys = torch.linspace(0.0,  h - 1.0, h, device=device, dtype=dtype)

        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # yy: (H,W) y-coords, xx: (H,W) action_embedding-coords
        grid_x = xx.reshape(1, 1, h * w)                # (1,1,HW)
        grid_y = yy.reshape(1, 1, h * w)
        self._grid_cache[key] = (grid_x, grid_y)
        return grid_x, grid_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        action_embedding: (B, C, H, W)
        returns: (B, 2*C) with concatenated [E_x, E_y] per channel.
        """
        assert x.ndim == 4, f"Expected (B,C,H,W), got {tuple(x.shape)}"
        B, C, H, W = x.shape
        tau = self.temperature.abs() + self.eps      # ensure positive τ

        # softmax over flattened spatial dimension
        x_flat = x.reshape(B, C, H * W) / tau
        attn = F.softmax(x_flat, dim=-1)             # (B, C, HW)

        # broadcast coordinate grids to (B, C, HW)
        gx, gy = self._get_grids(H, W, x.device, x.dtype)  # (1,1,HW)
        gx = gx.expand(B, C, -1)                    # (B, C, HW)
        gy = gy.expand(B, C, -1)

        # expected coordinates
        ex = (attn * gx).sum(dim=-1)                # (B, C)
        ey = (attn * gy).sum(dim=-1)                # (B, C)

        return torch.cat([ex, ey], dim=1)           # (B, 2C)


if __name__ == "__main__":
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)
    pool = SpatialSoftmax2d(normalize=True, temperature=1.0)
    y = pool(x)
    assert y.shape == (B, 2*C)
    print("SpatialSoftmax2d OK:", y.shape)
