import torch.nn as nn


class Downsample1d(nn.Module):
    """Strided 1D convolution halving the temporal length."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        """Downsample (B, C, T) to (B, C, T/2)."""
        return self.conv(x)


class Upsample1d(nn.Module):
    """Transposed 1D convolution doubling the temporal length."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        """Upsample (B, C, T) to (B, C, 2T)."""
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d followed by group normalization and Mish activation."""

    def __init__(self, input_channels, output_channels, kernel_size, num_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                input_channels, output_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(num_groups, output_channels),
            nn.Mish(),
        )

    def forward(self, x):
        """Apply convolution, normalization, and activation to (B, C, T)."""
        return self.block(x)
