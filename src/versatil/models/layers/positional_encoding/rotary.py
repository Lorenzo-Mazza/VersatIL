import torch
from torch import nn


class RotaryPositionalEncoding(nn.Module):
    """Base class for rotary positional encoding, from https://arxiv.org/pdf/2104.09864."""

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        base_frequency: float = 10000.0,
        learnable_frequencies: bool = False,
    ):
        """Initialize rotary positional encoding frequencies.

        Args:
            embedding_dimension: Full model embedding dimension.
            num_heads: Number of attention heads.
            base_frequency: Base frequency for geometric spacing.
            learnable_frequencies: Whether frequency bands are trainable.

        Raises:
            ValueError: If attention dimensions or frequencies are invalid.
        """
        super().__init__()
        if embedding_dimension <= 0:
            raise ValueError(
                f"embedding_dimension must be positive, got {embedding_dimension}."
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if embedding_dimension % num_heads != 0:
            raise ValueError(
                f"embedding_dimension ({embedding_dimension}) must be divisible "
                f"by num_heads ({num_heads})."
            )
        if base_frequency <= 0.0:
            raise ValueError(f"base_frequency must be positive, got {base_frequency}.")
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.head_dimension = embedding_dimension // num_heads

        if self.head_dimension % 2 != 0:
            raise ValueError("head_dimension must be even for rotary encoding")

        frequencies = self._compute_frequencies(
            dimension=self.head_dimension, base_frequency=base_frequency
        )
        self.register_parameter(
            "frequencies",
            nn.Parameter(frequencies, requires_grad=learnable_frequencies),
        )

    @staticmethod
    def _compute_frequencies(dimension: int, base_frequency: float) -> torch.Tensor:
        """Compute frequency bands with interleaved repetition for ``apply_rotation``.

        Args:
            dimension: Embedding dimension per head.
            base_frequency: Base frequency for computation.

        Returns:
            Frequency tensor of shape (dimension,) with interleaved pairs.
        """
        half_dimension = dimension // 2
        exponents = torch.arange(half_dimension, dtype=torch.float32) / half_dimension
        frequencies = 1.0 / (base_frequency**exponents)
        frequencies = frequencies.repeat_interleave(2)
        result: torch.Tensor = frequencies
        return result

    @staticmethod
    def _compute_frequencies_half(
        dimension: int, base_frequency: float
    ) -> torch.Tensor:
        """Compute frequency bands without repetition for ``apply_rotation_half``.

        Returns (dimension // 2,) frequencies matching the Gemma/LLaMA split-half
        convention where cos/sin are broadcast across the full head dimension.

        Args:
            dimension: Embedding dimension per head.
            base_frequency: Base frequency for computation.

        Returns:
            Frequency tensor of shape (dimension // 2,).
        """
        half_dimension = dimension // 2
        exponents = torch.arange(half_dimension, dtype=torch.float32) / half_dimension
        result: torch.Tensor = 1.0 / (base_frequency**exponents)
        return result

    @staticmethod
    def apply_rotation(
        tensor: torch.Tensor, sine: torch.Tensor, cosine: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary transformation using interleaved even/odd convention.

        Pairs even and odd indices: ``[-odd, even]`` interleaved back.
        This is the original RoFormer/VersatIL convention.

        Args:
            tensor: Input (B, num_heads, L, head_dim).
            sine: Sine components matching sequence + head_dim shape.
            cosine: Cosine components matching sequence + head_dim shape.

        Returns:
            Rotated tensor of same shape.
        """
        even_indices = tensor[..., 0::2]
        odd_indices = tensor[..., 1::2]
        rotated_pairs = torch.stack([-odd_indices, even_indices], dim=-1)
        rotated_pairs = rotated_pairs.flatten(-2)
        rotated_tensor = (tensor * cosine) + (rotated_pairs * sine)
        return rotated_tensor

    @staticmethod
    def apply_rotation_half(
        tensor: torch.Tensor, sine: torch.Tensor, cosine: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary transformation using split-half convention.

        Splits the last dimension in half: ``[-second_half, first_half]``.

        Args:
            tensor: Input (B, num_heads, L, head_dim).
            sine: Sine components matching sequence + head_dim shape.
            cosine: Cosine components matching sequence + head_dim shape.

        Returns:
            Rotated tensor of same shape.
        """
        first_half = tensor[..., : tensor.shape[-1] // 2]
        second_half = tensor[..., tensor.shape[-1] // 2 :]
        rotated = torch.cat([-second_half, first_half], dim=-1)
        return (tensor * cosine) + (rotated * sine)


class RotaryPositionalEncoding1D(RotaryPositionalEncoding):
    """Rotary positional encoding for 1D sequences."""

    def compute_rotation_components(
        self, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes sine and cosine components for 1D sequence positions.

        Args:
            seq_len: Sequence length.

        Returns:
            Tuple of (sine, cosine) tensors of shape (seq_len, head_dim).
        """
        device = self.frequencies.device
        position_indices = torch.arange(seq_len, device=device)
        angles = position_indices[:, None] * self.frequencies[None, :]
        sine_components = torch.sin(angles)
        cosine_components = torch.cos(angles)
        return sine_components, cosine_components


class RasterRotaryPositionalEncoding2D(RotaryPositionalEncoding):
    """Rotary encoding over flattened raster positions of a 2D grid.

    Matches the DFormerv2 reference convention: every token is rotated by its
    flattened index ``row * width + column`` with a single frequency band
    spanning the full head dimension, spaced as
    ``1 / base_frequency ** linspace(0, 1, head_dim // 2)`` with the endpoint
    included. Pretrained DFormerv2 checkpoints require exactly this scheme.
    """

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        base_frequency: float = 10000.0,
        learnable_frequencies: bool = False,
    ):
        """Initialize raster rotary encoding with endpoint-spaced frequencies.

        Args:
            embedding_dimension: Full model embedding dimension.
            num_heads: Number of attention heads.
            base_frequency: Base frequency for geometric spacing.
            learnable_frequencies: Whether frequency bands are trainable.
        """
        super().__init__(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            base_frequency=base_frequency,
            learnable_frequencies=learnable_frequencies,
        )
        half_dimension = self.head_dimension // 2
        exponents = torch.linspace(0, 1, half_dimension)
        frequencies = 1.0 / (base_frequency**exponents)
        with torch.no_grad():
            self.frequencies.copy_(frequencies.repeat_interleave(2))

    def compute_rotation_components(
        self, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes sine and cosine components for raster grid positions.

        Args:
            height: Grid height.
            width: Grid width.

        Returns:
            Tuple of (sine, cosine) tensors of shape (H, W, head_dim).
        """
        device = self.frequencies.device
        position_indices = torch.arange(height * width, device=device)
        angles = position_indices[:, None] * self.frequencies[None, :]
        sine_components = torch.sin(angles).reshape(height, width, -1)
        cosine_components = torch.cos(angles).reshape(height, width, -1)
        return sine_components, cosine_components


class RotaryPositionalEncoding2D(RotaryPositionalEncoding):
    """Rotary positional encoding for 2D spatial grids."""

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        base_frequency: float = 10000.0,
        learnable_frequencies: bool = False,
    ):
        """Initialize rotary positional encoding for 2D grids.

        Args:
            embedding_dimension: Full model embedding dimension.
            num_heads: Number of attention heads.
            base_frequency: Base frequency for geometric spacing.
            learnable_frequencies: Whether frequency bands are trainable.

        Raises:
            ValueError: If per-axis head dimensions are invalid.
        """
        super().__init__(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            base_frequency=base_frequency,
            learnable_frequencies=learnable_frequencies,
        )
        self.half_head_dim = self.head_dimension // 2
        if self.half_head_dim % 2 != 0:
            raise ValueError("half_head_dimension must be even for 2D rotary encoding")
        freq_set = self._compute_frequencies(
            self.half_head_dim, base_frequency=base_frequency
        )
        with torch.no_grad():
            self.frequencies.copy_(torch.cat([freq_set, freq_set]))

    def compute_rotation_components(
        self, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes sine and cosine components for 2D grid positions.

        Args:
            height: Grid height.
            width: Grid width.

        Returns:
            Tuple of (sine, cosine) tensors of shape (H, W, head_dim).
        """
        # Use frequencies' device - parameters are automatically moved when module.to(device) is called
        device = self.frequencies.device
        # y positions (rows), x positions (cols)
        y_pos = torch.arange(height, device=device)[:, None].repeat(1, width)
        x_pos = torch.arange(width, device=device)[None, :].repeat(height, 1)
        # Split frequencies: first half for y, second for x
        freq_y = self.frequencies[: self.half_head_dim]
        freq_x = self.frequencies[self.half_head_dim :]
        angles_y = y_pos[..., None] * freq_y[None, None, :]
        angles_x = x_pos[..., None] * freq_x[None, None, :]
        angles = torch.cat([angles_y, angles_x], dim=-1)
        sine_components = torch.sin(angles)
        cosine_components = torch.cos(angles)
        return sine_components, cosine_components
