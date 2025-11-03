"""Constants and enums for training configuration."""

from enum import Enum


class PrecisionType(str, Enum):
    """PyTorch Lightning precision types.

    See: https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision
    """

    FP32 = "32"  # Full 32-bit floating point
    FP16_MIXED = "16-mixed"  # Mixed precision with float16
    BF16_MIXED = "bf16-mixed"  # Mixed precision with bfloat16
    FP16_TRUE = "16-true"  # Pure float16 (not recommended)
    BF16_TRUE = "bf16-true"  # Pure bfloat16 (not recommended)
    FP64 = "64"  # Double precision (rarely needed)


class Float32MatmulPrecision(str, Enum):
    """Float32 matrix multiplication precision for Tensor Cores.

    Controls the precision of float32 matrix multiplications on GPUs with Tensor Cores.

    See: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    """

    HIGHEST = "highest"  # FP32 (no Tensor Cores, most precise, slowest)
    HIGH = "high"  # TF32 + FP32 fallback (good balance)
    MEDIUM = "medium"  # TF32 (recommended, ~8x faster, minimal precision loss)