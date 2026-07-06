"""Constants and enums for training configuration."""

from enum import StrEnum

import torch

# Reserved parameter-group name for optimizer parameters not matched by any
# explicit ``params_pattern``. Cannot be used as a custom group name."""
OPTIMIZER_UNMATCHED_GROUPS_NAME = "unmatched"


class PrecisionType(StrEnum):
    """PyTorch Lightning precision types.

    See: https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision
    """

    INT8 = "8"  # 8-bit precision (only for quantized inference)
    FP32 = "32"  # Full 32-bit floating point
    FP16_MIXED = "16-mixed"  # Mixed precision with float16
    BF16_MIXED = "bf16-mixed"  # Mixed precision with bfloat16
    FP16_TRUE = "16-true"  # Pure float16 (not recommended)
    BF16_TRUE = "bf16-true"  # Pure bfloat16 (not recommended)
    FP64 = "64"  # Double precision (rarely needed)

    @classmethod
    def _missing_(cls, value: str) -> "PrecisionType | None":
        """Accept Lightning's canonical precision strings.

        ``Trainer.precision`` normalizes ``32`` to ``"32-true"`` and ``64``
        to ``"64-true"``; map those back onto the enum members.
        """
        aliases = {
            "32-true": cls.FP32,
            "64-true": cls.FP64,
            "16": cls.FP16_MIXED,
            "bf16": cls.BF16_MIXED,
        }
        return aliases.get(value)

    def get_model_dtype(self) -> torch.dtype:
        """Get the dtype to convert model parameters to for this precision type.

        For mixed precision types, converts to the lower precision dtype to avoid
        dtype mismatch errors during inference (e.g., bfloat16 input vs float32 bias).

        Returns:
            torch.dtype to convert model to

        Raises:
            NotImplementedError: For INT8 precision (requires specialized quantization)
        """
        if self == PrecisionType.INT8:
            raise NotImplementedError(
                "INT8 precision requires post-training quantization. "
                "Use the versatil.quantization module or the post_training_compress endpoint."
            )
        dtype_map = {
            PrecisionType.FP32: torch.float32,
            PrecisionType.FP16_MIXED: torch.float16,
            PrecisionType.BF16_MIXED: torch.bfloat16,
            PrecisionType.FP16_TRUE: torch.float16,
            PrecisionType.BF16_TRUE: torch.bfloat16,
            PrecisionType.FP64: torch.float64,
        }
        return dtype_map[self]

    def is_mixed(self) -> bool:
        """Check if this precision type autocasts compute around float32 weights.

        Returns:
            True for the mixed half-precision types, where trainable parameters
            must stay in float32 storage so optimizer updates are not rounded
            away by the low-precision dtype.
        """
        return self in (
            PrecisionType.FP16_MIXED,
            PrecisionType.BF16_MIXED,
        )

    def autocast(self, device_type: str) -> torch.autocast:
        """Return an autocast context matching this precision.

        Enabled only for the mixed half-precision types, where forward passes
        outside the Lightning training loop must reproduce the training-time
        autocast over mixed float32/low-precision parameters. For all other
        precisions the returned context is a no-op.

        Args:
            device_type: Device type string for ``torch.autocast`` (e.g.
                ``"cuda"`` or ``"cpu"``).
        """
        return torch.autocast(
            device_type=device_type,
            dtype=self.get_model_dtype() if self.is_mixed() else None,
            enabled=self.is_mixed(),
        )

    def should_convert_model(self) -> bool:
        """Check if model should be converted to a specific dtype for this precision.

        Returns:
            True if model should be converted (for mixed/true half precision types)
        """
        return self in (
            PrecisionType.FP16_MIXED,
            PrecisionType.BF16_MIXED,
            PrecisionType.FP16_TRUE,
            PrecisionType.BF16_TRUE,
        )


class CheckpointFilename(StrEnum):
    """Standard filenames within a training checkpoint directory."""

    CONFIG = "config.yaml"
    TOKENIZER_DIR = "tokenizer"
    DEFAULT_CHECKPOINT = "last.ckpt"


class CheckpointKey(StrEnum):
    """Keys used within checkpoint state dicts and normalizer params."""

    STATE_DICT = "state_dict"
    INPUT_STATS = "input_stats"
    RAW_POLICY_STATE_DICT = "raw_policy_state_dict"


class CompileMode(StrEnum):
    """torch.compile optimization modes.

    See: https://pytorch.org/docs/stable/generated/torch.compile.html
    """

    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"
    MAX_AUTOTUNE_NO_CUDAGRAPHS = "max-autotune-no-cudagraphs"


class Float32MatmulPrecision(StrEnum):
    """Float32 matrix multiplication precision for Tensor Cores.

    Controls the precision of float32 matrix multiplications on GPUs with Tensor Cores.

    See: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    """

    HIGHEST = "highest"  # FP32 (no Tensor Cores, most precise, slowest)
    HIGH = "high"  # TF32 + FP32 fallback (good balance)
    MEDIUM = "medium"  # TF32 (recommended, ~8x faster, minimal precision loss)
