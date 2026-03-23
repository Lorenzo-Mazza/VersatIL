"""Quantization config strategies for torchao integration.

`PT2EStrategy`: graph-level quantization with operator fusion via torch.export.
`QuantizeApiStrategy`: eager mode quantization via `torchao` `quantize_()`.
"""

from torchao.quantization.quant_api import AOBaseConfig

from versatil.quantization.backends.base import BasePT2EBackend


class PT2EStrategy:
    """PT2E quantization strategy. Graph-level with operator fusion."""

    def __init__(self, pt2e_backend: BasePT2EBackend) -> None:
        """Initialize with a PT2E backend.

        Args:
            pt2e_backend: Backend providing quantization config and
                environment context.
        """
        self.pt2e_backend = pt2e_backend

    @property
    def needs_calibration(self) -> bool:
        """Static PT2E requires calibration, dynamic does not."""
        return not self.pt2e_backend.is_dynamic


class QuantizeApiStrategy:
    """quantize_() API strategy. Eager mode, no operator fusion."""

    def __init__(self, quantize_config: AOBaseConfig) -> None:
        """Initialize with a torchao AOBaseConfig instance.

        Args:
            quantize_config: torchao quantization config.
        """
        self.quantize_config = quantize_config
