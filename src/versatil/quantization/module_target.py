"""Torch Module-level targets for quantization workflows."""

from torchao.quantization.quant_api import AOBaseConfig

from versatil.quantization.pt2e.backends.base import BasePT2EBackend


class QuantizationModuleTarget:
    """Base class for a quantized policy submodule target."""

    def __init__(self, module_path: str) -> None:
        """Initialize the target.

        Args:
            module_path: Dotted path to the target module, or ``""`` for root.
        """
        self.module_path = module_path

    @property
    def label(self) -> str:
        """Return a readable module label for logs and errors.

        Returns:
            ``module_path`` for submodule targets, or ``"(root)"`` for the
            full-policy target.
        """
        return self.module_path or "(root)"

    def contains_module(self, module_name: str) -> bool:
        """Return whether a named module is inside this target.

        Args:
            module_name: Fully qualified module name from ``named_modules()``.

        Returns:
            Whether ``module_name`` is the target module itself or a child of
            the target module. The root target contains every module.
        """
        if self.module_path == "":
            return True
        return module_name == self.module_path or module_name.startswith(
            self.module_path + "."
        )

    def overlaps(self, other: "QuantizationModuleTarget") -> bool:
        """Return whether two targets can select the same submodule.

        Args:
            other: Target to compare against this target.

        Returns:
            Whether either target is root, both targets are the same path, or
            one target is nested under the other.
        """
        if self.module_path == "" or other.module_path == "":
            return True
        return (
            self.module_path == other.module_path
            or self.module_path.startswith(other.module_path + ".")
            or other.module_path.startswith(self.module_path + ".")
        )


class EagerQuantizationModuleTarget(QuantizationModuleTarget):
    """Target using an eager quantization config."""

    def __init__(
        self,
        module_path: str,
        quantize_config: AOBaseConfig,
    ) -> None:
        """Initialize an eager quantization target.

        Args:
            module_path: Dotted path to the target module, or ``""`` for root.
            quantize_config: torchao eager quantization config applied to this
                target.
        """
        super().__init__(module_path=module_path)
        self.quantize_config = quantize_config


class PT2EQuantizationModuleTarget(QuantizationModuleTarget):
    """Target using a PyTorch 2 Export backend quantizer config."""

    def __init__(
        self,
        module_path: str,
        pt2e_backend: BasePT2EBackend,
    ) -> None:
        """Initialize a PT2E quantization target.

        Args:
            module_path: Dotted path to the target module, or ``""`` for root.
            pt2e_backend: PT2E backend that creates the quantizer for this
                target.
        """
        super().__init__(module_path=module_path)
        self.pt2e_backend = pt2e_backend

    @property
    def needs_calibration(self) -> bool:
        """Return whether this target requires calibration batches.

        Returns:
            ``True`` for static PT2E backends and ``False`` for dynamic PT2E
            backends.
        """
        return not self.pt2e_backend.is_dynamic
