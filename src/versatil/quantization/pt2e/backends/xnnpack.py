"""XNNPACK backend for PT2E quantization."""

import importlib
from collections.abc import Generator
from contextlib import contextmanager
from types import ModuleType

import torch
from torch.fx import Node
from torchao.quantization.pt2e.quantizer import Quantizer

from versatil.quantization.constants import PT2EBackendName
from versatil.quantization.pt2e.backends.base import BasePT2EBackend

_XNNPACK_QUANTIZER_MODULE = "executorch.backends.xnnpack.quantizer.xnnpack_quantizer"
_XNNPACK_OPERATOR_TARGETS = (
    torch.ops.aten.linear.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
)


class XNNPACKPT2EBackend(BasePT2EBackend):
    """XNNPACK PT2E backend for ExecuTorch deployment."""

    def __init__(
        self,
        is_dynamic: bool = False,
        is_qat: bool = False,
        is_per_channel: bool = True,
    ) -> None:
        """Initialize XNNPACK PT2E quantization settings.

        Args:
            is_dynamic: Use dynamic activation quantization.
            is_qat: Use quantization-aware training observers.
            is_per_channel: Use per-channel symmetric weight quantization.
        """
        self._is_dynamic = is_dynamic
        self._is_qat = is_qat
        self._is_per_channel = is_per_channel

    @property
    def name(self) -> str:
        """Serialized PT2E backend name."""
        return PT2EBackendName.XNNPACK.value

    @property
    def is_dynamic(self) -> bool:
        """Whether this backend uses dynamic activation quantization."""
        return self._is_dynamic

    @property
    def is_qat(self) -> bool:
        """Whether this backend uses QAT observer configuration."""
        return self._is_qat

    @property
    def is_per_channel(self) -> bool:
        """Whether this backend uses per-channel weight quantization."""
        return self._is_per_channel

    @property
    def supported_device_types(self) -> tuple[str, ...]:
        """XNNPACK ExecuTorch artifacts run on CPU."""
        return ("cpu",)

    def create_quantizer(self, module_path: str) -> Quantizer:
        """Create an XNNPACK quantizer targeting a module path.

        Args:
            module_path: Dotted path to the target submodule.
                Empty string means global quantization.

        Returns:
            Configured XNNPACK quantizer.
        """
        xnnpack_quantizer = _load_xnnpack_quantizer_module()
        quantizer = xnnpack_quantizer.XNNPACKQuantizer()
        config = xnnpack_quantizer.get_symmetric_quantization_config(
            is_per_channel=self._is_per_channel,
            is_qat=self._is_qat,
            is_dynamic=self._is_dynamic,
        )
        for operator_target in _XNNPACK_OPERATOR_TARGETS:
            quantizer.set_operator_type(operator_target, config)
        quantizer.set_filter_function(_XNNPACKNodeFilter(module_path=module_path))
        return quantizer

    @contextmanager
    def environment_context(self) -> Generator[None]:
        """Yield without changing process environment."""
        yield

    def activate_environment(self) -> None:
        """No-op because XNNPACK deployment does not use torch.compile env."""


def _load_xnnpack_quantizer_module() -> ModuleType:
    """Load ExecuTorch XNNPACK quantizer module on demand."""
    return importlib.import_module(
        _XNNPACK_QUANTIZER_MODULE
    )  # This avoids a hard dependency on executorch for the entire versatil package, only requiring it when this backend is used.


class _XNNPACKNodeFilter:
    """Callable node filter for XNNPACK PT2E annotation."""

    def __init__(self, module_path: str) -> None:
        """Initialize the filter.

        Args:
            module_path: Dotted target submodule path. Empty string targets the
                root policy.
        """
        self._module_path = module_path

    def __call__(self, node: Node) -> bool:
        """Return whether the XNNPACK quantizer should consider a node.

        Args:
            node: FX node considered by the XNNPACK quantizer.

        Returns:
            Whether the node has a floating tensor output and belongs to the
            configured module scope.
        """
        return _node_outputs_float_tensor(node=node) and _node_matches_module_path(
            node=node,
            module_path=self._module_path,
        )


def _node_outputs_float_tensor(node: Node) -> bool:
    """Return whether a node output is safe for activation observers.

    Args:
        node: FX node considered by the XNNPACK quantizer.

    Returns:
        ``False`` for tensor outputs with non-floating dtypes, otherwise
        ``True``. Nodes without tensor metadata are left to the backend
        quantizer's own pattern checks.
    """
    value = node.meta.get("val")
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return True
    return dtype.is_floating_point


def _node_matches_module_path(node: Node, module_path: str) -> bool:
    """Return whether a node belongs to the requested module path.

    Args:
        node: FX node considered by the XNNPACK quantizer.
        module_path: Dotted target submodule path. Empty string targets root.

    Returns:
        Whether the node belongs to the requested module scope.
    """
    if module_path == "":
        return True
    module_path_prefix = f"{module_path}."
    for module_name, _module_type in node.meta.get("nn_module_stack", {}).values():
        normalized_name = _normalize_exported_module_path(module_name=module_name)
        if normalized_name == module_path:
            return True
        if normalized_name.startswith(module_path_prefix):
            return True
    return False


def _normalize_exported_module_path(module_name: str) -> str:
    """Normalize torch.export module names to policy-relative paths.

    Args:
        module_name: Module name stored in FX node metadata.

    Returns:
        Module path without the exported ``L['self'].`` prefix.
    """
    exported_self_prefix = "L['self']."
    if module_name.startswith(exported_self_prefix):
        return module_name[len(exported_self_prefix) :]
    return module_name
