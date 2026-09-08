"""X86 Inductor backend for PT2E quantized operator lowering."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from functools import partial

import torch._inductor.config as inductor_config
from torch import fx, nn
from torchao.quantization.pt2e.quantizer import Quantizer
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

from versatil.quantization.constants import FXNodeOp, PT2EBackendName
from versatil.quantization.pt2e.backends.base import BasePT2EBackend

_CUDA_VISIBLE_DEVICES_KEY = "CUDA_VISIBLE_DEVICES"
_TORCHINDUCTOR_FREEZING_KEY = "TORCHINDUCTOR_FREEZING"
_SOURCE_FN_STACK = "source_fn_stack"
_NN_MODULE_STACK = "nn_module_stack"
_EXPORTED_SELF_PREFIX = "L['self']."
_PATTERN_MODULE_TYPES = {
    f"{module_type.__module__}.{module_type.__qualname__}": module_type
    for module_type in (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.ReLU,
        nn.LeakyReLU,
        nn.Tanh,
        nn.GELU,
    )
}


def _nodes_match_module_path(nodes: list[fx.Node], module_path: str) -> bool:
    """Check whether every pattern node belongs to a module or its descendants.

    Args:
        nodes: Operator nodes belonging to one candidate quantization pattern.
        module_path: Policy-relative module path; an empty path selects the root.

    Returns:
        Whether all nodes have a matching exported module scope.

    Note:
        Direct calls to module helper methods can omit the parent module's own
        stack entry. Child entries retain their qualified paths.
    """
    if module_path == "":
        return True
    descendant_prefix = f"{module_path}."
    return all(
        any(
            (normalized := name.removeprefix(_EXPORTED_SELF_PREFIX)) == module_path
            or normalized.startswith(descendant_prefix)
            for name, _ in node.meta.get(_NN_MODULE_STACK, {}).values()
        )
        for node in nodes
    )


class _PerCallX86InductorQuantizer(X86InductorQuantizer):
    """Match x86 patterns by module hierarchy and shared-layer invocation.

    Note:
        Exported ``nn_module_stack`` keys identify individual calls to a shared
        layer. This adapter copies those identifiers into ``source_fn_stack`` for
        TorchAO's linear and convolution pattern matcher. Each denoising step then
        forms a separate source group while sharing the original layer weights.
        Module targets include descendant paths reached through helper methods.
    """

    def annotate(self, model: fx.GraphModule) -> fx.GraphModule:
        """Apply hierarchical module targets before operator and global settings.

        Args:
            model: Exported graph whose operators receive quantization settings.

        Returns:
            Graph with module, operator and global quantization annotations.

        Note:
            TorchAO preserves annotations assigned by earlier configuration passes.
            Its inherited pass handles operator settings, global settings and output
            propagation after these module selections.
        """
        for module_path, quantization_config in self.module_name_qconfig.items():
            self._annotate_with_config(
                model=model,
                quantization_config=quantization_config,
                filter_fn=partial(_nodes_match_module_path, module_path=module_path),
            )
        return super().annotate(model=model)

    def transform_for_annotation(self, model: fx.GraphModule) -> fx.GraphModule:
        """Fill missing source metadata from exported layer invocation identifiers.

        Args:
            model: Exported graph before quantization observers are inserted.

        Returns:
            The same graph with source metadata added to recognized layer calls.
        """
        for node in model.graph.nodes:
            if node.op != FXNodeOp.CALL_FUNCTION.value or node.meta.get(
                _SOURCE_FN_STACK
            ):
                continue
            module_stack = node.meta.get(_NN_MODULE_STACK)
            if not module_stack:
                continue
            invocation, (_, module_type) = next(reversed(module_stack.items()))
            source_type = (
                _PATTERN_MODULE_TYPES.get(module_type)
                if isinstance(module_type, str)
                else module_type
            )
            if source_type not in _PATTERN_MODULE_TYPES.values():
                continue
            node.meta[_SOURCE_FN_STACK] = [(invocation, source_type)]
        return model


class X86InductorBackend(BasePT2EBackend):
    """X86 Inductor backend for PT2E quantization and lowering."""

    @property
    def name(self) -> str:
        """Serialized PT2E backend name."""
        return PT2EBackendName.X86_INDUCTOR.value

    def __init__(
        self,
        is_dynamic: bool = False,
        is_qat: bool = False,
        reduce_range: bool = False,
    ) -> None:
        """Initialize X86 Inductor backend configuration.

        Args:
            is_dynamic: Use dynamic activation quantization.
            is_qat: Use quantization-aware training observers.
            reduce_range: Reduce quantization range for older CPUs
                without VNNI.
        """
        self._is_dynamic = is_dynamic
        self._is_qat = is_qat
        self._reduce_range = reduce_range

    @property
    def is_dynamic(self) -> bool:
        """Whether this backend uses dynamic activation quantization."""
        return self._is_dynamic

    @property
    def is_qat(self) -> bool:
        """Whether this backend uses QAT observer configuration."""
        return self._is_qat

    @property
    def supported_device_types(self) -> tuple[str, ...]:
        """X86 inductor only supports CPU inference."""
        return ("cpu",)

    def create_quantizer(self, module_path: str) -> Quantizer:
        """Create an X86InductorQuantizer targeting a specific module.

        Args:
            module_path: Dotted path to the target submodule.
                Empty string means global (whole model).

        Returns:
            Configured X86InductorQuantizer.
        """
        quantizer = _PerCallX86InductorQuantizer()
        config = get_default_x86_inductor_quantization_config(
            is_dynamic=self._is_dynamic,
            is_qat=self._is_qat,
            reduce_range=self._reduce_range,
        )
        if module_path == "":
            quantizer.set_global(config)
        else:
            quantizer.set_module_name_qconfig(module_path, config)
        return quantizer

    @contextmanager
    def environment_context(self) -> Generator[None]:
        """Enable CPU compilation, constant weights and the C++ wrapper temporarily.

        Note:
            Inductor freezing makes weights available as constants for quantized
            kernel selection. The configuration applies to the current process;
            environment variables provide the same settings to compiler processes.
            Original configuration and environment values are restored on exit.

        Yields:
            None.
        """
        saved = {
            _CUDA_VISIBLE_DEVICES_KEY: os.environ.get(_CUDA_VISIBLE_DEVICES_KEY),
            _TORCHINDUCTOR_FREEZING_KEY: os.environ.get(_TORCHINDUCTOR_FREEZING_KEY),
        }
        saved_cpp_wrapper = inductor_config.cpp_wrapper
        saved_freezing = inductor_config.freezing
        os.environ[_CUDA_VISIBLE_DEVICES_KEY] = ""
        os.environ[_TORCHINDUCTOR_FREEZING_KEY] = "1"
        inductor_config.cpp_wrapper = True
        inductor_config.freezing = True
        try:
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            inductor_config.cpp_wrapper = saved_cpp_wrapper
            inductor_config.freezing = saved_freezing

    def activate_environment(self) -> None:
        """Enable CPU compilation, constant weights and the C++ wrapper globally."""
        os.environ[_CUDA_VISIBLE_DEVICES_KEY] = ""
        os.environ[_TORCHINDUCTOR_FREEZING_KEY] = "1"
        inductor_config.cpp_wrapper = True
        inductor_config.freezing = True
