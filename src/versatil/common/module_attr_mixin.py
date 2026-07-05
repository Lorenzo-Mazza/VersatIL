"""Mixin for ``nn.Module`` device and dtype introspection."""

import torch
import torch.nn as nn


class ModuleAttrMixin(nn.Module):
    """Expose module-level ``device`` and ``dtype`` properties.

    A non-persistent reference buffer lets stateless and lazily initialized
    modules track ``.to(...)`` calls without adding trainable parameters or
    polluting checkpoints.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "_module_attr_reference",
            torch.empty(0),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        """Return the module's current device."""
        return self._module_attr_reference.device

    @device.setter
    def device(self, device: torch.device | str) -> None:
        """Move the module to ``device``."""
        self.to(device=torch.device(device))

    @property
    def dtype(self) -> torch.dtype:
        """Return the module's current floating-point dtype."""
        return self._module_attr_reference.dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype) -> None:
        """Cast the module to ``dtype``."""
        self.to(dtype=dtype)
