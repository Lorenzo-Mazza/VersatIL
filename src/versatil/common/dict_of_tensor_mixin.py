"""Mixin for modules that store and load nested ``ParameterDict`` state."""

from typing import cast

import torch
import torch.nn as nn

from versatil.common.module_attr_mixin import ModuleAttrMixin


class DictOfTensorMixin(ModuleAttrMixin):
    """Store nested tensors under ``params_dict`` and rebuild them on load."""

    def __init__(self, params_dict: nn.ParameterDict | None = None) -> None:
        super().__init__()
        self.params_dict = (
            params_dict if params_dict is not None else nn.ParameterDict()
        )

    @staticmethod
    def _add_tensor_to_parameter_dict(
        parameter_dict: nn.ParameterDict,
        parameter_keys: list[str],
        value: torch.Tensor,
    ) -> None:
        """Insert ``value`` into ``parameter_dict`` under a nested key path.

        Args:
            parameter_dict: Destination dictionary to mutate.
            parameter_keys: Ordered nested key path, such as
                ``["input_stats", "mean"]``.
            value: Tensor loaded from a checkpoint.
        """
        current_key = parameter_keys[0]
        if len(parameter_keys) == 1:
            parameter_dict[current_key] = nn.Parameter(
                value.clone(),
                requires_grad=False,
            )
            return

        if current_key not in parameter_dict:
            parameter_dict[current_key] = nn.ParameterDict()
        nested_parameter_dict = cast("nn.ParameterDict", parameter_dict[current_key])
        DictOfTensorMixin._add_tensor_to_parameter_dict(
            parameter_dict=nested_parameter_dict,
            parameter_keys=parameter_keys[1:],
            value=value,
        )

    @classmethod
    def _parameter_dict_from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
    ) -> nn.ParameterDict:
        """Build ``params_dict`` from checkpoint entries matching ``prefix``.

        Args:
            state_dict: Checkpoint state dictionary.
            prefix: Full state-dict prefix ending in ``"params_dict."``.

        Returns:
            Nested parameter dictionary reconstructed from matching entries.
        """
        parameter_dict = nn.ParameterDict()
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            parameter_keys = key.removeprefix(prefix).split(".")
            cls._add_tensor_to_parameter_dict(
                parameter_dict=parameter_dict,
                parameter_keys=parameter_keys,
                value=value,
            )
        return parameter_dict

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, str],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Rebuild nested ``ParameterDict`` entries before state loading.

        Args:
            state_dict: Checkpoint state dictionary passed by PyTorch.
            prefix: Module prefix for keys owned by this instance.
            local_metadata: Module metadata passed by PyTorch's load hook.
            strict: Whether PyTorch is loading the checkpoint strictly.
            missing_keys: Mutable missing-key list managed by PyTorch.
            unexpected_keys: Mutable unexpected-key list managed by PyTorch.
            error_msgs: Mutable error list managed by PyTorch.
        """
        self.params_dict = self._parameter_dict_from_state_dict(
            state_dict=state_dict,
            prefix=f"{prefix}params_dict.",
        )
