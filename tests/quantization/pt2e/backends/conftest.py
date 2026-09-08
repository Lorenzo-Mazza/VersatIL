"""Fixtures for PT2E quantizer backend tests."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import fx, nn

from versatil.quantization.constants import FXNodeOp
from versatil.quantization.pt2e.backends.x86_inductor import (
    _PerCallX86InductorQuantizer,
)


class _RepeatedLinearModel(nn.Module):
    """Apply a shared linear/GELU block repeatedly, then project its output."""

    def __init__(self, iterations: int) -> None:
        """Set the number of applications of the shared block."""
        super().__init__()
        self.iterations = iterations
        self.block = nn.Sequential(nn.Linear(8, 8), nn.GELU())
        self.readout = nn.Linear(8, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform input rows through the shared block and output layer.

        Args:
            inputs: Activations with shape (batch_size, 8).

        Returns:
            Activations with shape (batch_size, 4).
        """
        for _ in range(self.iterations):
            inputs = self.block(inputs) + inputs  # (batch_size, 8)
        return self.readout(inputs)  # (batch_size, 8) -> (batch_size, 4)


class _HelperBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Linear(8, 8)

    def forward_language_model(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.language_model(inputs)  # (batch_size, 8)


class _HelperDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_backbone = _HelperBackbone()
        self.vlm_backbone_other = _HelperBackbone()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.vlm_backbone.forward_language_model(inputs=inputs),  # (batch_size, 8)
            self.vlm_backbone_other.forward_language_model(
                inputs=inputs
            ),  # (batch_size, 8)
        )


class _HelperPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _HelperDecoder()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(inputs)  # ((batch_size, 8), (batch_size, 8))


@pytest.fixture
def helper_method_policy_factory(
    rng: np.random.Generator,
) -> Callable[[], nn.Module]:
    def factory() -> nn.Module:
        policy = _HelperPolicy()
        with torch.no_grad():
            for parameter in policy.parameters():
                values = rng.uniform(-0.1, 0.1, size=parameter.shape).astype(np.float32)
                parameter.copy_(
                    torch.from_numpy(values)
                )  # (output_dim, input_dim) or (output_dim,)
        return policy.eval()

    return factory


@pytest.fixture
def module_scope_nodes_factory() -> Callable[..., list[MagicMock]]:
    def factory(module_paths: list[str], serialized_prefix: bool) -> list[MagicMock]:
        nodes = []
        for module_path in module_paths:
            node = MagicMock(spec=fx.Node)
            path = f"L['self'].{module_path}" if serialized_prefix else module_path
            node.meta = {"nn_module_stack": {path: (path, nn.Linear)}}
            nodes.append(node)
        return nodes

    return factory


@pytest.fixture
def repeated_linear_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    def factory(iterations: int) -> nn.Module:
        model = _RepeatedLinearModel(iterations=iterations)
        for parameter in model.parameters():
            values = rng.uniform(-0.1, 0.1, size=parameter.shape).astype(np.float32)
            parameter.data.copy_(
                torch.from_numpy(values)
            )  # (output_dim, input_dim) or (output_dim,)
        return model.eval()

    return factory


@pytest.fixture
def source_metadata_graph_factory() -> Callable[..., MagicMock]:
    def factory(existing_source: bool, serialized_type: bool) -> MagicMock:
        node = MagicMock(spec=fx.Node)
        node.op = FXNodeOp.CALL_FUNCTION.value
        node.meta = {
            "nn_module_stack": {
                "linear@1": (
                    "decoder.linear",
                    "torch.nn.modules.linear.Linear" if serialized_type else nn.Linear,
                )
            }
        }
        if existing_source:
            node.meta["source_fn_stack"] = [("already_recorded_call", nn.Linear)]
        graph = MagicMock(spec=fx.GraphModule)
        graph.graph.nodes = [node]
        return graph

    return factory


@pytest.fixture
def source_metadata_quantizer_factory() -> Callable[[], _PerCallX86InductorQuantizer]:
    def factory() -> _PerCallX86InductorQuantizer:
        with patch(
            "versatil.quantization.pt2e.backends.x86_inductor."
            "X86InductorQuantizer.__init__",
            return_value=None,
        ):
            return _PerCallX86InductorQuantizer()

    return factory
