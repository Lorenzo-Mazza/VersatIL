"""Guards for running module forward passes without lasting side effects."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def module_side_effects_guard(module: nn.Module) -> Iterator[None]:
    """Restore module buffers and RNG state after a warmup forward pass.

    BatchNorm running statistics and quantizer observer buffers update in
    train mode even under no_grad, and the pass consumes RNG draws; buffers
    created inside the guard (lazy materialization) are kept.

    Args:
        module: Module whose pre-existing buffers are snapshotted.
    """
    rng_state = torch.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    buffers_before = {
        name: buffer.detach().clone() for name, buffer in module.named_buffers()
    }
    try:
        yield
    finally:
        with torch.no_grad():
            for name, buffer in module.named_buffers():
                if name in buffers_before:
                    buffer.copy_(buffers_before[name])
        torch.set_rng_state(rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
