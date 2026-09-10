"""Distributed training strategies used by VersatIL."""

from contextlib import nullcontext

import torch
from pytorch_lightning.strategies import DDPStrategy
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel


class DefaultStreamDDPStrategy(DDPStrategy):
    """Initialize DDP on the stream used by ordinary training.

    PyTorch Lightning 2.6.1 initializes DDP on a side CUDA stream, while
    subsequent forward and backward passes use the default stream. PyTorch
    reports that mismatch and inserts an avoidable synchronization. This
    strategy carries the upstream Lightning fix until it is available in a
    safe release.
    """

    def _setup_model(self, model: Module) -> DistributedDataParallel:
        """Wrap a model with DDP on the appropriate CUDA stream.

        Args:
            model: Model to wrap with DistributedDataParallel.

        Returns:
            DistributedDataParallel-wrapped model.
        """
        device_ids = self.determine_ddp_device_ids()
        stream_context = nullcontext()
        if device_ids is not None:
            if torch.cuda.is_current_stream_capturing():
                stream_context = torch.cuda.stream(torch.cuda.Stream())
                torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
            else:
                stream_context = torch.cuda.stream(torch.cuda.default_stream())

        with stream_context:
            return DistributedDataParallel(
                module=model,
                device_ids=device_ids,
                **self._ddp_kwargs,
            )
