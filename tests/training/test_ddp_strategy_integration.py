"""Two-process behavioral tests for the VersatIL DDP strategy."""

import warnings
from pathlib import Path

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from versatil.training.ddp_strategy import DefaultStreamDDPStrategy


def _run_ddp_training_step(
    local_rank: int,
    world_size: int,
    rendezvous_path: str,
) -> None:
    """Run one synchronized optimization step in a spawned worker.

    Args:
        local_rank: CUDA device and process rank.
        world_size: Number of participating processes.
        rendezvous_path: Shared file used to initialize the process group.
    """
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_path}",
        rank=local_rank,
        world_size=world_size,
    )
    try:
        warnings.filterwarnings(
            action="error",
            message="The AccumulateGrad node's stream does not match.*",
        )
        model = torch.nn.Linear(
            in_features=2,
            out_features=1,
            bias=False,
            device=device,
        )
        strategy = DefaultStreamDDPStrategy(
            parallel_devices=[device],
            find_unused_parameters=False,
        )
        distributed_model = strategy._setup_model(model=model)
        optimizer = torch.optim.SGD(
            params=distributed_model.parameters(),
            lr=0.1,
        )
        inputs = torch.full(
            size=(2, 2),
            fill_value=float(local_rank + 1),
            device=device,
        )
        targets = torch.zeros(size=(2, 1), device=device)

        loss = torch.nn.functional.mse_loss(distributed_model(inputs), targets)
        loss.backward()
        optimizer.step()

        local_weights = model.weight.detach()
        gathered_weights = [torch.empty_like(local_weights) for _ in range(world_size)]
        distributed.all_gather(gathered_weights, local_weights)
        for weights in gathered_weights[1:]:
            torch.testing.assert_close(weights, gathered_weights[0])
    finally:
        distributed.destroy_process_group()


@pytest.mark.integration
@pytest.mark.requires_gpu
def test_two_gpu_step_synchronizes_parameters_without_stream_warning(
    tmp_path: Path,
) -> None:
    """Verify real NCCL synchronization and the default-stream behavior."""
    world_size = 2
    if torch.cuda.device_count() < world_size:
        pytest.skip("requires two visible CUDA devices")

    rendezvous_path = tmp_path / "ddp_rendezvous"
    multiprocessing.spawn(
        fn=_run_ddp_training_step,
        args=(world_size, str(rendezvous_path)),
        nprocs=world_size,
        join=True,
    )
