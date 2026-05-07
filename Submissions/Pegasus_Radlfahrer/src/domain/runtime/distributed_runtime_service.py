from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class DistributedRuntimeContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    device_name: str

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


class DistributedRuntimeService:
    """Initializes torch.distributed from torchrun environment variables when requested."""

    def read_environment(self) -> DistributedRuntimeContext:
        rank = int(os.getenv("RANK", "0"))
        local_rank = int(os.getenv("LOCAL_RANK", str(rank)))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        return DistributedRuntimeContext(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=None,
            device_name="cpu",
        )

    def initialize(self, selected_device: str, use_libuv: bool = True) -> DistributedRuntimeContext:
        context = self.read_environment()
        if not context.is_distributed:
            return DistributedRuntimeContext(
                rank=context.rank,
                local_rank=context.local_rank,
                world_size=context.world_size,
                backend=None,
                device_name=selected_device,
            )

        if selected_device == "mps":
            raise ValueError("Distributed Data Parallel via torchrun is not supported on MPS in this project")

        if selected_device == "cuda":
            device_count = torch.cuda.device_count()
            if context.local_rank >= device_count:
                raise ValueError(
                    f"LOCAL_RANK={context.local_rank} exceeds available CUDA devices ({device_count})"
                )
            torch.cuda.set_device(context.local_rank)
            backend = "nccl"
            device_name = f"cuda:{context.local_rank}"
        else:
            backend = "gloo"
            device_name = "cpu"

        if not dist.is_initialized():
            previous_use_libuv = os.environ.get("USE_LIBUV")
            os.environ["USE_LIBUV"] = "1" if use_libuv else "0"
            try:
                if backend == 'nccl':
                    dist.init_process_group(backend=backend, device_id=torch.device(device_name))
                else:
                    dist.init_process_group(backend=backend)
            finally:
                if previous_use_libuv is None:
                    os.environ.pop("USE_LIBUV", None)
                else:
                    os.environ["USE_LIBUV"] = previous_use_libuv

        return DistributedRuntimeContext(
            rank=context.rank,
            local_rank=context.local_rank,
            world_size=context.world_size,
            backend=backend,
            device_name=device_name,
        )

    def barrier(self, context: DistributedRuntimeContext) -> None:
        if context.is_distributed and dist.is_initialized():
            if context.backend == 'nccl':
                dist.barrier(device_ids=[context.local_rank])
            else:
                dist.barrier()

    def shutdown(self, context: DistributedRuntimeContext) -> None:
        # We intentionally do not destroy the process group here.
        # Repeatedly initializing and destroying NCCL communicators in a loop 
        # (e.g., during hyperparameter search) causes socket exhaustion and hangs.
        pass
