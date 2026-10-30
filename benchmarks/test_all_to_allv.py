"""Standalone correctness test for cpu_all_to_allv custom op.

Run with:
  torchrun --nproc_per_node=4 benchmarks/test_all_to_allv.py

Validates that all_to_allv sends/receives exactly the right rows from each
peer, using gloo (CPU backend).
"""
import os
import sys

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Each rank r has (r+1)*2 rows of hidden_size=8.
    hidden = 8
    local_rows = (rank + 1) * 2
    # Fill with rank id so we can verify origin after exchange.
    local = torch.full((local_rows, hidden), float(rank), dtype=torch.bfloat16)

    # Gather the sizes from all ranks so every rank knows the split plan.
    sizes_tensor = torch.zeros(world, dtype=torch.int64)
    sizes_tensor[rank] = local_rows
    dist.all_reduce(sizes_tensor)
    sizes = sizes_tensor.tolist()

    # -----------------------------------------------------------------------
    # Test 1: all_gatherv via cpu_all_gatherv (reference, padded path)
    # -----------------------------------------------------------------------
    # Register the group manually the same way CpuCommunicator does.
    from vllm.distributed.device_communicators.cpu_communicator import (
        _CPU_EP_GROUP_REGISTRY,
    )
    _CPU_EP_GROUP_REGISTRY["test_group"] = dist.GroupMember.WORLD

    gathered = torch.ops.vllm.cpu_all_gatherv(local, sizes, "test_group")
    expected_rows = sum(int(s) for s in sizes)
    assert gathered.shape == (expected_rows, hidden), (
        f"[rank {rank}] all_gatherv shape mismatch: {gathered.shape}"
    )
    # Check that every rank's block has the right fill value.
    offset = 0
    for r in range(world):
        block = gathered[offset : offset + int(sizes[r])]
        assert (block == float(r)).all(), (
            f"[rank {rank}] all_gatherv wrong value in block {r}: {block}"
        )
        offset += int(sizes[r])
    if rank == 0:
        print(f"[PASS] all_gatherv: gathered shape {gathered.shape}", flush=True)

    # -----------------------------------------------------------------------
    # Test 2: all_to_allv — each rank sends its full local chunk to rank 0
    #         and receives nothing from others (a gather to rank 0 pattern).
    # -----------------------------------------------------------------------
    # input_split_sizes: rank r sends sizes[r] rows to rank 0, 0 to others.
    if rank == 0:
        # Rank 0 receives sizes[r] rows from every rank r.
        input_split_sizes = [0] * world
        input_split_sizes[0] = int(sizes[0])  # self-send
        output_split_sizes = [int(s) for s in sizes]
        inp = local  # own data only
    else:
        # Other ranks send all their rows to rank 0, receive nothing.
        input_split_sizes = [0] * world
        input_split_sizes[0] = local_rows  # send to rank 0
        output_split_sizes = [0] * world
        inp = local

    result = torch.ops.vllm.cpu_all_to_allv(
        inp, input_split_sizes, output_split_sizes, "test_group"
    )

    if rank == 0:
        assert result.shape[0] == sum(int(s) for s in sizes), (
            f"[rank 0] all_to_allv wrong row count: {result.shape[0]}"
        )
        offset = 0
        for r in range(world):
            block = result[offset : offset + int(sizes[r])]
            assert (block == float(r)).all(), (
                f"[rank 0] all_to_allv wrong value from rank {r}: {block}"
            )
            offset += int(sizes[r])
        print(
            f"[PASS] all_to_allv gather-to-rank0: received shape {result.shape}",
            flush=True,
        )
    else:
        assert result.shape[0] == 0, (
            f"[rank {rank}] expected empty result, got {result.shape}"
        )

    dist.barrier()
    if rank == 0:
        print("All tests passed.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
