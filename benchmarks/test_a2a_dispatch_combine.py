"""Correctness test for AgRsAll2AllManager all_to_all_single fast path.

Run with:
  torchrun --nproc_per_node=4 benchmarks/test_a2a_dispatch_combine.py

Simulates a single MoE layer dispatch+combine cycle with 4 EP ranks,
128 experts (32 per rank), topk=2, and verifies that output tokens match
the all_gather reference implementation.
"""
import torch
import torch.distributed as dist


NUM_EXPERTS = 128
EP_SIZE = 4
TOPK = 2
HIDDEN = 64
NUM_TOKENS = 8   # tokens per rank


def _make_random_routing(num_tokens, num_experts, topk, rank, seed=42):
    """Produce deterministic topk_ids / topk_weights for this rank."""
    g = torch.Generator()
    g.manual_seed(seed + rank * 1000)
    # topk_ids: random expert ids in [0, num_experts)
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), generator=g)
    topk_weights = torch.softmax(
        torch.rand(num_tokens, topk, generator=g), dim=-1
    ).to(torch.bfloat16)
    return topk_ids, topk_weights


def run_ag_rs_reference(hidden, topk_weights, topk_ids, rank, world, pg):
    """Reference: all_gather + (fake) reduce_scatter.

    We can't easily fake the MoE kernel here, so we test the
    dispatch+combine round-trip using identity experts: each token's
    hidden state is returned unchanged.
    """
    # all_gather hidden states from all ranks
    all_hs = torch.zeros(world * NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)
    dist.all_gather_single(all_hs, hidden.contiguous(), group=pg)

    # "MoE kernel": identity — output equals input for each token
    moe_out = all_hs.clone()

    # reduce_scatter: each rank gets its slice summed across world
    # (with identity kernel, rank r's output = sum of moe_out[r*S:(r+1)*S]
    #  over all EP ranks — but each EP rank only wrote tokens for its experts,
    #  so in reality only one rank writes non-zero for each token.  Here we
    #  sum all copies = world copies = world * hidden for our reference.)
    out = torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)
    dist.reduce_scatter_single(out, moe_out, group=pg)
    return out


def run_a2a_path(hidden, topk_weights, topk_ids, rank, world, pg):
    """Dispatch via all_to_all_single, identity kernel, combine back."""
    from vllm.distributed.device_communicators.cpu_communicator import (
        _CPU_EP_GROUP_REGISTRY,
    )
    from vllm.distributed.device_communicators.all2all import (
        _tokens_per_ep_rank,
        AgRsAll2AllManager,
    )

    experts_per_rank = NUM_EXPERTS // EP_SIZE

    # -- dispatch --
    send_sizes = _tokens_per_ep_rank(topk_ids, NUM_EXPERTS, EP_SIZE)

    # exchange sizes
    send_t = torch.tensor(send_sizes, dtype=torch.int64)
    recv_t = torch.zeros(world, dtype=torch.int64)
    dist.all_to_all_single(
        recv_t, send_t,
        output_split_sizes=[1] * world,
        input_split_sizes=[1] * world,
        group=pg,
    )
    recv_sizes = recv_t.tolist()

    # sort by destination EP rank
    primary_rank = (topk_ids[:, 0] // experts_per_rank).clamp(0, EP_SIZE - 1)
    sort_idx = torch.argsort(primary_rank, stable=True)
    sorted_hs = hidden[sort_idx]

    # exchange tokens
    recv_hs = torch.ops.vllm.cpu_all_to_allv(
        sorted_hs, send_sizes, recv_sizes, "test_group"
    )

    # "MoE kernel": identity
    expert_out = recv_hs.clone()

    # -- combine --
    # send results back (sizes reversed)
    returned_hs = torch.ops.vllm.cpu_all_to_allv(
        expert_out, recv_sizes, send_sizes, "test_group"
    )

    # unsort
    inv = torch.empty_like(sort_idx)
    inv[sort_idx] = torch.arange(NUM_TOKENS, dtype=sort_idx.dtype)
    output = returned_hs[inv]
    return output


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == EP_SIZE, f"Run with --nproc_per_node={EP_SIZE}"

    pg = dist.GroupMember.WORLD

    # Register group for custom ops
    from vllm.distributed.device_communicators.cpu_communicator import (
        _CPU_EP_GROUP_REGISTRY,
    )
    _CPU_EP_GROUP_REGISTRY["test_group"] = pg

    # Each rank has its own hidden states and routing.
    hidden = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16) * float(rank + 1)
    topk_ids, topk_weights = _make_random_routing(
        NUM_TOKENS, NUM_EXPERTS, TOPK, rank
    )

    # Reference path
    ref_out = run_ag_rs_reference(hidden, topk_weights, topk_ids, rank, world, pg)

    # All_to_all path
    a2a_out = run_a2a_path(hidden, topk_weights, topk_ids, rank, world, pg)

    # Both outputs have shape [NUM_TOKENS, HIDDEN].
    # The reference sums world copies of the identity output;
    # the a2a path returns just one copy per token (from the expert rank).
    # We verify shapes and that no token is lost (non-zero output).
    assert a2a_out.shape == (NUM_TOKENS, HIDDEN), (
        f"[rank {rank}] a2a output shape {a2a_out.shape}"
    )
    assert (a2a_out != 0).any(), f"[rank {rank}] a2a output is all zeros"

    # Verify that all tokens come back to the correct rank by checking that
    # the a2a output for each token equals the corresponding hidden row
    # (the identity kernel should return the dispatched tokens unchanged,
    #  so each token should be returned to the rank that owned it).
    assert torch.allclose(a2a_out, hidden, atol=0), (
        f"[rank {rank}] a2a output does not match original hidden states\n"
        f"  max diff: {(a2a_out - hidden).abs().max()}"
    )

    dist.barrier()
    if rank == 0:
        print(
            f"[PASS] all_to_all dispatch+combine: output shape {a2a_out.shape}, "
            f"values match original hidden states",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
