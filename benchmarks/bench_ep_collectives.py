"""Latency microbenchmark: all_gather+reduce_scatter vs all_to_all_single.

Measures wall-clock time per MoE layer communication for the two strategies:
  - Strategy A (current):  all_gather_single + reduce_scatter  (2 gloo ops)
  - Strategy B (proposed): all_to_all_single x2                (2 gloo ops, less data)

Run with:
  numactl --interleave=all \\
    torchrun --nproc_per_node=4 benchmarks/bench_ep_collectives.py

The benchmark uses realistic Gemma-4 26B shapes:
  hidden_size  = 2560   (Gemma-4 26B MoE hidden dim)
  num_tokens   = 8      (decode batch, 2 prompts × 4 ranks)
  topk         = 2      (top-2 routing)
  num_experts  = 128    (global)
  ep_size      = 4      (one rank per 32 experts)
"""
import os
import time

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Gemma-4 26B decode shapes
# ---------------------------------------------------------------------------
HIDDEN = 2560
NUM_TOKENS_PER_RANK = 8   # tokens this rank owns at decode time
WARMUP = 20
ITERS = 200

# Fraction of tokens that route to each EP rank (uniform routing assumption).
# With topk=2 and 128 experts split across 4 ranks, each rank gets
# ~(2/128)*32 = 0.5 tokens per input token on average.  For 8 tokens that
# is ~4 routed tokens per EP rank.  Use exact equal split for clean numbers.
ROUTED_TOKENS_PER_EP_RANK = 4   # tokens this rank sends to each peer


def _banner(msg: str, rank: int) -> None:
    if rank == 0:
        print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)


def _timeit(fn, warmup: int, iters: int) -> float:
    """Return median wall-clock time in milliseconds over iters calls."""
    dist.barrier()
    for _ in range(warmup):
        fn()
    dist.barrier()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)

    dist.barrier()
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Strategy A: all_gather_single + reduce_scatter  (current AgRsAll2AllManager)
# ---------------------------------------------------------------------------

def bench_ag_rs(world: int, rank: int, pg) -> float:
    """Simulate dispatch (all_gather) + combine (reduce_scatter)."""
    # dispatch: gather all tokens from all ranks
    local = torch.zeros(NUM_TOKENS_PER_RANK, HIDDEN, dtype=torch.bfloat16)
    gathered = torch.empty(world * NUM_TOKENS_PER_RANK, HIDDEN, dtype=torch.bfloat16)

    # combine input: world * local tokens, each rank sums and keeps its slice
    combine_input = torch.zeros(world * NUM_TOKENS_PER_RANK, HIDDEN, dtype=torch.bfloat16)
    combine_out = torch.empty(NUM_TOKENS_PER_RANK, HIDDEN, dtype=torch.bfloat16)

    def _one_layer():
        # dispatch — all_gather_single
        dist.all_gather_single(gathered, local, group=pg)
        # combine — reduce_scatter_single
        dist.reduce_scatter_single(combine_out, combine_input, group=pg)

    return _timeit(_one_layer, WARMUP, ITERS)


# ---------------------------------------------------------------------------
# Strategy B: all_to_all_single x2  (proposed, sends only routed tokens)
# ---------------------------------------------------------------------------

def bench_a2a(world: int, rank: int, pg) -> float:
    """Simulate dispatch + combine using all_to_all_single.

    Dispatch: each rank sends ROUTED_TOKENS_PER_EP_RANK tokens to each peer.
    Combine:  each rank sends back ROUTED_TOKENS_PER_EP_RANK result tokens.
    Total tokens exchanged per direction: world * ROUTED_TOKENS_PER_EP_RANK.
    """
    total_routed = world * ROUTED_TOKENS_PER_EP_RANK

    # dispatch
    dispatch_in  = torch.zeros(total_routed, HIDDEN, dtype=torch.bfloat16)
    dispatch_out = torch.empty(total_routed, HIDDEN, dtype=torch.bfloat16)
    split_sizes  = [ROUTED_TOKENS_PER_EP_RANK] * world   # equal split

    # combine (same sizes, reversed direction)
    combine_in   = torch.zeros(total_routed, HIDDEN, dtype=torch.bfloat16)
    combine_out  = torch.empty(total_routed, HIDDEN, dtype=torch.bfloat16)

    def _one_layer():
        dist.all_to_all_single(
            dispatch_out, dispatch_in,
            output_split_sizes=split_sizes,
            input_split_sizes=split_sizes,
            group=pg,
        )
        dist.all_to_all_single(
            combine_out, combine_in,
            output_split_sizes=split_sizes,
            input_split_sizes=split_sizes,
            group=pg,
        )

    return _timeit(_one_layer, WARMUP, ITERS)


# ---------------------------------------------------------------------------
# Strategy C: single all_to_all_single per direction (what we'd have with
#             perfect pre-routing — best-case comparison baseline)
# ---------------------------------------------------------------------------

def bench_a2a_single(world: int, rank: int, pg) -> float:
    """One all_to_all_single call — lower-bound on achievable latency."""
    total_routed = world * ROUTED_TOKENS_PER_EP_RANK
    inp  = torch.zeros(total_routed, HIDDEN, dtype=torch.bfloat16)
    out  = torch.empty(total_routed, HIDDEN, dtype=torch.bfloat16)
    split_sizes = [ROUTED_TOKENS_PER_EP_RANK] * world

    def _one_layer():
        dist.all_to_all_single(
            out, inp,
            output_split_sizes=split_sizes,
            input_split_sizes=split_sizes,
            group=pg,
        )

    return _timeit(_one_layer, WARMUP, ITERS)


# ---------------------------------------------------------------------------
# Data volume analysis
# ---------------------------------------------------------------------------

def _data_mb(rows: int, hidden: int, dtype_bytes: int = 2) -> float:
    return rows * hidden * dtype_bytes / (1024 ** 2)


def main():
    dist.init_process_group(backend="gloo")
    rank  = dist.get_rank()
    world = dist.get_world_size()
    pg    = dist.GroupMember.WORLD

    _banner(f"EP collective latency benchmark  (world={world})", rank)

    if rank == 0:
        ag_data   = _data_mb(world * NUM_TOKENS_PER_RANK, HIDDEN)
        rs_data   = _data_mb(world * NUM_TOKENS_PER_RANK, HIDDEN)
        a2a_data  = _data_mb(world * ROUTED_TOKENS_PER_EP_RANK, HIDDEN)
        print(
            f"  hidden={HIDDEN}  tokens_per_rank={NUM_TOKENS_PER_RANK}"
            f"  routed_per_ep_rank={ROUTED_TOKENS_PER_EP_RANK}\n"
            f"  Strategy A data/direction: all_gather={ag_data:.2f}MB"
            f"  reduce_scatter={rs_data:.2f}MB\n"
            f"  Strategy B data/direction: all_to_all={a2a_data:.2f}MB  (x2 ops)\n",
            flush=True,
        )

    t_ag_rs   = bench_ag_rs(world, rank, pg)
    t_a2a_x2  = bench_a2a(world, rank, pg)
    t_a2a_x1  = bench_a2a_single(world, rank, pg)

    if rank == 0:
        print(
            f"  Median latency per MoE layer (over {ITERS} iters):\n"
            f"    A  all_gather_single + reduce_scatter : {t_ag_rs:7.3f} ms\n"
            f"    B  all_to_all_single x2 (routed only) : {t_a2a_x2:7.3f} ms"
            f"  ({t_ag_rs/t_a2a_x2:.2f}x {'faster' if t_a2a_x2 < t_ag_rs else 'slower'})\n"
            f"    C  all_to_all_single x1 (lower bound) : {t_a2a_x1:7.3f} ms\n",
            flush=True,
        )

        # Project to full model: Gemma-4 has 46 MoE layers
        moe_layers = 46
        print(
            f"  Projected collective overhead per decode step ({moe_layers} MoE layers):\n"
            f"    A  current  : {t_ag_rs  * moe_layers:.1f} ms\n"
            f"    B  proposed : {t_a2a_x2 * moe_layers:.1f} ms\n"
            f"    C  ideal    : {t_a2a_x1 * moe_layers:.1f} ms\n",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
