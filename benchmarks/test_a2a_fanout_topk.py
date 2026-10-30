"""Unit test for _fanout_topk and the dispatch+combine round-trip with topk>1.

Tests the two helper functions in all2all.py that enable the fast all_to_all
path for models like Gemma-4 (topk=8, 128 experts, EP=4):

  _fanout_topk  — replicates each token to every EP rank that owns ≥1 of its
                  topk experts, sorted by destination rank.

  Simulated dispatch+combine round-trip — verifies that scatter_add correctly
  reassembles partial expert outputs back into the original token order.

Run:
    python benchmarks/test_a2a_fanout_topk.py
"""

import sys
import torch


# ---------------------------------------------------------------------------
# Inline the functions under test so the test is self-contained and
# does not require a distributed environment.
# ---------------------------------------------------------------------------

def _fanout_topk(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_states: torch.Tensor,
    num_experts: int,
    ep_size: int,
):
    num_tokens, topk = topk_ids.shape
    experts_per_rank = (num_experts + ep_size - 1) // ep_size

    dest_rank = (topk_ids // experts_per_rank).clamp(0, ep_size - 1)

    routes_to = torch.zeros(
        num_tokens, ep_size, dtype=torch.bool, device=topk_ids.device
    )
    for k in range(topk):
        routes_to.scatter_(1, dest_rank[:, k : k + 1], True)

    token_idx, rank_idx = routes_to.nonzero(as_tuple=True)

    order = torch.argsort(rank_idx, stable=True)
    token_idx = token_idx[order]
    rank_idx = rank_idx[order]

    send_sizes = [int((rank_idx == r).sum().item()) for r in range(ep_size)]

    fanned_hs = hidden_states[token_idx]
    fanned_weights = topk_weights[token_idx]
    fanned_ids = topk_ids[token_idx]

    return fanned_hs, fanned_weights, fanned_ids, send_sizes, token_idx, rank_idx


def test_fanout_topk1():
    """With topk=1 every token maps to exactly 1 rank: fanout == original."""
    T, num_experts, ep_size, H = 8, 4, 4, 16
    torch.manual_seed(0)
    topk_ids = torch.randint(0, num_experts, (T, 1))
    topk_weights = torch.ones(T, 1)
    hs = torch.randn(T, H)

    fanned_hs, fanned_tw, fanned_ti, send_sizes, token_idx, rank_idx = _fanout_topk(
        topk_ids, topk_weights, hs, num_experts, ep_size
    )

    assert fanned_hs.shape[0] == T, f"topk=1: expect F==T, got {fanned_hs.shape[0]}"
    assert sum(send_sizes) == T
    # Each token appears exactly once.
    counts = torch.zeros(T, dtype=torch.long)
    counts.scatter_add_(0, token_idx, torch.ones_like(token_idx))
    assert (counts == 1).all(), "topk=1: each token should appear exactly once"
    print("PASS test_fanout_topk1")


def test_fanout_topk_gemma4_shapes():
    """Gemma-4 decode: T=32 tokens, 128 experts, topk=8, EP=4 (32 experts/rank)."""
    T, num_experts, topk, ep_size, H = 32, 128, 8, 4, 2048
    experts_per_rank = num_experts // ep_size  # 32

    torch.manual_seed(42)
    # Each token selects 8 unique experts from [0, 127].
    topk_ids = torch.stack([
        torch.randperm(num_experts)[:topk] for _ in range(T)
    ])  # [T, 8]
    topk_weights = torch.rand(T, topk)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)  # normalize
    hs = torch.randn(T, H)

    fanned_hs, fanned_tw, fanned_ti, send_sizes, token_idx, rank_idx = _fanout_topk(
        topk_ids, topk_weights, hs, num_experts, ep_size
    )

    F = fanned_hs.shape[0]
    assert sum(send_sizes) == F
    # Each token appears between 1 and ep_size times.
    counts = torch.zeros(T, dtype=torch.long)
    counts.scatter_add_(0, token_idx, torch.ones_like(token_idx))
    assert (counts >= 1).all() and (counts <= ep_size).all()

    # Verify sorted by rank.
    assert (rank_idx[:-1] <= rank_idx[1:]).all(), "fan-out rows must be sorted by rank"

    # Verify send_sizes are contiguous blocks matching rank_idx.
    offset = 0
    for r, sz in enumerate(send_sizes):
        assert (rank_idx[offset:offset + sz] == r).all()
        offset += sz

    # Verify fanned_hs rows match original hs.
    assert torch.allclose(fanned_hs, hs[token_idx])

    print(
        f"PASS test_fanout_topk_gemma4_shapes  "
        f"F={F} (avg {F/T:.1f}x fanout, max {counts.max().item()}x)"
    )


def test_combine_scatter_add():
    """Simulate the full dispatch -> expert processing -> combine round-trip.

    Each EP rank processes only the experts it owns.  The combine uses
    scatter_add to accumulate partial weighted sums back into output.

    We verify that the result equals torch.einsum('tk,tkh->th', w, expert_out)
    computed directly, where expert_out[t,k] = token t processed by expert k.
    """
    T, num_experts, topk, ep_size, H = 8, 8, 4, 4, 16
    experts_per_rank = num_experts // ep_size  # 2

    torch.manual_seed(7)
    topk_ids = torch.stack([
        torch.randperm(num_experts)[:topk] for _ in range(T)
    ])  # [T, topk]
    topk_weights = torch.rand(T, topk)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    hs = torch.randn(T, H)

    # "Reference" output: each (token, expert) pair produces a unique output
    # vector.  expert_output[t, k] is what expert topk_ids[t,k] produces for
    # token t.  The combined output is the weighted sum across k.
    torch.manual_seed(99)
    expert_output = torch.randn(T, topk, H)  # [T, topk, H]
    reference = (topk_weights.unsqueeze(-1) * expert_output).sum(dim=1)  # [T, H]

    # Simulate what happens rank-by-rank:
    # Each rank r receives only the fan-out rows destined for it, runs its
    # experts, and produces a partial weighted sum.

    _, _, fanned_ti, send_sizes, token_idx, rank_idx = _fanout_topk(
        topk_ids, topk_weights, hs, num_experts, ep_size
    )

    # Simulate per-rank expert processing and produce fanned-out partial sums.
    # For each fan-out row (orig token t going to rank r), sum the weighted
    # expert outputs for experts on rank r.
    fanned_partial = torch.zeros(token_idx.shape[0], H)
    for i in range(token_idx.shape[0]):
        t = token_idx[i].item()
        r = rank_idx[i].item()
        for k in range(topk):
            eid = topk_ids[t, k].item()
            if eid // experts_per_rank == r:
                fanned_partial[i] += topk_weights[t, k].item() * expert_output[t, k]

    # Combine: scatter_add fanned_partial back into output.
    output = torch.zeros(T, H)
    idx = token_idx.unsqueeze(1).expand_as(fanned_partial)
    output.scatter_add_(0, idx, fanned_partial)

    # The scatter_add sum should match the reference weighted sum.
    assert torch.allclose(output, reference, atol=1e-5), (
        f"max diff = {(output - reference).abs().max().item()}"
    )
    print("PASS test_combine_scatter_add")


def test_fanout_all_same_rank():
    """Edge case: all tokens route to the same rank (topk=1, expert 0)."""
    T, num_experts, ep_size, H = 4, 8, 4, 8
    topk_ids = torch.zeros(T, 1, dtype=torch.long)  # all expert 0 → rank 0
    topk_weights = torch.ones(T, 1)
    hs = torch.randn(T, H)

    _, _, _, send_sizes, token_idx, rank_idx = _fanout_topk(
        topk_ids, topk_weights, hs, num_experts, ep_size
    )

    assert send_sizes[0] == T
    assert all(send_sizes[r] == 0 for r in range(1, ep_size))
    assert (rank_idx == 0).all()
    print("PASS test_fanout_all_same_rank")


def test_fanout_zero_tokens():
    """Edge case: empty batch (T=0)."""
    T, num_experts, ep_size, H = 0, 8, 4, 8
    topk_ids = torch.zeros(T, 2, dtype=torch.long)
    topk_weights = torch.zeros(T, 2)
    hs = torch.zeros(T, H)

    fanned_hs, _, _, send_sizes, token_idx, rank_idx = _fanout_topk(
        topk_ids, topk_weights, hs, num_experts, ep_size
    )

    assert fanned_hs.shape[0] == 0
    assert sum(send_sizes) == 0
    print("PASS test_fanout_zero_tokens")


if __name__ == "__main__":
    failed = 0
    for test_fn in [
        test_fanout_topk1,
        test_fanout_topk_gemma4_shapes,
        test_combine_scatter_add,
        test_fanout_all_same_rank,
        test_fanout_zero_tokens,
    ]:
        try:
            test_fn()
        except Exception as e:
            print(f"FAIL {test_fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    if failed:
        print(f"\n{failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("\nAll tests PASSED")
