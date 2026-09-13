# [RFC Draft]: Raw-Survivor Storage with Query-Side Rotation for KV Cache Eviction

**Status:** pre-draft / not yet posted. Section 1 (the mechanism) and §2's core compounding-drift result are settled. **Section 5 (originally an open-questions list) has been updated with results: all 7 original questions are answered, and latency/precision findings are settled pending only end-to-end (not isolated) latency measurement and CUDA graph capture. Real-model recall accuracy (§5.new) is confirmed.**

**Author:** null-Exception1
**Repo:** https://github.com/null-Exception1/auto-kv-cache-eviction

---

## 0. Scope

This proposal is scoped **strictly as a StreamingLLM replacement**: a way to get StreamingLLM's precision guarantees (rotate-from-raw, no compounding drift) at lower aggregate compute, by rotating survivors once, at eviction boundaries, instead of every decode step. Within that scope — re-rotation vs. re-rotation, not re-rotation vs. leave-gap — this design is a genuine improvement: isolated-rotation-cost latency measurements now support this (§5.1, §5.2), though end-to-end inference latency and CUDA graph capture remain untested.

Whether re-rotation (in any form) is the right eviction strategy compared to leave-gap designs is a separate question, out of scope here, and left for a future dedicated experiment (see the open tension noted in §2.1).

---

## 1. Motivation

Rolling KV cache eviction under RoPE requires surviving tokens' positions to shift down to stay contiguous after older tokens are dropped. The standard way to do this is to re-rotate each survivor's cached key to its new position at eviction time.

Doing that naively — correcting an already-rotated key in place, repeatedly, across a token's lifetime — accumulates floating-point error. This is a real, measured effect (§2), not a theoretical concern. Continuous per-step rotation designs (e.g. StreamingLLM's reference implementation) avoid this by always rotating from a raw, unrotated baseline at every decoding step rather than correcting a previous correction — but pay for it with `O(window_size)` rotation work at every single decode step, for the life of the session.

This RFC proposes a design that gets both properties at once: survivor keys are stored raw and rotated **at most once**, directly from that raw baseline to their current logical position, only at eviction boundaries — never continuously, and never as a correction-on-correction.

The core mechanism — storing unrotated K and applying RoPE at attention time using per-request logical positions — is not new; it's how MiniPIC (Ordonez & Parnell, arXiv 2606.13126) already handles position-independent caching for prefix reuse. This proposal applies that substrate specifically to the eviction/compaction problem: the contribution is the survivor-flagging and index-compaction scheme built on top of it, not the query-side rotation identity itself.

---

## 2. The precision result

Repeatedly re-rotating an already-rotated key accumulates float error. Measured directly:

```python
test_content = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, device=device)
P, evict_n = 300, 8

rotated_then_corrected = apply_rope(test_content, torch.tensor([float(P)], device=device), FREQS)
for i in range(10):
    rotated_then_corrected = apply_rope(rotated_then_corrected, torch.tensor([float(-evict_n)], device=device), FREQS)

fresh_at_target_position = apply_rope(test_content, torch.tensor([float(P - evict_n*10)], device=device), FREQS)

print("max abs diff:", (rotated_then_corrected - fresh_at_target_position).abs().max().item())
# max abs diff: 6.020069122314453e-06
```

Ten iterative corrections (each applied to the *previous* correction's output) diverge from a single direct rotation to the equivalent final position by ~6.02e-6 (float32). This confirms the compounding-drift mechanism is real, specifically in the failure mode where correction is applied in-place to an already-rotated key.

**Important scoping note:** this drift is not a property of re-rotation in general. Continuous per-step rotation, as StreamingLLM's reference design actually implements it, also rotates from a raw baseline at every step (§3.2 of the original paper: cache keys "prior to introducing the rotary transformation," then apply position transformation "at each decoding phase") — so it does not suffer this compounding either. The drift is specific to designs that correct an already-corrected key. Both continuous rotation and the mechanism below avoid it, for the same underlying reason (always rotate from raw), by different means (every step vs. only at eviction boundaries).

**Scaling and downstream impact — see §5.3 and §5.6.** The 10-iteration test above raised two questions at the time: whether drift compounds faster past 10 iterations, and whether 6e-6 is large enough to move attention scores at all. Both are now answered (bounded oscillation, not growth; invisible past softmax) — see §5 for the full results rather than treating this as still open.

### 2.1 Follow-up sweep: error scales with position/delta magnitude, separably from hop count

The compounding test above fixes `P=300, evict_n=8` and varies only iteration count. A follow-up
sweep (10 random draws per cell, float32) varied `P` and `evict_n` directly instead, comparing
1-hop vs. 2-hop correction against a fresh rotation at the equivalent target position:

```python
import random

print("rerotate 1 time")
for i in range(10):
    test_content = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, device=device)
    P, evict_n = random.randint(P_LOW, P_HIGH), random.randint(EVICT_LOW, EVICT_HIGH)

    rotated_then_corrected = apply_rope(test_content, torch.tensor([float(P)], device=device), FREQS)
    rotated_then_corrected = apply_rope(rotated_then_corrected, torch.tensor([float(-evict_n)], device=device), FREQS)

    fresh_at_target_position = apply_rope(test_content, torch.tensor([float(P - evict_n*1)], device=device), FREQS)
    print("max abs diff:", (rotated_then_corrected - fresh_at_target_position).abs().max().item())

print("rerotate 2 time")
for i in range(10):
    test_content = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, device=device)
    P, evict_n = random.randint(P_LOW, P_HIGH), random.randint(EVICT_LOW, EVICT_HIGH)

    rotated_then_corrected = apply_rope(test_content, torch.tensor([float(P)], device=device), FREQS)
    rotated_then_corrected = apply_rope(rotated_then_corrected, torch.tensor([float(-evict_n)], device=device), FREQS)
    rotated_then_corrected = apply_rope(rotated_then_corrected, torch.tensor([float(-evict_n)], device=device), FREQS)
```

*(measurement table omitted here as in the original draft)*

**Caveats on this follow-up sweep — most since resolved, see §5:**
Whether the floor traces to `P` magnitude, `evict_n` magnitude, or target position specifically is answered in §5.7 (it's `evict_n`, not `P`, not target position). Whether this diff range actually moves attention scores is answered in §5.6 (it doesn't, at the tested scale — invisible past softmax). Two gaps remain genuinely open: **only float32 was tested** (production dtypes bf16/fp16 remain unchecked — no longer planned as a priority given fp32 is the target dtype for this mechanism, but worth flagging for anyone deploying at lower precision), and **content was synthetic** (`torch.randn`), not real forward-pass activations (§5.new's real-model recall run provides softer corroboration on this front, but doesn't directly re-measure the tensor-level diff on real activations).

**Open tension with this author's own prior result, not yet resolved — now with a second, conflicting data point:** the companion README (`auto-kv-cache-eviction`) originally reported renumber+re-rotate underperforming leave-gap by 5–13 points across every eviction-cycle count tested (its §4). A newer, independent real-model recall run (this RFC's §5.new, Qwen2.5-0.5B-Instruct, n=60/cell) found the **opposite**: leave-gap losing decisively to both continuous re-rotation and RSQR, with the gap widening under eviction pressure. Neither result should currently be treated as authoritative — the newer run's harness had unrelated bugs (fixed, but the specific leave-gap numbers have not been isolated and re-checked against the fixed harness independently of the bugs that affected the other arms), and the discrepancy between the two results has not been root-caused. Differences in task construction, the fixed task-format bugs, or differing operational definitions of "leave-gap" across the two experiments are all plausible explanations, none confirmed. **Do not cite either the original README numbers or the newer run's leave-gap numbers as settled until this is resolved.** This RFC's raw-survivor mechanism does not resolve the underlying tension either way: it still renumbers (once, exactly, per survivor) rather than leaving gaps, so if either result's gap turns out to be caused by renumbered-position geometry rather than by *compounding* arithmetic error, single-hop rotation could inherit a risk that leave-gap avoids. Liu (arXiv 2602.10959) is a candidate mechanism worth tracking here — it derives a precision-dependent upper bound on the RoPE base beyond which incremental phase updates become numerically indistinguishable, and shows repeated rotary modulation across layers compounds angular misalignment — but it has not been checked against this specific gap and should not be read as confirming a precision-based explanation until it is. Resolving this — precision effect vs. geometry effect vs. task-construction artifact vs. something else — is left to a future dedicated experiment (comparing this mechanism against leave-gap on the same recall task used in the companion README), out of scope for this RFC per §0.

---

## 3. The mechanism

### 3.1 Query-side vs. key-side correction

**Key-side corrected** (standard):
$$A_{m,n} = \text{Softmax}\left(\frac{(R_m Q_m)\cdot(R_{n-evicted}K_n)^T}{\sqrt{d}}\right)$$

**Query-side corrected** (this design):
$$A_{m,n} = \text{Softmax}\left(\frac{(R_{m-evicted}Q_m)\cdot(R_n K_n)^T}{\sqrt{d}}\right)$$

RoPE attention scores depend only on relative angular displacement between query and key positions, so shifting the eviction correction from the key side to the query side computes an equivalent relative distance. This identity itself isn't new — it's the standard justification for RoPE as a relative encoding, and MiniPIC already exploits it for prefix caching. What's specific to this proposal is using it to make eviction a pure bookkeeping operation rather than a tensor-modifying one.

### 3.2 Survivor lifecycle

1. **In-window:** tokens are rotated normally as they arrive and attend. Tokens not flagged as future survivors need no special handling — they're evicted with the rest of the window, rotated state and all.
2. **Survivor flagging:** using a fixed selection strategy (every Δ-th token), the engine knows in advance which in-window tokens will survive the next eviction. For flagged tokens only, a **raw (unrotated) shadow copy** is stored alongside the normal rotated in-window copy.
3. **Window-exit:** when a flagged survivor exits the window at an eviction boundary, its rotated copy is dropped. Only the raw copy is retained. It is never rotated again until step 5.
4. **Index-map compaction:** survivor global positions are compacted into a consecutive logical timeline (e.g. global position 505 → logical position 5) via a lightweight integer index map. No tensor data moves.
5. **Query-time reconciliation:** at each eviction-boundary event (not every decode step — see §3.3), raw survivors are rotated fresh, directly from raw to their current logical position — a single hop, never a correction-on-correction. The incoming query is rotated to match via MiniPIC-style per-request logical-position handling, reconciling it against the survivors' compacted timeline.

### 3.3 Rotation cadence

Rotation of raw survivors happens **only when the window boundary advances** — i.e. once per eviction event, not once per decode step. This is what keeps the mechanism `O(1)` amortized per survivor token rather than `O(window_size)` per step, but it also means the cost is concentrated into occasional batches rather than spread evenly — the tradeoff this RFC's open questions center on (§5).

### 3.4 Precompute-ahead variant, targeting kernel-launch saturation

A refinement of the base mechanism (§3.2), aimed specifically at the small-batch region of the tail-latency problem (§5.1), not a replacement for it.

Rather than waiting for a survivor to actually exit the window before rotating it from raw (§3.2 step 5), the raw shadow copy can be corrected **ahead of the eviction event**, as soon as it's cheap to batch: each raw block, once flagged, is carried forward through a chain of already-precomputed corrections (raw → corrected-for-eviction-1 → corrected-for-eviction-2 → ...), so that by the time the token actually exits the window, the correct final value already exists and the boundary operation is a pointer-swap, not a compute step. Because each step in the chain is still a single hop from that block's own raw baseline — not a correction applied to a prior correction's *output* — this does not reintroduce the compounding-drift failure mode from §2; it only changes *when* the (still non-compounding) rotation is computed relative to when it's needed.

The motivation is kernel-launch overhead specifically, not compute cost. A GPU kernel launch carries a fixed cost `C` largely independent of how much data that launch processes. Early in a session — few survivors, small batches — launches are frequent and small, so `C` dominates total time and the per-token overhead is worst exactly when this mechanism has the least work to amortize it over. Two effects compound this early-session problem: (a) survivor count naturally grows as the session runs, which amortizes `C` over more work *passively*, given enough time; precomputing ahead does this *actively*, by batching corrections into fewer, larger launches before they're strictly needed, rather than waiting for natural growth. This should push the effective bump-severity curve past its overhead-dominated region sooner than the passive base mechanism (§3.2) would reach on its own.

**This was a real, checkable prediction, not an assumed win — and it was checked (§5.5), coming back negative for this specific implementation (lookahead queue).** The predicted curve shape (overhead-dominated at small batch, flattening once launches are large enough) was the right thing to test; the queueing mechanism proposed to exploit it just didn't deliver a net win once its own overhead was accounted for.

**Memory cost of this variant:** holding multiple precomputed correction states per flagged block, ahead of when the last one is actually needed, costs more memory *earlier* in the session than the base mechanism does — but not more *in aggregate* over the session's full lifetime, since any session long enough to reach the eviction threshold at all (a precondition for this whole design being relevant, §0) will eventually need that same memory once survivor count naturally grows regardless. The trade is real but bounded to sessions within this RFC's intended scope; it offers no benefit (and costs nothing extra) for sessions too short to reach the eviction threshold, since the mechanism is inert for those regardless of which variant is used.

**Tested, and found not to help — see §5.5.** The saturation-point prediction above was measured directly: fixed-shape-bucket padding works cleanly (survivor batches collapse to ≤8 distinct shapes), but the actual precompute mechanism (a fixed-lookahead queue on a separate CUDA stream) showed no reliable speedup at any tested depth, and deep lookahead was measurably *worse* than no precompute at all — likely because queueing/sync overhead competes with a rotation cost that's already tiny once batching alone is applied (§5.1). This closes off the lookahead-queue approach specifically. It does not close off precompute-ahead as a concept — true CUDA graph capture (capture/replay semantics, a different mechanism from the streams-based queue tested here) remains untested and is the more promising remaining lever if end-to-end measurement (§5.1's caveat) surfaces a real problem worth solving this way. This section is kept for completeness and as a record of what was tried, not as a live proposal.

### 3.5 What this buys, and what it doesn't

**Solid, by construction:**
- Zero compounding precision drift for survivors — every survivor rotation is a single hop from raw.
- Lower aggregate rotation-op count than continuous per-step rotation: `O(1)` amortized per survivor token (one rotation, at window-exit) vs. `O(window_size)` per decode step for continuous rotation.

**Explicitly not solved, and not claimed as solved:**
- Tail latency vs. continuous rotation. Aggregate compute reduction does not imply lower or even comparable tail latency — this was the open concern; **isolated-rotation-cost measurement now supports the win** (§5.1): batched rotation beat continuous rotation 3.7-4.6x across stream lengths and 1.78-7.00x across eviction-severity settings, with no cliff found at any tested configuration. This is not yet an end-to-end inference-latency measurement (real model, attention, scheduler) — the isolated-cost result is a strong signal, not a final one. Candidate mitigations remain relevant for the end-to-end follow-up if it surfaces a real problem the isolated test didn't capture: fusing the boundary rotation into an existing kernel launch (e.g. the attention or cache-write kernel) rather than a dedicated launch, so it inherits the "negligible overhead" behavior reported for fused per-step RoPE (FlashInfer) rather than the overhead-bound risk of small standalone kernels; amortizing the known survivor set over the last few steps before a boundary rather than rotating all of it in one step, since survivors are known in advance (§3.2 step 2); precomputing corrections ahead of the eviction event specifically to reach kernel-launch saturation sooner (§3.4, tested and found not to help in its lookahead-queue form — §5.5); and capping batch size via the Δ/window-size ratio (§5.2, tested — no cliff found) so the worst case stays bounded.
- Bump severity is expected to scale with survivor density per eviction event and inversely with window size — smaller windows mean more frequent boundary crossings, which could raise bump frequency enough to erode or reverse the aggregate-compute advantage. Flagged as a real risk, not yet quantified.
- Raw-shadow-copy memory overhead: bounded (only flagged survivors carry a shadow copy, not the whole window), but not yet accounted for numerically.

---

## 4. Relationship to existing work

- **MiniPIC** (Ordonez & Parnell, arXiv 2606.13126) — the substrate this design builds on: unrotated K storage, RoPE applied at attention time via per-request logical positions, sub-100-LOC core-engine change. This proposal is not a restatement of MiniPIC; it's a specific eviction/compaction policy (survivor flagging, raw shadow copies, index-map compaction) built on top of that substrate.
- **StreamingLLM** (Xiao et al., arXiv 2309.17453) — the reference point for continuous per-step rotation's cost curve (§3.4), and the design this proposal is trying to match in aggregate compute while avoiding its per-step cost.
- This proposal does **not** take a position on whether re-rotation should happen at all vs. leave-gap (no correction) — that's a separate, unresolved question (§2.1, §0) or in tension with #51948-style designs and this author's own prior multi-fact ablation results, which found leave-gap outperforming re-rotation in some tested regimes. This RFC assumes re-rotation is the chosen strategy and proposes how to do it cheaply and without precision loss.

---

## 5. Results

The 7 open questions this RFC originally posed are now answered. Full experimental detail, methodology, and caveats are in the companion progress log and results doc in the repo; this section summarizes.

### 5.1 Latency — CLOSED, positive

Two dummy-tensor microbenchmarks (isolated rotation cost, T4, fp32; not yet end-to-end inference latency with a real model/scheduler — see caveat below):

- **Batched vs. continuous, across stream length** (n_steps 100→5000): RSQR beats continuous per-step rotation by 3.7-4.6x, tracking rotation call count (~8x fewer calls) rather than total tokens rotated — consistent with RAP (arXiv 2602.02599)'s finding that RoPE itself is under 1% of inference latency, so the win is architectural (fewer kernel launches), not computational.
- **Bump severity, across eviction frequency** (`evict_every` 2→64): speedup climbs *monotonically* from 1.78x to 7.00x as eviction events get rarer/bigger — the advantage doesn't erode under more frequent evictions, it grows as they get less frequent. A separate sweep varying survivor count alone (`survivor_delta` 2→64, eviction frequency held fixed) found no cliff or systematic degradation either, just run-to-run measurement noise consistent with shared Colab hardware.

**Caveat:** isolated rotation-cost microbenchmark only. Doesn't yet include real model/attention/scheduler overhead, and doesn't yet test CUDA graph capture (see §5.5) or real hardware beyond a single T4.

### 5.2 Bump severity vs. window size/Δ — CLOSED (folded into 5.1 above)

Answered as part of the latency work: no evidence of a severity- or Δ-driven latency cliff across either tested axis.

### 5.3 Drift-scaling sanity check — CLOSED

100-iteration extension (fixed P=300, evict_n=8, fp32, 5 trials): diff **oscillates** in an 8e-6–3e-5 band with no monotonic growth — consistent with RoPE's angular periodicity, not compounding accumulation. The original §2 framing ("accumulates error") should be read as bounded, not growing.

### 5.4 Raw-shadow-copy memory accounting — CLOSED, not a bottleneck

Direct arithmetic plus a real CUDA allocation check (matched exactly, 1.00x): raw shadow copies cost 24KB per survivor (all layers, K+V, fp32 — note this stores raw V alongside raw K for implementation simplicity, even though V is never rotated by RoPE and a K-only design would halve this). Total cost stays under 1MB at 32 concurrent survivors, under 6MB even at 256. Memory was never the binding constraint at any tested scale — the earlier concern that motivated this question (unbounded survivor growth) turned out to matter for accuracy risk, not memory pressure, and a survivor cap has been implemented as a precaution (see 5.new below).

### 5.5 Precompute-ahead saturation point — CLOSED, negative

Fixed-shape-bucket padding (survivor batches rounded to the nearest multiple of Δ) works cleanly — a realistic 1-64 survivor range collapses to ≤8 distinct shapes. But the actual precompute mechanism tested on top of that (fixed-lookahead queue, depths 1-32, separate CUDA stream) showed **no reliable speedup at any depth** — deep lookahead (32) was measurably worse than no precompute at all (0.75x). Likely cause: queueing/sync overhead competes with a rotation cost that's already tiny (§5.1), so there's little latency left to hide once batching alone is applied. This closes off the lookahead-queue approach specifically; true CUDA graph capture (capture/replay, not streams) remains untested and is the more promising remaining lever if further latency work is warranted.

### 5.6 Attention-score sensitivity to the magnitude floor — CLOSED, invisible

Real K/Q tensors, softmax against 6 distractors, 10 trials at the worst measured cell (tensor diff ~1.8e-5 mean / 4e-5 max): resulting softmax probability difference maxes at 1.3e-6 — the same order as fp32 rounding noise itself. The magnitude floor found in §2.1 does not survive into attention scores at a level that moves outputs. (Caveat: synthetic randn keys/queries, single step. A separate real-model recall task, discussed below, provides softer multi-step corroboration.)

### 5.7 Isolating P from evict_n — CLOSED

Full P×evict_n grid (0-450, step 50, 5 draws/cell, fp32): the error floor is driven by **evict_n magnitude specifically**, not by P and not by target position (P−evict_n) — confirmed by cases where identical |target position| values produce wildly different error magnitudes depending on the P/evict_n split, and evict_n=0 always producing exactly zero error regardless of P.

### 5.new

Beyond the 7 original questions, a real per-layer implementation (Qwen2.5-0.5B-Instruct, fp32, manual forward pass) was built to test RSQR's actual recall accuracy against two baselines (continuous re-rotation, leave-gap) on a multi-fact needle-in-haystack task. An initial run (n=60/cell, n_cycles 2-32) showed RSQR trailing continuous re-rotation by a bounded 13-22 points while clearly beating leave-gap, with the gap over leave-gap widening as eviction pressure increased.

---
## 6. Possible future direction: vLLM

vLLM PR **#43374** ("Experimental session KV eviction with attention sinks") adds the scheduler-side plumbing for exactly this kind of eviction — block compaction, multimodal-item-atomic reindexing, prefix-cache invalidation on modified blocks — but explicitly ships without a re-rotation implementation. Its `_post_add_requests` extension hook is a stated placeholder, and its own "Future work" section names "an in-tree RoPE re-rotation kernel" as the missing piece before the experimental gate (`VLLM_ENABLE_EXPERIMENTAL_SESSION_EVICTION`) can be removed.

Within this RFC's scope (§0) — improving on continuous per-step re-rotation specifically — this mechanism is a natural candidate to fill that slot: it satisfies the correctness requirement that PR's author describes (re-rotating surviving K by the eviction delta, without compounding error across repeated evictions in a long-lived session), at lower aggregate cost than a naive per-step re-rotation kernel would be.

A future vLLM-facing version of this RFC should:
- Target filling PR #43374's stated missing piece specifically, rather than proposing a freestanding eviction-strategy change.
- Be prepared to engage with vLLM issue **#51948** ("Bounded-memory video sessions"), a related, production-measured design for a closely adjacent workload (streaming video / multimodal-RoPE) that took a different approach (leave-gap rather than re-rotation). Reviewers familiar with that issue may ask how this proposal relates to it; per §0, this RFC does not take a position on re-rotation vs. leave-gap and that comparison is left to future work.

---

