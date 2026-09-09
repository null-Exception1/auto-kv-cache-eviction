# [RFC Draft]: Raw-Survivor Storage with Query-Side Rotation for KV Cache Eviction

**Status:** pre-draft / not yet posted. Section 1 (the mechanism) and §2's core compounding-drift result are settled. §2.1 is a follow-up sweep, not yet fully rigorous (see its own caveats), that surfaces a separate magnitude-driven error term the original §2 test didn't exercise — treat §2 as settled only for the specific claim it measured (compounding, fixed P), not as a complete precision picture. Section 5 is the open-questions list — the latency claim and the new §2.1 follow-ups are unmeasured and flagged as such throughout.
**Author:** null-Exception1
**Repo:** https://github.com/null-Exception1/auto-kv-cache-eviction

---

## 0. Scope

This proposal is scoped **strictly as a StreamingLLM replacement**: a way to get StreamingLLM's precision guarantees (rotate-from-raw, no compounding drift) at lower aggregate compute, by rotating survivors once, at eviction boundaries, instead of every decode step. Within that scope — re-rotation vs. re-rotation, not re-rotation vs. leave-gap — this design is a genuine improvement, conditional on the latency-variance question in §5.

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

**Not yet measured:** whether this scales linearly past 10 iterations or compounds faster, and whether 6e-6 is large enough to move attention scores or downstream accuracy at all — it may be real but negligible in practice. A `range(1, 100)` sweep logging max_diff per step is a cheap next step before leaning on this number further.

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

**Caveats on this follow-up sweep, matching the rigor gaps already flagged for §2 and README §5:**
P and `evict_n` were varied together per draw, not independently, so the floor cannot yet be
attributed to `P` magnitude vs. `evict_n` magnitude vs. target-position `P - evict_n` specifically.
Only float32 was tested (production dtypes bf16/fp16 remain unchecked, per §5 item 4 below).
Content was synthetic (`torch.randn`), not real forward-pass activations. And critically, none of
this has been checked against attention-score sensitivity — whether a diff in the 1e-5–1e-4 range
actually moves softmax output enough to matter for the recall task, or is still downstream-
invisible, is unmeasured and is the next thing that would make this result decision-relevant
rather than just a tensor-diff curiosity.

**Open tension with this author's own prior result, not yet resolved:** the companion README (`auto-kv-cache-eviction`) reports renumber+re-rotate underperforming leave-gap by 5–13 points across every eviction-cycle count tested (its §4). That README's own §5 precision check — composing two rotations and comparing against a fresh embedding at the target position, diff 4.77e-7 — is explicitly framed there as evidence *against* rotation-arithmetic error being the cause, i.e. as pointing toward an attention-propagation/geometry effect instead. This RFC's raw-survivor mechanism does not resolve that tension: it still renumbers (once, exactly, per survivor) rather than leaving gaps, so if the README's gap is actually caused by renumbered-position geometry rather than by *compounding* arithmetic error, single-hop rotation inherits the same risk that leave-gap avoids. Liu (arXiv 2602.10959) is a candidate mechanism worth tracking here — it derives a precision-dependent upper bound on the RoPE base beyond which incremental phase updates become numerically indistinguishable, and shows repeated rotary modulation across layers compounds angular misalignment — but it has not been checked against this specific gap and should not be read as confirming a precision-based explanation until it is. Resolving this — precision effect vs. geometry effect vs. something else — is left to a future dedicated experiment (comparing this mechanism against leave-gap on the same recall task used in the companion README), out of scope for this RFC per §0.

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

### 3.4 What this buys, and what it doesn't

**Solid, by construction:**
- Zero compounding precision drift for survivors — every survivor rotation is a single hop from raw.
- Lower aggregate rotation-op count than continuous per-step rotation: `O(1)` amortized per survivor token (one rotation, at window-exit) vs. `O(window_size)` per decode step for continuous rotation.

**Explicitly not solved, and not claimed as solved:**
- Tail latency vs. continuous rotation. Aggregate compute reduction does not imply lower or even comparable tail latency. Continuous rotation's cost is small, constant-shaped, and spread evenly — which is why StreamingLLM's own reported curve (31→65ms/token, cache size 256→4096, Llama-2-7B on an A6000) is mild rather than spiky. This design's cost is zero most steps and then a batch-sized chunk at eviction boundaries — a latency-*variance* tradeoff, not a latency-*reduction* one, and it hasn't been measured. Candidate mitigations if the batch-shaped cost proves to matter in practice: fusing the boundary rotation into an existing kernel launch (e.g. the attention or cache-write kernel) rather than a dedicated launch, so it inherits the "negligible overhead" behavior reported for fused per-step RoPE (FlashInfer) rather than the overhead-bound risk of small standalone kernels; amortizing the known survivor set over the last few steps before a boundary rather than rotating all of it in one step, since survivors are known in advance (§3.2 step 2); and capping batch size via the Δ/window-size ratio (§5 item 2) so the worst case stays bounded. These are directions to try if measurement shows a real problem, not solutions to build preemptively.
- Bump severity is expected to scale with survivor density per eviction event and inversely with window size — smaller windows mean more frequent boundary crossings, which could raise bump frequency enough to erode or reverse the aggregate-compute advantage. Flagged as a real risk, not yet quantified.
- Raw-shadow-copy memory overhead: bounded (only flagged survivors carry a shadow copy, not the whole window), but not yet accounted for numerically.

---

## 4. Relationship to existing work

- **MiniPIC** (Ordonez & Parnell, arXiv 2606.13126) — the substrate this design builds on: unrotated K storage, RoPE applied at attention time via per-request logical positions, sub-100-LOC core-engine change. This proposal is not a restatement of MiniPIC; it's a specific eviction/compaction policy (survivor flagging, raw shadow copies, index-map compaction) built on top of that substrate.
- **StreamingLLM** (Xiao et al., arXiv 2309.17453) — the reference point for continuous per-step rotation's cost curve (§3.4), and the design this proposal is trying to match in aggregate compute while avoiding its per-step cost.
- This proposal does **not** take a position on whether re-rotation should happen at all vs. leave-gap (no correction) — that's a separate, unresolved question (§2.1, §0) or in tension with #51948-style designs and this author's own prior multi-fact ablation results, which found leave-gap outperforming re-rotation in some tested regimes. This RFC assumes re-rotation is the chosen strategy and proposes how to do it cheaply and without precision loss.

---

## 5. Open questions (in priority order)

1. **Batched (eviction-boundary) rotation latency vs. StreamingLLM's continuous 31→65ms/token curve.** Does a per-eviction-event rotation batch, amortized per token between events, come in above or below that continuous-cost curve? This is the load-bearing unmeasured claim in this proposal.
2. **Bump severity as a function of window size and survivor density.** Does a smaller window (more frequent eviction boundaries) degrade tail latency enough to erode the aggregate-compute win? Needs a sweep over window size and Δ (survivor selection interval), not a single configuration.
3. **Drift-scaling sanity check.** Extend the 10-iteration test in §2 to ~100 iterations to confirm the 6.02e-6 figure is linear and not accelerating. **Partially addressed by §2.1:** a follow-up sweep varying `P` and `evict_n` magnitude (rather than iteration count) shows a separate, magnitude-driven error floor that dominates at larger `P`/`evict_n`, plus a compounding term on top of it that becomes visible at that same larger magnitude. Still open: isolating `P` from `evict_n` from target-position magnitude, and the ~100-iteration extension at fixed magnitude originally proposed here.
4. **Raw-shadow-copy memory accounting.** Quantify the memory cost of holding a raw copy for flagged survivors while still in-window, as a function of Δ and window size.
5. **Whether precompute is worth reintroducing.** If (1) shows batched rotation is too slow at the tail, precomputing the raw→logical-position rotation ahead of the eviction boundary (during idle compute) is still available — now precomputing an exact, non-compounding operation rather than hiding a lossy one. Worth revisiting only if (1) shows it's needed.
6. **Attention-score sensitivity to the §2.1 magnitude floor.** Does a K-tensor diff in the 1e-5–1e-4 range (as measured at `P∈[800,1000]`) move softmax attention scores or downstream recall accuracy at all, or is it invisible past that point? This is the load-bearing question raised by §2.1 and is a prerequisite for treating the magnitude floor as anything more than a tensor-diff curiosity.
7. **Isolate P magnitude from evict_n magnitude in the §2.1 sweep.** The follow-up sweep varied both together; a cleaner sweep holding one fixed while varying the other would attribute the error floor to absolute position, correction size, or target position specifically — relevant for whether frequent-small-δ eviction is meaningfully safer than infrequent-large-δ eviction.

---

## 6. Possible future direction: vLLM

vLLM PR **#43374** ("Experimental session KV eviction with attention sinks") adds the scheduler-side plumbing for exactly this kind of eviction — block compaction, multimodal-item-atomic reindexing, prefix-cache invalidation on modified blocks — but explicitly ships without a re-rotation implementation. Its `_post_add_requests` extension hook is a stated placeholder, and its own "Future work" section names "an in-tree RoPE re-rotation kernel" as the missing piece before the experimental gate (`VLLM_ENABLE_EXPERIMENTAL_SESSION_EVICTION`) can be removed.

Within this RFC's scope (§0) — improving on continuous per-step re-rotation specifically — this mechanism is a natural candidate to fill that slot: it satisfies the correctness requirement that PR's author describes (re-rotating surviving K by the eviction delta, without compounding error across repeated evictions in a long-lived session), at lower aggregate cost than a naive per-step re-rotation kernel would be.

A future vLLM-facing version of this RFC should:
- Target filling PR #43374's stated missing piece specifically, rather than proposing a freestanding eviction-strategy change.
- Be prepared to engage with vLLM issue **#51948** ("Bounded-memory video sessions"), a related, production-measured design for a closely adjacent workload (streaming video / multimodal-RoPE) that took a different approach (leave-gap rather than re-rotation). Reviewers familiar with that issue may ask how this proposal relates to it; per §0, this RFC does not take a position on re-rotation vs. leave-gap and that comparison is left to future work.

---

