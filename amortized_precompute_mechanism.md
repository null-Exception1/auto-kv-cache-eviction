# [RFC Draft]: Amortized precomputed re-rotation for KV eviction in long-running sessions

**Status:** pre-draft / not yet posted. Written to capture the reasoning before writing it down loses shape. Treat every claim below as labeled — some are measured, most are not yet.

**Author:** null-Exception1
**Repo:** https://github.com/null-Exception1/auto-kv-cache-eviction

---

## 1. Motivation

Two existing pieces of vLLM work already stake out positions on how to handle KV eviction in long-running sessions, and they disagree:

- **#43374** (gtiwa, "Experimental session KV eviction with attention sinks") ships the plumbing for evicting a token range from a live session, and explicitly scopes RoPE re-rotation of survivors as unimplemented future work. It ships no position-correction default.
- **#51948** (nvbfalk, "Bounded-memory video sessions") measures leave-gap (no re-rotation at all) against a re-send baseline for continuous video captioning, and finds it works well — flat 32ms TTFT, 24-hour sessions, no measurable accuracy loss on their tested tasks. Leave-gap is scoped explicitly to M-RoPE / multimodal sessions.

My own small study (this repo) ran a parallel comparison — full replay vs. renumber+re-rotate vs. leave-gap — on standard 1D-RoPE **text** sessions, and found leave-gap outperforming re-rotation by 5–13 points across every eviction-cycle count tested. This is consistent with, not contradictory to, #51948 — but it raises the question of whether "leave gaps" is actually the right default everywhere, or whether it's a choice that looks free in the regimes both of us tested and stops being free outside them.

This RFC proposes a third position-correction option — **amortized precomputed re-rotation** — and is explicit that it is not yet justified by data. It's justified by three separable hypotheses, laid out below with what's actually been tested.

---

## 2. The three claims, and what's proven vs. hypothesized

### Claim A — Leave-gap accumulates unbounded position growth; re-rotation doesn't.

**Status: structurally true by construction, not yet measured end-to-end.**

In #51948's design, evicting old frames frees memory but does not renumber survivors — new incoming tokens keep climbing in position value regardless of how much has been evicted. This is why their design needs periodic full-replay "consolidation" before position values approach the model's trained horizon. If survivors are renumbered (re-rotated) at every eviction instead, the position axis stays compact indefinitely, and the consolidation step is never needed.

This is a mechanical consequence of the two designs, not something that needs a novel experiment to establish. What's unverified is whether it matters in practice — i.e., whether sessions actually run long enough, without natural restarts, for the consolidation cost to be worth avoiding.

### Claim B — RoPE re-rotation avoids the literature-documented "long-term decay" that leave-gap's unbounded position growth risks running into.

**Status: literature-grounded hypothesis, not measured by either #51948 or this repo.**

There is a documented body of work on RoPE long-term decay: perplexity increases and needle-in-haystack retrieval degrades progressively as position values climb, with position aliasing setting in around 40–50% of a model's trained context length in some studies. Neither #51948 (which consolidates before approaching that zone) nor this repo's own study (max 28 mean evictions, far short of any decay threshold) has run long enough, un-consolidated, to actually observe this. This is the single highest-value experiment this RFC depends on and has not yet run.

**Experiment needed:** run a session long enough — un-consolidated — that leave-gap's climbing position counter enters the documented decay zone, and compare against continuously re-rotated (compact-position) survivors at the same point. This is the one result that would justify re-rotation as intrinsically necessary rather than optional.

### Claim C — Amortized precomputation converts a fixed, occasional accuracy cost into a fixed, occasional non-event, and this is where the practical value is.

**Status: partially supported by this repo's own paired trial data; the core precompute mechanism is unimplemented and unbenchmarked.**

This is the most concrete of the three claims and the actual reason to build the mechanism, independent of whether Claim B holds.

- Re-rotation error is a function of the rotation angle `δ` applied per token, not of how many tokens are rotated together in one event (`K_corrected = apply_rope(K_cached, -δ, FREQS)`, applied uniformly). Rotating 100 survivors by `-δ` costs the same per-token precision loss as rotating 10 survivors by the same `-δ`.
- Consistent with that: B-corrected accuracy in this repo's own trial data does not trend downward as eviction count increases (82.7% → 89.3% → 88.7% → 91.3% across n_cycles 4/8/12/16) — if the cost compounded across events, degradation should worsen with more evictions, and it doesn't.
- This implies the accuracy gap between re-rotate and leave-gap is a **fixed, roughly per-event cost**, not something that accumulates. A larger window between eviction events buys more good-accuracy time without making any individual swap more disruptive — **provided the per-eviction token count (and thus δ) stays fixed as window size grows**, which is a scheduler design constraint to enforce explicitly, not a given.
- **What amortized precompute actually buys:** if the swap cost is fixed and unavoidable, doing it synchronously at eviction time risks a latency spike sized to the retained window (worst-case at large windows). Precomputing the rotated-position K for surviving tokens *ahead of the swap*, during the idle compute headroom of the preceding block, and simply pointer-swapping at the eviction boundary turns that synchronous cost into a scheduled non-event — at the price of holding two versions of surviving K in memory until the swap lands.

**Not yet done:**
- The precompute/pointer-swap mechanism itself is unimplemented — this repo currently only has the synchronous re-rotation reference implementation.
- No latency profiling has been run comparing synchronous rerotation cost against a real per-frame/per-token latency budget, at any realistic window size. The premise that synchronous rerotation is expensive enough to be spike-worthy is currently a guess, not a measurement — see Open Questions.
- No memory-cost accounting for the double-buffered K during the precompute window.

---

## 3. What this is not claiming

- **Not** claiming leave-gap is wrong for #51948's use case (continuous video captioning, M-RoPE, sessions that consolidate periodically). Their measured results stand on their own terms.
- **Not** claiming re-rotation is proven more accurate than leave-gap on any tested config — this repo's own data shows the opposite, at the eviction counts tested.
- **Not** claiming video sessions are inherently more compute-expensive for RoPE than text sessions — this was an early assumption in developing this proposal and there's currently no evidence for it. What actually determines cost is window size at eviction time, not domain (video vs. text) or total stream length.
- **Not yet a vLLM engine change proposal.** No scheduler/config API integration has been prototyped.

---

## 4. Relationship to existing work

- Builds on the eviction plumbing from **#43372** / **#43374** (gtiwa) — `evict_session_token_range()`, `SingleTypeKVCacheManager.evict_and_compact`, the `GPUModelRunner._post_add_requests` hook. #43374 explicitly leaves the re-rotation kernel as future work; this proposal is one candidate shape for that work, not the only one.
- Engages with **#51948** (nvbfalk) — this proposal does not contest their M-RoPE / leave-gap result on its own terms; it asks whether a different correction scheme is warranted for sessions that run long enough to approach RoPE's documented decay zone, which their consolidation step is specifically designed to avoid encountering.
- OpenVINO GenAI ships leave-gap as the default (`apply_rotation=False`) for the same reason cited across this thread (rotation cost) — consistent with, not contradicted by, this proposal, since Claim C's whole point is to make rotation cost close to free via amortization.
- **Counter-evidence to engage with:** "Stateful KV Cache Management for LLMs" (arXiv 2511.04686) argues the opposite in its settings — that non-contiguous eviction can worsen performance and contiguous retention is safer. Not yet reconciled here.

---

## 5. Open questions / experiments needed before this becomes a real proposal

In priority order:

1. **Decay-zone test (Claim B).** Run un-consolidated leave-gap far enough that positions enter the documented RoPE decay range; compare against continuously-compacted re-rotation at the same point. This is the load-bearing experiment for the whole RFC — without it, Claim B is a citation, not a result.
2. **Synchronous rerotation latency profiling (Claim C premise).** Time a single synchronous `apply_rope` counter-rotation call at realistic window/token sizes against a real per-frame or per-token latency budget (e.g. against #51948's own 32ms TTFT figure as a reference point). Determines whether the lag-spike premise is real at any window size worth caring about.
3. **Duty-cycle / recovery-time measurement.** Requires a harness change: multiple accuracy checkpoints *within* a trial (immediately post-swap, and at fixed token offsets after), not just one query at trial's end as in the current repo. Needed to confirm the "fixed per-event cost, doesn't compound" read of the existing B-corrected data, which is currently inferred from an aggregate trend, not a direct measurement of recovery.
4. **Statistical significance on existing data.** Per-trial log is already paired (same stream, same seed, scored by all three strategies) — a McNemar or paired bootstrap test costs no new GPU time and would sharpen the current leave-gap vs. re-rotate comparison.
5. **Precompute mechanism implementation + memory accounting.** Build the actual precompute/pointer-swap path; measure the double-buffer memory cost against the latency savings it buys.
6. **Facts inside the eviction-eligible window.** Both this repo's study and (implicitly) the design here assume important content can be pinned outside the eviction range. A companion run where target content is itself evictable is needed before any of this generalizes to a real scheduler, which evicts by raw position with no pinning concept today.

---

## 6. Sketch: what the mechanism would look like

*(Unimplemented, unverified against any real scheduler API — placeholder only.)*

```
[ normal forward pass, current position ]
        │
        ▼
[ alongside: precompute rotated-position K for surviving
  tokens at their post-eviction (-δ) position, using idle
  compute headroom within the current block ]
        │
        ▼
[ block boundary reached ] ── swap ready? ── yes ──▶ [ pointer-swap
        │ no                                           block for
        ▼                                               precomputed
[ hold synchronous fallback / delay swap ]               version ]
```

Config surface (placeholder, unchecked against vLLM internals): fixed per-eviction token count independent of window size (to keep δ constant per Claim C), a precompute lead-time parameter, and a memory ceiling on double-buffered survivors.

---

*Written up so the reasoning survives past tonight — not ready to post anywhere yet. Section 5 is the actual to-do list.*
