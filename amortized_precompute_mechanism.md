# [RFC Draft]: Amortized Precomputed Re-rotation for KV Eviction in Long-Running Sessions

**Status:** pre-draft, not yet posted to vLLM. Every claim below is labeled by evidence type:
measured (this repo's own data), cited (verified against primary sources, 2026-09-08), or
judgment (a design opinion, explicitly not a result). Two experiments specific to this proposal's
actual mechanism — synchronous rerotation latency at eviction time, and the duty-cycle recovery
curve — are still unrun and are the two highest-priority items in the roadmap (§6).

**Author:** null-Exception1
**Repo:** https://github.com/null-Exception1/auto-kv-cache-eviction

---

## 1. Motivation

Two existing pieces of vLLM work stake out positions on how to handle KV eviction in long-running
sessions, and they disagree:

- **#43374** (gtiwa, "Experimental session KV eviction with attention sinks") ships the plumbing
  for evicting a token range from a live session — `evict_session_token_range()`, the
  `SingleTypeKVCacheManager.evict_and_compact` path, the `GPUModelRunner._post_add_requests`
  hook — but explicitly scopes RoPE re-rotation of survivors as unimplemented future work. It
  ships no position-correction default.
- **#51948** (nvbfalk, "Bounded-memory video sessions") measures leave-gap (survivors keep their
  original absolute positions; no re-rotation at all) against a re-send baseline for continuous
  video captioning, and finds it works well — flat 32ms TTFT, 24-hour sessions, no measurable
  accuracy loss on their tested tasks. Leave-gap is scoped explicitly to M-RoPE / multimodal
  sessions, and relies on periodic full-replay "consolidation" to keep position values from
  climbing indefinitely.

The original StreamingLLM paper (Xiao et al., arXiv 2309.17453, §3.2) — the paper both #43374 and
#51948 build on — takes a third position: it caches Keys prior to rotation and re-applies RoPE at
every decoding step using the token's position *within the cache*, not its original absolute
position, calling this crucial for performance. That makes renumber-and-re-rotate the field's
canonical design for this exact problem, not a novel alternative to a leave-gap default. What the
reference implementation does *not* do is batch that correction — it re-rotates the full rolling
window continuously, every step. Its reported decoding latency scales linearly with cache size,
against a quadratically-scaling recomputation baseline, for up to a 22.2× speedup (Fig. 10, Llama-2
and Llama-2-13B, A6000) — real evidence that continuous rerotation doesn't blow up catastrophically
as cache size grows, though it doesn't by itself establish whether that cost is smooth or spiky at
the single-token level.

My own small study (this repo) ran a parallel comparison — full replay vs. renumber+re-rotate vs.
leave-gap — on standard 1D-RoPE **text** sessions, and found leave-gap outperforming re-rotation by
5–13 points across every eviction-cycle count tested, for a fact structurally protected from
eviction. This is the single most load-bearing result this repo has produced, and it complicates
rather than confirms this RFC's premise — it does not straightforwardly support renumbering being
the right default in this text-only, protected-fact setting. §2 (Claim B) holds this tension
explicitly rather than arguing around it.

This RFC proposes a fourth option for the vLLM eviction path this thread has been discussing —
**amortized precomputed re-rotation**, batched by a fixed Δ token count, with the eviction window's
trailing edge intentionally allowed to overshoot by up to Δ tokens between batches — as a way to
get renumbering's structural benefits (bounded position growth, avoidance of the documented RoPE
decay wall) without paying continuous per-step correction's aggregate compute cost. It is not yet
justified as a net win: it is justified by one measured mechanism-level result (Claim C, below) and
three separable, still-open hypotheses.

---

## 2. The four claims, and what's proven vs. hypothesized

### Claim A — Leave-gap accumulates unbounded position growth; renumbering doesn't.

**Evidence type: structural (true by construction) + one directly relevant, verified, text-domain
measurement.**

In #51948's design, evicting old frames frees memory but does not renumber survivors — new
incoming tokens keep climbing in position value regardless of how much has been evicted. This is
why their design needs periodic full-replay consolidation before position values approach the
model's trained horizon. If survivors are renumbered at every eviction instead, the position axis
stays compact indefinitely and consolidation is never needed. This is a mechanical consequence of
how the two designs are built, and it matches StreamingLLM's own canonical recipe (§1).

**Verified supporting evidence:** Poudel (2025, arXiv 2511.04686) measured Llama-3-8B-Instruct
directly and found generation quality degrades sharply once accumulated KV cache size approaches
the model's trained context window (8192 tokens) — a real, measured wall, independent of GPU
memory. Since leave-gap's position counter climbs with every token processed regardless of
eviction, a leave-gap session reaches that wall in fewer real tokens than a renumbered scheme
would.

**Still open:** whether sessions run long enough, without natural restarts, for this wall (or the
consolidation cost that avoids it) to actually matter in realistic deployments — see Claim D and
the roadmap.

### Claim B — Renumbering avoids literature-documented long-term RoPE decay; this repo's own data
shows leave-gap winning in the one configuration actually tested, and that tension is the real
open question this RFC needs to resolve.

**Evidence type: the general decay phenomenon is verified across multiple independent sources.
Whether renumbering is the fix that matters for this proposal's specific eviction pattern is
genuinely unresolved.**

Four independent, verified sources point toward renumbering mattering once positions run
unbounded:

1. **Wang, Min & Zou (arXiv 2601.15300, Jan 2026)** measured Qwen2.5-7B directly: performance
   holds steady, then collapses catastrophically (F1 0.55–0.56 → 0.3) past a critical threshold of
   **40–50% of trained context length**, with 43.2% as one specific cross-validated data point
   within that range. This is a cliff, not a gradual slope — a real margin-of-safety distinction.
   Same model family as this repo's own test model (Qwen2.5-0.5B).
2. **StreamingLLM's own canonical design** (Xiao et al., 2023) renumbers the rolling cache at
   every decoding step, treating compact positions as necessary — predates both #43374 and #51948.
3. **Still (O'Neill et al., arXiv 2606.07878, June 2026)**, verified directly against the paper's
   architecture figure: their iterative compaction pipeline explicitly includes an "un-rotate K"
   step before compression and a "re-rotate Ck" step after, confirming a third, independent, recent
   system treats RoPE un/re-rotation as necessary for iterative long-horizon KV handling.
4. **Decouple-and-Cache (Pang et al., arXiv 2605.01858, May 2026)**, verified against Appendix
   D.2: an ablation on real streaming video (StreamingBench, LLaVA-OV-7B) comparing
   position-agnostic (renumbered) encoding against standard position-aware (leave-gap-style)
   caching found **79.12% (renumbered) vs. 77.84% (leave-gap-style)**, attributed to positions
   reaching out-of-distribution values under real eviction discontinuity.

This also reconciles the apparent tension with #51948: nvbfalk's leave-gap works well specifically
because they periodically consolidate, resetting the position counter before it can reach the
OOD/overflow zone Decouple-and-Cache's ablation probes. Both results are consistent — leave-gap is
safe as long as something resets the position counter before decay sets in; the two papers just
use different reset mechanisms.

**A fifth verified source adds a necessary precision, not simple counter-evidence:** Poudel (2025,
arXiv 2511.04686, also cited under Claim A) documents that scattered, attention-score-based
eviction followed by compaction can scramble a model's sense of relative position, because
compacting non-adjacent survivors makes tokens that were thousands of positions apart appear
adjacent — and recommends contiguous-block retention as mitigation. That recommendation is
structurally closer to both this proposal's and #51948's eviction pattern (contiguous blocks) than
it is a vote against renumbering specifically. The risk it documents is real but scoped to a
different eviction pattern (non-contiguous/scored) than the one this RFC and #51948 both use.

**The actual unresolved tension:** four sources point toward renumbering mattering once positions
run unbounded. This repo's own text-domain data, on the one configuration tested, shows leave-gap
winning by 5–13 points. Both can be true without contradiction if the deciding factor is something
not yet isolated — modality (text vs. video), task structure (this repo protected the recalled
fact from eviction; Decouple-and-Cache's task did not), or session length relative to the decay
threshold (this repo's longest test never approached anything like the 40–50% threshold found in
Claim A/B's evidence). **This is the central open question this RFC exists to resolve, not a
premise it assumes going in.**

### Claim C — Amortized precomputation converts a fixed, occasional accuracy cost into a fixed,
occasional non-event.

**Evidence type: the strongest-supported claim in this document — backed directly by this repo's
own paired trial data. The precompute mechanism itself is unimplemented and unbenchmarked.**

- Re-rotation error is a function of the rotation angle δ applied per token, not of how many
  tokens are rotated together in one event (`K_corrected = apply_rope(K_cached, -δ, FREQS)`,
  applied uniformly). Rotating 100 survivors by −δ costs the same per-token precision loss as
  rotating 10 survivors by the same −δ.
- Consistent with that: B-corrected accuracy in this repo's own trial data does not trend downward
  as eviction count increases (82.7% → 89.3% → 88.7% → 91.3% across n_cycles 4/8/12/16, n=150 per
  point) — if the cost compounded across events, degradation should worsen with more evictions,
  and it doesn't.
- A separate, isolated precision check confirms the rotation math itself is exact at machine
  precision: composing `apply_rope(·, P)` then `apply_rope(·, −δ)` matches embedding fresh content
  directly at `P−δ` to within 4.77e-7 (float32 noise floor) at one tested `(P, δ)` pair — this one
  point does not establish a full sweep, and has not been tested at production bf16/fp16
  precision.
- This implies the accuracy gap between re-rotate and leave-gap, whichever direction it runs, is a
  fixed, roughly per-event cost, not something that accumulates. A larger window between eviction
  events buys more good-accuracy time without making any individual swap more disruptive, provided
  the per-eviction token count (and thus δ) stays fixed as window size grows.
- **What amortized precompute buys, if a real per-event cost exists:** if the swap cost is fixed
  and unavoidable, doing it synchronously at eviction time risks concentrating latency onto
  specific tokens. Precomputing the rotated-position K for surviving tokens ahead of the swap,
  during the compute headroom of the preceding block, and pointer-swapping at the eviction
  boundary turns that synchronous cost into a scheduled non-event — at the price of holding two
  versions of surviving K in memory until the swap lands.

**Confound to rule out before trusting this at production precision:** Wang et al. (arXiv
2411.13476, verified) show numerical error from bf16's limited precision accumulates specifically
as context length grows, with the first token contributing disproportionately. This repo's
accuracy numbers were generated in float32, so this confound hasn't contaminated the existing
result — but the "fixed, doesn't compound" read should not be assumed to transfer to a bf16
production deployment without re-checking.

**Not yet done:** the precompute/pointer-swap mechanism itself is unimplemented (this repo
currently only has the synchronous reference implementation); no latency profiling has been run
comparing synchronous rerotation cost against a real per-token latency budget on real hardware, at
any window size; no memory-cost accounting for the double-buffered K during the precompute window.

### Claim D — Continuous renumbering is the safer default for this RFC's target regime;
periodic consolidation is only preferable when idle windows are guaranteed.

**Evidence type: design judgment, not a measured result.**

- Periodic consolidation's cost is opportunistic — it depends on an idle window arriving before
  the position counter reaches the decay zone. The sessions most likely to run continuously without
  a natural break are exactly the sessions where scheduling a consolidation pass is hardest. The
  failure mode is silent quality decay under sustained load, with no obvious trigger to point to
  afterward.
- Continuous (or batched) renumbering doesn't have that race condition — deterministic and bounded
  by construction, independent of whether a traffic gap ever arrives.
- Where consolidation is still the better choice: deployments with guaranteed idle windows by
  protocol design (e.g. turn-based chat with real gaps between messages). There, consolidation's
  simplicity — no double-buffering, no per-eviction bookkeeping — is a legitimate win over
  renumbering's small but permanent cost.
- This RFC's target regime (video streams, always-on agents, sessions with no guaranteed idle
  time) is specifically the case where consolidation's dependency on idle time is weakest.

**Not yet measured:** how often consolidation windows actually fail to arrive in production
traffic, or how severe the resulting decay is when they don't.

---

## 3. What this is not claiming

- **Not** claiming leave-gap is wrong for #51948's use case. Their measured results stand on their
  own terms.
- **Not** claiming re-rotation is proven more accurate than leave-gap on any tested config — this
  repo's own data shows the opposite, for a fact structurally protected from eviction. This is the
  central open tension in §2, not something argued around.
- **Not** claiming video sessions are inherently more compute-expensive, more information-dense,
  or otherwise mechanically different from text sessions in a way that would make this repo's
  leave-gap-wins result stop applying to streaming. No evidence was found for that claim, and
  "streaming has more/faster information" is not on its own a mechanism that predicts which
  correction scheme wins.
- **Not yet a vLLM engine change proposal.** No scheduler/config API integration has been
  prototyped.

---

## 4. Relationship to existing work

- Builds on the eviction plumbing from **#43374** (gtiwa) — see §1. #43374 explicitly leaves the
  re-rotation kernel as future work; this proposal is one candidate shape for that work, not the
  only one.
- Engages with **#51948** (nvbfalk) — does not contest their M-RoPE/leave-gap result on its own
  terms; asks whether a different correction scheme is warranted for text sessions, where this
  repo's own data currently suggests leave-gap may already be sufficient.
- OpenVINO GenAI ships leave-gap as the default (`apply_rotation=False`) for the same cost reasons
  discussed here — consistent with this repo's own text-domain result, not contradicted by it.
- **Decouple-and-Cache** (Pang et al., arXiv 2605.01858) — quantifies the leave-gap-vs-renumber gap
  on real streaming video (§2, Claim B). Its mechanism is closer to StreamingLLM's "store
  pre-rotation, apply position transform at use time with reassigned contiguous IDs" than to this
  proposal's "rotate once, then apply a corrective delta to already-rotated keys" — functionally
  similar in spirit, not the same mechanism.
- **Poudel (2025, arXiv 2511.04686)** — documents both the context-window-wall risk (Claim A) and
  a scattered-eviction failure mode scoped to a different eviction pattern than this proposal's
  (a precision to Claim B, not counter-evidence).
- **Closest related work, needs explicit distinction:** Still (O'Neill et al., arXiv 2606.07878)
  independently treats RoPE un-rotate/re-rotate as necessary (§2, Claim B). Differences from this
  proposal:
  - Still is a **trained** per-layer Perceiver producing a compressed latent KV representation;
    this proposal is a **training-free** geometric operation (exact rotation), no learned
    parameters.
  - Still's lookahead buffer (one raw chunk held ahead of compaction, described in the paper as
    "the standard procedure used by iterative results") amortizes a learned compactor's cost
    ahead of the critical path. This proposal's precompute-ahead-of-swap buffer amortizes a fixed,
    exact per-event rotation cost. Whether Still's specific rationale is approximation-error
    compounding or something else is not fully confirmed from the paper's body text — the
    mechanisms may differ in failure mode (learned-model drift vs. scheduling latency) even though
    both are "do the expensive thing ahead of time" in spirit.

---

## 5. Sketch: what the mechanism would look like

*(Unimplemented, unverified against any real scheduler API — placeholder only.)*

The window's leading edge is continuous, always attached to the latest token — new tokens enter
one at a time as normal. The trailing edge is intentionally not strictly bounded at every instant:
it's allowed to overshoot the nominal window size by up to Δ tokens between eviction events, and
only snaps back when a Δ-sized eviction batch fires. This is a deliberate choice to batch rotation
work, not an oversight — the overshoot tokens are headed for eviction/compaction anyway, so a few
extra tokens of memory headroom between batches is an acceptable cost. Batching is cheaper in
aggregate compute (`N × window_size / Δ` total rotation-ops over a session) than continuous
per-step correction (`N × window_size`), unless slot-relative addressing makes per-step rotation
O(1) — unverified, and a substantially harder kernel-level property to build than the batched
approach.

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

Config surface (placeholder, unchecked against vLLM internals): fixed per-eviction token count
independent of window size (to keep δ constant per Claim C), a precompute lead-time parameter, and
a memory ceiling on double-buffered survivors, including the Δ-token overshoot headroom.

---

## 6. Roadmap

Ordered by what would most change whether this proposal is worth pursuing, not by ease.

### Phase 0 — free, no GPU time
1. **Statistical significance on existing data.** Per-trial log is already paired (same stream,
   same seed, scored by all three strategies) — a McNemar or paired bootstrap test on the existing
   150-trial-per-point data costs nothing and sharpens the current leave-gap vs. re-rotate
   comparison before any new experiment is designed around it.

### Phase 1 — resolves the central tension (§2, Claim B)
2. **Unpin the target fact.** Re-run this repo's existing paired comparison (full replay /
   renumber+re-rotate / leave-gap) with the target fact itself evictable, instead of structurally
   protected. This isolates whether leave-gap's win in the current data is a property of the
   correction scheme or an artifact of testing only content that never needed correcting.
3. **Push toward the decay threshold.** Re-run the same comparison at eviction-cycle counts high
   enough to approach the 40–50%-of-context-length range identified for Qwen2.5-7B (Claim B,
   citation 1) scaled to this repo's Qwen2.5-0.5B test model. Current tests never ran long enough,
   un-consolidated, to enter anything like that zone.

### Phase 2 — validates this proposal's specific mechanism, not just renumbering in general
4. **Synchronous rerotation latency profiling.** Time a single synchronous `apply_rope`
   counter-rotation call at realistic window/token sizes against a real per-token latency budget,
   on real hardware. Nothing in the literature reviewed for this RFC profiles this directly for a
   batched, eviction-triggered correction — this needs to be run from scratch, not compared
   against a borrowed number.
5. **Duty-cycle / recovery-time measurement.** Requires a harness change: multiple accuracy
   checkpoints within a trial (immediately post-swap, and at fixed token offsets after), not just
   one query at trial's end as in the current repo. Needed to directly confirm the "fixed
   per-event cost, doesn't compound" read of the existing data, which is currently inferred from
   an aggregate trend rather than measured as a recovery curve.
6. **Precompute mechanism implementation + memory accounting.** Build the actual precompute/
   pointer-swap path; measure the double-buffer memory cost (including Δ-token overshoot
   headroom) against whatever latency savings Phase 2's profiling shows.

### Phase 3 — generalization and deployment realism
7. **bf16 precision re-check.** Re-run Claim C's precision comparison at production bf16/fp16,
   not just the current float32 point, to rule out the confound documented in arXiv 2411.13476.
8. **Consolidation-window reliability in real traffic (Claim D).** Measure how often guaranteed
   idle windows actually materialize in realistic streaming traffic patterns, and how severe decay
   gets when they don't — the evidence that would turn Claim D from a design judgment into a
   measured argument.
9. **Real scheduler integration.** No scheduler/config API prototyping has happened yet; this
   depends on Phases 1–2 landing first, since it's premature to integrate a mechanism whose core
   cost/benefit tradeoff isn't yet measured.

Phase 1 is the gating step for whether this RFC's central claim is even directionally right for
text. Phase 2 is the gating step for whether this proposal's specific mechanism (vs. plain
continuous renumbering) is worth the added complexity. Nothing past Phase 2 is worth doing until
those land.

---

## 7. Verification log

Citation-checking pass performed 2026-09-08, against primary sources, not summaries or secondary
citations.

| Claim | Source | Result |
|---|---|---|
| Decouple-and-Cache 79.12 vs 77.84 | arXiv 2605.01858, Appendix D.2 | Confirmed exact, direct quote match. |
| Qwen2.5-7B degradation threshold | arXiv 2601.15300 | Confirmed. Headline threshold is **40–50%** of trained context length, with 43.2% as one cross-validated data point within that range — not itself the headline figure. |
| Still's un-rotate/re-rotate architecture | arXiv 2606.07878 | Confirmed directly from the paper's architecture figure and body text. |
| Still's lookahead-buffer rationale | arXiv 2606.07878 | Buffer's existence and "standard procedure" status confirmed; the specific approximation-error-compounding rationale is not fully confirmed from available text — left flagged as open in §4. |
| Poudel, context-window wall + scattered-eviction risk | arXiv 2511.04686 | Confirmed, and split correctly across two claims — a Claim A support and a Claim B precision-scoping, not a single undifferentiated counter-citation. |
| StreamingLLM linear-vs-quadratic latency scaling, 22.2× speedup | arXiv 2309.17453, Fig. 10 | Confirmed. (An earlier internal draft had attached specific per-token millisecond figures to this citation that do not appear in the primary source; those figures have been removed and are not part of this document.) |
| BFloat16/RoPE precision interaction | arXiv 2411.13476 | Confirmed — matches the accumulation-with-context-length and first-token-contribution claims in Claim C almost verbatim. |

---

*Section 6 is the actual to-do list. Nothing past Phase 1 is worth building until the central
tension in Claim B is resolved one way or the other.*
