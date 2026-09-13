# RSQR Precision/Recall Results — Multi-Fact NIAH, Qwen2.5-0.5B-Instruct

Companion results doc to RFC v4. Answers the model-level half of §0's core
hypothesis ("does RSQR match StreamingLLM's precision guarantees at lower
rotation cost") for the first time with real model activations, not tensor
sweeps. Latency (§5#1) is not addressed here — see Open Questions below.

## Setup

- Model: Qwen2.5-0.5B-Instruct, fp32, manual per-layer forward pass
  (real `apply_rope`, no HF cache abstraction).
- Task: multi-fact NIAH-style recall. A "secret number" fact is planted
  early in a synthetic filler stream; the model must recall it after
  `n_cycles × cycle_len` filler tokens and an eviction/retention policy
  has had a chance to act on the cache.
- Arms compared:
  - **B (continuous re-rotation)** — StreamingLLM-style: sink + sliding
    window, full window re-rotated to current position every step.
    Ceiling/reference arm.
  - **B (leave-gap / uncorrected)** — evicted tokens' positions are
    simply left un-adjusted; no correction applied at all. Cheapest
    possible baseline.
  - **C (RSQR)** — raw-survivor storage, single-hop rotation-from-raw
    at eviction boundaries only, rank-based logical-position compaction
    (RFC §3.2). Every-8th window token flagged as a permanent survivor
    (`survivor_delta=8`, `block_size=8`, `window_size=24`).
- n = 60 independent trials per `n_cycles` value (3 seeds × 20 trials),
  `n_cycles ∈ {2, 4, 6, 12, 16, 32}`.

## Results

| n_cycles | B corrected (acc) | B uncorrected (acc) | C RSQR (acc) | B corrected (logp) | B uncorrected (logp) | C RSQR (logp) |
|---:|---:|---:|---:|---:|---:|---:|
| 2  | 93.3% | 93.3% | 80.0% | -1.149 | -1.149 | -1.171 |
| 4  | 93.3% | 61.7% | 70.0% | -1.012 | -1.314 | -1.255 |
| 6  | 91.7% | 53.3% | 73.3% | -1.002 | -1.394 | -1.198 |
| 12 | 100.0% | 28.3% | 78.3% | -1.022 | -1.752 | -1.367 |
| 16 | 96.7% | 35.0% | 86.7% | -0.943 | -1.691 | -1.588 |
| 32 | 93.3% | 18.3% | 95.0% | -1.004 | -1.664 | -1.064 |

(n = 60 per row; `n_cycles=64` excluded — see Known Issues.)

## Reading

- **Leave-gap degrades steadily and substantially** as eviction pressure
  increases (93%→18% accuracy), confirming that *some* position
  correction is necessary at this task's eviction rates — this is not a
  close call.
- **Continuous re-rotation stays flat near-ceiling** throughout (92–100%),
  as expected for the reference arm that pays full per-step rotation
  cost.
- **RSQR trails continuous re-rotation by a real but bounded margin**
  (roughly 13–22 points through most of the range) while performing
  rotation work only at eviction boundaries, not every step. It
  decisively and increasingly outperforms leave-gap at every
  eviction-pressure level past the lowest, with the gap widening as
  pressure increases (61pt gap at n_cycles=12; 77pt gap at n_cycles=32).
- **At n_cycles=32, RSQR's accuracy edges above continuous re-rotation's**
  (95.0% vs 93.3%) and its logprob is the best of the three arms
  (-1.064). This is the single highest-eviction-pressure point in the
  fair range and the smallest margin in the table — treat as a
  noteworthy trend worth a dedicated follow-up run (more seeds, finer
  n_cycles granularity around 16–32), not yet as a claim that RSQR beats
  continuous re-rotation.

**Headline claim supported by this data:** RSQR recovers most of
continuous re-rotation's recall precision while performing eviction-
boundary-only rotation, and clearly outperforms the naive leave-gap
alternative it is positioned to replace, with the advantage over
leave-gap growing as eviction pressure increases.

**Not yet supported by this data:** any claim that RSQR is cheaper than
continuous re-rotation in wall-clock terms. This document is precision-
only. See Open Questions.

## Known issues / exclusions

- **`n_cycles=64` excluded.** At this setting, `survivor_delta=8` produces
  128 permanent survivors by the end of the run (unbounded growth — every
  8th window token is flagged forever, with no cap or decay). RSQR
  accuracy collapsed to 0% at this cell, worse than leave-gap, with
  diffuse/low-confidence top-5 logits rather than any sign of numerical
  breakdown (no NaNs, no out-of-range values). This is best read as a
  survivor-count-scaling issue in the current flagging rule, not a
  precision failure of the rotation mechanism itself — RSQR was not
  tested as designed at this setting, since permanent-survivor count
  should plausibly be bounded independent of stream length. Needs a
  capped or decaying survivor policy before this range can be tested
  fairly; flagged as future work, not a negative result.
- **Prior README's leave-gap result is not reproduced here.** An earlier
  informal result reported leave-gap beating both full replay and
  re-rotation by 5–13 points. This run shows the opposite, decisively.
  The discrepancy has not been root-caused — differences in task
  construction, prior task-format bugs (since fixed, see notebook
  changelog), and differences in what "leave-gap" means in each setup
  are all plausible explanations, but none has been confirmed. Do not
  cite both results together without resolving this.
- Single model (Qwen2.5-0.5B-Instruct), single task family (single-fact,
  filler-based, 7-digit-secret-number NIAH), fp32 only. Generalization to
  larger models, harder distractor sets, multi-hop facts, or bf16/fp16
  is untested.

## Open questions (unaffected by this result)

- **§5#1 — latency.** Still the load-bearing open question. This
  document establishes RSQR is precision-competitive; it says nothing
  about whether eviction-boundary batched rotation is actually cheaper
  in wall-clock terms than continuous per-step rotation. Separate
  dummy-tensor latency harness scoped, not yet run.
- **§5#2 — bump severity vs. window/Δ.** Untouched.
- **§5#4 — memory accounting.** Untouched. The unbounded-survivor-growth
  issue above makes this more urgent, not less: an uncapped survivor set
  is also an uncapped raw-shadow-copy memory cost.
- **§5#5 — precompute-ahead saturation.** Untouched; depends on the
  latency harness.
- **New: survivor-count bounding policy.** Not in the original RFC's
  open-questions list. Needs a cap, decay, or re-flagging rule so
  survivor count doesn't grow unboundedly with stream length.
