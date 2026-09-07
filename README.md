# Autonomous threshold-triggered KV eviction for text sessions — a preliminary study

**Status: research note / pilot replication, not an RFC.** This repository reports a small,
single-model empirical comparison of three KV-cache eviction correction schemes on a synthetic
text recall task. It is not yet a proposal for a vLLM engine change — see "What this is (and
isn't)" below.

## Relationship to existing work

This builds on [#43372](https://github.com/vllm-project/vllm/pull/43372) (session KV truncation
plumbing) and [#43374](https://github.com/vllm-project/vllm/pull/43374) ("Experimental session KV
eviction with attention sinks", @gtiwa), which introduced `evict_session_token_range()`, the
scheduler-side `SingleTypeKVCacheManager.evict_and_compact` path, and the
`GPUModelRunner._post_add_requests` hook for position-correction logic. #43374 explicitly scopes
the RoPE re-rotation kernel as future work.

It also engages with [#51948](https://github.com/vllm-project/vllm/issues/51948) (@nvbfalk), which
independently reaches the opposite default for a different model class; evict without
re-rotating, leaving position gaps in place and scopes that finding to multimodal RoPE (M-RoPE),
explicitly stating it was measured on that model class.

**What this adds:** a small empirical test of whether "gaps are benign" also holds outside M-RoPE,
on a standard 1D-RoPE text model, plus a working (preliminary, not fully verified) re-rotation
implementation.

## Scope

**In scope:** text sessions using standard (1D) RoPE (Qwen2.5, Llama 3, Mistral-style models).

**Explicitly out of scope:** M-RoPE / multimodal sessions. #51948's result was measured on that
model class; this work does not extend or contest that finding on its own terms.

It only asks whether the *same design choice* (leave gaps, don't re-rotate) also holds for 1D RoPE text models,
which #51948 never claimed to cover.

## What this is (and isn't)

This is a pilot study, run on one 0.5B model, one hyperparameter configuration, with a synthetic
task, using a hand-rolled reference attention loop rather than vLLM's actual runtime. It is a
useful data point, not a validated engine design. Concretely, it does **not** yet establish:

- that the result holds at production model scale
- that it holds across sink/window/block configurations (only one was tested)
- that it holds when the *recalled content itself* is eligible for eviction (see "Important
  caveat" below — in this experiment it never was)
- statistical significance of the accuracy gaps reported (see Methodology)

Treat the result as: *"on this model and config, leaving position gaps didn't hurt a
downstream-of-the-fact recall task, and tracked or beat the corrected/replayed alternatives."*
That's worth reporting and building on — it isn't yet a case for shipping anything.

## Important caveat on what was actually measured

In every strategy tested, the three target facts are held in a cache region that is **never
evicted**. Only the filler tokens *between* the facts and the query get evicted and
corrected/left-gapped. This isolates the effect of positional discontinuity from the effect of
losing the fact itself — a reasonable thing to isolate — but it means this experiment does not
yet test the case a real threshold-triggered scheduler would eventually hit: the fact itself aging
out of the window. See "Open questions."

## Results (see METHODOLOGY.md for full detail, task description, and limitations)

Qwen2.5-0.5B-Instruct, synthetic multi-fact recall (3 named facts, 1 queried per trial),
`sink_size=6, window_size=24, block_size=8`, n=150 trials per point across 3 seeds:

| n_cycles | mean evictions | A: full replay | B: renumber + re-rotate | B: leave gap, no re-rotation |
|---|---|---|---|---|
| 4 | 4.0 | 84.0% | 82.7% | 90.0% |
| 8 | 12.0 | 90.0% | 89.3% | 95.3% |
| 12 | 20.0 | 87.3% | 88.7% | 96.0% |
| 16 | 28.0 | 90.7% | 91.3% | 95.3% |

(n_cycles=2 excluded — eviction did not reliably trigger at that config; see METHODOLOGY.md.)

Read with the sample-size caveat in mind (n=150/point, no significance test run yet): re-rotation
tracks full replay closely, and leave-gap does at least as well as both on every row tested here.

## Open questions this work does not resolve

- **Full rerotation verification.** Only one `(P, δ)` pair was checked against a fresh-embedding
  control (see METHODOLOGY.md). A swept range, including large δ and production dtypes
  (bf16/fp16, not just float32), is still needed.
- **Facts inside the eviction window.** The pinned-fact design above needs a companion run where
  the target fact itself can be evicted, which is closer to what a real scheduler would do.
- **Statistical significance.** Per-trial results are paired across strategies (same stream, same
  seed) — a paired test (McNemar / paired bootstrap) on the existing trial log would sharpen this
  considerably and costs no additional GPU time.
- **Scale and config generalization.** One model, one hyperparameter point.
- **Shared-block eviction under prefix caching.** #43374 already flags `evict_and_compact()`
  raising on `ref_cnt > 1` as unsolved; this work doesn't touch it.

## Repo contents

- `METHODOLOGY.md` — task design, strategies compared, full results, precision check, related
  work, and limitations.
- `qwen_multi_fact_eviction.ipynb` — the experiment notebook.
- A future-work sketch of what a vLLM integration might look like is kept in METHODOLOGY.md,
  clearly marked as unimplemented and unassessed.
