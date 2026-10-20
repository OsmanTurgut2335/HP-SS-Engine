# Phase 4 — Final System & Scaling

This directory is **only for new Phase 4 artifacts**. Earlier phases remain available under the historical repository paths; we will **not copy Phase 1–3 notebooks, CSVs or figures into this directory**.

- `notebooks/`: Phase 4 pipeline, correctness, scaling and demo notebooks
- `benchmarks/`: Phase 4 raw measurements and summaries (uploaded after real execution)
- `figures/`: Phase 4 generated scaling/correctness plots (uploaded after real execution)
- `report/`: Phase 4 findings and instructor checklist

## Workflow

1. **4A — Final search pipeline:** use real 20K Qwen responses and the cached MiniLM 20K×384 vectors; train/persist TruncatedSVD k=100; compare exact 384-dim and reduced 100-dim BLAS search; demonstrate actual user query text via the *same MiniLM model*. Separate offline preprocessing and per-query latency.
2. **4B — Correctness:** compare optimized exact search against a reference implementation on the same normalized vectors, scores within 1e-5 and top-k IDs; separately report reduced-space ranking overlap, not claim exact equivalence.
3. **4C — Scaling:** N=1K/5K/10K/20K; at least 50 queries, warmups and >=5 timed repetitions per condition, save raw CSV/summary and scaling plot; report whether linear and the measured bottleneck.
4. **4D — Live demo:** text query → embedding → transform → top-k IDs, scores and source texts; show fastest tested system and a full-dimensional reference option.
5. **4E — Final synthesis:** answer 10× corpus hypothesis without claiming it was measured.

**Git caveat:** branches inherit prior committed files. Phase-specific directories give clear separation of *new* artifacts without rewriting earlier history. Use the Phase 4 directory to navigate this phase.
