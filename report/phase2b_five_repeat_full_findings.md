# Phase 2B — Verified full five-repeat rerun findings (2026-10-21)

**Status: both notebooks completed.** Verified directly against saved Drive CSV contents: [Notebook 07](../notebooks/07_phase2b_dimensionality_reduction.ipynb), [Notebook 08](../notebooks/08_phase2b_decision_validation.ipynb). Original earlier Phase 2B measurements are retained as historical exploratory runs.

## Compliance and artifacts

- Fit-only: [135 records](../benchmarks/phase2b_fit_only_full_5run.csv), **27** N/k/method groups, each with repeat indices **0–4**.
- Search diagnostic: [675 records](../benchmarks/phase2b_retrieval_full_5run.csv), **135** N/k/method/query groups, each with repeats **0–4**.
- Fit+reconstruction: [180 raw records](../benchmarks/phase2b_methods_raw_full_5run.csv), **36** N/k/method groups, each with repeats **0–4**; [36-row median summary](../benchmarks/phase2b_methods_full_5run.csv).
- NumPy 1K full SVD: [5 raw measurements](../benchmarks/phase2b_full_svd_reference_1k_raw_full_5run.csv) and [reference summary](../benchmarks/phase2b_full_svd_reference_1k_full_5run.csv).
- Original full-run figures: [methods comparison](../figures/phase2b_comparison_full_5run.png), [fit-only and retrieval decision](../figures/phase2b_decision_full_5run.png).
- Code: notebook 07 performs one untimed warmup per N/k/method; notebook 08 uses three untimed warmups for each fit and each timed retrieval group. Five full timed repetitions follow. Pilot repeats are fewer and **not used for final conclusions**.

## Representative measured results — N=20K, k=100

**Fit-only medians** (not reconstruction, search or embedding inference):

| Method | Median seconds |
|---|---:|
| NumPy full SVD | 1.436 |
| TruncatedSVD | 3.280 |
| PCA | 3.212 |

Full SVD produces all singular vectors in one decomposition; the **same five full SVD times** are repeated across k=50/100/200 in the CSV for reference, not claimed as 15 independent SVD runs. In this Colab session and matrix shape, full SVD is faster than randomized TruncatedSVD; do **not** claim TruncatedSVD always wins. A full SVD outlier reached 5.286 s (20K); medians mitigate it.

**Fit+reconstruction medians**:

| Method | Median seconds | Relative Frobenius error | Projected storage |
|---|---:|---:|---:|
| Custom power iteration | 5.740 | 0.34428 | 7.629 MiB |
| TruncatedSVD | 4.239 | 0.34442 | 7.629 MiB |
| PCA | 1.935 | 0.34366 | 7.629 MiB |

These fit+reconstruct timings are a **different workload** from fit-only. Power iteration hit its maximum **35 iterations** at N=20K,k=100, so convergence is **not established**.

**Retrieval diagnostic**: five fixed in-corpus queries, five repeats each, k=100, N=20K. Original 384-dim median search 2.544 ms, TruncatedSVD 100-dim median 0.510 ms, PCA 100-dim median 0.526 ms. Mean top-10 overlap with original ranking: TruncatedSVD **94%**, PCA **90%**. These are *ranking-agreement diagnostics*, **not** externally judged relevance accuracy; small query sample, query embeddings drawn from corpus.

**Design decision:** k=100 TruncatedSVD remains a defensible **provisional** representation option: 20K×100 float32 storage 7.629 MiB versus original 20K×384 29.297 MiB (~74% saving), measured ~94% top-10 overlap on five in-corpus queries, and faster search after transformation. But TruncatedSVD has higher fit cost than full SVD in this environment; final Phase 4 should measure actual end-to-end query pipeline and baseline top-k correctness at tolerance 1e-5, using a larger 50–200 query workload. **Reduced dimensions change rankings**; exact baseline top-k correctness must be evaluated on the full-dimensional exact search, not asserted for approximate reduced retrieval.

## Limitations and next steps

1. Timing medians vary with Colab runtime, BLAS threading, garbage collection and session load; compare matched groups within the same experiment.
2. Fit-only and fit+reconstruct measurements are not directly interchangeable.
3. Warmup iterations and timed repetitions are verified in source code and CSV; warmup is not itself stored as a separate timing.
4. Notebook 07 uses custom power iteration; reaching max iterations is not evidence of convergence.
5. Phase 4: pipeline + N=1K/5K/10K/20K scaling, 50–200 queries, five timed iterations and warmup, exact top-k correctness against baseline within 1e-5; distinguish exact 384d retrieval and reduced 100d approximation.
