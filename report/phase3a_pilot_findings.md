# Phase 3A — ThreadPool vs BLAS pilot findings (2026-10-20)

**Status: PILOT ONLY.** Verified from actual Google Drive CSV files; three timed repetitions per configuration. No full benchmark results yet.

## Source artifacts
- [Raw pilot CSV](https://drive.google.com/file/d/1lJoiLfv3FllaPd2PhnVi1Kr6IzJkq9yh/view)
- [Summary pilot CSV](https://drive.google.com/file/d/10otA_j0f9L1UnMaXW_AbYsOirZBMIp5M/view)
- [Pilot figure](https://drive.google.com/file/d/16xhs6INcH_tZWkFucuMTdo2rORnUp5vF/view)
- [Notebook](../notebooks/10_phase3a_threadpool_vs_blas.ipynb)

## Setup
Real normalized MiniLM embeddings (384 dimensions); 1000/5000 docs × 8/32 in-corpus queries; top-10; BLAS limited to one thread; ThreadPool workers 1/2/4; CPU count reported **2**. Three repeats, 4 workload combinations × 5 variants × 3 = **60 measured timings**. Search time excludes embedding generation and disk I/O. Code checks top-10 ID agreement across variants.

## Measured median timings (milliseconds)

| Documents | Queries | Sequential | Batched BLAS | Thread 1 | Thread 2 | Thread 4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 8 | 1.084 | 0.994 | 2.201 | 1.770 | 2.170 |
| 1000 | 32 | 4.037 | 2.759 | 7.637 | 4.852 | 10.010 |
| 5000 | 8 | 5.703 | 5.078 | 7.116 | 5.145 | 5.826 |
| 5000 | 32 | 24.646 | 16.992 | 27.753 | 18.876 | 20.365 |

At 5000 docs × 32 queries: Batched BLAS **1.450×**, Thread 2 **1.306×**, Thread 4 **1.210×**, Thread 1 **0.888×** vs sequential NumPy. At 1000 docs × 32 queries Thread 4 was **0.403×** (substantially slower), showing that thread overhead can dominate.

## Preliminary interpretation (not definitive)

- Batched BLAS won all four pilot workloads against sequential NumPy. Batching reduces Python call overhead and uses matrix-matrix multiplication, without worker orchestration.
- ThreadPool 2 workers improved 5000×8 and 5000×32, but lost at both 1000-doc workloads. Threads are **not universally faster**.
- ThreadPool 4 was slower than 2 in every pilot workload; runtime reported only two CPU cores. Oversubscription and scheduling overhead are plausible explanations, not experimentally isolated causes.
- One-worker ThreadPool was slower than direct sequential in all four workloads, reflecting task/executor overhead.
- **Fairness:** ThreadPool parallelizes independent NumPy searches; batched BLAS uses a different operation shape. This is an end-to-end batch-search comparison of alternative implementations, not a claim that thread scheduling alone explains all differences.
- Correctness is asserted in notebook before timing for identical top-10 IDs, but the screenshot only shows the graph; the full-run output still needs review.

## Next steps
- [ ] Run full `PILOT=False` experiment (1K/5K/10K/20K docs, 16/64/128 queries, workers 1/2/4/8, BLAS 1/2 threads, five repeats).
- [ ] Verify full correctness, full raw CSV, summary and figure, then update findings and bottleneck logs.
- [ ] Discuss GIL, BLAS thread competition, Amdahl, memory bandwidth, two-core limit and code complexity using the full data.
- [ ] Compare against matched best Phase 2 search implementation; do not directly compare TF-IDF pure Python or Gram matrix timings as equivalent searches.
