# Phase 3 Submission Evidence Index

**Branch:** `feature/phase3-parallelism` — all linked repository paths below are on this branch. Measured artifacts are preserved as CSV and original Colab PNG figures; reports separate unlike workloads and document limitations.

## Start here
- [Final five-strategy comparison and instructor checklist](phase3_final_comparison.md)
- [Phase 3D same-vector Numba and pure-Python findings](phase3d_full_findings.md)

## Experiments and notebooks

| Phase | Notebook | Findings | Raw CSV | Summary CSV | Original PNG |
|---|---|---|---|---|---|
| 3A NumPy/BLAS/ThreadPool | [10](../notebooks/10_phase3a_threadpool_vs_blas.ipynb) | [Full](phase3a_full_findings.md) | [Raw](../benchmarks/phase3a_threadpool_raw_full.csv) | [Summary](../benchmarks/phase3a_threadpool_summary_full.csv) | [PNG](../figures/phase3a_threadpool_speedup_full.png) |
| 3B ProcessPool vs others | [11](../notebooks/11_phase3b_processpool.ipynb) | [Full](phase3b_full_findings.md) | [Raw](../benchmarks/phase3b_processpool_raw_full.csv) | [Summary](../benchmarks/phase3b_processpool_summary_full.csv) | [PNG](../figures/phase3b_processpool_speedup_full.png) |
| 3C asyncio I/O and CPU control | [12](../notebooks/12_phase3c_asyncio.ipynb) | [Full](phase3c_full_findings.md) | [Raw](../benchmarks/phase3c_asyncio_raw_full.csv) | [Summary](../benchmarks/phase3c_asyncio_summary_full.csv) | [PNG](../figures/phase3c_asyncio_io_speedup_full.png) |
| 3D matched MiniLM Python/NumPy/Numba | [13](../notebooks/13_phase3d_same_vectors_baseline_numba.ipynb) | [Full](phase3d_full_findings.md) | [Raw](../benchmarks/phase3d_same_vectors_raw_full.csv) | [Summary](../benchmarks/phase3d_same_vectors_summary_full.csv) | [PNG](../figures/phase3d_same_vectors_full.png) |

- [Phase 3B process startup/first-search CSV](../benchmarks/phase3b_processpool_startup_full.csv)
- [Phase 3D correctness CSV](../benchmarks/phase3d_same_vectors_correctness_full.csv)
- [Phase 3A pilot findings](phase3a_pilot_findings.md), [Phase 3B pilot findings](phase3b_pilot_findings.md), [Phase 3C pilot findings](phase3c_pilot_findings.md)
- [Phase 2C original Gram-matrix Numba notebook](../notebooks/09_phase2c_vectorise_vs_loop.ipynb), [Phase 2 findings](phase2_findings_and_open_items.md)
- [Phase 1 pure-Python TF-IDF notebook](../notebooks/05_phase1_pure_python_profiling.ipynb), [Phase 1 findings](phase1_findings_and_open_items.md)

## Read this before comparing speedups

1. Phase 3B CPU search methods are matched **within Phase 3B**. Phase 3D CPU search methods are matched **within Phase 3D**. Different Colab sessions show different absolute times; do not pool them.
2. Historical Phase 1 TF-IDF and Phase 3 MiniLM vector spaces differ. Phase 3D provides an additional same-MiniLM bridge, not a replacement for historical results.
3. Phase 3C asyncio's ~39.1× is simulated I/O waiting, **not** CPU similarity-search or real API embedding generation.
4. Numba `prange` is now tested on the same top-k search task in Phase 3D; the older Phase 2C experiment tested a different Gram-matrix operation.
5. Batched BLAS is the current speed/complexity recommendation, not a universal winner for every workload.
6. At near-equal float32 scores, top-10 **ordering** can change due to accumulation differences; Phase 3A documents this. Phase 3D tested top-10 correctness on its specific queries.
7. Original PNGs and CSVs were uploaded to GitHub from the verified Drive outputs; SVG charts were independently generated from verified summary values.
