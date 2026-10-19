# Phase 2 — Matrix operations and dimensionality reduction

This branch starts from `feature/phase1-baseline` and preserves all Phase 1 results. The source of truth for requirements is the instructor's assignment, not only the starter code.

## Mandatory checklist

### 2A — Sparse vs Dense
- Real corpus at **1K, 5K, 10K, 20K** documents.
- Sparse TF-IDF in CSR format and **dense semantic embeddings** (not merely `toarray()` on TF-IDF).
- Measure memory footprints and three dot-product cases: sparse–sparse, sparse–dense, dense–dense.
- Implement each dot product ourselves, verify against reference operations, show code and discuss representation choice.
- Note that a sparse–dense dot product requires matching feature dimensions; compare like-for-like by making an explicit common-dimensionality experimental matrix, while separately measuring real TF-IDF CSR and real dense embedding memory.

### 2B — Dimensionality reduction
- Compare at least two approaches, **including own power-iteration SVD** and sklearn TruncatedSVD; NumPy full SVD or PCA may be added on tractable subsets.
- Report time versus components `k`, time versus matrix size, reconstruction Frobenius error at `k=50,100,200`, and memory.
- Explain final choice, TruncatedSVD vs full SVD, PCA vs SVD, and practical crossover k.

### 2C — Vectorization and JIT
- Identical matrix operation across four implementations: pure Python element-wise, NumPy `@`, Numba `@njit`, Numba `@njit(parallel=True)` with `prange`.
- Verify output equivalence, benchmark after Numba warmup, report speedup against Python loops.

## Google Drive project workspace

- [Phase 2 root](https://drive.google.com/drive/folders/13gbotBBHdDsMqdYgUucLhF7b2-4duQIt)
- [2A](https://drive.google.com/drive/folders/1X5PTywZ1jmOTxI1M-GYSbf4c1cCSohCk)
- [2B](https://drive.google.com/drive/folders/1V4E2twjCmyY9Tw1oiI7WqAen27aZ285O)
- [2C](https://drive.google.com/drive/folders/1fqOMuCbmBSdftOsbLPagg006IYy1vUXD)
- [Figures](https://drive.google.com/drive/folders/1PGNkpC53tvOSDF2RQ0uNyk7BcQwuxPJY)
- [Benchmarks](https://drive.google.com/drive/folders/1Cy5DDjOunpRv45zyNegMaDPQf0SXc7tZ)

**Status:** Branch and folder hierarchy created. No Phase 2 measurements have yet been taken; never present projections as measured values.

## Phase 1 baseline scaling note

Pure Python baseline at 100/500/1000 documents measured medians of 29.685/125.223/228.967 ms. The 20K estimate of ~4.6 s per query is only a linear extrapolation, not a benchmark. For Phase 2, use required 1K/5K/10K/20K sizes; if practical, extend the pure Python baseline at matching representation and query inputs, with fewer repetitions documented when expensive.

## Phase 2A 1K pilot — verified Drive artifacts

- Raw CSV: https://drive.google.com/file/d/1I7UJcxIhq0-uZmLq9CtD-nvl10UpXaXm/view (75 rows, **three** workloads, 5 queries × 5 repeats).
- Memory CSV: https://drive.google.com/file/d/1sNhb4sD99B4sD40vwxFmDKsZmnw_LzNO/view (1K only).
- Memory figure: https://drive.google.com/file/d/15WUfBEKUbaZHiYq-7cW9Nw94X-L5qtb7/view
- Latency figure: https://drive.google.com/file/d/1feqmRY5DTqPz09j5fc1uViTB8E0OaDPt/view
- Cached 20K × 384 embedding: https://drive.google.com/file/d/1zS2QlNooedY4qBYyh5h6Um4fppBkj16w/view (30,720,128 bytes).
- Measured median latency, 1K: TF-IDF sparse–sparse **67.943 ms**, embedding sparse–dense **342.490 ms**, embedding dense–dense **13.243 ms**.
- Memory, 1K: TF-IDF CSR **0.2947 MiB**, dense embedding **1.4648 MiB**, embedding CSR **2.9335 MiB**.
- **Important:** CSV lacks the subsequently added `sparse_sparse_embedding` matched-space workload, meaning the pilot ran an older notebook revision. Do not claim the four-workload comparison is finished. Rerun current notebook to obtain 100 rows at 1K, then 400 rows across 1K/5K/10K/20K.
- First Step 2 failure was unsorted TF-IDF CSR indices in custom two-pointer dot; fixed by calling `sort_indices()`. No GPU is needed for the CPU search benchmark; embedding creation was performed on CPU due to Colab GPU quota.
- Interpretation: TF-IDF sparse vs embedding dense is a **cross-representation** comparison and not an isolated format speedup; dense embeddings stored as CSR take more memory because they are nearly fully populated.

## Phase 2A completed — 2026-10-20

Verified 400 measured query times (four workloads × four scales × 25 runs), four memory rows and two updated figures. See [full report](report/phase2a_sparse_dense_results.md), [raw CSV](benchmarks/raw_results/phase2a_dot_products_raw.csv), [memory CSV](benchmarks/raw_results/phase2a_memory.csv), [figures](figures/phase2a_figures.md). Phase 2B and 2C remain pending. Earlier 1K-only pilot was superseded by this full run.
