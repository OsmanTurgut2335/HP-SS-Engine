# Instructor starter-code alignment audit

Source: https://github.com/adaskin/hpc-for-ai-course/tree/main/projects/project1-hp-ss-engine/starter_code

Instructor README explicitly describes files as skeletons and asks students to fill TODOs and adapt them to their dataset. Existing independently implemented notebooks need not be thrown away.

## Phase 1 correspondence

| Starter | Existing project | Status |
|---|---|---|
| phase1_baseline/naive_search.py cosine_similarity_python | notebooks/05_phase1_pure_python_profiling.ipynb pure_python_cosine | Implemented via three explicit element-wise Python loops |
| phase1_baseline/naive_search.py search_naive | pure_python_topk | Implemented, k=10 |
| phase1_baseline/profiling_template.py | cProfile and line_profiler in notebook 05 | Both run; outputs saved |
| benchmark_utils.py repeated timings | benchmarks/raw_results/pure_python_cosine_phase1_raw.csv | 5 queries x 5 repetitions per scale |
| benchmark_utils.py plots | Google Drive figures (indexed by figures/phase1_figures.md) | Completed |

The first CSR/SciPy naive notebook is distinct and does NOT satisfy the pure Python arithmetic requirement by itself; notebook 05 does.

## Phase 2 checklist based on actual starter files

- phase2_matrix_ops/sparse_dense_compare.py: implement sparse-sparse, sparse-dense (manual nonzero iteration), dense-dense dot products; measure time and memory and justify design.
- phase2_matrix_ops/svd_power_iteration.py: own power-iteration-based SVD and reconstruction error; compare reduction methods, dimensions and scaling.
- phase2_matrix_ops/vectorise_vs_loop.py: implement Gram matrix in pure Python loops, NumPy @, Numba serial and Numba parallel; validate and benchmark each.
- Reuse benchmark_utils.py timing conventions or clearly document equivalent methods. Benchmark and compare on identical inputs.

## Phase 3 checklist from starter files

- thread_pool_search.py
- process_pool_search.py
- async_search.py
- ray_search.py

Assignment requires at least three parallel approaches, so choose appropriate implementations based on measured constraints; templates are illustrative.

## Project reconciliation

Preserve existing 20K Qwen corpus, Phase 1 timings, profiler outputs and plots. Do not claim direct speedups across different representation sizes. Starter files can be copied into a dedicated reference directory (retaining attribution), while the actual implementations remain reproducible and runnable against our dataset.

## Remaining important work

Phase 2 and Phase 3 experiments; further sizes 5K/10K/20K; correctness, recall@k, memory, speedup, figure captions and final 10-15 page report.
