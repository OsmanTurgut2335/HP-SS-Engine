# HP-SS-Engine

High-Performance Semantic Similarity Search Engine — HPC for AI course project (Fall 2026).

## Goal
Build and profile a semantic similarity search engine, then measure the effects of sparse/dense representations, dimensionality reduction, JIT compilation, and parallel execution.

## Project phases
1. **Baseline & profiling:** Pure Python cosine similarity search, cProfile and line_profiler; identify the top three bottlenecks.
2. **Representations & optimization:** Sparse TF-IDF CSR versus dense embeddings, custom dot products, power-iteration SVD, and Python/NumPy/Numba comparisons.
3. **Parallelism:** Compare at least three parallel strategies and analyze speedups and limitations.
4. **Integration & scaling:** End-to-end pipeline, correctness checks, and 1K/5K/10K/20K document benchmarks.

## Layout
- `notebooks/` — Colab notebooks for each phase
- `src/` — reusable Python modules
- `data/` — dataset metadata and small examples (large corpora stay on Google Drive)
- `benchmarks/` — profiling evidence, benchmark CSVs and analysis logs
- `figures/` — figures used in the report
- `report/` — final report and presentation artifacts

## Reproducibility
Use Google Colab Free CPU as the initial runtime. Record package versions, CPU/RAM, thread settings, random seeds, input sizes, warmups, and at least five timed iterations per benchmark. Keep embedding generation separate from search-only timings. Never report unmeasured speedups.

## Storage
GitHub tracks code, small results, and documentation. Google Drive holds large raw datasets, generated embeddings, and large experimental outputs.

**Status:** Project scaffold in progress; no benchmark measurements have been collected yet.
