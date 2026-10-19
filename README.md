# HP-SS-Engine — High-Performance Similarity Search Engine

Course project: HPC for AI, Fall 2026. This repository implements and benchmarks a text similarity search system, beginning with a deliberately slow pure-Python baseline and progressively testing matrix representations, dimensionality reduction, vectorization/JIT and parallelism.

## Current status

- **Dataset:** 20,000 English responses generated with Qwen2.5-0.5B-Instruct; unique IDs and prompts verified. Full JSONL is stored on Google Drive, not committed to GitHub.
- **Phase 1 completed:** explicit Python-loop cosine, top-10 retrieval, correctness checks, `cProfile`, `line_profiler`, and 75 timed searches at 100/500/1,000 documents. Median times: **29.685 / 125.223 / 228.967 ms**.
- **Phase 2 in preparation:** new `feature/phase2-matrix-ops` branch and dedicated Drive folders. No Phase 2 benchmark results yet.
- **Phase 3/4:** not yet implemented.

## Repository layout

| Path | Purpose |
|---|---|
| `notebooks/03_generate_corpus_resumable.ipynb` | Resumable Qwen text generation |
| `notebooks/04_similarity_baseline.ipynb` | Earlier Python-loop/SciPy CSR baseline (not pure Python arithmetic) |
| `notebooks/05_phase1_pure_python_profiling.ipynb` | Assignment-compliant pure-Python cosine baseline, profiling and plots |
| `benchmarks/raw_results/` | Measured CSV results |
| `benchmarks/profiling/` | cProfile and line_profiler outputs |
| `benchmarks/analysis/` | Experiment/development logs |
| `figures/phase1_figures.md` | Links to measured Phase 1 plots on Drive |
| `report/dataset_20k_results.md` | Corpus audit, preprocessing, benchmark findings |
| `report/starter_code_alignment.md` | Mapping to instructor starter code |
| `PHASE2_PLAN.md` | Phase 2 requirements and Drive workspace |

## Getting started (Google Colab)

1. Open [Phase 1 notebook](https://colab.research.google.com/github/OsmanTurgut2335/HP-SS-Engine/blob/feature/phase2-matrix-ops/notebooks/05_phase1_pure_python_profiling.ipynb).
2. Set up the full corpus in Google Drive at `MyDrive/HP-SS-Engine/data/raw/qwen_corpus_20k_progress.jsonl`. [Existing corpus on Drive](https://drive.google.com/file/d/14kR6SrY0hQ89JUw3awoheUe_tHl-gwcz/view).
3. Run notebook cells top to bottom. The notebook mounts Drive, installs Phase 1 dependencies, loads the corpus, checks numerical correctness, profiles the algorithm and saves CSV/PNG outputs.
4. For a local Python environment, create a virtual environment and run `python -m pip install -r requirements.txt`. Colab-specific Drive mounting requires Colab; local execution of notebooks needs path adjustments.

**Reproducibility:** Fixed 512-feature TF-IDF representation for the pure Python Phase 1 benchmark; 5 queries × 5 repetitions per corpus size; 10 results per query. Profiling instrumentation is not included in timed benchmark runs. Runs are CPU calculations, even if Colab provides a T4 GPU. CPU model, installed versions and thread settings should be recorded in future experiment manifests.

## Phase 1 findings

| Documents | Median search time | Mean search time |
|---:|---:|---:|
| 100 | 29.685 ms | 30.106 ms |
| 500 | 125.223 ms | 121.429 ms |
| 1,000 | 228.967 ms | 236.177 ms |

The `line_profiler` reports approximately 99.7% of the inclusive `pure_python_topk` time on its cosine-similarity call line; sorting contributes about 0.1%. Major bottlenecks are Python element-wise dot loops, repeated norm loops and loop overhead.

[Raw Phase 1 CSV](benchmarks/raw_results/pure_python_cosine_phase1_raw.csv) · [cProfile](benchmarks/profiling/pure_python_1k_cprofile.txt) · [line_profiler](benchmarks/profiling/pure_python_1k_line_profile.txt) · [Figures](figures/phase1_figures.md)

## Phase 2 and instructor requirements

See [PHASE2_PLAN.md](PHASE2_PLAN.md). The assignment requires:
- **2A:** Sparse TF-IDF CSR versus real dense embeddings; memory at 1K/5K/10K/20K and independently implemented sparse–sparse, sparse–dense, dense–dense products.
- **2B:** At least two dimensionality reduction methods, including our own power-iteration SVD; time, memory and reconstruction error at k=50/100/200.
- **2C:** Identical operation in pure Python loops, NumPy, Numba serial and Numba parallel; benchmark speedups against Python.

## Dependencies

See [requirements.txt](requirements.txt). It lists project-wide dependencies, including optional libraries for generation and future phases. These are **not yet frozen to tested exact versions**. Before submission we will capture a clean Colab environment, pin compatible versions, and verify the install and reproduction steps.

## Data and results policy

Large JSONL and future embeddings stay in Drive. Code, lightweight raw CSV results, profiling outputs, documentation and links to figures are tracked in GitHub. Do not compare timing results obtained using different vector dimensionalities or representations as if they were like-for-like speedups. No estimated timing is presented as measured data.
