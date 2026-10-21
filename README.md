# HP-SS-Engine — High-Performance Similarity Search Engine

Course project: HPC for AI, Fall 2026. This repository implements and benchmarks a text similarity search system, beginning with a deliberately slow pure-Python baseline and progressively testing matrix representations, dimensionality reduction, vectorization/JIT and parallelism.

## FINAL SYSTEM — CURRENT SUBMISSION STATUS (2026-10-21)

**All four implementation phases are completed.** The earlier sections of this README document the historical incremental development process; for current results use the links below.

- [Phase 4 final verified findings](report/phase4_final_findings.md): correctness, 1K–20K scaling, measured 20K tradeoffs, limitations and live-demo evidence.
- [Phase 4B full correctness CSV](benchmarks/phase4b_correctness_summary_full_20k.csv) and [Top-10 overlap plot](figures/phase4b_svd_top10_overlap_full_20k.png).
- [Phase 4C 60 raw timed measurements](benchmarks/phase4c_scaling_raw_full.csv), [summary](benchmarks/phase4c_scaling_summary_full.csv), [environment](benchmarks/phase4c_environment_full.json) and [scaling plot](figures/phase4c_scaling_full.png).
- [Phase 4D live-demo Colab notebook](notebooks/17_phase4d_live_demo.ipynb) ([open in Colab](https://colab.research.google.com/github/OsmanTurgut2335/HP-SS-Engine/blob/feature/phase4-final-system/notebooks/17_phase4d_live_demo.ipynb)).

### Reproduce the final 20K live demo

1. Use **free Google Colab, CPU runtime**. The source corpus and cached models are on the connected project Google Drive; the repository intentionally does not include large model/data files.
2. Confirm these existing paths under `MyDrive/HP-SS-Engine/`:
   - `data/raw/qwen_corpus_20k_progress.jsonl` (20K Qwen responses)
   - `phase2_matrix_ops/2A_sparse_dense/qwen_20k_minilm_384_float32.npy` (20K×384 float32)
   - `phase4_final_system/models/truncated_svd_k100_full_20k.joblib`
   - `phase4_final_system/models/reduced_index_k100_full_20k.npy`
3. Open `notebooks/17_phase4d_live_demo.ipynb` in Colab; run all setup cells, mount Drive, and wait for `READY`.
4. In the last cell, type a fresh English query and click **Search top 10**. The results and individual timings are computed live.
5. To reproduce measurements, run notebook `15_phase4b_correctness.ipynb` and `16_phase4c_scaling.ipynb` with `PILOT=False`. Scaling uses 100 queries, two warmups, five timed repeats, four corpus sizes, one BLAS thread. Run Phase 4A full first if saved SVD artifacts are absent.

**Important:** The 20K scaling result 190.792 ms is for **100 batched queries** including Top-10 ranking. A single user-observed live query took 69.91 ms end-to-end in one run; that is not a replicated benchmark. Phase 4B's 77% external Top-10 overlap measures agreement with exact 384d ranking, not ground-truth relevance.

### Phase branches

- [Phase 1](https://github.com/OsmanTurgut2335/HP-SS-Engine/tree/feature/phase1-baseline)
- [Phase 2 (isolated)](https://github.com/OsmanTurgut2335/HP-SS-Engine/tree/feature/phase2-matrix-ops)
- [Phase 3 (isolated)](https://github.com/OsmanTurgut2335/HP-SS-Engine/tree/feature/phase3-parallelism)
- [Phase 4 final system](https://github.com/OsmanTurgut2335/HP-SS-Engine/tree/feature/phase4-final-system)

**Final technical report:** [12-page PDF](report/HP_SS_Engine_Final_Technical_Report.pdf) with seven evidence figures. **Still pending:** 15-minute presentation + 5-minute Q&A and rehearsal of the 30-second live demo. The PDF is also saved in the Google Drive project report folder.

---

## Historical development status

- **Dataset:** 20,000 English responses generated with Qwen2.5-0.5B-Instruct; unique IDs and prompts verified. Full JSONL is stored on Google Drive, not committed to GitHub.
- **Phase 1 completed:** explicit Python-loop cosine, top-10 retrieval, correctness checks, `cProfile`, `line_profiler`, and 75 timed searches at 100/500/1,000 documents. Median times: **29.685 / 125.223 / 228.967 ms**.
- **Phase 2A completed:** 1K/5K/10K/20K, 400 measured searches, custom dot-product methods, real 384-dimensional MiniLM embeddings, memory and latency figures. [Results](report/phase2a_sparse_dense_results.md) · [Figures](figures/phase2a_figures.md). Phase 2B/2C complete; see linked full reports and notebooks below.
- **Phase 3/4:** implemented and benchmarked. See the current final-system guide below.

## Repository layout

| Path | Purpose |
|---|---|
| `notebooks/03_generate_corpus_resumable.ipynb` | Resumable Qwen text generation |
| `notebooks/04_similarity_baseline.ipynb` | Earlier Python-loop/SciPy CSR baseline (not pure Python arithmetic) |
| `notebooks/05_phase1_pure_python_profiling.ipynb` | Assignment-compliant pure-Python cosine baseline, profiling and plots |
| `notebooks/06_phase2a_sparse_dense.ipynb` | Phase 2A: real semantic embeddings, CSR TF-IDF, custom dot products, benchmark and figures |
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

## Phase 2A results

[Full measured analysis](report/phase2a_sparse_dense_results.md) · [Raw timing CSV](benchmarks/raw_results/phase2a_dot_products_raw.csv) · [Memory CSV](benchmarks/raw_results/phase2a_memory.csv) · [Figures](figures/phase2a_figures.md). At 20K, TF-IDF CSR occupies 6.701 MiB, dense embeddings 29.297 MiB, embedding CSR 58.670 MiB. Same-embedding dense–dense search median 149.586 ms, versus 6066.276 ms for custom embedding sparse–sparse. This comparison is specific to these algorithms and CPU environment.

## Phase 2B notebook preparation (historical notes; full benchmarks completed)

[Notebook 07 — Dimensionality Reduction](notebooks/07_phase2b_dimensionality_reduction.ipynb) follows instructor `starter_code/phase2_matrix_ops/svd_power_iteration.py` using real cached 20K × 384 MiniLM embeddings. Implements custom block power-iteration SVD, sklearn TruncatedSVD and PCA, and a 1K NumPy full-SVD reference. Pilot is enabled by default (1K, k=50). After correctness, full run will measure N=1K/5K/10K/20K, k=50/100/200, fit+reconstruction time, Frobenius error, projected array storage and plots. **Historical planning note:** Phase 2B full experiments were subsequently completed; see the full results section below.


## Phase 2B complete: measured artifacts

- [Full reduction results](report/phase2b_full_results.md) and [decision analysis](report/phase2b_decision_results.md).
- [Full reduction CSV](benchmarks/raw_results/phase2b_methods_full.csv), [fit-only CSV](benchmarks/raw_results/phase2b_fit_only_full.csv), [retrieval CSV](benchmarks/raw_results/phase2b_retrieval_full.csv), [memory CSV](benchmarks/raw_results/phase2b_projected_matrix_memory.csv), [full SVD reference](benchmarks/raw_results/phase2b_full_svd_reference_1k.csv).
- [Figure index](figures/phase2b_figures.md) (PNGs stored in Google Drive).
- [Notebook 07](notebooks/07_phase2b_dimensionality_reduction.ipynb) and [Notebook 08](notebooks/08_phase2b_decision_validation.ipynb).
- Provisional design choice: TruncatedSVD k=100. At 20K, five in-corpus queries had 94% mean top-10 overlap; this is not ground-truth relevance accuracy. Phase 2C is next.
