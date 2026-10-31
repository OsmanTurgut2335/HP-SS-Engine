# HP-SS-Engine — High-Performance Semantic Similarity Search

**YZM 513 / HPC for AI — Fall 2026** · 20,000 Qwen-generated English documents · MiniLM 384d · TruncatedSVD 100d · CPU BLAS Top-10 retrieval.

**Status:** Phases 0–4 implemented and benchmarked. Public 20K dataset, embeddings and saved search model/index published in [Release v1.0-final](https://github.com/OsmanTurgut2335/HP-SS-Engine/releases/tag/v1.0-final). The 15-minute presentation deck is a separate deliverable and is not yet in this repository.

## Start here

- **[Reproduction guide — read before running](docs/REPRODUCIBILITY.md)**: independent rebuild, Drive paths, dependency installation, asset validation and known limitations.
- **[Live demo in Google Colab](https://colab.research.google.com/github/OsmanTurgut2335/HP-SS-Engine/blob/feature/phase4-final-system/notebooks/17_phase4d_live_demo.ipynb)**: the updated presentation interface. **Automatically downloads and SHA-256 verifies all four public Release assets; no Drive needed.**
- **[12-page technical report](report/HP_SS_Engine_Final_Technical_Report.pdf)**: existing report in repository; any newer review copy should be checked before replacing this artifact.
- [Final Phase 4 findings](report/phase4_final_findings.md) · [Benchmark CSVs](benchmarks/) · [Figures](figures/) · [Presentation evidence notes](report/presentation_notes.md).

**Version note:** Release tag `v1.0-final` was created before the latest GitHub-first notebook and documentation fixes. Its **four downloadable binary assets are correct**; for the current source code use the linked `feature/phase4-final-system` branch, **not** the Release's auto-generated source ZIP. A final source tag can be created once the presentation and final PDF are frozen.

## Final system

```text
20,000 generated text documents
        |
        v
MiniLM 384-dimensional document embeddings (offline)
        |
        v
TruncatedSVD k=100 -> L2-normalized 20K x 100 index (offline)
        |
New text query -> MiniLM 384d -> saved SVD 100d -> normalize
        |
        v
NumPy / BLAS cosine scores -> deterministic Top-10 -> real text excerpts
```

**Search** is CPU-based; a T4 GPU was used for generating the Qwen corpus, not for the measured CPU search. All methods use real, cached vectors rather than synthetic random search inputs.

## Results at a glance

| Experiment | Measured result | Interpretation |
|---|---|---|
| Phase 1 pure Python | 228.967 ms at 1K documents | TF-IDF 512d baseline, not directly comparable to MiniLM |
| Phase 2B memory | 29.297 -> 7.629 MiB at 20K | 384d -> 100d, ~74% storage reduction |
| Phase 3D matched 20K x 64 queries | Sequential NumPy 436.527 ms; batched BLAS 226.436 ms; Numba serial 942.034 ms; parallel 583.025 ms | Same-vector comparison |
| Phase 4B exact correctness | Score error within 1e-5; exact Top-10 matches reference | Exact 384d BLAS compared to float64 reference |
| Phase 4B approximate Top-10 overlap | 86.9% in-corpus, 77% external | SVD vs original 384d ranking, **not human relevance accuracy** |
| Phase 4C 20K x 100 queries | Sequential 384d 779.859 ms; batched 384d 208.815 ms; batched SVD 100d 190.792 ms | Two warmups, five timed repeats, one BLAS thread |

These figures come from **different experiments and workloads**; compare methods within their own benchmark. Interactive Phase 4D query times vary and are not substituted for replicated Phase 4C benchmarks.

## Source code and evidence map

| Phase | Notebook(s) | Evidence |
|---|---|---|
| Corpus generation | [03](notebooks/03_generate_corpus_resumable.ipynb) | [Dataset audit](report/dataset_20k_results.md) |
| Baseline/profiling | [05](notebooks/05_phase1_pure_python_profiling.ipynb) | [Profiling](benchmarks/profiling/) |
| Sparse/dense | [06](notebooks/06_phase2a_sparse_dense.ipynb) | [Results](report/phase2a_sparse_dense_results.md) |
| SVD and PCA | [07](notebooks/07_phase2b_dimensionality_reduction.ipynb), [08](notebooks/08_phase2b_decision_validation.ipynb) | [5-repeat findings](report/phase2b_five_repeat_full_findings.md) |
| Vectorization/JIT | [09](notebooks/09_phase2c_vectorise_vs_loop.ipynb) | [Figures](figures/) |
| ThreadPool/BLAS | [10](notebooks/10_phase3a_threadpool_vs_blas.ipynb) | [Findings](report/phase3a_full_findings.md) |
| ProcessPool | [11](notebooks/11_phase3b_processpool.ipynb) | [Findings](report/phase3b_full_findings.md) |
| asyncio | [12](notebooks/12_phase3c_asyncio.ipynb) | [Findings](report/phase3c_full_findings.md) |
| Matched NumPy/Numba | [13](notebooks/13_phase3d_same_vectors_baseline_numba.ipynb) | [Findings](report/phase3d_full_findings.md) |
| Final pipeline | [14](notebooks/14_phase4a_final_pipeline.ipynb) | [Final findings](report/phase4_final_findings.md) |
| Correctness | [15](notebooks/15_phase4b_correctness.ipynb) | [CSV](benchmarks/phase4b_correctness_summary_full_20k.csv) |
| Scaling | [16](notebooks/16_phase4c_scaling.ipynb) | [Raw CSV](benchmarks/phase4c_scaling_raw_full.csv) · [Figure](figures/phase4c_scaling_full.png) |
| Live demo | [17](notebooks/17_phase4d_live_demo.ipynb) | Interactive Colab |

## Installation and storage

**Quick start in a new Colab session:** [Open the public live demo notebook](https://colab.research.google.com/github/OsmanTurgut2335/HP-SS-Engine/blob/feature/phase4-final-system/notebooks/17_phase4d_live_demo.ipynb) and choose **Runtime → Run all**. It downloads four Release assets, verifies SHA-256, downloads the public MiniLM model, and opens a Top-10 search widget. Network access is required; Google Drive is not.

**Local asset download:**

```bash
python scripts/download_release_assets.py --root ./HP-SS-Engine
python scripts/preflight_assets.py --root ./HP-SS-Engine
python scripts/smoke_test_search.py --root ./HP-SS-Engine
```


- `requirements.txt` — search, profiling and benchmark dependencies, compatible version ranges.
- `requirements-generation.txt` — optional Qwen corpus-generation dependencies.
- `scripts/preflight_assets.py` — validates the 20K corpus, 384d embeddings, fitted SVD and 100d index.
- `scripts/smoke_test_search.py` — exercises CPU Top-10 retrieval using a saved document embedding (no external model download).
- The four exact large assets are in [public Release v1.0-final](https://github.com/OsmanTurgut2335/HP-SS-Engine/releases/tag/v1.0-final), **not in Git history**. [Download and verify](scripts/download_release_assets.py) them without Google Drive credentials. See [asset inventory](docs/FINAL_ASSETS.md).
- The live demo notebook uses the Release directly. Historical benchmark notebooks 00–16 may still use the original Colab Drive cache paths; see [reproduction guide](docs/REPRODUCIBILITY.md).

## Known limitations

The independent public-Release live demo has not yet been certified by a fresh end-to-end Colab run. The `requirements.txt` ranges are not a byte-for-byte environment lockfile; the original environment JSON recorded Python 3.13.16 and NumPy 2.1.3, but not every library version. Phase 1 and Phase 2A figures were copied from Drive into GitHub; source notebooks and raw timing CSVs remain linked for regeneration. Phase 4D measures a single query per interaction; Phase 4C is the controlled five-repeat benchmark. Exact 384d correctness and approximate SVD ranking overlap are distinct properties.
