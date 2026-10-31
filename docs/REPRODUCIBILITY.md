# Reproduction and Drive/GitHub asset policy

**Source of truth:** GitHub branch `feature/phase4-final-system` holds all project notebooks, code, reports, figures and benchmark tables. **Google Drive is an artifact cache**, not a separate source-code repository. Large binary assets are intentionally excluded from GitHub via `.gitignore`.

## 1. What is in GitHub and what is not?

| Artifact | GitHub | Google Drive |
|---|---|---|
| All notebooks 00–17, scripts, README, dependency files | Yes | Not required |
| Raw benchmark CSVs, environment JSON, profiling text | Yes | May also have run copies |
| Final plots and final report | Yes | May also have copies |
| Qwen 20K JSONL corpus (~10.8 MB) | No | `data/raw/qwen_corpus_20k_progress.jsonl` |
| MiniLM 20K×384 float32 (~30.7 MB) | No | `phase2_matrix_ops/2A_sparse_dense/qwen_20k_minilm_384_float32.npy` |
| Saved SVD k=100 model and 20K×100 index | No | `phase4_final_system/models/` |

**Access limitation:** The Drive folder is not a public GitHub artifact. A reader without access to the author's Drive **cannot run the cached-asset demo immediately**. The notebooks below allow independent reconstruction in the reader's own Google Drive. If the instructor needs the *exact same cached corpus/index*, share those files directly with the instructor or provide a controlled artifact archive. Do not claim the current repository is a zero-download, fully self-contained checkout.

## 2. Rebuild on your own Google Drive (independent path)

1. Open [Google Colab](https://colab.research.google.com/) and mount **your own** Drive. All notebooks use `/content/drive/MyDrive/HP-SS-Engine`; create that folder in your Drive. Use the GitHub Colab links below, not copies of stale notebooks.
2. [Notebook 03 — generate Qwen corpus](../notebooks/03_generate_corpus_resumable.ipynb): select a **T4 GPU** runtime. Generate 20,000 English responses and write `data/raw/qwen_corpus_20k_progress.jsonl`. This is the expensive offline generation stage. Sampling settings are fixed in the notebook, but hardware/library versions can affect exact output text.
3. [Notebook 06 — MiniLM embeddings](../notebooks/06_phase2a_sparse_dense.ipynb): produce `phase2_matrix_ops/2A_sparse_dense/qwen_20k_minilm_384_float32.npy`. The notebook also benchmarks sparse/dense methods; let the embedding cache complete before Phase 4.
4. [Notebook 14 — final pipeline](../notebooks/14_phase4a_final_pipeline.ipynb): change `PILOT=True` to `PILOT=False`, then **Run all**. It fits/saves the full 20K `truncated_svd_k100_full_20k.joblib` and `reduced_index_k100_full_20k.npy`.
5. [Notebook 15 — correctness](../notebooks/15_phase4b_correctness.ipynb): set `PILOT=False` and rerun; check exact 384d score tolerance `1e-5` and separate SVD Top-10 overlap.
6. [Notebook 16 — scaling](../notebooks/16_phase4c_scaling.ipynb): set `PILOT=False`, rerun; 100 queries, two warmups, five timed repetitions, corpus sizes 1K/5K/10K/20K, BLAS threads=1.
7. [Notebook 17 — live demo](../notebooks/17_phase4d_live_demo.ipynb): run all setup cells, then enter an English query and press **Search Top 10**.

**Direct Colab:** https://colab.research.google.com/github/OsmanTurgut2335/HP-SS-Engine/blob/feature/phase4-final-system/notebooks/17_phase4d_live_demo.ipynb

## 3. Using existing cached artifacts

If the instructor has been given access to the Drive assets, place them under the exact paths listed in the table above. The original Drive folder: https://drive.google.com/drive/folders/1v3QHC93L1Y0XLYmvyQDh08wbxHvk7LVn . Access is subject to the owner's permissions.

After mounting Drive, validate the four essential files using `scripts/preflight_assets.py` (from a cloned repo):

```bash
python scripts/preflight_assets.py --root /content/drive/MyDrive/HP-SS-Engine
```

The preflight checks document IDs, nonempty responses, embedding/index shapes and dtypes, SVD component shape, and reduced-index normalization. **It does not substitute for running the correctness benchmarks.**

## 4. Installation

The current code is notebook-first. Notebook setup cells install their own dependencies; for local development:

```bash
python -m pip install -r requirements.txt
# Only if generating the corpus from scratch:
python -m pip install -r requirements-generation.txt
```

Reference recorded Phase 4C environment: Python 3.13.16, NumPy 2.1.3, Linux, two logical CPUs, BLAS threads=1. Exact versions of other packages were not logged, so `requirements.txt` specifies compatible ranges rather than claiming a fully frozen environment. CPU timing results can vary across machines and sessions.

## 5. Evidence, correctness and fair comparisons

- Benchmark CSVs are versioned in [benchmarks/](../benchmarks/); plots are in [figures/](../figures/).
- Phase 1's pure Python baseline uses TF-IDF; do **not** compare it directly with final MiniLM/SVD search as a same-vector speedup.
- Phase 3D is the controlled same-vector comparison.
- Phase 4B verifies float32 BLAS against a float64 reference (tolerance `1e-5`). The 100d SVD index is approximate and does **not** promise identical Top-10 rankings.
- Phase 4C's 190.792 ms is a **100-query batch median**, not the latency of a single interactive query.
- Phase 4D shows a real interactive query and a per-run timing breakdown. Single demo times are not five-repeat benchmarks.

## 6. Data integrity and cleanup policy

Never delete raw JSONL, cached embeddings, model/index files or raw timing CSVs to reduce folder clutter. These are the reproducibility assets. GitHub is authoritative for code and documentation; Drive holds the heavy data/cache only. Do not treat the two locations as automatic bidirectional synchronization.
