# Reproducing HP-SS-Engine (GitHub-first)

**Source of truth:** the [GitHub repository](https://github.com/OsmanTurgut2335/HP-SS-Engine/tree/feature/phase4-final-system) contains all source notebooks, benchmark CSV/JSON, figures, report and documentation. The **four exact 20K data/model artifacts** are published as public downloadable assets in [Release v1.0-final](https://github.com/OsmanTurgut2335/HP-SS-Engine/releases/tag/v1.0-final). **Google Drive is not required for the final live demo.**

## Fastest: run live search in a new Colab account (no Drive)

1. Open [Notebook 17 in Colab](https://colab.research.google.com/github/OsmanTurgut2335/HP-SS-Engine/blob/feature/phase4-final-system/notebooks/17_phase4d_live_demo.ipynb).
2. Choose **Runtime → Run all**. The first cell installs dependencies and downloads the four public Release files into `/content/HP-SS-Engine/`. It checks their **published SHA-256 digests** and byte sizes. This requires internet access for GitHub, PyPI and the MiniLM model on Hugging Face; **no Google Drive permission**.
3. Wait for the MiniLM model to download and the warmup to pass. Enter a new English query and click **Search Top 10**.
4. The displayed timing is for that single live query; the five-repeat batch measurements are in Phase 4C and must not be conflated.

## Download and validate all four assets locally

Python 3.11–3.13 is a reasonable installation target, but a fresh dependency-resolve/run has not been certified across all these versions. In a local checkout:

```bash
python -m pip install -r requirements.txt
python scripts/download_release_assets.py --root ./HP-SS-Engine
python scripts/download_release_assets.py --root ./HP-SS-Engine --verify-only
python scripts/preflight_assets.py --root ./HP-SS-Engine
```

This puts the 20K corpus, MiniLM embeddings, SVD model and normalized 100d index in the paths expected by Phase 4 notebooks. The downloader reuses valid files, replaces invalid files and checks SHA-256 before accepting a download. **The preflight validates shapes, IDs and normalization**; it is not a substitute for rerunning benchmark experiments. The `.joblib` model uses Python pickle serialization; only load this trusted published asset.

| Asset | Target under `./HP-SS-Engine` |
|---|---|
| 20K corpus | `data/raw/qwen_corpus_20k_progress.jsonl` |
| 20K × 384 float32 MiniLM | `phase2_matrix_ops/2A_sparse_dense/qwen_20k_minilm_384_float32.npy` |
| TruncatedSVD k=100 | `phase4_final_system/models/truncated_svd_k100_full_20k.joblib` |
| 20K × 100 float32 index | `phase4_final_system/models/reduced_index_k100_full_20k.npy` |

[File sizes, SHA-256 hashes and direct download URLs](FINAL_ASSETS.md).

## Reproducing all experiments (separate from live demo)

The original notebooks 00–16 were authored and benchmarked in Google Colab with Drive as an **output/cache directory**. Many still mount `/content/drive/MyDrive/HP-SS-Engine`. If reproducing *those exact notebooks*, mount your **own** Drive and place the downloaded assets in that same project folder. This is an original-experiment storage convention, **not a restriction on downloading the data or using Notebook 17**.

- 03: regenerate 20K Qwen responses (T4 GPU recommended; expensive and not necessary for using the Release corpus).
- 05: pure Python cosine and profiling.
- 06: sparse/dense representations and MiniLM embeddings.
- 07–09: SVD/PCA and vectorization/JIT.
- 10–13: ThreadPool, ProcessPool, asyncio and matched NumPy/Numba.
- 14: final pipeline; set `PILOT=False` for 20K. If saved model/index exist, it loads them.
- 15: correctness; set `PILOT=False`.
- 16: scaling; set `PILOT=False`; two warmups and five timed repetitions.
- 17: independent GitHub Release live demo; **no Drive mount**.

Original experiment outputs are versioned in [benchmarks](../benchmarks/); figures in [figures](../figures/). Some notebooks also write their own Drive cache.

## Reproducibility and methodological limits

- Reference measured Phase 4C environment: Python 3.13.16, NumPy 2.1.3, two logical CPUs, one BLAS thread. Other installed package versions were not all recorded; `requirements.txt` contains compatibility ranges rather than a verified lockfile.
- Phase 1 uses 512d TF-IDF and is **not** a matched-input speedup comparison with final MiniLM/SVD.
- Phase 3D uses matched 384d vectors for the fair NumPy/Numba comparison.
- Phase 4B exact 384d Top-10 matches a float64 reference within 1e-5 score tolerance; approximate SVD100 rankings differ by design.
- Phase 4C 190.792 ms is the median **100-query batch** runtime, not a single live-query latency.
- The public Release's file names, sizes and SHA-256 digests were checked via GitHub's release metadata. A fresh Colab run and a full end-to-end download/semantic-search execution must still be performed to certify runtime compatibility.
