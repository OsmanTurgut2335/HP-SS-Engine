# Final 20K corpus and model assets — public GitHub Release

**All four files are published** in [v1.0-final](https://github.com/OsmanTurgut2335/HP-SS-Engine/releases/tag/v1.0-final). They are not Git-tracked binaries, so a normal `git clone` stays small. They can be downloaded **without Drive permissions** using [the downloader](../scripts/download_release_assets.py).

| Asset | Size (bytes) | SHA-256 | Direct GitHub download |
|---|---:|---|---|
| Qwen 20K JSONL corpus | 10,830,847 | `00fa40516cb2934e797dfb52e1906ccfcd395d489cf6f5213bc272b2a8940a25` | [Download](https://github.com/OsmanTurgut2335/HP-SS-Engine/releases/download/v1.0-final/qwen_corpus_20k_progress.jsonl) |
| MiniLM 20K × 384 | 30,720,128 | `207e8271de8d8787a1ac4694ca9369721e38ad47d0190a573d08fa0f2510dea9` | [Download](https://github.com/OsmanTurgut2335/HP-SS-Engine/releases/download/v1.0-final/qwen_20k_minilm_384_float32.npy) |
| TruncatedSVD k100 model | 155,543 | `d220f7378fa80c95fa8527dc4439c28061707fbd052dd2e8afcc300e89337e68` | [Download](https://github.com/OsmanTurgut2335/HP-SS-Engine/releases/download/v1.0-final/truncated_svd_k100_full_20k.joblib) |
| Normalized 20K × 100 index | 8,000,128 | `d269d94082c84dfd0838b61a8faeeac83f51fb01b8bbec8959f5352aba5b8321` | [Download](https://github.com/OsmanTurgut2335/HP-SS-Engine/releases/download/v1.0-final/reduced_index_k100_full_20k.npy) |

## Quick use

```bash
python scripts/download_release_assets.py --root ./HP-SS-Engine
python scripts/preflight_assets.py --root ./HP-SS-Engine
```

The Colab [live demo notebook](../notebooks/17_phase4d_live_demo.ipynb) automatically downloads and verifies all four files into `/content/HP-SS-Engine`. **No Google Drive account or mount required.**

Security: `joblib.load` uses pickle under the hood. Never load a model file from an untrusted source. The SHA-256 check binds the download to the exact asset digest published in the GitHub Release metadata.
