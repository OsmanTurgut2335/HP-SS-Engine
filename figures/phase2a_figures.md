# Phase 2A figures

Both PNGs are generated from actual 1K/5K/10K/20K experiment CSVs by [notebook 06](../notebooks/06_phase2a_sparse_dense.ipynb).

- **Figure 2A-1 — Representation memory footprint:** [PNG on Drive](https://drive.google.com/file/d/15WUfBEKUbaZHiYq-7cW9Nw94X-L5qtb7/view). Shows CSR TF-IDF, dense MiniLM embeddings and the same embeddings stored as CSR. 20K: 6.701 / 29.297 / 58.670 MiB.
- **Figure 2A-2 — Custom dot-product latency:** [PNG on Drive](https://drive.google.com/file/d/1feqmRY5DTqPz09j5fc1uViTB8E0OaDPt/view). Median of 25 runs per workload per corpus size. Four curves: TF-IDF sparse–sparse, embedding sparse–sparse, embedding sparse–dense, embedding dense–dense.

Source data: [timings](../benchmarks/raw_results/phase2a_dot_products_raw.csv) and [memory](../benchmarks/raw_results/phase2a_memory.csv). Interpretation and caveats: [Phase 2A report](../report/phase2a_sparse_dense_results.md).

PNG binary files are in Google Drive, linked from GitHub. The notebook can regenerate them.
