# Final system assets — verified Google Drive files

All four files below were verified to exist in the project Google Drive on 2026-10-31. They are **not yet publicly readable or included in the GitHub checkout**. Access depends on Google Drive permissions.

| Artifact | Size (bytes) | Drive file |
|---|---:|---|
| 20K Qwen JSONL corpus | 10,830,847 | [qwen_corpus_20k_progress.jsonl](https://drive.google.com/file/d/14kR6SrY0hQ89JUw3awoheUe_tHl-gwcz/view) |
| 20K×384 MiniLM embeddings | 30,720,128 | [qwen_20k_minilm_384_float32.npy](https://drive.google.com/file/d/1zS2QlNooedY4qBYyh5h6Um4fppBkj16w/view) |
| Fitted SVD k=100 | 155,543 | [truncated_svd_k100_full_20k.joblib](https://drive.google.com/file/d/1AawoFeL2KfWL7rPmFG-FIW29QB4fEHCv/view) |
| 20K×100 final index | 8,000,128 | [reduced_index_k100_full_20k.npy](https://drive.google.com/file/d/1GGwsLxpvcIHT1QT0C3JSg2rN7Xe5R-v7/view) |

## Place in the Colab project Drive folder

```text
HP-SS-Engine/
├── data/raw/qwen_corpus_20k_progress.jsonl
├── phase2_matrix_ops/2A_sparse_dense/qwen_20k_minilm_384_float32.npy
└── phase4_final_system/models/
    ├── truncated_svd_k100_full_20k.joblib
    └── reduced_index_k100_full_20k.npy
```

Run `python scripts/preflight_assets.py --root /content/drive/MyDrive/HP-SS-Engine` after copying files into your own Drive.

**Public access:** A file link is not proof of public access. For public, link-only read access, the owner must set **Share → General access → Anyone with the link → Viewer** on each file (or a common containing folder) and verify using an incognito window. The available Drive connector cannot set public anonymous link access; it can share individual files with a named email address. Do not claim public access before verification.

**GitHub distribution:** For an independent download without Drive access, attach these four binaries to a GitHub Release (preferred) or upload them to the repository and revise `.gitignore` as needed. The existing repo currently ignores `*.npy` and `*.joblib`. If publishing a release, update the download links in this manifest to the actual release asset URLs. `joblib` is a pickle-based format: load only the trusted file supplied with this project, never untrusted third-party joblib assets.
