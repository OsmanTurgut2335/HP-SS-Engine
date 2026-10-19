# Phase 2B — verified pilot (2026-10-20)

Real cached MiniLM 384-dimensional embeddings; N=1000, k=50, CPU, one run per method. These are **pilot measurements**, not the full scaling experiment.

| Method | Fit + reconstruction time (s) | Absolute Frobenius error | Relative Frobenius error | Reduced matrix MiB |
|---|---:|---:|---:|---:|
| Custom block power iteration (35 iterations) | 0.185005 | 15.653665 | 0.495012 | 0.190735 |
| sklearn TruncatedSVD (randomized) | 0.172626 | 15.654336 | 0.495034 | 0.190735 |
| sklearn PCA (randomized) | 0.109396 | 15.582773 | 0.492771 | 0.190735 |

NumPy full SVD reference on 1K: decomposition 0.110495 s; rank-50 relative Frobenius error 0.494949; rank-100 0.328016; rank-200 0.159176. NumPy SVD time is *fit only*, so not directly comparable to fit+reconstruction timings above. PCA centers columns; error objectives differ from uncentered SVD. Errors are not retrieval accuracy metrics.

**Interpretation:** own power iteration is close to optimal rank-50 reference (0.495012 vs 0.494949 relative error). PCA was fastest on this single pilot run, but this is not sufficient to choose a final method. Projected float32 storage 0.190735 MiB vs original 1.464844 MiB (87% smaller); this excludes fitting/model memory. The pilot plot contains only one point per method and does not establish scaling.

## Artifacts

- [Pilot CSV (Drive)](https://drive.google.com/file/d/1mdPh0UA775bpJEcAyPrL1ismSLVHmJ6u/view)
- [Full-SVD 1K reference CSV (Drive)](https://drive.google.com/file/d/1RAqCYMkra24xIdEm-J7EZpILsaC6uqqG/view)
- [Projected matrix memory CSV (Drive)](https://drive.google.com/file/d/13B8qZ4gUX_YMfgxYw8fdEI5r_9_3r8mV/view)
- [Pilot PNG (Drive)](https://drive.google.com/file/d/1wlQaYTFO-uZj4ZiDC8ZNPPEwQwCIIptx/view)
- [Notebook](../notebooks/07_phase2b_dimensionality_reduction.ipynb)

Next: full k=50/100/200 and N=1K/5K/10K/20K; record results before conclusions. Fit+reconstruction timing is not pure decomposition time; the report must label it accurately.
