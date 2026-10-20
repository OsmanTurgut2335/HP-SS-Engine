# Phase 3D: same-vector baseline and Numba — full results

Completed on 2026-10-20. Same real 384-dimensional normalized MiniLM vectors and same top-10 retrieval task. Five timed repetitions per configuration; 76 summary configurations, 58 correctness records.

## 20,000 documents × 64 queries (median)

| Method | Milliseconds | Speedup vs sequential NumPy |
|---|---:|---:|
| Sequential NumPy | 436.527 | 1.000x |
| Batched BLAS | 226.436 | 1.928x |
| Numba serial | 942.034 | 0.463x |
| Numba parallel (2 threads) | 583.025 | 0.749x |

Numba parallel is about 1.62x faster than Numba serial, but slower than optimized NumPy/BLAS. This is a **matched search workload**, unlike the earlier Phase 2C Gram-matrix experiment. Numba compilation was excluded from timing.

At 20K×16: sequential 82.809 ms; batched BLAS 50.771 ms; Numba serial 234.815 ms; Numba parallel 133.211 ms.

## Pure Python bridge to Phase 1

At 1,000 documents × 4 queries on **the same MiniLM vectors**, Phase 1-style pure Python cosine was 1687.633 ms and sequential NumPy 0.687793 ms (~2454x difference). **Important:** Python recomputes both vector norms per pair, while NumPy uses pre-normalized vectors and dot products. Thus this is an implementation comparison, not a pure Python interpreter-only speedup. Large pure-Python jobs were intentionally skipped; historical Phase 1 TF-IDF measurements were not mixed with these MiniLM timings.

## Correctness

All 58 checks matched both the top-10 document ID **set and order** against sequential NumPy. The largest measured absolute float32 score difference was 1.0132789611816406e-06. This does not guarantee identical rankings for all future queries; Phase 3A separately found a near-tie rank swap.

## Evidence

- [Notebook](../notebooks/13_phase3d_same_vectors_baseline_numba.ipynb)
- [Full raw timings on Drive](https://drive.google.com/file/d/1d4qFoMUf_KIWUwEGtOHU7sW5E7prh13j/view)
- [Full summary on Drive](https://drive.google.com/file/d/1Nx6cOokz3onDr8Sg5VnM3sCQQ0ZNTjME/view)
- [Full correctness checks on Drive](https://drive.google.com/file/d/1sHZAaaZoP61TMhrIsCs9OByzNNwL0W2Q/view)
- [Full chart on Drive](https://drive.google.com/file/d/1n1EnRbmv_SNtTJ_g0UMJI9RQtYthTY7z/view)

See [Phase 3 final comparison](phase3_final_comparison.md) for the five-strategy synthesis and remaining methodological limitations.
