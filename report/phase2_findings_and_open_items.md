# Phase 2 — Verified Findings and Open Items (2026-10-20)

This is a durable findings register for the final report and presentation. All numbers are measured; limitations are explicit.

## 2A — Sparse vs Dense

- Tested real 20K-text corpus at 1K, 5K, 10K, 20K. TF-IDF CSR has 9,617 features; MiniLM dense embeddings have 384 features.
- At 20K: TF-IDF CSR 6.701 MiB; dense embedding 29.297 MiB; embedding CSR 58.670 MiB. CSR is beneficial for zero-heavy TF-IDF but can be worse for near-dense embeddings due to stored indices.
- At 20K, manual dot products: TF-IDF sparse–sparse 560.687 ms; embedding sparse–sparse 6066.276 ms; embedding sparse–dense 3054.274 ms; embedding dense–dense 149.586 ms. These are the specific implementations measured, not universal rankings.
- **Fairness:** TF-IDF and embeddings are different feature spaces. Compare sparse–sparse, sparse–dense and dense–dense on the *same embedding data* for direct representation effects; discuss TF-IDF separately.
- **Instructor answer:** Sparse is useful for many zeros; dense is better for near-dense vectors and BLAS-based arithmetic. Show custom dot-product code rather than only sklearn calls.
- Sources: [2A report](../report/phase2a_sparse_dense_results.md), [raw CSV](../benchmarks/raw_results/phase2a_dot_products_raw.csv), [memory CSV](../benchmarks/raw_results/phase2a_memory.csv).

## 2B — Dimensionality reduction

- Tested custom block power-iteration SVD, randomized TruncatedSVD, randomized PCA on real 20K × 384 embeddings, k=50/100/200 and N=1K/5K/10K/20K. Also NumPy full SVD reference and supplemental fit-only comparison.
- At 20K, k=100: relative Frobenius reconstruction error power 0.34428; TruncatedSVD 0.34442; PCA 0.34366. Initial time is **fit plus reconstruction**, not pure fitting: power 5.162 s; TruncatedSVD 4.038 s; PCA 2.023 s.
- Storage at 20K: original float32 384d 29.297 MiB, projected k=100 7.629 MiB (~74% smaller), excluding model/working memory.
- Supplemental query benchmark at 20K k=100: original 384d median 2.334 ms; TruncatedSVD 0.409 ms (~5.7× ranking speedup in matched experiment), PCA 0.324 ms. Mean top-10 overlap vs original ranking: TruncatedSVD 94%, PCA 90%. Only **five in-corpus queries**, so overlap is not ground-truth recall or relevance accuracy. Training and embedding inference excluded from search latency.
- **Provisional decision:** TruncatedSVD k=100 balances storage, ranking time and top-10 agreement. k=200 yields 96% overlap for both TruncatedSVD/PCA but doubles k=100 storage; choice not yet fully validated with held-out queries.
- **Important counterexample:** Fit-only median NumPy full SVD on 20K × 384 = 0.8333 s, while TruncatedSVD k50 1.5589 s, k100 2.4875 s, k200 3.9897 s. Theoretically truncated SVD can win by computing only top-k components, but it **did not beat full SVD at 20K for any tested k** on this CPU. Largest tested winning k: N=1K k100, N=5K k50, N=20K none. Do not claim universal crossover.
- PCA centers feature columns; TruncatedSVD normally does not. Custom power iteration hit its 35-iteration cap at 20K; do not assert convergence.
- Sources: [full report](../report/phase2b_full_results.md), [decision report](../report/phase2b_decision_results.md), [full CSV](../benchmarks/raw_results/phase2b_methods_full.csv), [fit-only CSV](../benchmarks/raw_results/phase2b_fit_only_full.csv), [retrieval CSV](../benchmarks/raw_results/phase2b_retrieval_full.csv), [memory CSV](../benchmarks/raw_results/phase2b_projected_matrix_memory.csv).

## 2C — Vectorization, loops and Numba

- Same Gram matrix `G=X.T @ X` implemented as pure Python triple loops, NumPy `@`, Numba `@njit` serial and `@njit(parallel=True)` with `prange`. Numerical agreement checked on real data. CPU with two BLAS/Numba threads; Numba JIT warm-up excluded.
- Five repeats per method/workload, 65 measured times total. Medians in ms:

| Matrix | Python loops | NumPy | Numba serial | Numba parallel |
|---|---:|---:|---:|---:|
| 100 × 32 | 39.770 | 0.027 | 0.149 | 0.165 |
| 300 × 64 | 487.238 | 0.101 | 1.679 | 1.183 |
| 1000 × 96 | not measured | 0.570 | 13.172 | 10.401 |
| 5000 × 128 | not measured | 2.970 | 428.132 | 376.013 |
| 5000 × 300 | not measured | 11.945 | 685.929 | 902.780 |

- At 300×64, measured Python/NumPy median speedup ~4808×. Do **not** extrapolate Python speedup at larger sizes; slow Python loops were deliberately skipped when estimated `m*n*n > 2,000,000`.
- NumPy was faster than both Numba implementations in all five tested shapes. BLAS is highly optimized and avoids interpreter overhead. Numba parallel beat serial at some intermediate shapes but lost at 100×32 and 5000×300. Possible factors: parallel overhead, memory access, thread scheduling, memory bandwidth; specific causes need further profiling. Amdahl's law limits parallel acceleration.
- This is a Gram matrix microbenchmark, **not end-to-end search latency**.
- Sources: [raw full CSV](../benchmarks/raw_results/phase2c_gram_timings_full.csv), [summary CSV on Drive](https://drive.google.com/file/d/1sHfZN9oh-QvUBUQpnHkDHvFxClVUR-6h/view), [full figure on Drive](https://drive.google.com/file/d/1WYQP6B4gYAF7rkIO5RnOI3EAC7P6ZUB6/view), [notebook](../notebooks/09_phase2c_vectorise_vs_loop.ipynb).

## Delivery checklist / unresolved

- [x] 2A, 2B, 2C experimental runs and underlying Drive artifacts.
- [x] Phase 2B raw CSVs, reports and figure index archived on GitHub.
- [ ] Fix GitHub Notebook 09 plotting line `zip(g.rows,g.columns)` to `zip(g['rows'],g['columns'])`; Colab user copy is already corrected and its plot exists.
- [ ] Mirror Phase 2C full summary CSV and pilot CSVs from Drive to GitHub (previous writes blocked).
- [ ] Add Phase 2C standalone report and figure index; unify final Phase 2 report answering every instructor question explicitly.
- [ ] Confirm all final notebooks, PNGs, report and eventual presentation in intended Drive/GitHub locations; no presentation has been created yet.
- [ ] Broader held-out query validation if time permits before final system choice.

**Do not lose these observations in the final report or presentation:** sparse can waste memory on dense embeddings; Full SVD can beat TruncatedSVD on a 384-feature matrix; Numba does not necessarily beat optimized BLAS; parallel Numba is not always faster; speedups and overlap must be qualified by what was actually measured.
