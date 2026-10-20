# Phase 3A — Full Benchmark Findings (2026-10-20)

**Status: measured full benchmark, 720 timings.** All values below are actual Drive CSV results; five repetitions per configuration. Phase 3A compares *search implementations* on normalized real 384-dimensional MiniLM embeddings, not Phase 2C Gram timings or Phase 1 TF-IDF baseline.

## Artifacts

- [Full raw timings CSV](https://drive.google.com/file/d/1h0I4ASNbhZ7TqSBGYn5o8ag1GsHRLhx8/view) — 720 rows
- [Full summary CSV](https://drive.google.com/file/d/1fxzm8_25SxDl_oaEzjhV0QFU6CLuV8hI/view) — 144 configurations
- [Full PNG](https://drive.google.com/file/d/1hQ7KLkgpEocGYsmybgnhoZ-2zHgdsb5t/view)
- [Notebook 10](../notebooks/10_phase3a_threadpool_vs_blas.ipynb)
- [Pilot findings](phase3a_pilot_findings.md)

## Design and fairness

- 4 corpus sizes (1K/5K/10K/20K), 3 query counts (16/64/128), BLAS threads 1/2, sequential NumPy, batched BLAS and ThreadPool workers 1/2/4/8, five repeats. CPU count reported **2**.
- All methods use same normalized corpus and in-corpus queries; search-only timing excludes embedding inference and disk I/O. Top-10 ID correctness asserted. For batched BLAS, top-10 **sets** are compared rather than exact ranking order due to near-tie float32 numerical differences; sequential vs ThreadPool exact ranking required.
- Speedup values in summary compare to sequential **with the same BLAS setting and corpus/query size**. Cross-BLAS comparison must use actual medians and the same reference.
- Batched BLAS uses a different matrix multiplication shape (matrix-matrix rather than repeated matrix-vector), so gains are not solely thread gains.

## Full benchmark 20K docs × 128 queries (median ms)

| Method | BLAS=1 | BLAS=2 |
|---|---:|---:|
| Sequential NumPy | 446.500 | 509.040 |
| Batched BLAS | **266.491** | 295.780 |
| ThreadPool 1 | 453.672 | 498.477 |
| ThreadPool 2 | 316.269 | 506.009 |
| ThreadPool 4 | 326.126 | 601.212 |
| ThreadPool 8 | 319.422 | 1106.968 |

- BLAS=1, batched vs sequential **1.675×**; ThreadPool 2 vs sequential **1.412×**, ThreadPool 4 **1.369×**, ThreadPool 8 **1.398×**.
- With BLAS=2, ThreadPool 8 takes 1106.968 ms, or ~2.48× **longer** than sequential BLAS=1 at 446.500 ms; oversubscription is a plausible cause, not separately profiled.
- BLAS=1 is faster than BLAS=2 for every listed 20K×128 variant; adding BLAS threads is not automatically faster.

## Other sizes and important anomaly

- BLAS=1, 1K×128: sequential 15.395 ms, batched 8.852 ms (**1.739×**), ThreadPool 2 17.972 ms (**0.857×**), ThreadPool 4 28.951 ms (**0.532×**).
- BLAS=1, 10K×128: sequential 192.486 ms, batched 117.331 ms (**1.641×**), ThreadPool 2 145.307 ms (**1.325×**).
- BLAS=1, 5K×128: sequential 336.182 ms, batched 202.558 ms, ThreadPool 2 80.433 ms (**4.180×**), ThreadPool 4 73.765 ms (**4.557×**). **Caution:** sequential at 5K×128 is slower than at 10K×128 (192.486 ms), which is non-monotonic; BLAS=2 sequential 5K×128 is only 110.850 ms. Possible runtime noise/thermal or scheduling effects. Do not generalize the 4.557× number or infer superlinear scaling without reruns and profiling.
- ThreadPool 1 adds overhead in most settings, but a few noisy cases are faster; avoid blanket claims.

## Numerical correctness finding — important

The first full run halted at `('batched_blas', 20000, 128, 0)` because exact top-10 **order** differed for one of 128 queries. Diagnostic rerun found max absolute float32 score difference **8.940697e-07** between per-query `corpus @ q` and batch `Q @ corpus.T`. Query index 105 returned the **same top-10 IDs**, with positions of document IDs **8335** and **9767** swapped due to nearly equal scores (~0.936343). BLAS matrix-vector and matrix-matrix paths can have different floating-point accumulation order; index tie-breakers only apply when scores are exactly equal. This is not evidence of wrong documents or a major numerical error.

Notebook was changed to require exact ordered IDs for sequential vs ThreadPool and matching top-10 ID **sets** for batched BLAS; the full benchmark then completed. **Further improvement for final correctness testing:** also assert `np.allclose` on the raw similarity scores with documented tolerance (e.g., atol=1e-6, rtol=1e-5) and explicitly measure top-k order/score differences. Set equality alone does not establish full ranking equivalence. Do not silently round all scores or claim byte-identical output.

## Interpretation for instructor

- **GIL:** NumPy's BLAS computations may release GIL, allowing thread-level query concurrency; Python-side task/ranking overhead remains.
- **BLAS & memory bandwidth:** batched matrix-matrix operations can use optimized kernels and improve throughput. Additional worker/BLAS threads can compete for memory bandwidth and CPU resources.
- **Amdahl's Law:** nonparallel overhead, ranking, worker startup, and resource limits cap speedup; linear speedup is not guaranteed.
- **Best speedup per unit code complexity, so far:** batched NumPy/BLAS is the provisional choice for this 20K search batch, as it is simpler and faster than tested ThreadPool configurations in the 20K×128 example. Revisit after ProcessPool and asyncio experiments.
- **10× corpus:** 200K×384 float32 embeddings alone ~293 MiB (plus other arrays). Cache, memory bandwidth, batch score matrices, and process copying become more important; no actual 200K benchmark performed.

## Follow-up
- [ ] Targeted rerun of anomalous 5K×128 configuration under controlled conditions; measure CPU/BLAS and warm-up effects.
- [ ] Strengthen batched correctness with `np.allclose` for full scores and report order/ID-set differences.
- [ ] Implement Phase 3B ProcessPool and 3C asyncio; compare matched search workloads where meaningful.
- [ ] Incorporate results in unified Phase 3 and final reports.
