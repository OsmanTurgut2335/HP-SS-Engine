# Phase 3B — ProcessPool pilot findings (2026-10-20)

**PILOT ONLY**, verified from Drive files; 3 timed repeats per configuration. Same real normalized 384-dimensional MiniLM corpus, BLAS threads=1. Sizes 1K/5K documents, 8/32 queries, workers 1/2, process chunksize 1/8. Comparison includes sequential NumPy, batched BLAS, and ThreadPool.

## Evidence
- [Raw CSV](https://drive.google.com/file/d/1oHNV9icwVu9edJI48MIg6wNjyXSyUCl0/view)
- [Summary CSV](https://drive.google.com/file/d/1Q4q0qqfDMNE8IIgAXUpHFDgqqGyDcS_H/view)
- [Process startup CSV](https://drive.google.com/file/d/1cvd-Eq2B1g5EN2xry_kqiBs_3V3OLEaA/view)
- [Figure](https://drive.google.com/file/d/1NfEnJPmm5JwwZdEccNubruBwH8nzEVLo/view)
- [Notebook](../notebooks/11_phase3b_processpool.ipynb)

## 5K docs × 32 queries, median ms

| Method | Median ms | Speedup vs sequential |
|---|---:|---:|
| Sequential NumPy | 23.675 | 1.000× |
| Batched BLAS | 20.301 | 1.166× |
| ThreadPool 2 | 20.116 | 1.177× |
| ProcessPool 1, chunk 8 | 35.641 | 0.664× |
| ProcessPool 2, chunk 8 | 37.206 | 0.636× |
| ProcessPool 2, chunk 1 | 58.481 | 0.405× |

The process pool **lost to sequential** for every pilot workload/configuration. Chunking often reduced task-dispatch overhead, but was not enough to beat batched NumPy or ThreadPool. ProcessPool startup and first-search cost at 5K×32 was 67.238 ms (1 worker) and 101.981 ms (2 workers); this was recorded **outside** steady-state timed repetitions.

## Interpretation and limitations

- NumPy/BLAS arithmetic already executes in native code and can release GIL, so processes do not automatically accelerate it.
- Process workers incur IPC, query serialization, scheduling, result transfer and startup; `chunksize=8` reduces per-task dispatch frequency relative to `chunksize=1`.
- Notebook uses Linux `fork` and corpus inherited copy-on-write. It does **not** pickle/copy the entire 20K corpus for each query. Do not misrepresent the measured process cost.
- ThreadPool and ProcessPool are measured in the same notebook to control the environment. CPU count and runtime allocation matter; T4 GPU is **not used** by NumPy, ThreadPool, ProcessPool, or CPU Numba here.
- Pilot only: three repetitions and at most 5K documents; no final verdict about larger 20K corpus until full benchmark.
- Phase 3A floating-point precision observation remains important: batched `Q @ X.T` and individual `X @ q` can differ at ~1e-6 float32, swapping nearly tied ranks while keeping same top-10 IDs. This is recorded in [Phase 3A full findings](phase3a_full_findings.md).

## Next
- [ ] Run full Phase 3B (`PILOT=False`) and verify 20K×128 and worker/chunksize scaling.
- [ ] Discuss GIL, Amdahl, BLAS, IPC, memory bandwidth and code complexity in consolidated report.
- [ ] Note that the notebook does not yet separately time full corpus pickling, and the process pool implementation uses native NumPy search rather than pure-Python CPU-bound search.
