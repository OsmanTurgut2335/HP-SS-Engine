# Phase 3 — Parallelism: Final Comparison and Design Decision
Date: 2026-10-20 | Course: YZM 513 | Branch: `feature/phase3-parallelism`

## Executive answer

**For our in-memory, normalized 384-dimensional MiniLM embedding search, batched NumPy/BLAS is the best measured speedup per unit of code complexity.** On the *same Phase 3B run*, 20,000 documents × 128 queries, BLAS threads=1, the sequential baseline was 670.172 ms and batched BLAS was 435.124 ms (**1.540×**). ThreadPool 4 workers was 489.070 ms (1.370×), ProcessPool 4 workers/chunksize 8 was 475.582 ms (1.409×). These are medians of 5 repeats; **do not combine absolute times from separate Colab sessions**.

We implemented/evaluated five strategies: (1) NumPy/BLAS, (2) Numba `parallel=True`/`prange`, (3) ThreadPoolExecutor, (4) ProcessPoolExecutor, (5) asyncio. **Important:** Numba's existing benchmark is a Gram-matrix workload, and asyncio's headline speedup is *simulated I/O*, not the same top-k retrieval workload. Therefore the five strategies are covered, but a strictly same-workload, five-way benchmark **has not been completed**. Whether the reused Phase 2C Numba implementation counts toward the Phase 3 bonus is subject to instructor evaluation.

## Instructor checklist: implementation, baseline, explanation, course concepts

| Strategy | Code | Measured workload and matched baseline | Why it helped / didn't |
|---|---|---|---|
| NumPy/BLAS | [Phase 3A notebook](../notebooks/10_phase3a_threadpool_vs_blas.ipynb), [Phase 3B notebook](../notebooks/11_phase3b_processpool.ipynb) | Phase 3B 20K×128: sequential 670.172 ms; batched 435.124 ms; **1.540×** | BLAS matrix–matrix batching amortizes overhead, optimized native kernels; minimal Python management |
| Numba parallel `prange` | [Phase 2C notebook](../notebooks/09_phase2c_vectorise_vs_loop.ipynb) | **Different task: Gram matrix**, 5K×300; NumPy 11.945 ms, Numba parallel 902.780 ms (~75.6× slower than NumPy); Numba serial 685.929 ms | Parallel-loop overhead, memory access/BLAS advantage, runtime-specific overhead. **Not** a top-k search speedup. Compilation/warmup and memory effects need explicit qualification |
| ThreadPoolExecutor | [Phase 3A notebook](../notebooks/10_phase3a_threadpool_vs_blas.ipynb), [Phase 3B notebook](../notebooks/11_phase3b_processpool.ipynb) | Phase 3B 20K×128: 4 workers 489.070 ms; **1.370×** vs sequential, but slower than batched BLAS | Native NumPy releases GIL, allowing some thread overlap; scheduling overhead and contention cap gains |
| ProcessPoolExecutor | [Phase 3B notebook](../notebooks/11_phase3b_processpool.ipynb) | Phase 3B 20K×128: 4 workers/chunk8 475.582 ms; **1.409×** vs sequential, slower than batched BLAS | Separate interpreters bypass GIL; IPC, task serialization, process startup, scheduling, memory and load balancing cost |
| asyncio | [Phase 3C notebook](../notebooks/12_phase3c_asyncio.ipynb) | **Simulated I/O**, 120 × 50ms: serial 6040.915 ms; concurrency 40 154.435 ms; **39.116×**. **CPU control** 20K×128: sequential 922.691 ms, asyncio.gather 844.110 ms; no genuine CPU parallelism | Overlaps waits via cooperative scheduling; does not parallelize CPU loops without yielding/offloading |

**Baseline caveat:** The Phase 1 pure-Python baseline uses 512-feature TF-IDF; Phase 3 uses 384-dimensional MiniLM embeddings. Thus Phase 1 timings (1K documents median 228.967 ms) are useful historical bottleneck context, **not an apples-to-apples parallel speedup denominator**. The closest valid baseline is sequential NumPy with *the same corpus, query set, and session*. The best Phase 2 implementation is NumPy/BLAS for the relevant dense operation, but the Phase 2C 5K×300 Gram-matrix benchmark and Phase 3 top-k search have different outputs. A strict instructor-compliant “against Phase 1 baseline and best Phase 2 version” search benchmark requires an additional same-input comparison, or explicitly qualifying the mismatched feature spaces and operations.

## Direct same-session comparison: Phase 3B full (20K docs × 128 queries, BLAS=1)

| Configuration | Median ms | Speedup vs sequential |
|---|---:|---:|
| Sequential NumPy | 670.172 | 1.000× |
| Batched BLAS | **435.124** | **1.540×** |
| ThreadPool 1 | 680.363 | 0.985× |
| ThreadPool 2 | 607.038 | 1.104× |
| ThreadPool 4 | 489.070 | 1.370× |
| ProcessPool 1, chunk1 | 914.898 | 0.733× |
| ProcessPool 2, chunk1 | 529.207 | 1.266× |
| ProcessPool 4, chunk1 | 532.331 | 1.259× |
| ProcessPool 4, chunk8 | **475.582** | **1.409×** |
| ProcessPool 4, chunk32 | 480.161 | 1.396× |

**No universal chunking rule:** ProcessPool 2/chunk1 was 529.207 ms, but 2/chunk8 was 736.794 ms and 2/chunk32 was 782.662 ms. Different workers/chunksize interact with scheduling and runtime noise; chunk8 is not globally optimal. ProcessPool 4 worker startup **plus first search** took 651.020 ms, **not pure startup** and excluded from steady-state medians.

### Phase 3A cross-check, not pooled with Phase 3B

At 20K×128 and BLAS=1, Phase 3A sequential 446.500 ms; batched BLAS 266.491 ms (1.675×), ThreadPool 2 316.269 ms (1.412×), ThreadPool 4 326.126 ms (1.369×). Phase 3B sequential 670.172 ms: Colab session conditions differ. Phase 3A 5K×128 showed anomalous non-monotonic timing; do not overgeneralize.

### Numerical correctness caveat

Phase 3A batched matrix–matrix vs per-query matrix–vector `float32` accumulation had maximum score difference **8.940697e-07**. Nearly tied documents 8335 and 9767 swapped order for one query, while top-10 **ID sets** matched. Use tolerance-based score validation and document tie-order policy. A full `np.allclose` score test remains outstanding.

## Course concepts

**GIL:** Pure Python bytecode is constrained by the GIL in threads, but NumPy/BLAS native kernels often release it. ProcessPool has separate interpreter processes and GILs, yet IPC may outweigh the benefit. `asyncio` schedules cooperative coroutines; it does not create CPU parallelism by itself.

**Amdahl's Law:** `S(p)=1/((1-f)+f/p)` is an ideal upper bound for parallelizable fraction `f`, ignoring overhead. Python task scheduling, ranking/top-k, process startup and I/O overheads further limit observed gains. More workers do not imply linear speedup.

**Memory bandwidth and cache:** A 20K×384 `float32` dense matrix occupies about 29.3 MiB; repeated matrix-vector searches read this data many times. Batch matrix multiplication can improve reuse and arithmetic intensity; concurrent workers compete for cache/bandwidth. Actual memory bandwidth was **not directly measured**.

**BLAS and oversubscription:** NumPy's optimized BLAS uses native vectorized kernels and may itself be multithreaded. Combining BLAS threads with outer ThreadPool/ProcessPool workers can oversubscribe CPUs. Phase 3A BLAS=2, ThreadPool 8 at 20K×128 took 1106.968 ms, vs sequential BLAS=1 446.500 ms; this is consistent with contention, not proof of a single cause.

**I/O concurrency:** In the Phase 3C *simulation*, 120 requests each sleeping 50 ms achieved 39.116× at 40 concurrency, almost ideal overlap. This is not an embedding generation or similarity-search acceleration. CPU negative control had high variability (sequential 677–1080 ms, asyncio 702–1082 ms across 5 repeats), and its apparent 1.093× is not evidence of parallel CPU execution.

## Design decision: best speedup per unit code complexity

**Choose batched NumPy/BLAS for the search core.** The vectorized expression `Q @ X.T` (followed by deterministic top-k ranking) is short and readable; on matched 20K×128 it is faster than ThreadPool and ProcessPool, and avoids process lifecycle, IPC, serialization and worker scheduling. It also remains a reliable baseline for any future specialized accelerator.

- ThreadPool: moderate code complexity, occasional improvement over per-query sequential; not consistently better than batching.
- ProcessPool: high complexity; useful for heavy pure-Python CPU work, but underperforms batched BLAS in this workload.
- Numba `prange`: extra JIT/runtime complexity and was slower than BLAS on the measured Gram matrix. It may still help custom loops lacking optimized NumPy equivalents.
- asyncio: appropriate **only for an I/O-bound embedding/API ingestion stage**; not a substitute for CPU search optimization.
- Dask/Ray not implemented: our current corpus fits in memory and one Colab machine; distributed scheduling complexity unlikely to pay off here. This is a design rationale, not an experimental finding.

### What if corpus were 10× larger?

20K→200K docs increases 384-dimensional float32 matrix from ~29.3 MiB to ~293 MiB. **Not measured**: expect greater memory traffic, cache pressure and longer search; batched BLAS remains the first method to test, with tuned query batch sizes. If RAM/cache pressure or throughput requirements become limiting, evaluate chunked exact search, approximate nearest neighbor (ANN), quantization, and potentially shared-memory or distributed strategies. A 10× corpus does **not automatically** make ProcessPool, Dask or Ray superior; benchmark on the target machine before changing the decision.

## Artifacts and outstanding work

- [Phase 1 findings](phase1_findings_and_open_items.md)
- [Phase 2 findings](phase2_findings_and_open_items.md)
- [Phase 3A full findings](phase3a_full_findings.md)
- [Phase 3B full findings](phase3b_full_findings.md)
- [Phase 3C full raw](https://drive.google.com/file/d/1lDJEqdLqYwQCpY8D_j8T5dqLnJJLCxrc/view), [summary](https://drive.google.com/file/d/1WjYnUZ0dJkGMbVSrU92UrfPGfFU7AmMp/view), [figure](https://drive.google.com/file/d/1QnpkYvmxHTajzZmd13xV6uJNV8nA_OR1/view)
- [Phase 3A full raw](https://drive.google.com/file/d/1h0I4ASNbhZ7TqSBGYn5o8ag1GsHRLhx8/view), [summary](https://drive.google.com/file/d/1fxzm8_25SxDl_oaEzjhV0QFU6CLuV8hI/view)
- [Phase 3B full raw](https://drive.google.com/file/d/1LV1DKFEoNvpiZk1mDcbk3-VOV5fr46RH/view), [summary](https://drive.google.com/file/d/1-pzqfzFE0mAjP-JV6kuxUBSU1SZXteie/view)

### To close before submission
- [ ] Confirm/commit Phase 3C pilot and full findings to GitHub (prior connector writes blocked).
- [ ] Strict apples-to-apples **all five methods on same search workload**, especially Numba `prange`, if needed to claim bonus fully; separate asyncio I/O task regardless.
- [ ] Explicit Phase 1 pure-Python and best Phase 2 search comparison using **matching embeddings and ranking semantics**, or disclose that historical results are not directly comparable.
- [ ] Fix Phase 2C plotting code `zip(g.rows,g.columns)` → `zip(g['rows'],g['columns'])` in GitHub notebook.
- [ ] Add tolerance-based numerical score test for batched vs per-query.
- [ ] Validate final tables and figures against CSVs, rerun anomalous configurations if feasible.
