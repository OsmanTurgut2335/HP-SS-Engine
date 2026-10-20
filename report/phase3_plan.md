# Phase 3 — Parallelism and Concurrency Plan

Date: 2026-10-20. Status: planned; **no Phase 3 benchmark results yet**.

## Five strategies (bonus target: more than three)

1. **NumPy / BLAS**: Phase 2 vectorized dense operations; use as the strong single-query/batch reference. Test BLAS thread settings explicitly.
2. **Numba `parallel=True` / `prange`**: Phase 2C already implements custom parallel Gram loops. Compare on matched Gram workload and report why BLAS may win. Do not conflate Gram timings with search timings.
3. **ThreadPoolExecutor**: parallelize independent query searches on a shared 20K×384 dense corpus; test worker counts 1/2/4/8, measure GIL release vs BLAS oversubscription.
4. **ProcessPoolExecutor**: compare CPU-bound Python tasks, batched tasks, and data sharing/pickling costs. A process pool is not guaranteed to beat BLAS; measure startup and serialization separately where feasible.
5. **asyncio**: test 100+ simulated I/O-bound query/embedding calls (serial, concurrent, bounded) with a CPU-search stage. Clearly distinguish simulated wait time from measured CPU search time; precomputed queries control.

**Not selected:** Dask and Ray. At 20K×384 float32, corpus vectors occupy ~29.3 MiB and fit in memory; Colab is a single machine. Dask out-of-core scheduling and Ray distributed object/task orchestration likely add overhead and complexity for this workload. Revisit for 10× or larger corpora, RAM pressure, multi-node services, or stateful actors; these are hypotheses until benchmarked.

## Experimental integrity

- Use real cached `qwen_20k_minilm_384_float32.npy` from `/content/drive/MyDrive/HP-SS-Engine/phase2_matrix_ops/2A_sparse_dense/`.
- Hold corpus, query IDs, top-k, normalization, and score/tie policy constant for comparable *search* benchmarks. Report vectorization vs worker-level parallelism separately.
- Sequential NumPy search and batched NumPy search are distinct baselines; report both, and keep BLAS thread settings explicit.
- Warm up; use at least five timed repetitions for full experiments, preserve raw CSV, runtime metadata, correctness assertions, figures and summary statistics. Pilot timings must not be presented as final.
- Distinguish setup cost, search-only latency, batch throughput, simulated I/O delay, serialization, memory overhead and speedup. Record worker counts and actual allocated CPU cores.
- Explain results through GIL, Amdahl's law, BLAS scheduling/oversubscription, memory bandwidth and parallel overhead. Never invent speedups.
- Decide best speedup **per unit code complexity**, and discuss whether 10× corpus changes the choice without pretending unmeasured 200K performance was tested.

## Planned work

- [ ] 3A: NumPy sequential/batch and ThreadPool pilot/full (20K embeddings; 1/2/4/8 workers).
- [ ] 3B: ProcessPool (pure Python and batched/shared-data variants).
- [ ] 3C: asyncio (I/O simulation and CPU-only control).
- [ ] 3D: consolidate prior Numba parallel Gram findings with correct workload separation.
- [ ] Cross-strategy comparison, code complexity rubric, 10× corpus discussion, findings register, Drive/GitHub artifacts and presentation.
