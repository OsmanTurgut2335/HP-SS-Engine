# Presentation Notes — verified evidence for a 15-minute talk

**Status:** Evidence and suggested story ready; actual slide deck and oral rehearsal are still pending. The live Colab demo is implemented and has been run successfully.

## Performance story (15 minutes + 5 minutes Q&A)

1. **Problem and corpus (1.5 min).** Search Top-10 among 20K real Qwen-generated English responses. MiniLM 384d; no fine-tuning. Qwen generation used T4; search benchmark is CPU.
2. **Naive baseline and profiling (1.5 min).** Pure Python 512d TF-IDF cosine, 1K documents: 228.967 ms. cProfile and line_profiler identify similarity loops as the dominant cost. Do not compare this directly with the final MiniLM search as same-vector speedup.
3. **Representation and matrix ops (1.5 min).** Sparse TF-IDF vs dense MiniLM; show actual memory numbers and explain different semantics/representations.
4. **Dimensionality reduction (2 min).** MiniLM 384d -> TruncatedSVD 100d; 29.297 -> 7.629 MiB. Own power-iteration SVD, NumPy full SVD, TruncatedSVD, PCA. Explain fit-only vs search; full NumPy SVD fit can beat randomized TruncatedSVD for this narrow matrix. Overlap is not relevance accuracy.
5. **Vectorization and Numba (1.5 min).** Gram A.T@A from Phase 2C, then controlled same-vector Phase 3D (20K x 64 queries): sequential NumPy 436.527 ms, batched BLAS 226.436 ms, Numba serial 942.034 ms, Numba parallel 583.025 ms.
6. **Parallelism (2 min).** ThreadPool, ProcessPool, asyncio; CPU BLAS releases GIL, Python threading overhead, multiprocessing IPC, asyncio simulated I/O gains but no CPU parallelism. Use within-phase comparisons only.
7. **Final system, correctness and scaling (2 min).** 384d float32 BLAS vs float64 reference within 1e-5; 100% exact Top-10 matches in Phase 4B. SVD100 mean overlap 86.9% in-corpus, 77% external. Phase 4C at 20K/100 queries: sequential 779.859 ms; batched 384d 208.815 ms; batched SVD100 190.792 ms. Non-monotonic scaling needs explanation, not claims of linearity.
8. **Live demo (1 min).** Open the updated notebook 17 in Colab with assets already loaded. Enter a new query and display real Top-10 texts and timing cards. Use ~30 seconds for actual interaction. Single-query timing is not the five-repeat batch benchmark.
9. **What surprised us / what we'd do with more time (2 min).** BLAS batch beat additional worker strategies; SVD saved much more memory than search latency; CPU and sorting overheads complicate scaling; future work 200K, ANN, controlled hardware and human-labeled relevance.

## Figures to use

- Phase 1 profiling plot and/or bottleneck screenshot — see `benchmarks/profiling/` and `figures/phase1_figures.md`.
- Phase 2A sparse/dense memory — `figures/phase2a_figures.md` (Drive-hosted image).
- Phase 2B `figures/phase2b_comparison_full_5run.png`.
- Phase 3D `figures/phase3d_same_vectors_full.png`.
- Phase 3A/B comparative figure (choose one, not both if crowded).
- Phase 4B `figures/phase4b_svd_top10_overlap_full_20k.png`.
- Phase 4C `figures/phase4c_scaling_full.png`.

## Professor questions / precautions

- **Why is TruncatedSVD chosen if full SVD fit was faster?** Memory/search/ranking tradeoff and pipeline choice; no direct full-SVD-top100 retrieval benchmark was done. Do not claim truncated always beats full.
- **Does SVD preserve Top-10 exactly?** No; 77% external overlap vs exact 384d, which is not a ground-truth relevance score.
- **Is 190.792 ms the live-query latency?** No, 100 batched queries including ranking; the demo separately times one query including encoding.
- **Why doesn't asyncio accelerate CPU work?** Its speedup is for simulated nonblocking I/O; CPU search coroutine contains no yielding await.
- **Why is the corpus not in GitHub?** Binary/raw assets are in Drive; reproducible rebuild instructions and paths are in `docs/REPRODUCIBILITY.md`.
- **What is the actual final parallelism choice?** Optimized NumPy/BLAS batching for multi-query workloads; the live demo uses one query and BLAS vector search, not ProcessPool or asyncio.

Every performance number in the slides should be checked against its own raw benchmark CSV. No cross-session speedup ratios.
