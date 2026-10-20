# Phase 1 — Verified Findings and Open Items (2026-10-20)

Permanent findings register for the final report and presentation. Grounded in [notebook 05](../notebooks/05_phase1_pure_python_profiling.ipynb) and [75 raw timings](../benchmarks/raw_results/pure_python_cosine_phase1_raw.csv). Avoid treating other implementations or vector spaces as directly comparable.

## Experiment and correctness

- Actual input corpus: 20,000 Qwen-generated responses, ordered by document ID. Profiling baseline uses TF-IDF transformed to a fixed **512-feature dense float32** representation; only the first 100, 500, or 1000 documents are searched. Five prompt queries, five repetitions per size = **75 timings**.
- Search code implements element-wise cosine using **three explicit Python loops** (dot product and two squared norms), scores each document and sorts descending for top-10. This is intentionally unoptimized to expose bottlenecks.
- Score comparisons against NumPy were asserted within 1e-5, and top-10 document IDs were checked for three queries over 100 documents.
- Benchmark timing excludes profiler overhead. CPU-only, warm-up before timings. The experiment is a **pure Python cosine microbenchmark**, not end-to-end 20K document embedding/search latency.

## Measured search latency

| Documents | Median search latency |
|---:|---:|
| 100 | 29.685 ms |
| 500 | 125.223 ms |
| 1000 | 228.967 ms |

- Search latency grows substantially with document count because cosine arithmetic is repeated for each document.
- The 1K run takes about 7.71 times as long as the 100-doc run (228.967 / 29.685), for 10 times as many documents; this is not a complexity proof.
- At 1000 docs, line profiling attributes about **99.7% of top-k runtime to cosine similarity computation**; sorting accounts for roughly **0.1%**. The Python element-wise loops dominate, not sorting. This is the main motivation for Phase 2 vectorization and optimized matrix operations.
- cProfile and line_profiler logs were saved by notebook to Drive `HP-SS-Engine/benchmarks/profiling/`: `pure_python_1k_cprofile.txt` and `pure_python_1k_line_profile.txt`. Preserve these as evidence of actual bottlenecks.
- Measured charts: [scaling PNG](https://drive.google.com/file/d/1N7Pjsv5ReqExrCd_gh5mp9Fi_Os6PBF1/view) and [latency boxplot](https://drive.google.com/file/d/1pYJnUPp6SDtpQLq4aZrYDq5L0DPYDYK3/view). [GitHub figure index](../figures/phase1_figures.md).

## Bottleneck interpretation and design consequences

1. **Primary bottleneck: per-document Python cosine loops.** Each document requires three Python loops over 512 features. Interpretation: repeated Python-level scalar operations create high overhead. Evidence: ~99.7% of profiled top-k time in cosine.
2. **Secondary cost: repeated norm calculations.** Each cosine call recomputes norms even though the TF-IDF vectors are already L2-normalized; this is avoidable work. This is an **algorithmic observation**, not an independently timed percentage. Cache/normalize vectors and use dot products in optimized search.
3. **Sorting and ranking.** Full sorting is O(N log N) and can be replaced by partial top-k selection, but measured sorting was only ~0.1% at 1K. Therefore it is not the first optimization priority; measure again once cosine arithmetic is optimized.

## Important caveats

- This Phase 1 512-dimensional TF-IDF baseline is **not an apples-to-apples speedup baseline** for Phase 2A's 384-dimensional MiniLM embeddings, which represent a different feature space and use different operations.
- The older `04_similarity_baseline.ipynb` includes SciPy operations and is not the pure-Python element-wise baseline. Do not conflate their timing numbers.
- Do not claim the baseline was timed at 5K, 10K or 20K: only 100, 500, 1000 documents were timed in notebook 05.
- Line-profile percentages reflect the profiled 1K workload, not all workloads or environments.
- The fixed 512-feature TF-IDF truncation makes the pure-Python benchmark tractable but does not capture full-vocabulary or dense embedding accuracy.

## Open items before submission

- [x] Pure Python cosine implementation, correctness checks, 75 timed measurements, cProfile and line_profiler, two PNG figures.
- [x] Raw timing CSV and figure index archived on Phase 1 GitHub branch.
- [ ] Verify both profiling TXT files are accessible in Drive and mirror or link them in GitHub.
- [ ] Integrate this findings register into the unified project report and presentation.
- [ ] Cross-check Phase 1 instructor rubric, including any required top-three bottleneck evidence or additional benchmarking scales.
- [ ] If Phase 1 and Phase 2 performance are compared, use a matched vector representation, input shape, and operation; otherwise present them as separate experiments.

**Preserve for the final presentation:** Python-level cosine computation, not sorting, was the dominant measured bottleneck; repeated norms are an avoidable secondary cost; later optimized methods should be evaluated using fair same-data comparisons.
