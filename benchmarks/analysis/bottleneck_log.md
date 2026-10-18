# Bottleneck and Optimization Log

Keep profiling evidence, explanations, unsuccessful experiments, and reproducible measurements. Never invent performance figures.

## Experiment template
- **Experiment ID / date / phase / commit:**
- **Question / hypothesis:**
- **Workload:** dataset, document count, embedding dimensions, dtype, queries, top-k
- **Environment:** CPU, RAM, Python and library versions, BLAS/thread settings
- **Baseline implementation:**
- **Profiling evidence:** top functions/lines, measured timings, profiler output path
- **Observed bottleneck:**
- **Root-cause hypothesis:**
- **Proposed change:**
- **Benchmark protocol:** warmups and at least five timed repetitions
- **Raw results path:**
- **Baseline median / optimized median / speedup:**
- **Correctness check:** tolerance, top-k agreement; recall@k if representation changes
- **Interpretation and confounders:**
- **Decision:** keep / revert / investigate
- **Figure or report reference:**

## Initial status
No performance measurements collected.
