# Presentation Notes — Evidence and Narrative

This is a running collection of *verified* decisions and results for the final 15-minute presentation. It is not the final slide deck. Link every performance claim to a raw benchmark or profiling artifact.

## 00 — Project initialization
**Problem:** Build a high-performance semantic similarity search engine and explain how its performance evolves through profiling and optimization.

**Architecture decision:**
- Google Colab Free CPU: run and benchmark implementations without paid GPU access.
- Google Drive: store large corpora, embeddings, and raw experimental artifacts.
- GitHub: track source code, lightweight results, documentation, and version history.
- GitHub main branch: stable state; later work can happen on feature branches and be merged by pull request.

**Reproducibility:** Setup notebook creates the project directory layout; benchmark logs document workload, machine, library versions, timing protocol, and correctness.

**Important caveats:**
- The initial GitHub scaffold was committed in several commits via the GitHub file API.
- GitHub and Drive are connected, but *not* automatically synchronized.
- No profiling results, performance claims, or speedup figures exist yet.

**Potential slide:** A simple Colab ↔ Drive / GitHub workflow diagram.
**Evidence:** README.md, notebooks/00_project_setup.ipynb, benchmarks/analysis/development_log.md.

## 01 — Baseline & profiling (pending)
- Dataset and document/query statistics:
- Naive Python implementation:
- Top three measured bottlenecks and profiler evidence:
- Baseline median latency:
- Chart / screenshot references:

## 02 — Representations, reduction & JIT (pending)
- Sparse vs dense memory and runtime:
- Custom dot product validation:
- Power-iteration SVD and comparison:
- NumPy / Numba results:

## 03 — Parallelism (pending)
- Three strategies and controlled settings:
- GIL, BLAS, Amdahl and memory bandwidth observations:
- Scaling results and negative findings:

## 04 — Final pipeline (pending)
- End-to-end diagram:
- 1K / 5K / 10K / 20K scaling:
- Accuracy and recall@k:
- Key takeaways and limitations:

## Experiment-to-slide checklist
1. What was the measured problem?
2. Which profiler/table proves it?
3. What change was made and why?
4. What changed in timing, memory, and correctness?
5. What limitation or tradeoff remains?
