# Development Log

Record every work session, important design decision, and supporting artifact.

## 2026-10-18 — Project setup
- Created Google Drive structure for notebooks, source, datasets, embeddings, benchmarks, figures, and report.
- Chose Google Colab Free CPU for initial execution.
- Created the initial Colab setup notebook.
- Connected Google Drive and GitHub for project management.
- Initialized the GitHub repository and documented the four phases.
- No benchmarks or measured speedups yet.

## Session template
- **Date / phase / commit:**
- **Objective:**
- **Changes:**
- **Files changed:**
- **Decisions and alternatives:**
- **Problems and resolutions:**
- **Evidence (profilers, CSV, figures):**
- **Verification:**
- **Next action:**

## 2026-10-19 — LLM corpus pilots
- Qwen2.5-0.5B-Instruct on Tesla T4; 10-response sequential pilot: 63.25 s generation, 19.56 s model load.
- 20-prompt comparison (max 100 generated tokens): batch 1 63.15 s; batch 4 19.85 s; batch 8 12.03 s. Batch 8 delivered about 5.25x throughput versus sequential.
- Batch 8 mean output length 93.7 tokens (assignment minimum is 50 tokens on average); 12/20 hit the generation cap, so some text may be incomplete.
- Decision: test shorter, complete answers on 100-200 prompts, then generate 15K documents and resume to 20K for 1K/5K/10K/20K search scaling.
- Keep unique document IDs and checkpoint data in Google Drive. Corpus-generation speed is not search-engine benchmark speed.
- Evidence: Google Drive data/raw/qwen_pilot_10_summary.json and qwen_pilot_batch_summary.json.

## 2026-10-19 — Production corpus generation notebook prepared
- Added `notebooks/03_generate_corpus_resumable.ipynb` on `feature/phase1-baseline`.
- Design: 20,000 deterministic unique topic/task/context prompts; produce first 15,000 and then resume with TARGET_DOCS=20000.
- Qwen2.5-0.5B-Instruct, T4 GPU, batch 8, request short 3-4 sentence answers; max_new_tokens=100 is a safety cap, not a length target.
- 100-document quality pilot before the main run; monitor mean token length, empty responses and cap hits.
- Checkpoint JSONL every 200 docs on Drive and validate IDs/prompt consistency before resuming; save generation status JSON.
- Not yet run: no claim that 15K documents exist or that the 100-doc pilot passed. Wait for user Colab execution and inspect artifacts.

## 2026-10-19 — First 15,000-document corpus complete
- Google Drive `data/raw/qwen_corpus_20k_progress.jsonl`: 15,000 rows, 15,000 unique IDs (0–14999), zero missing IDs, zero empty responses.
- Production status: 7381.57 s, batch 8, Qwen2.5-0.5B-Instruct on T4, mean 50.44 tokens, ~2.03 docs/s.
- Direct corpus audit: median 45 tokens, min 9, max 128, 284 repeated response texts, 247 responses at >=127 tokens, 14,778 responses ending in punctuation. Text duplicates are not duplicate IDs.
- Colab session disconnected before final audit cell, but checkpoint file and full corpus persisted to Drive.
- Next: plan additional 5,000 docs; improve average token margin and assess text duplicates before search benchmarks. Preserve source and record any filtering decisions.

## 2026-10-19 — Final 5K production prepared
- Updated `notebooks/03_generate_corpus_resumable.ipynb`: `TARGET_DOCS=20000`, revised response-length instruction to ~65–85 tokens.
- Existing 15K JSONL corpus is preserved; resume logic checks IDs and generates only missing 5K (IDs 15000–19999).
- Before production, verify Colab prints `Found 15000 existing documents. Target: 20000.` and `Remaining: 5000.`
- Final 5K production has NOT been run or verified yet. After run, audit token mean, duplicate text and completeness, then save actual metrics.

## 2026-10-19 — Corpus complete, 20K audit
- All 20,000 JSONL rows parsed successfully; IDs 0..19999 and 20,000 prompts unique, zero empty responses, zero malformed rows.
- Mean 58.128 model tokens, median 53, 288 extra exact duplicate response texts, 678 responses >=127 tokens; 19,378 end in punctuation.
- Final 5K: 3192.16 seconds, mean 81.194 tokens, 1.566 docs/s. Full generation: 10573.73 seconds (dataset creation only).
- Archived generation CSV at benchmarks/raw_results/corpus_generation_summary.csv and draft dataset report at report/dataset_20k_results.md.
- Next phase: 100 query prompts, 1K/5K/10K/20K cosine retrieval baseline, profiler and repeatable timings.

## 2026-10-19 — First CPU similarity-search baseline
- Colab T4 environment, but SciPy TF-IDF CSR search runs on **CPU**, not GPU.
- Naive Python per-document sparse cosine top-10, N=1000, 5 query indices x 5 repeats, 25 raw measurements saved in Drive and mirrored to `benchmarks/raw_results/naive_cosine_tfidf_1k.csv`.
- Median 359.895 ms/query, mean 368.495 ms, range 271.401–522.266 ms. Search timing excludes TF-IDF construction.
- cProfile for one 1K query: 0.698s naive function, 362891 total function calls (including profiler/IPython overhead). Dominant cumulative paths: sparse multiply 0.376s, getrow 0.175s, sum 0.134s. Cumulative times overlap and should not be added.
- Next: benchmark vectorized CSR matrix-vector reference on the same query/scale, validate top-10 scores and IDs, then plan 5K/10K/20K comparisons.

## 2026-10-19 — Pure Python Phase 1 completed
- 512-feature dense TF-IDF with explicit nested Python loops; 75 timed runs (5 queries x 5 repeats x 3 scales).
- Median latency: 100 docs 29.685ms, 500 docs 125.223ms, 1000 docs 228.967ms.
- cProfile: cosine 1000 calls, 0.284s self time. line_profiler: cosine call line 99.7% of top-k inclusive time, sorting ~0.1%; three multiply/accumulate lines each ~17-18% of cosine time.
- Saved raw CSV and both profiler text files to GitHub; two PNG figures saved in Google Drive figures folder. Report draft updated.
- Next: compare optimized dense methods using identical 512-feature vectors and queries; do not compare this directly with full-vocabulary CSR baseline.

## 2026-10-20 — Phase 2A complete
- 20K MiniLM 384-d embeddings cached, sparse TF-IDF 9,617 features; CPU benchmark.
- 1K/5K/10K/20K × four dot-product workloads × 25 runs = 400 actual measurements, CSV mirrored to GitHub.
- 20K medians: TF-IDF SS 560.687ms, embedding SS 6066.276ms, embedding SD 3054.274ms, embedding DD 149.586ms.
- 20K memory: TF-IDF CSR 6.701MiB, dense embeddings 29.297MiB, embedding CSR 58.670MiB.
- Two Drive PNG figures indexed in GitHub; complete results in report/phase2a_sparse_dense_results.md. Next Phase 2B dimensionality reduction.
