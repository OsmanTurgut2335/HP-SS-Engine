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
