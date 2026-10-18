# Phase 1 — Batched LLM corpus pilot

This notebook compares sequential versus batched generation on the **same 20 prompts** with Qwen2.5-0.5B-Instruct. It targets short responses (roughly 60–75 generated tokens) and saves measurements to Google Drive.

**Assignment requirement:** average text length >= 50 tokens (not 50 words per response). Measure actual output lengths before deciding on full-corpus generation.

**Important:** GPU batching uses padding; time and generated text may differ between sequential and batch modes. Benchmark throughput, length, nonempty responses, and truncation. This is a corpus-generation feasibility test, not a search benchmark.
