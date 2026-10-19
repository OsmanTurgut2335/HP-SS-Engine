# Dataset preparation and experiment notes (working draft)

## Corpus and provenance
- Assignment: high-performance semantic similarity search; 5K–20K documents, average text length at least 50 tokens, with search scaling tests at 1K/5K/10K/20K.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, run on Google Colab Tesla T4.
- Prompt generation: deterministic combinations of 80 topics, 10 task styles and 25 contexts; 20,000 distinct prompts and document IDs.
- Current corpus: `data/raw/qwen_corpus_20k_progress.jsonl` on Google Drive; **15,000 records** at time of audit. Target is 15K + 5K additional records.
- Storage: JSONL with id, prompt, response, response_tokens, model. Resume/checkpoint mechanism saves status periodically.

## Measured preparation results
| Experiment | Workload | Duration | Throughput | Mean tokens |
|---|---|---:|---:|---:|
| Sequential pilot | 10 responses | 63.25 s generation | — | — |
| Batch 1 pilot | 20 responses | 63.15 s | 0.317 docs/s | 93.5 |
| Batch 4 pilot | 20 responses | 19.85 s | 1.008 docs/s | 95.1 |
| Batch 8 pilot | 20 responses | 12.03 s | 1.662 docs/s | 93.7 |
| Short-answer quality pilot | 100 responses | 69.0 s | 1.45 docs/s | 50.0 |
| Production stage 1 | 15,000 responses | 7381.57 s | 2.03 docs/s | 50.44 |

Batch 8 vs batch 1 pilot: 5.25x throughput improvement for **corpus generation**, not for similarity search.

## Corpus audit (15K)
- 15,000 unique IDs, zero missing IDs, zero empty responses.
- Mean 50.44 tokens, median 45 tokens; 284 extra occurrences of duplicate response text; 247 outputs at or near the 128-token generation ceiling.
- The average length currently meets the >=50-token condition, with a narrow margin. Re-evaluate after adding the final 5K.
- Response duplication and truncation are data-quality limitations; report any filtering/repair and preserve raw corpus.

## Benchmark integrity and reporting plan
- Do not mix LLM generation times with retrieval/search latency.
- For each search implementation: record corpus size, vector dimensions, query count, top-k, hardware, versions, dtype, thread settings, warmup, >=5 timing repetitions, and raw per-run measurements.
- Compare naive Python cosine search, TF-IDF CSR, dense embeddings, dimensionality reduction, NumPy, Numba, and parallel strategies as required.
- Add search profiling to `benchmarks/profiling/`, raw timing CSVs to `benchmarks/raw_results/`, derived analysis to `benchmarks/analysis/`, and plots to `figures/`.
- Figures will be generated from recorded benchmark CSVs, not illustrative or fabricated numbers.

## Next steps
1. Finish 5K additional documents and repeat the corpus audit.
2. Generate query set and document embeddings/TF-IDF representations.
3. Implement and profile Phase 1 naive baseline.
4. Run reproducible search benchmarks and add charts as results become available.

**Status:** Working report notes, not a final 10–15 page report.
