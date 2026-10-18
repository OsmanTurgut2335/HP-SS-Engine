# Pilot 01 — Qwen generation feasibility (measured)

Date: 2026-10-18 (UTC)
Source: Google Drive `data/raw/qwen_pilot_10_summary.json` and `qwen_pilot_10.jsonl`.

## Setup
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Device: Tesla T4
- Generation: deterministic (`do_sample=False`), max 128 new tokens
- Prompts: 10 diverse English questions
- Model loading: 19.5562 seconds

## Measured results
- 10 records, 10 nonempty responses, no exact duplicate response texts
- Total sequential generation time: 63.2521 seconds
- Median generation time: 5.8775 seconds per response
- 7/10 responses hit the 128-token cap; some are truncated mid-sentence
- Generated token counts: 128, 102, 128, 108, 128, 128, 128, 128, 128, 104

## Extrapolation only (not measured)
Using the median per-response generation time:
- 5,000 responses: 8.16 hours
- 10,000 responses: 16.33 hours
- 20,000 responses: 32.65 hours

These estimates exclude model loading, retries, runtime disconnects, and other overhead; they are **not** measured full-dataset runtimes.

## Interpretation / next action
Pilot succeeded technically, but 128-token truncation harms document quality and sequential 20K generation is likely impractical on Colab Free. Test batching, controlled output length, resume/checkpoint strategy, and corpus quality before committing to large-scale generation. Keep generation time separate from similarity-search timing.
