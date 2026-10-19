# Dataset preparation — report-ready draft

## Construction
We generated a 20,000-document English corpus with Qwen2.5-0.5B-Instruct on a Google Colab Tesla T4. Each input prompt was deterministically constructed from one of 80 topics, 10 task styles, and 25 contexts, providing 20,000 unique prompt combinations. The corpus is stored as UTF-8 JSON Lines with document ID, prompt, response, response-token count, and model identifier. Generation used batch size 8 and resumable checkpoints to Google Drive. A first stage generated 15,000 responses, and a second stage appended the remaining 5,000 without overwriting earlier records.

## Audit and quality
A complete audit of the stored JSONL verified 20,000 rows, exactly one record for every ID from 0 to 19,999, 20,000 unique prompts, no malformed JSON, and no empty responses. The overall mean length is **58.13 model tokens per response**, above the required minimum of 50. The median is 53. There are 288 additional occurrences of exactly repeated response texts and 678 outputs at or near the 128-token generation limit. These limitations will be disclosed when interpreting retrieval quality.


## Preprocessing and representation
The raw JSONL responses are preserved unchanged. We validated record integrity, checked for empty responses and duplicate texts, and retained all 20,000 document IDs. For sparse retrieval, scikit-learn's `TfidfVectorizer` performs **lowercasing** (`lowercase=True`, its default) and **word tokenization** (default token pattern) during vectorization; `norm='l2'` normalizes TF-IDF document and query vectors so their dot product yields cosine similarity. We use `dtype=np.float32` to reduce memory usage. These operations do not rewrite the original response strings. We do not apply additional stopword removal, stemming, lemmatization, or blanket punctuation stripping. Dense embedding experiments will use the original text with the embedding model's tokenizer. Vectorization time will be distinguished from similarity-search query latency in the performance evaluation.

## Resource use
The initial 15K required 7,381.57 s (2.03 docs/s); the final 5K required 3,192.16 s (1.57 docs/s). The aggregate generation time was 10,573.73 s. These figures characterize corpus preparation and must not be conflated with similarity search runtime. The experiment uses the same fixed corpus for subsequent representation and retrieval benchmarks.

## Reproducibility
The generation notebook, prompt mapping, model name, random seed, output schema, and checkpoint configuration are version-controlled. Search benchmarking will report vector representation, corpus scale, query set, top-k, hardware, BLAS/Numba thread settings, warmup, repeated timing runs, and correctness/recall checks.

## Pending results
Search baseline profiling, sparse/dense representation comparisons, SVD, parallel speedups, recall@k, and the final performance figures have not yet been measured; they will be added from actual experiment output.

## Phase 1 baseline and profiling protocol
The initial CSR/SciPy baseline measured 25 CPU query executions at 1,000 documents: median 359.895 ms, mean 368.495 ms, range 271.401–522.266 ms. Function-level cProfile attributed cumulative time primarily to repeated sparse multiply, row extraction, and sum operations; cumulative times overlap. This baseline uses Python iteration with SciPy arithmetic and therefore is **not** the pure Python nested-loop baseline required by the assignment.

Notebook `05_phase1_pure_python_profiling.ipynb` implements explicit element-wise Python cosine loops, validates scores against NumPy, profiles with both cProfile and line_profiler, and will save 25 repetitions per scale at 100/500/1,000 documents. To keep pure Python profiling tractable, it uses a separate fixed 512-feature dense TF-IDF representation. Figures are generated **only after** real measurements are run, so their measured values remain pending. Results using this representation cannot be directly compared for speedup against the earlier full-vocabulary CSR baseline.

## Phase 1 measured pure-Python baseline (2026-10-19)
Executed `05_phase1_pure_python_profiling.ipynb` on Colab CPU with fixed 512-feature TF-IDF dense vectors and Python nested loops (three loops per cosine). For each corpus scale, five queries were run five times (25 timings per scale), excluding vectorization, warmup and profiling overhead.

| Documents | Median query latency (ms) | Mean (ms) | Minimum (ms) | Maximum (ms) |
|---:|---:|---:|---:|---:|
| 100 | 29.685 | 30.106 | 15.461 | 53.626 |
| 500 | 125.223 | 121.429 | 78.180 | 185.954 |
| 1000 | 228.967 | 236.177 | 80.660 | 520.501 |

`cProfile` identified 1,000 calls to `pure_python_cosine` as the principal cost (0.284 s own time, 0.287 s cumulative). `line_profiler` found 99.7% of the inclusive time of `pure_python_topk` on the cosine-call line, with only about 0.1% on sorting. Within cosine, the three elementwise multiplication/accumulation lines account for 17.3%, 17.9%, and 17.4% of cosine line-profile time; their associated loop overhead accounts for 15.8%, 16.1%, and 15.2%. This yields three concrete optimization targets: repeated Python elementwise dot loop, repeated per-document norm loops, and Python-level loop iteration overhead. Profiled durations include profiler overhead and must not be substituted for benchmark latency.

Raw timings: `benchmarks/raw_results/pure_python_cosine_phase1_raw.csv`. Profiler outputs: `benchmarks/profiling/pure_python_1k_cprofile.txt` and `pure_python_1k_line_profile.txt`. Figures generated in Google Drive `figures/phase1_pure_python_scaling.png` and `figures/phase1_pure_python_latency_boxplot.png`. This 512-feature dense baseline is not directly comparable for speedup to the earlier full-vocabulary CSR/SciPy baseline.

## Representative sample items

The following are excerpts from actual corpus records, retaining the original document IDs and prompts:

| ID | Prompt | Response excerpt |
|---:|---|---|
| 19995 | Explain a reliability challenge in API authentication for a transportation provider. | A primary reliability challenge in API authentication for a transportation provider is ensuring that user credentials are securely transmitted over the network... |
| 19998 | Describe a simple example involving solar power for a small business. | Solar power can significantly reduce energy costs by offsetting electricity bills through renewable energy sources like photovoltaics (PV)... |
| 19999 | Describe a practical application of content moderation in an environmental project. | Content moderation is crucial in environmental projects to ensure that all submitted materials are accurate, reliable, and free from errors... |

These are English LLM-generated responses, not scraped web pages. Full items are in the Drive JSONL file; excerpts are shown here to keep the report compact.

## Phase 1 deliverables and figure references

- **Pure Python nested-loop cosine:** implemented in `notebooks/05_phase1_pure_python_profiling.ipynb`, with score/top-10 correctness checks.
- **Function profiler:** `benchmarks/profiling/pure_python_1k_cprofile.txt`.
- **Line profiler:** `benchmarks/profiling/pure_python_1k_line_profile.txt`.
- **Raw baseline measurements:** `benchmarks/raw_results/pure_python_cosine_phase1_raw.csv`, 75 rows (100/500/1000 docs, 5 queries x 5 repeats).
- **Figure 1, scaling:** https://drive.google.com/file/d/1N7Pjsv5ReqExrCd_gh5mp9Fi_Os6PBF1/view
- **Figure 2, latency boxplot:** https://drive.google.com/file/d/1pYJnUPp6SDtpQLq4aZrYDq5L0DPYDYK3/view

Both figures were created by the notebook from measured data, and are indexed in `figures/phase1_figures.md`. They currently reside in Drive. A separate screenshot of the profiling output may be inserted in the final PDF report.

**Phase 1 conclusion:** The line profiler attributes 99.7% of the inclusive top-k function time to the call performing cosine calculations. Element-wise dot and norm loops dominate; sorting is negligible at this scale. Therefore the next optimization targets are reducing Python loop overhead and reusing/avoiding redundant norm computations. The current dense 512-feature baseline is the appropriate like-for-like reference for those experiments.
