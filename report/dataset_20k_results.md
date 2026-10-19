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
