# Phase 2A — Sparse vs Dense: verified results

## Setup and reproducibility

- Corpus: 20,000 English Qwen responses, IDs 0–19999.
- Sparse representation: float32 TF-IDF, L2-normalized, CSR, vocabulary **9,617** features.
- Dense representation: sentence-transformers/all-MiniLM-L6-v2, float32 L2-normalized, **384** dimensions; generated once and cached on Google Drive.
- All search benchmarks: CPU; Colab GPU quota prevented T4 use, but GPU is not required for these CPU benchmarks.
- Query workload: 5 prompts, 5 repetitions each, 4 corpus sizes: 1K/5K/10K/20K, top-10 retrieval; 400 timing observations. Each search loops through corpus documents; conversion and embedding generation are excluded from query timings.
- Custom dot functions: sparse–sparse (two-pointer on sorted CSR indices), sparse–dense (nonzero index loop), dense–dense (elementwise NumPy multiply and sum; no dot or @). Scores checked against reference dot products at 1K pilot; first unsorted-CSR bug corrected by `sort_indices()`.
- Four workloads: (a) TF-IDF sparse–sparse; (b) embedding sparse–sparse using embedding CSR copy; (c) embedding sparse–dense using embedding CSR copy; (d) embedding dense–dense using original embedding dense. **Only (b), (c), (d) are directly comparable as same-vector workloads.** They also use different custom arithmetic algorithms; speed ratios reflect both format and implementation.

## Memory footprint (MiB, array storage only)

| Documents | TF-IDF CSR | Embedding dense | Embedding stored CSR |
|---:|---:|---:|---:|
| 1,000 | 0.295 | 1.465 | 2.934 |
| 5,000 | 1.472 | 7.324 | 14.668 |
| 10,000 | 2.956 | 14.648 | 29.335 |
| 20,000 | 6.701 | 29.297 | 58.670 |

CSR bytes include `data`, `indices` and `indptr`; dense bytes are the float32 array's `nbytes`. Python object overhead, vectorizer, model weights and runtime allocations are excluded. Sparse TF-IDF has 868,306 nonzero entries at 20K; embedding CSR has 7,680,000, i.e. essentially fully populated.

## Search time (median milliseconds per query, n=25 per cell)

| Documents | TF-IDF sparse–sparse | Embedding sparse–sparse | Embedding sparse–dense | Embedding dense–dense |
|---:|---:|---:|---:|---:|
| 1,000 | 25.579 | 259.653 | 151.438 | 8.598 |
| 5,000 | 143.842 | 1947.305 | 765.174 | 36.576 |
| 10,000 | 254.387 | 3628.050 | 1531.437 | 73.295 |
| 20,000 | 560.687 | 6066.276 | 3054.274 | 149.586 |

At 20K, dense–dense embedding was **40.55× faster** than embedding sparse–sparse and **20.42× faster** than embedding sparse–dense *in these particular implementations*. Do not generalize this to all sparse/dense libraries or models. The TF-IDF sparse–sparse latency is a different representation/workload and must not be used for a like-for-like speedup.

Timing variability was noticeable; some outliers raised the means above medians. Report medians with all 25 raw measurements retained. GPU embedding inference performance was not benchmarked.

## Why the results make sense

TF-IDF vectors contain many zeros, so CSR stores only a few terms per document and saves substantial memory. Semantic embeddings have nearly all 384 coordinates nonzero; CSR stores values *and* their indices, making it about 2× larger than dense. The custom two-pointer and Python sparse-index loops also have substantial Python overhead, whereas dense elementwise multiplication uses fast NumPy operations.

**Design decision:** retain sparse TF-IDF as a memory-efficient lexical retrieval option and dense float32 embeddings for semantic similarity. The final retrieval method requires further accuracy/recall and optimized search benchmarks. Phase 2B will test dimensionality reduction on dense embeddings.

## Figures and raw data

- [Memory figure PNG (Drive)](https://drive.google.com/file/d/15WUfBEKUbaZHiYq-7cW9Nw94X-L5qtb7/view)
- [Manual dot latency PNG (Drive)](https://drive.google.com/file/d/1feqmRY5DTqPz09j5fc1uViTB8E0OaDPt/view)
- [Raw timing CSV](../benchmarks/raw_results/phase2a_dot_products_raw.csv)
- [Memory CSV](../benchmarks/raw_results/phase2a_memory.csv)
- [Notebook](../notebooks/06_phase2a_sparse_dense.ipynb)

The PNGs are stored in the Phase 2 Drive workspace; the figure index in GitHub links to them. **No performance values are synthetic.**
