# Phase 2B full dimensionality reduction experiment

36 measured runs using real 20,000 x 384 MiniLM embeddings. Four N sizes: 1000, 5000, 10000, 20000; k: 50, 100, 200; three methods: custom power iteration, randomized TruncatedSVD, randomized PCA.

At N=20000, fitting plus reconstruction (seconds):

| k | Power iteration | TruncatedSVD | PCA |
|---|---:|---:|---:|
| 50 | 3.776 | 1.322 | 1.376 |
| 100 | 5.162 | 4.038 | 2.023 |
| 200 | 8.022 | 5.358 | 3.294 |

At N=20000, relative Frobenius errors:

| k | Power iteration | TruncatedSVD | PCA |
|---|---:|---:|---:|
| 50 | 0.50149 | 0.50157 | 0.49986 |
| 100 | 0.34428 | 0.34442 | 0.34366 |
| 200 | 0.18449 | 0.18469 | 0.18452 |

Projected float32 storage at N=20000: original 384d 29.297 MiB; k=50 3.815 MiB; k=100 7.629 MiB; k=200 15.259 MiB. These figures exclude model and peak working memory. The power iteration algorithm reached its iteration cap in the 20K runs. Reconstruction error is not search accuracy. These initial timings include reconstruction; the separate fit-only benchmark is in the decision report.

See [full measurements](../benchmarks/raw_results/phase2b_methods_full.csv), [projected memory](../benchmarks/raw_results/phase2b_projected_matrix_memory.csv), [decision report](phase2b_decision_results.md), and [full plot](https://drive.google.com/file/d/1f-hZNVM2xCIfoV_PJLQPtVPFCTBgKkyX/view).
