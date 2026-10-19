# Phase 2B — supplemental design decision validation

## Scope

Real cached 20K x 384 MiniLM embeddings. Fit-only comparison of NumPy full SVD, randomized TruncatedSVD and randomized PCA at N=1000/5000/20000, k=50/100/200, three repetitions, CPU with two BLAS threads. 81 recorded rows: NumPy full SVD is calculated once per repetition and its timing repeated for the three k values, so there are 63 unique fits. Retrieval experiment has 675 rows: 5 in-corpus query IDs, 5 repetitions, 3 N, 3 k, 3 representations. Excludes self-match. Top-10 overlap is **not** human-judged relevance or recall.

## Fit-only median seconds

| N | k | NumPy full SVD | TruncatedSVD | PCA |
|---:|---:|---:|---:|---:|
| 1000 | 50 | 0.2655 | 0.3114 | 0.3006 |
| 1000 | 100 | 0.2655 | 0.1708 | 0.2144 |
| 1000 | 200 | 0.2655 | 0.4655 | 0.4211 |
| 5000 | 50 | 0.7565 | 0.5234 | 0.5394 |
| 5000 | 100 | 0.7565 | 0.8439 | 1.0215 |
| 5000 | 200 | 0.7565 | 1.3025 | 1.1515 |
| 20000 | 50 | 0.8333 | 1.5589 | 1.5879 |
| 20000 | 100 | 0.8333 | 2.4875 | 2.3474 |
| 20000 | 200 | 0.8333 | 3.9897 | 3.6415 |

TruncatedSVD's largest **tested** k beating full SVD: N=1000: k=100; N=5000: k=50; N=20000: none. This is not a continuous crossover or a universal threshold. Full SVD is fast for only 384 columns on this CPU. The earlier fit-plus-reconstruction numbers must not be compared as fit-only.

## Search results at 20K

Each group has 25 timings; the table shows median milliseconds and mean overlap with the original 384-d top-10 ranking.

| k | Representation | Median search ms | Mean top-10 overlap |
|---:|---|---:|---:|
| 50 | original 384 | 1.833 | 1.00 |
| 50 | TruncatedSVD | 0.266 | 0.82 |
| 50 | PCA | 0.212 | 0.78 |
| 100 | original 384 | 2.334 | 1.00 |
| 100 | TruncatedSVD | 0.409 | 0.94 |
| 100 | PCA | 0.324 | 0.90 |
| 200 | original 384 | 3.673 | 1.00 |
| 200 | TruncatedSVD | 1.747 | 0.96 |
| 200 | PCA | 1.561 | 0.96 |

The original 384-d search was repeated independently at each k, hence different median baseline times due to run variability. Relative speedups must use the matching k run, and are not a universal guarantee. TruncatedSVD k=100 is a **provisional** balanced choice: 74% projected matrix storage reduction, 94% top-10 overlap, faster measured ranking than original 384-d. k=200 increases overlap to 96% but uses twice the projected storage. PCA is somewhat faster in the measured ranking but has lower overlap at k=100. Five in-corpus queries are insufficient for robust search-quality claims. Query embedding and fitting time excluded from search timings.

## Instructor questions

1. **Which method and why?** Provisional TruncatedSVD k=100, based on storage, ranking speed, and overlap; verify on held-out query set and task relevance labels before final commitment.
2. **Why is TruncatedSVD sometimes faster than full SVD?** Randomized top-k approximation avoids calculating all components. For this 384-column matrix it did not always win; at 20K none of tested k values beat full SVD.
3. **PCA vs SVD?** PCA centers feature columns before decomposition; sklearn TruncatedSVD does not center.
4. **Practical maximum winning k?** For N=1000 tested k=100, N=5000 k=50, N=20000 none; limited to tested grid and machine.

## Artifacts

[Fit-only CSV on Drive](https://drive.google.com/file/d/1AAeyMEn6LFZ35kxKnirJPeR2ZOnTF592/view) · [Retrieval raw CSV](../benchmarks/raw_results/phase2b_retrieval_full.csv) · [Decision PNG](https://drive.google.com/file/d/1-ki40ovez2Lhah2K3xP4bXOjBcvzuH2n/view) · [Notebook](../notebooks/08_phase2b_decision_validation.ipynb).
