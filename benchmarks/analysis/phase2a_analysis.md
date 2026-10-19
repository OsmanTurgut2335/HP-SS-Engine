# Phase 2A benchmark analysis

Completed all mandatory corpus scales 1K, 5K, 10K, 20K; 4 workloads × 5 queries × 5 repeats = 400 timed queries. Verified Drive files and copied CSVs to GitHub.

At 20K median latency (ms): TF-IDF sparse–sparse 560.687; embedding sparse–sparse 6066.276; embedding sparse–dense 3054.274; embedding dense–dense 149.586. Embedding dense–dense is 40.55× faster than embedding sparse–sparse **for our measured custom implementations**, not a universal speedup. TF-IDF is a different representation.

At 20K matrix storage MiB: TF-IDF CSR 6.701; embedding dense 29.297; embedding CSR 58.670. Sparse representation is advantageous only when many coordinates are zero.

Two plots in Drive: [memory](https://drive.google.com/file/d/15WUfBEKUbaZHiYq-7cW9Nw94X-L5qtb7/view), [latency](https://drive.google.com/file/d/1feqmRY5DTqPz09j5fc1uViTB8E0OaDPt/view). See [report](../../report/phase2a_sparse_dense_results.md).

Bug fixed during pilot: sparse two-pointer method assumes sorted CSR column indices; explicit sort_indices() added. CPU benchmark used because Colab T4 was unavailable; 20K embeddings were generated and cached.
