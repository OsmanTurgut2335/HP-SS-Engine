# Phase 3B — Full ProcessPool Findings (2026-10-20)

**FULL benchmark completed and verified** from Drive: 840 timed rows (168 configurations × 5 repeats), 36 process startup+first-search records. Same real normalized MiniLM 20K×384 embeddings; document sizes 1K, 5K, 10K, 20K; 16, 64, 128 queries; BLAS threads fixed at 1. Implementations: sequential NumPy, batched BLAS, ThreadPool 1/2/4, ProcessPool 1/2/4 with chunksize 1/8/32.

## Evidence
- [Full raw CSV](https://drive.google.com/file/d/1LV1DKFEoNvpiZk1mDcbk3-VOV5fr46RH/view)
- [Full summary CSV](https://drive.google.com/file/d/1-pzqfzFE0mAjP-JV6kuxUBSU1SZXteie/view)
- [Full startup CSV](https://drive.google.com/file/d/1koL-vNPRYFTRcSYbEc1qchQ5Y5-Y_NLz/view)
- [Notebook](../notebooks/11_phase3b_processpool.ipynb)
- [Pilot](phase3b_pilot_findings.md)
- [Phase 3A full](phase3a_full_findings.md)

## Matched 20K docs × 128 queries, medians

| Method | Median ms | Speedup vs sequential (same run) |
|---|---:|---:|
| Sequential | 670.172 | 1.000× |
| Batched BLAS | **435.124** | **1.540×** |
| ThreadPool 1 | 680.363 | 0.985× |
| ThreadPool 2 | 607.038 | 1.104× |
| ThreadPool 4 | 489.070 | 1.370× |
| ProcessPool 1, chunk=1 | 914.898 | 0.733× |
| ProcessPool 1, chunk=8 | 689.899 | 0.971× |
| ProcessPool 1, chunk=32 | 666.600 | 1.005× |
| ProcessPool 2, chunk=1 | 529.207 | 1.266× |
| ProcessPool 2, chunk=8 | 736.794 | 0.910× |
| ProcessPool 2, chunk=32 | 782.662 | 0.856× |
| ProcessPool 4, chunk=1 | 532.331 | 1.259× |
| ProcessPool 4, chunk=8 | **475.582** | **1.409×** |
| ProcessPool 4, chunk=32 | 480.161 | 1.396× |

Startup **plus first search** at 20K×128: 1 worker 749.536 ms; 2 workers 609.667 ms; 4 workers 651.020 ms. These are not pure startup times and are excluded from steady-state timing.

## Findings

1. At the largest workload, ProcessPool 4/chunk8 beats sequential by 1.409× but is still slower than batched BLAS (475.582 vs 435.124 ms); ThreadPool 4 is 489.070 ms. Thus ProcessPool can help at scale but does not dominate.
2. Chunking matters, but **is not monotonic**. For 4 workers, chunk8 improves over chunk1 (475.582 vs 532.331 ms). For 2 workers, chunk1 (529.207 ms) outperforms chunk8 (736.794 ms) and chunk32 (782.662 ms). Avoid saying larger chunks always help. Scheduling, load balancing and runtime noise can affect these outcomes; reruns may be warranted.
3. GIL: ProcessPool bypasses per-process Python GIL, but NumPy BLAS often releases GIL, making process-based NumPy less compelling than for pure-Python CPU work. Process task IPC, serialization and worker lifecycle add overhead.
4. This ProcessPool design uses Linux fork with copy-on-write inherited corpus; the entire corpus is **not** serialized per query. Query vectors and result IDs cross process boundaries. Results may differ on spawn-based environments.
5. Amdahl's Law, limited CPU cores, memory bandwidth and scheduling cap speedup. Process startup and first query is separately tracked; for one-shot use, include that overhead in end-to-end evaluation.
6. **Across-run comparability warning:** Phase 3A 20K×128 sequential BLAS1 median was 446.500 ms, Phase 3B 670.172 ms. Runtime/hardware load changed. Compare Phase 3B strategies against Phase 3B sequential, not by mixing absolute times across Colab sessions.
7. Phase 3A numerical finding persists: BLAS matrix-vector vs matrix-matrix float32 accumulation yielded max absolute score difference 8.940697e-07 and swapped ranks of near-tied IDs 8335 and 9767, but same top-10 ID set. See [Phase 3A full findings](phase3a_full_findings.md). Full score-level `np.allclose` test remains recommended.
8. **Provisional speed/complexity winner for this workload:** batched NumPy/BLAS, simplest and fastest in same-run 20K×128. ProcessPool worthwhile to explore for heavier pure-Python per-query work or different CPU resources; untested here.

## Open follow-ups
- [ ] Confirm figure `phase3b_processpool_speedup_full.png` exists and link it.
- [ ] Re-test anomalous chunk/worker combinations with repeated randomized ordering.
- [ ] Compare process memory footprint and separately benchmark serialization cost if assignment requires.
- [ ] Phase 3C asyncio simulated I/O (not directly comparable CPU acceleration); consolidate five strategies and note Numba Phase 2C Gram vs search workload distinction.
