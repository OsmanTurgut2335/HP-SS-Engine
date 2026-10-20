# Phase 3C Full Findings

Completed 2026-10-20. Verified 35 raw measurements, 7 summary configurations, five repeats.

Simulated I/O: 120 requests x 50ms `asyncio.sleep`; median times: serial 6040.915ms; concurrency 5 1212.027ms (4.984x), 10 608.049ms (9.935x), 20 305.656ms (19.764x), 40 154.435ms (39.116x). These measure wait overlap, **not** real API embedding generation or CPU search.

CPU control, 20K documents x 128 queries: sequential 922.691ms, asyncio.gather 844.110ms. The async search function contains no await in CPU work; tasks do not execute CPU work in parallel. Samples varied substantially; 1.093x median difference must not be attributed to asyncio CPU parallelism.

Data: [raw](https://drive.google.com/file/d/1lDJEqdLqYwQCpY8D_j8T5dqLnJJLCxrc/view), [summary](https://drive.google.com/file/d/1WjYnUZ0dJkGMbVSrU92UrfPGfFU7AmMp/view), [figure](https://drive.google.com/file/d/1QnpkYvmxHTajzZmd13xV6uJNV8nA_OR1/view), [notebook](../notebooks/12_phase3c_asyncio.ipynb).

See [final comparison](phase3_final_comparison.md) for all five methods and limitations.
