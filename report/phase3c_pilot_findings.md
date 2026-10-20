# Phase 3C Pilot Findings

Verified 18 timing rows, six configurations, three repeats. Simulated 40 requests x 20ms: serial 813.310ms; concurrency 5 163.634ms (4.970x), 10 82.431ms (9.867x), 20 41.608ms (19.547x). CPU control 5K docs x16 queries: sequential 17.116ms; asyncio.gather 16.013ms. No genuine CPU parallelism.

[Raw](https://drive.google.com/file/d/1GBK9xXM-yiX_6IntRBCTSkkiGVCdP7ei/view) | [Summary](https://drive.google.com/file/d/1zvshl3ipUyqHsIWQAPFLGLhRjH4ppQLW/view) | [Figure](https://drive.google.com/file/d/1D2SwYkI9r83djU3x_kUTkDgGFuJr3dXF/view). See [full findings](phase3c_full_findings.md).
