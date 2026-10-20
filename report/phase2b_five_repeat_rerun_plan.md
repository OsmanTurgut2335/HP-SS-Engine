# Phase 2B — Five-repeat benchmark correction (pending execution)

Instructor requires **warmup runs and at least 5 timed iterations** for all final benchmarks. Earlier Phase 2B measurements did not meet this universally: notebook 07 fit+reconstruction recorded one timing per configuration, and notebook 08 fit-only recorded three. The prior results remain historical/exploratory; **do not represent them as compliant final five-run results**.

## Updated notebooks

1. [07 — Dimensionality reduction](../notebooks/07_phase2b_dimensionality_reduction.ipynb): `PILOT=False` runs 5 timed fit+reconstruction repetitions for every N/k/method after one untimed warmup; includes custom block power iteration, TruncatedSVD, PCA, and 1K full SVD reference. Saves new `phase2b_methods_raw_full_5run.csv`, `phase2b_methods_full_5run.csv`, `phase2b_full_svd_reference_1k_raw_full_5run.csv` and figures to Drive. Custom power iteration at 20K×200 may take substantial Colab CPU time; do not silently omit a method or extrapolate skipped measurements.
2. [08 — Decision validation](../notebooks/08_phase2b_decision_validation.ipynb): `PILOT=False` runs 5 timed fit-only repetitions after 3 untimed warmups for each N/k/method; retrieval diagnostic uses 5 timed queries per query ID after warmups. New `phase2b_fit_only_full_5run.csv` and `phase2b_retrieval_full_5run.csv`. The five fixed in-corpus queries measure overlap, **not relevance**; Phase 4 should use 50–200 search queries as specified by the assignment.

## Completion criteria

- Execute both notebooks in Colab, verify full-run CSV files and their `repeat` columns (0–4), inspect failed/unfinished workloads, archive final CSV and figures to GitHub, update Phase 2 findings and Development/Bottleneck logs.
- No new measurements have been claimed yet; updating code alone is not execution.
- For fairness, interpret Phase 2B `fit+reconstruct` separately from `fit-only`, and compare medians **within matched hardware sessions**.
- Then proceed to Phase 4: end-to-end embedding → reduction → BLAS search → top-k, scaling N=1K/5K/10K/20K, correct top-k vs baseline at tolerance 1e-5, warmups and >=5 repeats, and live demo.
