#!/usr/bin/env python3
"""Offline CPU smoke test of the exact four published Release assets.

Run after downloading assets:
  python scripts/smoke_test_search.py --root ./HP-SS-Engine

Uses a saved document embedding as query to test model + index + ranking.
It does not download the MiniLM text encoder or replace the live text-query demo.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("./HP-SS-Engine"))
    args = p.parse_args()
    root = args.root
    with (root / "data/raw/qwen_corpus_20k_progress.jsonl").open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    docs.sort(key=lambda row: row["id"])
    assert len(docs) == 20000
    embeddings = np.load(root / "phase2_matrix_ops/2A_sparse_dense/qwen_20k_minilm_384_float32.npy", mmap_mode="r")
    svd = joblib.load(root / "phase4_final_system/models/truncated_svd_k100_full_20k.joblib")
    index = np.load(root / "phase4_final_system/models/reduced_index_k100_full_20k.npy", mmap_mode="r")
    assert embeddings.shape == (20000, 384)
    assert index.shape == (20000, 100)
    # Same SVD transform + normalization used for text queries.
    q = normalize(svd.transform(np.asarray(embeddings[0:1], dtype=np.float32)), axis=1)[0].astype(np.float32)
    with threadpool_limits(limits=1):
        scores = index @ q
    order = np.lexsort((np.arange(len(scores)), -scores))[:10]
    assert len(order) == 10 and len(set(order.tolist())) == 10
    assert np.isfinite(scores).all()
    print("PASS: 20K corpus + 384d embeddings + SVD100 + normalized index + CPU Top-10")
    for rank, i in enumerate(order, 1):
        doc = docs[int(i)]
        print(f"{rank:2d}. document_id={doc['id']}, cosine={scores[i]:.6f}, excerpt={doc['response'][:90]!r}")
    print("NOTE: this is an asset/search smoke test, not a benchmark or human relevance evaluation.")

if __name__ == "__main__":
    main()
