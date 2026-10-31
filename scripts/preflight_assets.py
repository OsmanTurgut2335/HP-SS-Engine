#!/usr/bin/env python3
"""Validate required HP-SS Engine assets before running Phase 4 notebooks.

Colab example:
  !python scripts/preflight_assets.py --root /content/drive/MyDrive/HP-SS-Engine
"""
import argparse
import json
import sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True,help="Drive project root")
    args=parser.parse_args()
    root=args.root
    corpus=root/"data/raw/qwen_corpus_20k_progress.jsonl"
    embedding=root/"phase2_matrix_ops/2A_sparse_dense/qwen_20k_minilm_384_float32.npy"
    model=root/"phase4_final_system/models/truncated_svd_k100_full_20k.joblib"
    index=root/"phase4_final_system/models/reduced_index_k100_full_20k.npy"
    files={"corpus":corpus,"embedding":embedding,"svd_model":model,"reduced_index":index}
    missing=[name for name,path in files.items() if not path.is_file()]
    for name,path in files.items():
        print(f"{'OK' if path.is_file() else 'MISSING':7} {name:14} {path}")
    if missing:
        print("\nMissing files: "+", ".join(missing))
        print("See docs/REPRODUCIBILITY.md; generate corpus -> embeddings -> Phase 4A.")
        return 2
    try:
        import numpy as np
        import joblib
        with corpus.open(encoding="utf-8") as f:
            rows=[json.loads(line) for line in f if line.strip()]
        rows.sort(key=lambda r:r["id"])
        assert len(rows)==20000, f"Expected 20K documents, got {len(rows)}"
        assert [r["id"] for r in rows]==list(range(20000)), "Document IDs/order incorrect"
        assert all(isinstance(r.get("response"),str) and r["response"].strip() for r in rows)
        X=np.load(embedding,mmap_mode="r")
        Z=np.load(index,mmap_mode="r")
        svd=joblib.load(model)
        assert X.shape==(20000,384) and X.dtype==np.float32,(X.shape,X.dtype)
        assert Z.shape==(20000,100) and Z.dtype==np.float32,(Z.shape,Z.dtype)
        assert svd.components_.shape==(100,384),svd.components_.shape
        assert np.isfinite(Z).all()
        assert np.allclose(np.linalg.norm(Z,axis=1),1,atol=1e-4)
        print("\nPASS: corpus, embeddings, SVD and normalized reduced index are consistent.")
        return 0
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 1

if __name__=="__main__":
    raise SystemExit(main())
