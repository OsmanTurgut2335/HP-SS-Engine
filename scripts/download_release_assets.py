#!/usr/bin/env python3
"""Download verified v1.0-final assets from the public GitHub Release.

Usage:
    python scripts/download_release_assets.py --root ./HP-SS-Engine
    python scripts/download_release_assets.py --root ./HP-SS-Engine --verify-only
No Google Drive credentials are required. Downloads are verified by SHA-256.
"""
import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

TAG = "v1.0-final"
BASE = "https://github.com/OsmanTurgut2335/HP-SS-Engine/releases/download/" + TAG + "/"
ASSETS = {
    "qwen_corpus_20k_progress.jsonl": (
        "data/raw/qwen_corpus_20k_progress.jsonl", 10830847,
        "00fa40516cb2934e797dfb52e1906ccfcd395d489cf6f5213bc272b2a8940a25"),
    "qwen_20k_minilm_384_float32.npy": (
        "phase2_matrix_ops/2A_sparse_dense/qwen_20k_minilm_384_float32.npy", 30720128,
        "207e8271de8d8787a1ac4694ca9369721e38ad47d0190a573d08fa0f2510dea9"),
    "truncated_svd_k100_full_20k.joblib": (
        "phase4_final_system/models/truncated_svd_k100_full_20k.joblib", 155543,
        "d220f7378fa80c95fa8527dc4439c28061707fbd052dd2e8afcc300e89337e68"),
    "reduced_index_k100_full_20k.npy": (
        "phase4_final_system/models/reduced_index_k100_full_20k.npy", 8000128,
        "d269d94082c84dfd0838b61a8faeeac83f51fb01b8bbec8959f5352aba5b8321"),
}

def valid(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == digest

def download(name, path, size, digest):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".hp-ss-", dir=str(path.parent))
    os.close(fd)
    try:
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "HP-SS-Engine-reproduction/1.0"})
        with urllib.request.urlopen(req, timeout=120) as src, open(tmp, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        if not valid(Path(tmp), size, digest):
            raise RuntimeError(f"Checksum/size mismatch: {name}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("./HP-SS-Engine"))
    parser.add_argument("--verify-only", action="store_true", help="Do not download; check existing files")
    args = parser.parse_args()
    failures = []
    for name, (relative, size, digest) in ASSETS.items():
        path = args.root / relative
        if valid(path, size, digest):
            print(f"OK      {relative} (SHA-256 verified)")
            continue
        if args.verify_only:
            print(f"MISSING/CORRUPT {relative}")
            failures.append(name)
            continue
        print(f"GET     {name} ({size:,} bytes)", flush=True)
        try:
            download(name, path, size, digest)
            print(f"OK      {relative} (SHA-256 verified)", flush=True)
        except Exception as exc:
            failures.append(name)
            print(f"FAILED  {name}: {exc}", file=sys.stderr)
    if failures:
        print(f"\nFAIL: {len(failures)} missing/invalid asset(s)", file=sys.stderr)
        return 1
    print("\nPASS: all four GitHub Release assets verified.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
