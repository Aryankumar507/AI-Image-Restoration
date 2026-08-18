#!/usr/bin/env python
"""Generate the synthetic paired training/validation dataset.

NOTE: this produces *stand-in* data, not real semiconductor imagery -- see
docs/ASSESSMENT.md (W-01) for why that is the project's largest open risk.
Replace with a real loader once KLA's dataset is released.

Usage (from the repo root, after `pip install -e .`):
    python scripts/generate_data.py --n_train 200 --size 256
"""
import argparse
from pathlib import Path

from kla_restoration.data_generator import build_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=200, help="Number of training pairs")
    ap.add_argument("--n_val", type=int, default=None,
                    help="Number of validation pairs (default: max(10, n_train//10))")
    ap.add_argument("--size", type=int, default=256, help="Ground-truth image size (square)")
    ap.add_argument("--scale_factor", type=int, default=2, help="Downsampling factor")
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "data"),
                    help="Root directory to write train/ and val/ into")
    args = ap.parse_args()

    n_val = args.n_val if args.n_val is not None else max(10, args.n_train // 10)
    out_dir = Path(args.out_dir)

    build_dataset(out_dir / "train", n_samples=args.n_train, size=args.size,
                  scale_factor=args.scale_factor, seed0=0)
    build_dataset(out_dir / "val", n_samples=n_val, size=args.size,
                  scale_factor=args.scale_factor, seed0=100000)


if __name__ == "__main__":
    main()
