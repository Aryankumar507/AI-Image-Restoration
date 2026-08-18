#!/usr/bin/env python
"""Compare defect preservation across the PoC training variants.

Runs the same audit (identical seeds, identical defect placements) against
several checkpoints and prints one table, so the effect of each training-data
change is isolated.

Variants under test:
    baseline    2 GT families, no defects in the target      (the shipped model)
    A_defects   2 GT families, defects present in the target
    B_families  5 GT families, no defects in the target
    AB_both     5 GT families, defects present in the target

Every model is audited on the SAME GT families (--kinds, default texture +
dendrite) because that is the only set the baseline ever saw; auditing it on
families it was never trained on would not be a fair comparison.

Usage (from the repo root, after `pip install -e .`):
    python scripts/poc_compare.py --n 60
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from defect_preservation import contrast, inject_defect, restore  # noqa: E402
from kla_restoration.data_generator import degrade, make_ground_truth  # noqa: E402
from kla_restoration.model import NAFNetLiteSR  # noqa: E402

import cv2  # noqa: E402


def audit(ckpt_path, n, size, kinds, device):
    """Return (per-defect-type recoveries, overall array, false-structure array)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = NAFNetLiteSR.from_checkpoint(ckpt).to(device).eval()

    recoveries, by_kind = [], {}
    rng = np.random.default_rng(0)          # same seed for every variant

    for i in range(n):
        kind_gt = str(rng.choice(kinds))
        gt_clean = make_ground_truth(size=size, kind=kind_gt, seed=10_000 + i)
        gt_def, mask, dkind = inject_defect(gt_clean, rng)

        deg = degrade(gt_def, rng=np.random.default_rng(50_000 + i))
        pred = restore(model, deg, device)

        c_true = contrast(gt_def, mask)
        if abs(c_true) > 1e-6:
            rec = contrast(pred, mask) / c_true
            recoveries.append(rec)
            by_kind.setdefault(dkind, []).append(rec)

    false_struct = []
    for i in range(n):
        kind_gt = str(rng.choice(kinds))
        gt_clean = make_ground_truth(size=size, kind=kind_gt, seed=70_000 + i)
        deg = degrade(gt_clean, rng=np.random.default_rng(80_000 + i))
        pred = restore(model, deg, device)
        hp_pred = cv2.Laplacian(pred, cv2.CV_32F)
        hp_gt = cv2.Laplacian(gt_clean, cv2.CV_32F)
        false_struct.append(float(
            np.clip(np.abs(hp_pred) - np.abs(hp_gt), 0, None).mean()
            / (np.abs(hp_gt).mean() + 1e-8)))

    return by_kind, np.array(recoveries), np.array(false_struct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--kinds", nargs="+", default=["texture", "dendrite"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    variants = [
        ("baseline",   REPO_ROOT / "checkpoints" / "nafnet_lite.pt"),
        ("A_defects",  REPO_ROOT / "checkpoints" / "poc_A_defects.pt"),
        ("B_families", REPO_ROOT / "checkpoints" / "poc_B_families.pt"),
        ("AB_both",    REPO_ROOT / "checkpoints" / "poc_AB_both.pt"),
        ("AB+DAL4",    REPO_ROOT / "checkpoints" / "poc_DAL4.pt"),
        ("AB+DAL8",    REPO_ROOT / "checkpoints" / "poc_DAL8.pt"),
        ("AB+DAL16",   REPO_ROOT / "checkpoints" / "poc_DAL16.pt"),
    ]

    rows = []
    for name, path in variants:
        if not path.is_file():
            print(f"  (skipping {name}: {path.name} not found)")
            continue
        by_kind, rec, fs = audit(path, args.n, args.size, tuple(args.kinds), args.device)
        rows.append((name, by_kind, rec, fs))

    print()
    print(f"Defect preservation across training variants "
          f"(n={args.n}, families: {', '.join(args.kinds)})")
    print("=" * 96)
    print(f"{'variant':<12} {'mean':>7} {'median':>7} {'<0.5':>7} {'<0.2':>7} "
          f"{'particle':>9} {'void':>8} {'break':>8} {'false-str':>10}")
    print("-" * 96)
    for name, by_kind, rec, fs in rows:
        def km(k):
            v = by_kind.get(k)
            return f"{np.mean(v):9.3f}" if v else f"{'--':>9}"
        print(f"{name:<12} {rec.mean():7.3f} {np.median(rec):7.3f} "
              f"{(rec < 0.5).mean():6.1%} {(rec < 0.2).mean():6.1%} "
              f"{km('particle')} {km('void')[:8]:>8} {km('break')[:8]:>8} "
              f"{fs.mean():10.3f}")
    print("-" * 96)
    print("mean/median: defect contrast recovery, 1.0 = faithful, 0.0 = erased")
    print("<0.5, <0.2 : fraction of defects substantially lost / effectively erased")
    print("false-str  : invented high-frequency detail on defect-free images (0 = none)")

    out = REPO_ROOT / "outputs" / "defect_audit" / "poc_comparison.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"n={args.n}, families: {', '.join(args.kinds)}\n\n")
        f.write(f"{'variant':<12} {'mean':>7} {'median':>7} {'<0.5':>7} {'<0.2':>7} {'false-str':>10}\n")
        for name, _, rec, fs in rows:
            f.write(f"{name:<12} {rec.mean():7.3f} {np.median(rec):7.3f} "
                    f"{(rec < 0.5).mean():6.1%} {(rec < 0.2).mean():6.1%} {fs.mean():10.3f}\n")
    print(f"\nwritten to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
