#!/usr/bin/env python
"""Defect preservation and hallucination audit (Idea D).

WHY THIS EXISTS
---------------
SSIM, PSNR and LPIPS all reward an output that LOOKS like clean semiconductor
imagery. None of them ask the question a metrology engineer actually cares
about:

    Does the restoration preserve real defects, and does it invent fake ones?

A model trained on a narrow family of structures can learn what a grid should
look like and regenerate an idealised one, scoring well on every standard
metric while erasing the very anomaly the inspection exists to find. In
metrology a hallucinated detail is worse than a blurry one.

METHOD
------
  1. Take a clean ground-truth image and inject a small synthetic defect
     (a bright particle, a dark void, or a broken line).
  2. Degrade the defective image through the normal pipeline.
  3. Restore it with the model.
  4. Measure, inside the defect region only:
       * CONTRAST RECOVERY -- how much of the defect's true contrast against
         its surroundings survives restoration. About 1.0 is faithful, near 0
         means the defect was cleaned away.
  5. Measure, over defect-free images:
       * FALSE STRUCTURE -- high-frequency detail present in the output but
         absent from the truth, i.e. structure the model invented.

Usage (from the repo root, after `pip install -e .`):
    python scripts/defect_preservation.py --n 40
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

from kla_restoration.data_generator import KINDS, degrade, make_ground_truth
from kla_restoration.model import NAFNetLiteSR

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "checkpoints" / "nafnet_lite.pt"
OUT_DIR = REPO_ROOT / "outputs" / "defect_audit"


def inject_defect(gt, rng):
    """Add one small synthetic defect. Returns (image, mask, kind)."""
    img = gt.copy()
    h, w = img.shape
    mask_u8 = np.zeros((h, w), dtype=np.uint8)

    kind = str(rng.choice(["particle", "void", "break"]))
    # Keep away from the border so the defect is fully visible.
    cy = int(rng.integers(h // 6, 5 * h // 6))
    cx = int(rng.integers(w // 6, 5 * w // 6))

    if kind == "particle":                      # bright contamination blob
        r = int(rng.integers(3, 7))
        cv2.circle(img, (cx, cy), r, 255.0, -1)
        cv2.circle(mask_u8, (cx, cy), r, 1, -1)
    elif kind == "void":                        # dark pit / missing material
        r = int(rng.integers(3, 7))
        cv2.circle(img, (cx, cy), r, 0.0, -1)
        cv2.circle(mask_u8, (cx, cy), r, 1, -1)
    else:                                       # break: erase a line segment
        half = int(rng.integers(4, 9))
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        img[y0:y1, x0:x1] = float(np.median(gt))
        mask_u8[y0:y1, x0:x1] = 1

    return img, mask_u8.astype(bool), kind


def restore(model, deg, device="cpu"):
    with torch.no_grad():
        t = torch.from_numpy(deg / 255.0).float()[None, None].to(device)
        return model(t).clamp(0, 1).squeeze().cpu().numpy() * 255.0


def contrast(img, mask):
    """Mean intensity difference between the defect region and its surround."""
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8), 1).astype(bool)
    ring &= ~mask
    if not mask.any() or not ring.any():
        return 0.0
    return float(img[mask].mean() - img[ring].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="Images per condition")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_panels", type=int, default=6)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT),
                    help="Model to audit (default: the submission checkpoint)")
    ap.add_argument("--tag", default="", help="Suffix for output filenames")
    ap.add_argument("--kinds", nargs="+", default=None,
                    help="GT families to audit on (default: all). Pin this "
                         "when comparing models trained on different sets.")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = NAFNetLiteSR.from_checkpoint(ckpt).to(args.device).eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kinds = tuple(args.kinds) if args.kinds else KINDS
    recoveries, by_kind = [], {}
    rng = np.random.default_rng(0)

    for i in range(args.n):
        kind_gt = str(rng.choice(kinds))
        gt_clean = make_ground_truth(size=args.size, kind=kind_gt, seed=10_000 + i)
        gt_def, mask, dkind = inject_defect(gt_clean, rng)

        deg = degrade(gt_def, rng=np.random.default_rng(50_000 + i))
        pred = restore(model, deg, args.device)

        # Defect contrast in the truth vs in the restoration, measured at GT
        # resolution (where the mask is defined).
        c_true = contrast(gt_def, mask)
        c_pred = contrast(pred, mask)
        if abs(c_true) > 1e-6:
            rec = c_pred / c_true
            recoveries.append(rec)
            by_kind.setdefault(dkind, []).append(rec)

        if i < args.save_panels:
            deg_v = cv2.resize(np.clip(deg, 0, 255).astype(np.uint8),
                               (args.size, args.size), interpolation=cv2.INTER_NEAREST)
            outline = (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), 1)
                       - mask.astype(np.uint8)).astype(bool)
            panel = []
            for img, name in ((gt_def, "GT+defect"), (deg_v, "degraded"),
                              (pred, "restored"), (gt_clean, "GT no defect")):
                v = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                v[outline] = (0, 0, 255)                  # mark the defect site
                cv2.rectangle(v, (0, 0), (v.shape[1], 22), (0, 0, 0), -1)
                cv2.putText(v, f"{name} [{dkind}]", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                panel.append(v)
            pfx = f"{args.tag}_" if args.tag else ""
            cv2.imwrite(str(OUT_DIR / f"{pfx}defect_{i:03d}_{dkind}.png"),
                        np.concatenate(panel, axis=1))

    # False structure: on CLEAN images, how much detail does the model invent?
    false_struct = []
    for i in range(args.n):
        kind_gt = str(rng.choice(kinds))
        gt_clean = make_ground_truth(size=args.size, kind=kind_gt, seed=70_000 + i)
        deg = degrade(gt_clean, rng=np.random.default_rng(80_000 + i))
        pred = restore(model, deg, args.device)
        hp_pred = cv2.Laplacian(pred, cv2.CV_32F)
        hp_gt = cv2.Laplacian(gt_clean, cv2.CV_32F)
        denom = np.abs(hp_gt).mean() + 1e-8
        false_struct.append(
            float(np.clip(np.abs(hp_pred) - np.abs(hp_gt), 0, None).mean() / denom))

    rec = np.array(recoveries)
    lines = [
        "Defect preservation & hallucination audit",
        "=" * 46,
        f"images per condition : {args.n}",
        f"GT families audited  : {', '.join(kinds)}",
        "",
        "DEFECT CONTRAST RECOVERY  (1.0 = faithful, 0.0 = defect erased)",
        f"  mean               : {rec.mean():.3f}",
        f"  median             : {np.median(rec):.3f}",
        f"  10th percentile    : {np.percentile(rec, 10):.3f}",
        f"  fraction < 0.5     : {(rec < 0.5).mean():.1%}   <- substantially lost",
        f"  fraction < 0.2     : {(rec < 0.2).mean():.1%}   <- effectively erased",
        "",
        "  by defect type:",
    ]
    for k in sorted(by_kind):
        v = np.array(by_kind[k])
        lines.append(f"    {k:9s} n={len(v):3d}  mean {v.mean():6.3f}   <0.5 {(v < 0.5).mean():5.1%}")

    fs = np.array(false_struct)
    lines += [
        "",
        "FALSE STRUCTURE on defect-free images  (0.0 = invents nothing)",
        f"  mean               : {fs.mean():.3f}",
        f"  90th percentile    : {np.percentile(fs, 90):.3f}",
        "",
        f"panels written to    : {OUT_DIR.relative_to(REPO_ROOT)}",
    ]

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "report.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
