#!/usr/bin/env python
"""Ablation over the four architectural/loss ideas (Slide 5 evidence).

Each variant is trained on identical data with identical hyperparameters, so
the only difference is the idea under test:

    baseline   A+B training data, plain NAFNet
    E          + Fourier magnitude loss           (periodic pitch fidelity)
    B          + log-domain speckle branch        (multiplicative -> additive)
    A          + FiLM degradation conditioning    (OOD adaptivity)
    ABE        all three
    +TTA       x8 self-ensemble, applied to the best variant at inference

Evaluated on 5 GT families with defects -- the set that resembles real
inspection data, NOT the narrow distribution any single variant trained on.

Usage (from the repo root, after `pip install -e .`):
    python scripts/idea_ablation.py --n 60
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from kla_restoration.data_generator import KINDS, degrade, make_ground_truth
from kla_restoration.model import NAFNetLiteSR, self_ensemble

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_eval_set(n, size, seed=4242):
    """Fixed evaluation set: all 5 families, defects present."""
    data, rng = [], np.random.default_rng(seed)
    for i in range(n):
        kind = str(rng.choice(KINDS))
        gt = make_ground_truth(size=size, kind=kind, seed=200_000 + i, defect_prob=0.5)
        deg = degrade(gt, rng=np.random.default_rng(300_000 + i))
        data.append((gt / 255.0, deg))
    return data


def score(ckpt_path, data, device, tta=False, lpips_fn=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = NAFNetLiteSR.from_checkpoint(ckpt).to(device).eval()

    ps, ss, lp = [], [], []
    for gtn, deg in data:
        t = torch.from_numpy(deg / 255.0).float()[None, None].to(device)
        with torch.no_grad():
            out = self_ensemble(model, t) if tta else model(t)
        p = out.clamp(0, 1).squeeze().cpu().numpy()
        ps.append(psnr(gtn, p, data_range=1.0))
        ss.append(ssim(gtn, p, data_range=1.0))
        if lpips_fn is not None:
            a = torch.from_numpy(gtn).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
            b = torch.from_numpy(p).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
            with torch.no_grad():
                lp.append(float(lpips_fn(a, b)))

    n_params = sum(v.numel() for v in ckpt["model"].values())
    return np.mean(ps), np.mean(ss), (np.mean(lp) if lp else float("nan")), n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_lpips", action="store_true")
    args = ap.parse_args()

    lpips_fn = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex", verbose=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"  (LPIPS unavailable: {exc})")

    data = build_eval_set(args.n, args.size)

    variants = [
        ("baseline (A+B data)", "nafnet_lite.pt", False),
        ("+ E  Fourier loss", "idea_E.pt", False),
        ("+ B  log-speckle", "idea_B.pt", False),
        ("+ A  FiLM degrad.", "idea_A.pt", False),
        ("+ A+B+E combined", "idea_ABE.pt", False),
        ("+ A+B+E + x8 TTA", "idea_ABE.pt", True),
    ]

    rows = []
    for label, fname, tta in variants:
        path = REPO_ROOT / "checkpoints" / fname
        if not path.is_file():
            print(f"  (skipping {label}: {fname} not found)")
            continue
        rows.append((label,) + score(path, data, args.device, tta, lpips_fn))

    print()
    print(f"Idea ablation (n={args.n}, 5 GT families + defects)")
    print("=" * 72)
    print(f"{'variant':<24} {'PSNR':>9} {'SSIM':>9} {'LPIPS':>9} {'params':>10}")
    print("-" * 72)
    base = rows[0] if rows else None
    for label, p, s, l, npar in rows:
        d = f"  ({p - base[1]:+.2f})" if base and label != base[0] else ""
        print(f"{label:<24} {p:8.2f}dB {s:9.4f} {l:9.4f} {npar/1e6:9.3f}M{d}")
    print("-" * 72)
    print("Evaluated on all 5 GT families with defects present -- deliberately")
    print("harder and broader than any single variant's training distribution.")

    out = REPO_ROOT / "outputs" / "idea_ablation.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Idea ablation (n={args.n}, 5 GT families + defects)\n\n")
        f.write(f"{'variant':<24} {'PSNR':>9} {'SSIM':>9} {'LPIPS':>9} {'params':>10}\n")
        for label, p, s, l, npar in rows:
            f.write(f"{label:<24} {p:8.2f}dB {s:9.4f} {l:9.4f} {npar/1e6:9.3f}M\n")
    print(f"\nwritten to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
