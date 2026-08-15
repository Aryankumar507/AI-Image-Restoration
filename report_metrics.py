"""
Full-image evaluation + visual report on the held-out synthetic validation
set. Compares our trained NAFNet-Lite-SR model against a naive bicubic
upsampling baseline (no denoising at all) to quantify the value the learned
model adds.

Note on LPIPS: the pretrained AlexNet/VGG backbone LPIPS needs is hosted on
download.pytorch.org, which is not reachable from this sandboxed environment
(network is allowlisted to pypi/npm/github only). We therefore report a
lightweight *edge-correlation* perceptual proxy (gradient-map cosine
similarity) in its place and flag this explicitly -- in a real submission
environment with unrestricted network access, swap in real `lpips.LPIPS()`
(the training/eval code already isolates this into `perceptual_metric()`
below, one function to replace).
"""
from pathlib import Path

import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

from model import NAFNetLiteSR


def perceptual_metric(a, b):
    """Edge-correlation proxy for LPIPS (see module docstring)."""
    ga = cv2.Laplacian(a, cv2.CV_32F)
    gb = cv2.Laplacian(b, cv2.CV_32F)
    num = (ga * gb).sum()
    den = (np.linalg.norm(ga) * np.linalg.norm(gb) + 1e-8)
    return 1 - (num / den)  # lower is better, like LPIPS


def bicubic_baseline(deg):
    h, w = deg.shape
    up = cv2.resize(deg, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return np.clip(up, 0, 255)


def main():
    ckpt = torch.load("checkpoints/nafnet_lite.pt", map_location="cpu")
    model = NAFNetLiteSR(width=ckpt["width"], enc_blocks=(1, 1), dec_blocks=(1, 1), middle_blocks=1)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_dir = Path("data/val")
    gt_files = sorted((val_dir / "gt").glob("*.npy"))

    rows = []
    Path("outputs/samples").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, gt_path in enumerate(gt_files):
            deg_path = val_dir / "degraded" / gt_path.name
            gt = np.load(gt_path).astype(np.float32)
            deg = np.load(deg_path).astype(np.float32)

            deg_t = torch.from_numpy(deg / 255.0).float().unsqueeze(0).unsqueeze(0)
            pred = model(deg_t).clamp(0, 1).squeeze().numpy() * 255.0

            bicubic = bicubic_baseline(deg)

            gt_n = gt / 255.0
            pred_n = pred / 255.0
            bic_n = np.clip(bicubic, 0, 255) / 255.0

            model_psnr = sk_psnr(gt_n, pred_n, data_range=1.0)
            model_ssim = sk_ssim(gt_n, pred_n, data_range=1.0)
            model_perc = perceptual_metric(gt, pred)

            bic_psnr = sk_psnr(gt_n, bic_n, data_range=1.0)
            bic_ssim = sk_ssim(gt_n, bic_n, data_range=1.0)
            bic_perc = perceptual_metric(gt, bicubic)

            rows.append((gt_path.stem, model_psnr, model_ssim, model_perc, bic_psnr, bic_ssim, bic_perc))

            if i < 6:
                # save side-by-side comparison
                deg_vis = np.clip(deg, 0, 255).astype(np.uint8)
                deg_vis = cv2.resize(deg_vis, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                panel = np.concatenate([
                    deg_vis,
                    bicubic.astype(np.uint8),
                    pred.astype(np.uint8),
                    gt.astype(np.uint8),
                ], axis=1)
                cv2.imwrite(f"outputs/samples/{gt_path.stem}_deg_bicubic_ours_gt.png", panel)

    rows = np.array([r[1:] for r in rows], dtype=np.float32)
    names = ["model_psnr", "model_ssim", "model_perc(lower better)", "bicubic_psnr", "bicubic_ssim", "bicubic_perc(lower better)"]
    means = rows.mean(axis=0)

    print(f"{'metric':30s} {'ours':>10s} {'bicubic-only':>14s}")
    print(f"{'PSNR (dB, higher better)':30s} {means[0]:10.2f} {means[3]:14.2f}")
    print(f"{'SSIM (higher better)':30s} {means[1]:10.4f} {means[4]:14.4f}")
    print(f"{'Edge-perceptual (lower better)':30s} {means[2]:10.4f} {means[5]:14.4f}")

    with open("outputs/metrics_report.txt", "w") as f:
        f.write(f"{'metric':30s} {'ours':>10s} {'bicubic-only':>14s}\n")
        f.write(f"{'PSNR (dB, higher better)':30s} {means[0]:10.2f} {means[3]:14.2f}\n")
        f.write(f"{'SSIM (higher better)':30s} {means[1]:10.4f} {means[4]:14.4f}\n")
        f.write(f"{'Edge-perceptual (lower better)':30s} {means[2]:10.4f} {means[5]:14.4f}\n")


if __name__ == "__main__":
    main()
