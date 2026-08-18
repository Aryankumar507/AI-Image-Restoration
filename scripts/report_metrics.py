"""
Full-image evaluation + visual report on the held-out validation set.
Compares the trained NAFNet-Lite-SR model against a naive bicubic upsampling
baseline (no denoising at all) to quantify the value the learned model adds.

Reports all three metrics KLA grades on: PSNR, SSIM and LPIPS. LPIPS uses the
real pretrained backbone (`lpips.LPIPS(net='alex')`), which downloads weights
on first use. If that download is unavailable, the script falls back to an
edge-correlation proxy and labels it clearly rather than silently reporting a
different metric under the same name (defect W-06).
"""
from pathlib import Path

import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

from kla_restoration.model import NAFNetLiteSR

# Repo root (scripts/ lives one level below it), so all default paths resolve
# regardless of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "checkpoints" / "nafnet_lite.pt"
VAL_DIR = REPO_ROOT / "data" / "val"
SAMPLES_DIR = REPO_ROOT / "outputs" / "samples"
METRICS_REPORT = REPO_ROOT / "outputs" / "metrics_report.txt"


_LPIPS_MODEL = None
_LPIPS_UNAVAILABLE = None


def _get_lpips():
    """Lazily construct the real LPIPS model, or record why we cannot.

    Returns the model, or None if the pretrained backbone is unreachable.
    """
    global _LPIPS_MODEL, _LPIPS_UNAVAILABLE
    if _LPIPS_MODEL is not None or _LPIPS_UNAVAILABLE is not None:
        return _LPIPS_MODEL
    try:
        import lpips
        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False)
        _LPIPS_MODEL.eval()
    except Exception as exc:                      # noqa: BLE001 - report any cause
        _LPIPS_UNAVAILABLE = str(exc)
        _LPIPS_MODEL = None
    return _LPIPS_MODEL


def lpips_metric(a, b):
    """True LPIPS between two [0,255] grayscale float arrays. Lower is better.

    LPIPS expects 3-channel input in [-1,1], so the grayscale image is
    replicated across channels and rescaled.
    """
    model = _get_lpips()
    if model is None:
        return None

    def _prep(x):
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32) / 255.0)
        t = t.clamp(0, 1) * 2 - 1                 # [0,255] -> [-1,1]
        return t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    with torch.no_grad():
        return float(model(_prep(a), _prep(b)).item())


def edge_proxy_metric(a, b):
    """Edge-correlation stand-in used only when real LPIPS is unavailable.

    NOT LPIPS -- reported under its own name so the two are never confused.
    """
    ga = cv2.Laplacian(a, cv2.CV_32F)
    gb = cv2.Laplacian(b, cv2.CV_32F)
    num = (ga * gb).sum()
    den = (np.linalg.norm(ga) * np.linalg.norm(gb) + 1e-8)
    return 1 - (num / den)  # lower is better, like LPIPS


def perceptual_metric(a, b):
    """Real LPIPS when available, else the clearly-labelled edge proxy."""
    val = lpips_metric(a, b)
    return val if val is not None else edge_proxy_metric(a, b)


def bicubic_baseline(deg):
    h, w = deg.shape
    up = cv2.resize(deg, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return np.clip(up, 0, 255)


def main():
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = NAFNetLiteSR.from_checkpoint(ckpt)
    model.eval()

    val_dir = VAL_DIR
    gt_files = sorted((val_dir / "gt").glob("*.npy"))

    rows = []
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

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
                cv2.imwrite(str(SAMPLES_DIR / f"{gt_path.stem}_deg_bicubic_ours_gt.png"), panel)

    rows = np.array([r[1:] for r in rows], dtype=np.float32)
    means = rows.mean(axis=0)

    # Name the metric that actually ran, so a proxy is never mistaken for LPIPS.
    if _LPIPS_MODEL is not None:
        perc_label = "LPIPS (lower better)"
    else:
        perc_label = "Edge-proxy, NOT LPIPS (lower)"

    lines = [
        f"{'metric':30s} {'ours':>10s} {'bicubic-only':>14s}",
        f"{'PSNR (dB, higher better)':30s} {means[0]:10.2f} {means[3]:14.2f}",
        f"{'SSIM (higher better)':30s} {means[1]:10.4f} {means[4]:14.4f}",
        f"{perc_label:30s} {means[2]:10.4f} {means[5]:14.4f}",
        f"\nImages evaluated: {len(rows)}",
    ]
    if _LPIPS_MODEL is None:
        lines.append(
            "WARNING: real LPIPS unavailable "
            f"({_LPIPS_UNAVAILABLE}); reported an edge-correlation proxy instead."
        )

    report = "\n".join(lines)
    print(report)
    METRICS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_REPORT, "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
