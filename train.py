"""
Training script for the KLA restoration model (reproduces submitted model).

Loss: L1 (pixel fidelity, robust to outlier speckle pixels vs L2) + a
differentiable SSIM term (structural similarity -- directly optimizes one of
KLA's three grading metrics), following the "combine perceptual loss with
pixel-level metrics" tip in the deck. We skip LPIPS during training (it
needs a pretrained VGG backbone and would dominate compute on CPU); we still
report LPIPS at evaluation time as a held-out perceptual metric.

Usage:
    python3 train.py --epochs 8 --data_dir data/train --val_dir data/val
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import NAFNetLiteSR


# ---------------------------------------------------------------- dataset --
class PairDataset(Dataset):
    """Loads (degraded, gt) .npy pairs produced by data_generator.py and
    extracts random square patches for training (keeps CPU training fast)."""

    def __init__(self, root, patch_size=64, train=True):
        self.gt_dir = Path(root) / "gt"
        self.deg_dir = Path(root) / "degraded"
        self.files = sorted(self.gt_dir.glob("*.npy"))
        self.patch_size = patch_size
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        gt_path = self.files[idx]
        deg_path = self.deg_dir / gt_path.name
        gt = np.load(gt_path)          # HxW, full res, [0,255]-ish
        deg = np.load(deg_path)        # (H/2)x(W/2), may exceed [0,255]

        ps = self.patch_size
        dh, dw = deg.shape
        if self.train:
            y = np.random.randint(0, dh - ps + 1)
            x = np.random.randint(0, dw - ps + 1)
        else:
            y = (dh - ps) // 2
            x = (dw - ps) // 2
        deg_patch = deg[y:y + ps, x:x + ps]
        gt_patch = gt[2 * y:2 * (y + ps), 2 * x:2 * (x + ps)]

        # normalize GT to [0,1]; degraded is normalized by the SAME constant
        # (255) so that the "exceeds ground truth range" property from the
        # spec is preserved as values >1.0 rather than being clipped away
        deg_t = torch.from_numpy(deg_patch / 255.0).float().unsqueeze(0)
        gt_t = torch.from_numpy(gt_patch / 255.0).float().unsqueeze(0)
        return deg_t, gt_t


# -------------------------------------------------------------- SSIM loss --
def gaussian_window(size, sigma, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    return window


def ssim(img1, img2, window_size=11, sigma=1.5):
    device = img1.device
    window = gaussian_window(window_size, sigma, device).unsqueeze(0).unsqueeze(0)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad)
    mu2 = F.conv2d(img2, window, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def combined_loss(pred, target, alpha=0.84):
    l1 = F.l1_loss(pred, target)
    s = 1 - ssim(pred.clamp(0, 1), target.clamp(0, 1))
    return alpha * s + (1 - alpha) * l1, l1.item(), s.item()


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/train")
    ap.add_argument("--val_dir", default="data/val")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--out", default="checkpoints/nafnet_lite.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    train_ds = PairDataset(args.data_dir, patch_size=args.patch_size, train=True)
    val_ds = PairDataset(args.val_dir, patch_size=args.patch_size, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = NAFNetLiteSR(width=args.width, enc_blocks=(1, 1), dec_blocks=(1, 1), middle_blocks=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e3:.1f}K | device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))

    best_val = float("inf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tr_loss, tr_l1, tr_ssim = 0.0, 0.0, 0.0
        for deg, gt in train_dl:
            deg, gt = deg.to(device), gt.to(device)
            pred = model(deg)
            loss, l1v, ssimv = combined_loss(pred, gt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tr_loss += loss.item()
            tr_l1 += l1v
            tr_ssim += ssimv
        n = len(train_dl)
        tr_loss, tr_l1, tr_ssim = tr_loss / n, tr_l1 / n, tr_ssim / n

        model.eval()
        val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
        with torch.no_grad():
            for deg, gt in val_dl:
                deg, gt = deg.to(device), gt.to(device)
                pred = model(deg).clamp(0, 1)
                loss, _, _ = combined_loss(pred, gt)
                val_loss += loss.item()
                mse = F.mse_loss(pred, gt).item()
                val_psnr += 10 * np.log10(1.0 / max(mse, 1e-10))
                val_ssim += ssim(pred, gt).item()
        nv = len(val_dl)
        val_loss, val_psnr, val_ssim = val_loss / nv, val_psnr / nv, val_ssim / nv

        dt = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} | train_loss {tr_loss:.4f} (l1 {tr_l1:.4f}, 1-ssim {tr_ssim:.4f}) "
              f"| val_loss {val_loss:.4f} val_psnr {val_psnr:.2f}dB val_ssim {val_ssim:.4f} | {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "width": args.width}, args.out)

    print(f"Best val_loss {best_val:.4f}, checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
