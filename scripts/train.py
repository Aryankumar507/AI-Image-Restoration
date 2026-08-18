"""
Training script for the KLA restoration model (reproduces submitted model).

Loss: L1 (pixel fidelity, robust to outlier speckle pixels vs L2) + a
differentiable SSIM term (structural similarity -- directly optimizes one of
KLA's three grading metrics), following the "combine perceptual loss with
pixel-level metrics" tip in the deck. We skip LPIPS during training (it
needs a pretrained VGG backbone and would dominate compute on CPU); we still
report LPIPS at evaluation time as a held-out perceptual metric.

Usage (from the repo root, after `pip install -e .`):
    python scripts/train.py --epochs 8

    # data_dir/val_dir default to <repo>/data/{train,val} and the checkpoint
    # to <repo>/checkpoints/nafnet_lite.pt; override with --data_dir etc.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from kla_restoration.model import NAFNetLiteSR

# Repo root (scripts/ lives one level below it), so default data and output
# paths resolve regardless of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- dataset --
def augment_pair(deg, gt, rng):
    """Apply the same random dihedral transform to a (degraded, gt) pair.

    The 8 symmetries of the square (4 rotations x optional flip). Both arrays
    MUST receive the identical transform, or the spatial correspondence the
    model learns from is silently destroyed.

    Semiconductor structures are strongly symmetric, so this is close to free
    accuracy on a small training set (defect W-03).
    """
    k = int(rng.integers(0, 4))          # number of 90-degree rotations
    if k:
        deg = np.rot90(deg, k)
        gt = np.rot90(gt, k)
    if rng.random() < 0.5:               # horizontal flip
        deg = np.fliplr(deg)
        gt = np.fliplr(gt)
    if rng.random() < 0.5:               # vertical flip
        deg = np.flipud(deg)
        gt = np.flipud(gt)
    # rot90/flip return views with negative strides; torch needs contiguous.
    return np.ascontiguousarray(deg), np.ascontiguousarray(gt)


class PairDataset(Dataset):
    """Loads (degraded, gt) .npy pairs produced by data_generator.py and
    extracts random square patches for training (keeps CPU training fast).

    When train=True, patches are randomly cropped and augmented with the
    dihedral group; when False, a deterministic centre crop is used so
    validation numbers are comparable across epochs.
    """

    def __init__(self, root, patch_size=64, train=True, augment=True, seed=0):
        self.gt_dir = Path(root) / "gt"
        self.deg_dir = Path(root) / "degraded"
        self.files = sorted(self.gt_dir.glob("*.npy"))
        self.patch_size = patch_size
        self.train = train
        self.augment = augment and train
        self._rng = np.random.default_rng(seed)

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
            y = int(self._rng.integers(0, dh - ps + 1))
            x = int(self._rng.integers(0, dw - ps + 1))
        else:
            y = (dh - ps) // 2
            x = (dw - ps) // 2
        deg_patch = deg[y:y + ps, x:x + ps]
        gt_patch = gt[2 * y:2 * (y + ps), 2 * x:2 * (x + ps)]

        if self.augment:
            deg_patch, gt_patch = augment_pair(deg_patch, gt_patch, self._rng)

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


def anomaly_weight_map(pred, target, gain=8.0, eps=1e-4):
    """Per-pixel loss weight that emphasises where the model is WRONG.

    Motivation (docs/ASSESSMENT.md): an L1+SSIM objective averaged over a whole
    image is nearly indifferent to a 5-pixel defect -- its gradient share is
    swamped. Adding defects to the training data lifted recovery only 0.15 ->
    0.30 for exactly this reason: the model saw anomalies but was never made to
    care about them.

        w = 1 + gain * normalized(|pred - target|)      [detached]

    WHY THE RESIDUAL, NOT THE TARGET'S OWN TEXTURE. The obvious formulation
    weights by local deviation of the target, |target - blur(target)|, on the
    theory that a defect is a local deviation from its surroundings. Measured,
    that fails: on a periodic grid the structure's own edges are as
    locally-deviant as the defect, so after per-image normalization the defect
    gets a weight ratio of 1.00x -- no emphasis whatsoever. The residual
    formulation reaches ~1500x on the same test, across all five GT families,
    because an erased defect is precisely a large prediction error.

    The weight is detached, so this reweights gradients without letting the
    model reduce the loss by manipulating the weights themselves.
    """
    with torch.no_grad():
        r = (pred - target).abs()
        # Per-image normalization: (N,1,1,1) so each sample scales by its own
        # statistics rather than by whatever else shares the batch.
        w = 1.0 + gain * (r / (r.mean(dim=(1, 2, 3), keepdim=True) + eps))
    return w


def fft_magnitude_loss(pred, target):
    """L1 between the FFT magnitude spectra of prediction and target (IDEA E).

    Semiconductor structures are strongly PERIODIC -- line/space gratings,
    contact-hole arrays, via grids. What a metrology engineer measures on such
    a structure is its PITCH, which lives in a handful of Fourier coefficients.
    A spatial-domain L1 can be small while the reconstructed pitch is subtly
    wrong, because a slight period error costs little per pixel but moves the
    spectral peak.

    Matching magnitude (not phase) keeps this a texture/periodicity constraint
    rather than a second copy of the pixel loss: phase carries position, which
    L1 already supervises.

    rfft2 is used since the input is real, halving the compute.
    """
    pf = torch.fft.rfft2(pred, norm="ortho")
    tf = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(pf.abs(), tf.abs())


def combined_loss(pred, target, alpha=0.5, defect_gain=0.0, fft_weight=0.0):
    """Weighted structural + pixel loss.

    alpha balances the two GRADED metrics: the SSIM term optimizes SSIM, the
    L1 term optimizes PSNR. The original 0.84 came from papers using MS-SSIM
    and under-weighted pixel fidelity; 0.5 treats both as equally graded.

    L1 rather than L2 because speckle is multiplicative and heavy-tailed, so
    squared error would let a few outlier pixels dominate the gradient.

    defect_gain > 0 enables the anomaly-weighted pixel term: the L1 residual is
    reweighted by anomaly_weight_map() so that badly-predicted pixels -- which
    is what an erased defect looks like -- contribute proportionally more
    gradient. This is a focal-loss-style reweighting for regression. The SSIM
    term is left unweighted: it is a windowed statistic, so per-pixel
    reweighting is not well-defined for it.

    NOTE on the absent clamp: an earlier version computed SSIM on
    `pred.clamp(0, 1)`. clamp() has ZERO gradient outside its range, so any
    pixel the model pushed past 1.0 received no structural gradient at all --
    precisely the out-of-range pixels speckle produces, which the problem
    statement calls out as expected. SSIM is well-defined for out-of-range
    inputs, so we simply do not clamp during training. Clamping remains
    correct at inference, where there are no gradients to kill.
    """
    if defect_gain > 0:
        w = anomaly_weight_map(pred, target, gain=defect_gain)
        l1 = ((pred - target).abs() * w).sum() / w.sum()
    else:
        l1 = F.l1_loss(pred, target)
    s = 1 - ssim(pred, target)
    total = alpha * s + (1 - alpha) * l1
    if fft_weight > 0:
        total = total + fft_weight * fft_magnitude_loss(pred, target)
    return total, l1.item(), s.item()


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "train"))
    ap.add_argument("--val_dir", default=str(REPO_ROOT / "data" / "val"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--code_dim", type=int, default=0,
                    help="FiLM degradation-code width (IDEA A). 0 disables. "
                         "Conditions every NAFBlock on estimated degradation "
                         "severity, for out-of-distribution robustness. Try 16.")
    ap.add_argument("--log_branch", type=int, default=0,
                    help="Log-domain speckle branch channels (IDEA B). 0 disables. "
                         "Speckle is multiplicative; log1p makes it additive, "
                         "which conv nets denoise far better. Try 8.")
    ap.add_argument("--fft_weight", type=float, default=0.0,
                    help="Weight on the FFT-magnitude loss (IDEA E). Semiconductor "
                         "structures are periodic, so matching the spectrum "
                         "constrains reconstructed PITCH. Try 0.05-0.2.")
    ap.add_argument("--defect_gain", type=float, default=0.0,
                    help="Anomaly-weighted L1 gain. 0 disables (plain L1). "
                         "Higher values make locally-anomalous pixels -- i.e. "
                         "defects -- contribute more gradient. Try 4-12.")
    ap.add_argument("--no_augment", action="store_true",
                    help="Disable dihedral (flip/rot90) augmentation")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Loss weight: alpha*(1-SSIM) + (1-alpha)*L1. Higher "
                         "favours SSIM, lower favours PSNR (default: 0.5)")
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--enc_blocks", type=int, nargs="+", default=[1, 1],
                    help="NAFBlocks per encoder stage (also sets U-Net depth)")
    ap.add_argument("--dec_blocks", type=int, nargs="+", default=[1, 1],
                    help="NAFBlocks per decoder stage (must match --enc_blocks length)")
    ap.add_argument("--middle_blocks", type=int, default=1,
                    help="NAFBlocks in the bottleneck")
    ap.add_argument("--out", default=str(REPO_ROOT / "checkpoints" / "nafnet_lite.pt"))
    args = ap.parse_args()

    if len(args.enc_blocks) != len(args.dec_blocks):
        ap.error("--enc_blocks and --dec_blocks must have the same length "
                 f"(got {len(args.enc_blocks)} and {len(args.dec_blocks)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    train_ds = PairDataset(args.data_dir, patch_size=args.patch_size, train=True,
                           augment=not args.no_augment)
    val_ds = PairDataset(args.val_dir, patch_size=args.patch_size, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = NAFNetLiteSR(
        width=args.width,
        enc_blocks=tuple(args.enc_blocks),
        dec_blocks=tuple(args.dec_blocks),
        middle_blocks=args.middle_blocks,
        code_dim=args.code_dim,
        log_branch=args.log_branch,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e3:.1f}K | device={device} | config={model.config}")

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
            loss, l1v, ssimv = combined_loss(pred, gt, alpha=args.alpha,
                                             defect_gain=args.defect_gain,
                                             fft_weight=args.fft_weight)
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
                # Clamp here (unlike training): validation mirrors inference,
                # where outputs are clamped before being written, and there
                # are no gradients to kill.
                pred = model(deg).clamp(0, 1)
                loss, _, _ = combined_loss(pred, gt, alpha=args.alpha)  # unweighted: comparable across runs
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
            # Persist the full architecture alongside the weights so inference
            # never has to guess it. "width" is kept for backward compat.
            torch.save({
                "model": model.state_dict(),
                "config": model.config,
                "width": args.width,
            }, args.out)

    print(f"Best val_loss {best_val:.4f}, checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
