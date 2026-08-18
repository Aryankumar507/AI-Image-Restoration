#!/usr/bin/env python3
"""
KLA Hackathon - Evaluation Script (submission component #1)

Standalone, non-notebook script per the mandatory submission format:
  - Accepts: path to test images directory, path to output directory
  - Loads the trained model and runs inference on all input images
  - Writes restored outputs to the specified directory
  - Runs with no manual edits (used as-is for benchmarking on H100)

Usage:
    python evaluate_submission.py --input_dir <test_dir> --output_dir <out_dir>

    # --checkpoint defaults to checkpoints/nafnet_lite.pt (resolved relative
    # to this file), so it need not be passed. Override it with:
    #   --checkpoint /path/to/other.pt

Input format expected: grayscale images (.png/.npy) of the degraded
(noisy, low-res) test samples, at native bit depth (8- or 16-bit).
Output: restored images written as 8-bit .png with the SAME base filename,
upsampled to the ground-truth resolution.

Output resolution: 2x the input by default, per the problem statement
(256->512 and 128->256). Override with --scale if a different ratio is
required; --scale 1 denoises at native resolution without upscaling.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import cv2

from kla_restoration.model import NAFNetLiteSR, apply_tlc, self_ensemble

# Repo root, so the default checkpoint resolves no matter what the caller's
# working directory is.
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "nafnet_lite.pt"


def load_image(path):
    """Load a degraded image at its NATIVE bit depth.

    Returns (array_float32, scale) where `scale` is the divisor that maps the
    stored integer range onto the [0,1] convention the model was trained with
    (see PairDataset in scripts/train.py, which normalizes by 255).

    Two subtleties, both of which silently wreck accuracy if got wrong:

      * cv2.IMREAD_GRAYSCALE force-converts to 8-bit, discarding most of the
        dynamic range of 16-bit inspection imagery. IMREAD_UNCHANGED preserves
        it, so we use that and reduce to single channel ourselves.
      * Speckle noise legally pushes values ABOVE the ground-truth range (the
        problem statement calls this out explicitly). We must NOT clip here --
        those >1.0 values are exactly what the model was trained to see.
    """
    if path.suffix.lower() == ".npy":
        # .npy is already float in the training convention (nominally [0,255],
        # may exceed it); no bit-depth inference applies.
        return np.load(path).astype(np.float32), 255.0

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"could not read image: {path}")

    # Drop any alpha channel, then reduce colour to single channel. The spec
    # says inputs are grayscale, so a 3-channel file is a storage artifact.
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    # Map the stored integer range onto the training scale.
    if arr.dtype == np.uint16:
        scale = 65535.0
    elif arr.dtype == np.uint8:
        scale = 255.0
    else:
        # float TIFF/EXR: assume already in the training convention.
        scale = 255.0

    return arr.astype(np.float32), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Path to directory of degraded test images")
    ap.add_argument("--output_dir", required=True, help="Path to directory to write restored images")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--scale", type=float, default=2.0,
        help="Output size relative to the input (default: 2, per the problem "
             "statement: 256->512 and 128->256). The network's SR head is "
             "natively 2x; any other value is applied by resampling its output.")
    ap.add_argument(
        "--tlc", type=int, default=0, metavar="WINDOW",
        help="Test-time Local Converter window (0 = off, the default). Replaces "
             "global LayerNorm statistics with a local window matching the "
             "training patch size. Measured neutral on this model (21.73 vs "
             "21.71 dB at w=64) because the U-Net downsamples twice, so deep "
             "blocks already see small feature maps; kept for larger inputs "
             "where the mismatch grows.")
    ap.add_argument(
        "--self_ensemble", action="store_true",
        help="x8 geometric self-ensemble: restore under all 8 symmetries of "
             "the square and average. Measured +0.39 dB PSNR / +0.013 SSIM at "
             "8x the inference cost -- a quality/speed dial, still milliseconds "
             "on an H100.")
    ap.add_argument(
        "--batch_size", type=int, default=1,
        help="Images per forward pass (default: 1). Measured fastest at 1 on a "
             "4 GB RTX 2050 (23 ms/image vs 202 ms at 8) because batched "
             "activations spill VRAM. Raise on high-VRAM hardware (H100).")
    args = ap.parse_args()

    if args.batch_size < 1:
        ap.error(f"--batch_size must be >= 1, got {args.batch_size}")
    if args.scale <= 0:
        ap.error(f"--scale must be positive, got {args.scale}")

    t_start = time.time()

    # weights_only=True is the default from PyTorch 2.6 onward and is required
    # for forward compatibility; our checkpoint holds only tensors and plain
    # Python scalars, so it loads cleanly under the restricted unpickler.
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    # Architecture comes from the checkpoint itself, so scaling the model up
    # never desynchronizes training from inference.
    model = NAFNetLiteSR.from_checkpoint(ckpt)
    if args.tlc:
        apply_tlc(model, window=args.tlc)
    model.to(args.device).eval()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".npy")])
    print(f"Found {len(files)} input images. Model+init time: {time.time()-t_start:.2f}s")

    # ---------------------------------------------------------------- speed --
    # Inference time is an explicit scoring criterion (defect W-05). Measured
    # on an RTX 2050 (4 GB), 40 images of 128x128:
    #
    #     batch_size=1  ->  23.0 ms/image     <- default
    #     batch_size=8  -> 201.5 ms/image
    #     batch_size=32 -> 471.7 ms/image
    #
    # Batching is SLOWER here: activations for a batch exceed free VRAM and
    # spill over PCIe. Larger batches may pay off on a high-VRAM card such as
    # the H100, so --batch_size remains available -- but the default is the
    # value that is measurably fastest on hardware we could actually test.
    #
    # Retained because they help unconditionally: channels_last + autocast
    # engage tensor cores, and a warmup pass keeps CUDA context and cuDNN
    # autotuning cost out of the measured window.
    use_cuda = torch.device(args.device).type == "cuda"
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def run_batch(stack):
        """Forward one batch (N,H,W), returning float32 predictions in [0,1]."""
        t = torch.from_numpy(stack).unsqueeze(1).to(args.device, non_blocking=True)
        if use_cuda:
            t = t.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=amp_dtype):
                out = self_ensemble(model, t) if args.self_ensemble else model(t)
        else:
            out = self_ensemble(model, t) if args.self_ensemble else model(t)
        return out.float().clamp(0, 1).squeeze(1).cpu().numpy()

    loaded = [(f,) + load_image(f) for f in files]

    # Warmup so context/autotune cost is not charged to the measured window.
    if use_cuda and loaded:
        _, warm_arr, warm_scale = loaded[0]
        run_batch((warm_arr / warm_scale).astype(np.float32)[None])
        torch.cuda.synchronize()

    t_infer0 = time.time()
    for i in range(0, len(loaded), args.batch_size):
        chunk = loaded[i:i + args.batch_size]
        # Only equal-sized images can share a batch; fall back to one-by-one
        # when a chunk is ragged.
        shapes = {arr.shape for _, arr, _ in chunk}
        groups = [chunk] if len(shapes) == 1 else [[c] for c in chunk]

        for group in groups:
            stack = np.stack([(arr / sc).astype(np.float32) for _, arr, sc in group])
            preds = run_batch(stack)

            for (f, arr, _), restored in zip(group, preds):
                # The SR head is natively 2x. For any other requested scale,
                # resample from the network's output to the target size.
                if args.scale != 2.0:
                    src_h, src_w = arr.shape[:2]
                    target = (int(round(src_w * args.scale)),
                              int(round(src_h * args.scale)))
                    if (restored.shape[1], restored.shape[0]) != target:
                        interp = cv2.INTER_AREA if args.scale < 2.0 else cv2.INTER_CUBIC
                        restored = cv2.resize(restored, target, interpolation=interp)

                restored_img = (restored * 255.0).clip(0, 255).astype(np.uint8)
                # Preserve the input stem EXACTLY. Scorers commonly pair
                # restored outputs to ground truth by filename stem, so any
                # suffix here risks every file failing to match.
                cv2.imwrite(str(out_dir / (f.stem + ".png")), restored_img)

    if use_cuda:
        torch.cuda.synchronize()
    t_infer1 = time.time()

    n = max(len(files), 1)
    print(f"Inference done: {len(files)} images in {t_infer1-t_infer0:.2f}s "
          f"({(t_infer1-t_infer0)/n*1000:.1f} ms/image)")
    print(f"Total end-to-end time (incl. model load): {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    main()
