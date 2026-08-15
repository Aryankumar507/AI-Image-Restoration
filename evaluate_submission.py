#!/usr/bin/env python3
"""
KLA Hackathon - Evaluation Script (submission component #1)

Standalone, non-notebook script per the mandatory submission format:
  - Accepts: path to test images directory, path to output directory
  - Loads the trained model and runs inference on all input images
  - Writes restored outputs to the specified directory
  - Runs with no manual edits (used as-is for benchmarking on H100)

Usage:
    python3 evaluate_submission.py --input_dir <test_dir> --output_dir <out_dir> \
        --checkpoint checkpoints/nafnet_lite.pt

Input format expected: grayscale images (.png/.npy) of the degraded
(noisy, low-res) test samples. Output: restored images written as .png,
same base filename, upsampled to the ground-truth resolution.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import cv2

from model import NAFNetLiteSR


def load_image(path):
    if path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Path to directory of degraded test images")
    ap.add_argument("--output_dir", required=True, help="Path to directory to write restored images")
    ap.add_argument("--checkpoint", default="checkpoints/nafnet_lite.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t_start = time.time()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = NAFNetLiteSR(width=ckpt["width"], enc_blocks=(1, 1), dec_blocks=(1, 1), middle_blocks=1)
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".npy")])
    print(f"Found {len(files)} input images. Model+init time: {time.time()-t_start:.2f}s")

    t_infer0 = time.time()
    with torch.no_grad():
        for f in files:
            deg = load_image(f)
            deg_t = torch.from_numpy(deg / 255.0).float().unsqueeze(0).unsqueeze(0).to(args.device)
            restored = model(deg_t).clamp(0, 1).squeeze().cpu().numpy()
            restored_img = (restored * 255.0).clip(0, 255).astype(np.uint8)
            out_path = out_dir / (f.stem + "_restored.png")
            cv2.imwrite(str(out_path), restored_img)
    t_infer1 = time.time()

    n = max(len(files), 1)
    print(f"Inference done: {len(files)} images in {t_infer1-t_infer0:.2f}s "
          f"({(t_infer1-t_infer0)/n*1000:.1f} ms/image)")
    print(f"Total end-to-end time (incl. model load): {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    main()
