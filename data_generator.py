"""
Synthetic paired-data generator for the KLA "AI-Based Restoration of Degraded
Images" challenge.

WHY THIS FILE EXISTS
---------------------
KLA's problem statement says the real paired dataset will be provided by KLA.
We do not have access to that dataset in this environment, so to actually
build, train and evaluate an end-to-end solution we synthesize procedural
"semiconductor-like" ground-truth images (periodic line/via grids similar to
the DRAM-style structures shown in the deck, plus grain/dendrite-like organic
textures matching Figure 1/2 in the deck) and apply the EXACT degradation
recipe described in the problem statement:

  1. Ground truth image, 512x512 (or 256x256), grayscale, high SNR.
  2. Degraded image = downsample(GT) -> add Gaussian blur (soft edges) ->
     add speckle (multiplicative) noise whose values may legally EXCEED the
     ground-truth intensity range.
  3. Degraded resolution is exactly 2x lower than GT (512->256, 256->128),
     per the spec.

This lets the rest of the pipeline (model, training, evaluation) be built
and validated end-to-end. When KLA's real dataset is released, only this
file needs to be swapped for a real-data loader -- model/training/eval code
is agnostic to where the pairs come from.
"""
import numpy as np
import cv2
from pathlib import Path


def _make_periodic_grid(size, pitch, line_width=2, seed=None):
    """DRAM/via-like periodic structure (matches 'texture' samples in deck)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 40, dtype=np.float32)
    for y in range(0, size, pitch):
        img[y:y + line_width, :] = 220
    for x in range(0, size, pitch):
        img[:, x:x + line_width] = 220
    # via dots at intersections
    for y in range(0, size, pitch):
        for x in range(0, size, pitch):
            cv2.circle(img, (x, y), max(1, line_width), 255, -1)
    # slight random jitter / defects to avoid perfectly periodic (unrealistic) data
    jitter = rng.normal(0, 3, img.shape).astype(np.float32)
    img = np.clip(img + jitter, 0, 255)
    return img


def _make_dendrite_texture(size, seed=None):
    """Organic branching texture (matches 'dendrite' sample in deck) using
    a simple diffusion-limited-aggregation-style random walk approximation
    (fast fractal noise stand-in, not real DLA, kept lightweight for CPU)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    n_branches = rng.integers(6, 14)
    for _ in range(n_branches):
        x, y = rng.integers(0, size, size=2)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.integers(size // 4, size // 2)
        for _ in range(length):
            angle += rng.normal(0, 0.35)
            x = int(np.clip(x + np.cos(angle) * 2, 0, size - 1))
            y = int(np.clip(y + np.sin(angle) * 2, 0, size - 1))
            cv2.circle(img, (x, y), rng.integers(1, 3), 200, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    base = rng.normal(60, 8, (size, size)).astype(np.float32)
    img = np.clip(base + img, 0, 255)
    return img


def make_ground_truth(size=512, kind="texture", seed=None):
    if kind == "texture":
        pitch = np.random.default_rng(seed).integers(16, 40)
        return _make_periodic_grid(size, pitch, seed=seed)
    else:
        return _make_dendrite_texture(size, seed=seed)


def degrade(gt, scale_factor=2, gaussian_sigma_range=(0.8, 2.0),
            speckle_var_range=(0.02, 0.08), seed=None):
    """Apply the KLA degradation recipe to a ground-truth image.

    gt: float32 HxW array, range [0,255]
    Returns degraded image at H/scale x W/scale, float32, range may exceed
    [0,255] due to speckle noise (this is intentional per the spec).
    """
    rng = np.random.default_rng(seed)
    h, w = gt.shape
    # 1) Gaussian blur -- mimics "Gaussian noise" degradation (softened edges,
    #    loss of fine structure) described in the deck
    sigma = rng.uniform(*gaussian_sigma_range)
    blurred = cv2.GaussianBlur(gt, (0, 0), sigma)

    # 2) Downsample (area interpolation approximates sensor binning)
    lr = cv2.resize(blurred, (w // scale_factor, h // scale_factor),
                     interpolation=cv2.INTER_AREA)

    # 3) Speckle (multiplicative) noise -- independent per-pixel, can push
    #    values beyond the original signal range (explicitly called out as
    #    expected behaviour in the problem statement)
    var = rng.uniform(*speckle_var_range)
    noise = rng.normal(0, np.sqrt(var), lr.shape).astype(np.float32)
    speckled = lr + lr * noise

    return speckled  # NOT clipped -- degraded range may exceed [0,255]


def generate_pair(size=512, scale_factor=2, kind=None, seed=None):
    rng = np.random.default_rng(seed)
    if kind is None:
        kind = rng.choice(["texture", "dendrite"])
    gt = make_ground_truth(size=size, kind=kind, seed=seed)
    degraded = degrade(gt, scale_factor=scale_factor, seed=seed)
    return gt, degraded, kind


def build_dataset(out_dir, n_samples=200, size=512, scale_factor=2, seed0=0):
    out_dir = Path(out_dir)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "degraded").mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        gt, deg, kind = generate_pair(size=size, scale_factor=scale_factor, seed=seed0 + i)
        np.save(out_dir / "gt" / f"{i:05d}_{kind}.npy", gt.astype(np.float32))
        np.save(out_dir / "degraded" / f"{i:05d}_{kind}.npy", deg.astype(np.float32))
    print(f"Wrote {n_samples} pairs to {out_dir}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    build_dataset("/home/claude/kla_restoration/data/train", n_samples=n, size=256, scale_factor=2, seed0=0)
    build_dataset("/home/claude/kla_restoration/data/val", n_samples=max(10, n // 10), size=256, scale_factor=2, seed0=100000)
