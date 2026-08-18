"""
Synthetic paired-data generator for the KLA "AI-Based Restoration of Degraded
Images" challenge.

WHY THIS FILE EXISTS
---------------------
KLA's real paired dataset is distributed during the hackathon and is not
available here, so we synthesize procedural "semiconductor-like" ground truth
(periodic via/line grids and organic dendrite textures) and degrade it to
build and validate the pipeline end to end.

DEGRADATION MODEL
-----------------
Rather than one fixed recipe, `degrade()` samples a RANDOMIZED high-order
pipeline per image, following Real-ESRGAN (Wang et al., 2021):

  * random ORDER of blur / speckle / downsample
  * random blur sigma, sometimes anisotropic
  * random speckle variance (multiplicative, may exceed the GT range)
  * optional Poisson shot noise and additive Gaussian read noise
  * random downsampling kernel (area / bilinear / bicubic / lanczos)

A single fixed recipe teaches the network to invert exactly that process,
which is a direct cause of out-of-distribution failure -- and KLA's test set
is explicitly out-of-distribution (defects W-01 and W-04).

When the real dataset is released, only this file needs replacing; the model,
training loop and evaluation code are agnostic to where pairs come from.
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


def _make_line_space(size, seed=None):
    """Line/space grating -- the most common metrology structure of all.

    Unidirectional, with a random pitch, duty cycle and rotation so the model
    cannot memorize one exact period.
    """
    rng = np.random.default_rng(seed)
    pitch = int(rng.integers(10, 34))
    duty = rng.uniform(0.3, 0.65)
    lw = max(1, int(pitch * duty))
    big = int(size * 1.5)
    img = np.full((big, big), 45, dtype=np.float32)
    for x in range(0, big, pitch):
        img[:, x:x + lw] = 215
    # Rotate to an arbitrary orientation, then centre-crop back to `size`.
    M = cv2.getRotationMatrix2D((big / 2, big / 2), rng.uniform(0, 180), 1.0)
    img = cv2.warpAffine(img, M, (big, big), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    o = (big - size) // 2
    img = img[o:o + size, o:o + size]
    return np.clip(img + rng.normal(0, 3, img.shape).astype(np.float32), 0, 255)


def _make_contact_array(size, seed=None):
    """Contact-hole / pillar array: a hex or square lattice of dots."""
    rng = np.random.default_rng(seed)
    pitch = int(rng.integers(14, 32))
    r = max(2, int(pitch * rng.uniform(0.18, 0.34)))
    hexagonal = rng.random() < 0.5
    bright_dots = rng.random() < 0.5
    bg, fg = (35, 225) if bright_dots else (215, 40)
    img = np.full((size, size), bg, dtype=np.float32)
    for j, y in enumerate(range(pitch // 2, size, pitch)):
        off = (pitch // 2) if (hexagonal and j % 2) else 0
        for x in range(pitch // 2 + off, size, pitch):
            cv2.circle(img, (x, y), r, fg, -1)
    return np.clip(img + rng.normal(0, 3, img.shape).astype(np.float32), 0, 255)


def _make_polygon_layout(size, seed=None):
    """Irregular polygon layout, as found in logic/routing layers.

    Deliberately aperiodic: this is the family that punishes a model which has
    learned to regenerate a periodic pattern instead of restoring what is
    actually present.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 40, dtype=np.float32)
    for _ in range(int(rng.integers(6, 16))):
        if rng.random() < 0.55:                      # rectilinear trace
            x, y = rng.integers(0, size, 2)
            w = int(rng.integers(size // 12, size // 3))
            h = int(rng.integers(3, 12))
            if rng.random() < 0.5:
                w, h = h, w
            cv2.rectangle(img, (x, y), (x + w, y + h), 210, -1)
        else:                                        # convex blob
            n = int(rng.integers(3, 7))
            cx, cy = rng.integers(size // 6, 5 * size // 6, 2)
            rad = int(rng.integers(size // 16, size // 7))
            ang = np.sort(rng.uniform(0, 2 * np.pi, n))
            pts = np.array([[cx + rad * np.cos(a), cy + rad * np.sin(a)] for a in ang],
                           dtype=np.int32)
            cv2.fillPoly(img, [pts], 225)
    return np.clip(img + rng.normal(0, 3, img.shape).astype(np.float32), 0, 255)


def inject_gt_defect(img, rng):
    """Add one small realistic defect to a ground-truth image.

    Training WITH defects present in the ground truth is the direct fix for the
    erasure failure: if anomalies only ever appear as things to be removed, the
    model learns that removing them is correct. Here they are part of the
    target, so reproducing them is what the loss rewards.
    """
    h, w = img.shape
    kind = rng.random()
    cy = int(rng.integers(h // 8, 7 * h // 8))
    cx = int(rng.integers(w // 8, 7 * w // 8))

    if kind < 0.34:                                   # bright particle
        cv2.circle(img, (cx, cy), int(rng.integers(2, 7)), 255.0, -1)
    elif kind < 0.67:                                 # dark void / pit
        cv2.circle(img, (cx, cy), int(rng.integers(2, 7)), 0.0, -1)
    else:                                             # break in a structure
        half = int(rng.integers(3, 10))
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        img[y0:y1, x0:x1] = float(np.median(img))
    return img


#: Ground-truth structure families this generator can synthesize.
KINDS = ("texture", "dendrite", "linespace", "contact", "polygon")


def make_ground_truth(size=512, kind="texture", seed=None, defect_prob=0.0):
    """Synthesize one clean ground-truth image.

    defect_prob > 0 injects a small anomaly with that probability, so defects
    appear in the TARGET the model is trained to reproduce.
    """
    if kind == "texture":
        pitch = np.random.default_rng(seed).integers(16, 40)
        img = _make_periodic_grid(size, pitch, seed=seed)
    elif kind == "linespace":
        img = _make_line_space(size, seed=seed)
    elif kind == "contact":
        img = _make_contact_array(size, seed=seed)
    elif kind == "polygon":
        img = _make_polygon_layout(size, seed=seed)
    else:
        img = _make_dendrite_texture(size, seed=seed)

    if defect_prob > 0:
        drng = np.random.default_rng(None if seed is None else seed + 31_337)
        if drng.random() < defect_prob:
            img = inject_gt_defect(img, drng)
    return img


def _apply_blur(img, rng, sigma_range):
    """Optical/defocus blur. Occasionally anisotropic, as real optics are."""
    sigma_x = rng.uniform(*sigma_range)
    if rng.random() < 0.25:
        sigma_y = rng.uniform(*sigma_range)
    else:
        sigma_y = sigma_x
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)


def _apply_speckle(img, rng, var_range):
    """Multiplicative speckle: y = x + x*n.

    Signal-dependent, and legally pushes values beyond the ground-truth range
    -- the problem statement calls this out as expected behaviour.
    """
    var = rng.uniform(*var_range)
    noise = rng.normal(0, np.sqrt(var), img.shape).astype(np.float32)
    return img + img * noise


def _apply_gaussian_noise(img, rng, sigma_range):
    """Additive sensor/read noise, independent of signal level."""
    sigma = rng.uniform(*sigma_range)
    return img + rng.normal(0, sigma, img.shape).astype(np.float32)


def _apply_poisson(img, rng, scale_range):
    """Shot noise from finite photon/electron counts.

    Variance scales with intensity, which is physically what a detector sees
    and is distinct from both speckle and read noise.
    """
    scale = rng.uniform(*scale_range)
    lam = np.clip(img, 0, None) * scale
    noisy = rng.poisson(lam).astype(np.float32) / max(scale, 1e-8)
    return noisy


def _apply_downsample(img, rng, scale_factor):
    """Resolution reduction with a randomly chosen resampling kernel.

    Real acquisition chains bin/resample in different ways; training on a
    single kernel (INTER_AREA) teaches the model to invert exactly that one,
    which is a direct cause of out-of-distribution failure (defect W-04).
    """
    h, w = img.shape
    kernel = rng.choice(["area", "bilinear", "bicubic", "lanczos"])
    interp = {
        "area": cv2.INTER_AREA,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }[str(kernel)]
    return cv2.resize(img, (w // scale_factor, h // scale_factor), interpolation=interp)


def degrade(gt, scale_factor=2, gaussian_sigma_range=(0.3, 3.0),
            speckle_var_range=(0.005, 0.15), read_noise_range=(0.0, 6.0),
            poisson_scale_range=(0.05, 1.5), randomize_order=True,
            p_gaussian_noise=0.5, p_poisson=0.35, seed=None, rng=None):
    """Apply a randomized high-order degradation to a ground-truth image.

    Follows the Real-ESRGAN "high-order degradation" strategy: rather than one
    fixed recipe, sample a pipeline per image so the model sees a broad family
    of degradations and generalizes instead of learning to invert one exact
    process (defects W-01 and W-04).

    Randomized per call:
      * ORDER of blur / speckle / downsample. The previous fixed order
        (blur -> downsample -> speckle) does not match sensor physics, where
        noise is typically injected at capture, before binning.
      * blur sigma, optionally anisotropic
      * speckle variance
      * presence and strength of additive read noise and Poisson shot noise
      * the downsampling kernel (area / bilinear / bicubic / lanczos)

    gt: float32 HxW array, nominally [0,255]
    Returns: float32 array at H/scale x W/scale, deliberately NOT clipped --
    the degraded range may exceed [0,255].

    Pass `rng` to share a generator with the caller; otherwise `seed` is used.
    Note: the caller must NOT reuse the ground-truth seed here, or the noise
    realization becomes deterministically correlated with image content.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    img = gt.astype(np.float32)

    def op_blur(x):
        return _apply_blur(x, rng, gaussian_sigma_range)

    def op_speckle(x):
        return _apply_speckle(x, rng, speckle_var_range)

    def op_down(x):
        return _apply_downsample(x, rng, scale_factor)

    if randomize_order:
        # Downsampling may occur at any point; blur and speckle likewise. The
        # only constraint is that downsampling happens exactly once.
        ops = [op_blur, op_speckle, op_down]
        order = rng.permutation(len(ops))
        ops = [ops[i] for i in order]
    else:
        ops = [op_blur, op_down, op_speckle]

    for op in ops:
        img = op(img)

    # Sensor noise applied after the main chain, at the final resolution.
    if rng.random() < p_poisson:
        img = _apply_poisson(img, rng, poisson_scale_range)
    if rng.random() < p_gaussian_noise:
        img = _apply_gaussian_noise(img, rng, read_noise_range)

    return img  # NOT clipped -- degraded range may exceed [0,255]


def generate_pair(size=512, scale_factor=2, kind=None, seed=None, defect_prob=0.0,
                  **degrade_kwargs):
    """Generate one (ground_truth, degraded) pair.

    IMPORTANT: the degradation uses a generator derived from, but distinct
    from, the ground-truth seed. Passing the same seed to both (as an earlier
    version did) makes the noise realization a deterministic function of image
    content, which the network can learn to exploit -- a subtle form of label
    leakage (defect W-01).
    """
    rng = np.random.default_rng(seed)
    if kind is None:
        kind = str(rng.choice(KINDS))
    gt = make_ground_truth(size=size, kind=kind, seed=seed, defect_prob=defect_prob)

    # Decorrelate the degradation stream from the content stream.
    deg_rng = np.random.default_rng(None if seed is None else seed + 999_983)
    degraded = degrade(gt, scale_factor=scale_factor, rng=deg_rng, **degrade_kwargs)
    return gt, degraded, str(kind)


def build_dataset(out_dir, n_samples=200, size=512, scale_factor=2, seed0=0,
                  defect_prob=0.0):
    out_dir = Path(out_dir)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "degraded").mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        gt, deg, kind = generate_pair(size=size, scale_factor=scale_factor,
                                      seed=seed0 + i, defect_prob=defect_prob)
        np.save(out_dir / "gt" / f"{i:05d}_{kind}.npy", gt.astype(np.float32))
        np.save(out_dir / "degraded" / f"{i:05d}_{kind}.npy", deg.astype(np.float32))
    print(f"Wrote {n_samples} pairs to {out_dir}")


if __name__ == "__main__":
    import sys

    # Repo root: this file lives at <repo>/src/kla_restoration/
    repo_root = Path(__file__).resolve().parents[2]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    build_dataset(repo_root / "data" / "train", n_samples=n, size=256, scale_factor=2, seed0=0)
    build_dataset(repo_root / "data" / "val", n_samples=max(10, n // 10), size=256, scale_factor=2, seed0=100000)
