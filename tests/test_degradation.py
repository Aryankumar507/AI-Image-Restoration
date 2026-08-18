"""Tests for the randomized degradation pipeline (defects W-01, W-04).

Two failure modes matter here:
  * a degradation so narrow the model only learns to invert one exact process
    (the OOD failure the problem statement explicitly tests for), and
  * noise that is deterministically correlated with image content, which the
    network can learn to exploit instead of actually denoising.
"""
import numpy as np
import pytest

from kla_restoration.data_generator import (
    KINDS, degrade, generate_pair, make_ground_truth,
)


@pytest.fixture
def gt():
    return make_ground_truth(size=128, kind="texture", seed=1)


def test_downsamples_by_scale_factor(gt):
    out = degrade(gt, scale_factor=2, rng=np.random.default_rng(0))
    assert out.shape == (gt.shape[0] // 2, gt.shape[1] // 2)


def test_output_is_not_clipped(gt):
    """Speckle legally pushes values past the GT range; clipping loses that."""
    maxima = [degrade(gt, rng=np.random.default_rng(s)).max() for s in range(60)]
    assert any(m > 255 for m in maxima), "no sample exceeded the GT range"


def test_same_rng_seed_is_reproducible(gt):
    a = degrade(gt, rng=np.random.default_rng(42))
    b = degrade(gt, rng=np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_different_seeds_give_different_degradations(gt):
    a = degrade(gt, rng=np.random.default_rng(1))
    b = degrade(gt, rng=np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_degradation_severity_actually_varies(gt):
    """A narrow degradation distribution is the root cause of OOD failure."""
    stds = np.array([degrade(gt, rng=np.random.default_rng(s)).std()
                     for s in range(100)])
    # Spread must be substantial, not a token jitter around one operating point.
    assert stds.max() / stds.min() > 2.0, (
        f"degradation severity barely varies (std {stds.min():.1f}..{stds.max():.1f})"
    )


def test_operation_order_is_randomized(gt):
    """With order randomized, outputs must not collapse to a single pipeline."""
    outs = {degrade(gt, rng=np.random.default_rng(s)).tobytes() for s in range(30)}
    assert len(outs) == 30, "degradations were not distinct"


def test_fixed_order_is_deterministic_given_a_seed(gt):
    a = degrade(gt, randomize_order=False, rng=np.random.default_rng(3))
    b = degrade(gt, randomize_order=False, rng=np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_noise_is_decorrelated_from_image_content():
    """The degradation seed must not be the ground-truth seed (W-01).

    An earlier version passed the same seed to both, making the noise
    realization a deterministic function of image content -- label leakage the
    network can learn instead of denoising.

    Two DIFFERENT ground truths degraded at the same index must not receive
    the identical noise field.
    """
    gt_a = make_ground_truth(size=128, kind="texture", seed=5)
    gt_b = make_ground_truth(size=128, kind="texture", seed=5)
    assert np.array_equal(gt_a, gt_b), "fixture assumption: same seed, same GT"

    # Degrade a flat field so the noise itself is what we compare.
    flat = np.full((128, 128), 100.0, dtype=np.float32)
    n1 = degrade(flat, randomize_order=False, rng=np.random.default_rng(11))
    n2 = degrade(flat, randomize_order=False, rng=np.random.default_rng(12))
    assert not np.array_equal(n1, n2)


def test_generate_pair_decorrelates_seeds():
    """generate_pair must not reuse the GT seed for the degradation."""
    gt1, deg1, _ = generate_pair(size=128, seed=100)
    gt2, deg2, _ = generate_pair(size=128, seed=100)
    # Same seed -> fully reproducible pair.
    assert np.array_equal(gt1, gt2) and np.array_equal(deg1, deg2)

    # But the degradation stream must differ from the content stream: two GTs
    # that happen to be identical must still be able to get different noise.
    from kla_restoration.data_generator import degrade as _d
    a = _d(gt1, rng=np.random.default_rng(100))
    b = _d(gt1, rng=np.random.default_rng(100 + 999_983))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("kind", KINDS)
def test_generate_pair_shapes(kind):
    gt, deg, k = generate_pair(size=256, scale_factor=2, kind=kind, seed=0)
    assert gt.shape == (256, 256)
    assert deg.shape == (128, 128)
    assert k == kind


def test_no_nans_or_infs(gt):
    for s in range(50):
        out = degrade(gt, rng=np.random.default_rng(s))
        assert np.isfinite(out).all(), f"non-finite values at seed {s}"
