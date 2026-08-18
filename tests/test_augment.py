"""Tests for training-time augmentation (defect W-03).

The silent-failure mode here is applying a different transform to the degraded
patch than to its ground truth, which destroys the spatial correspondence the
model learns from while still training without error.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train import augment_pair  # noqa: E402


def _paired_arrays(n=16, scale=2):
    """A degraded/GT pair where GT is an exact `scale`x upsample of degraded.

    That exact relationship lets us verify the same geometric transform was
    applied to both: it must still hold afterwards.
    """
    deg = np.arange(n * n, dtype=np.float32).reshape(n, n)
    gt = np.kron(deg, np.ones((scale, scale), dtype=np.float32))
    return deg, gt


@pytest.mark.parametrize("seed", range(12))
def test_same_transform_applied_to_both(seed):
    """GT must remain the exact 2x upsample of degraded after augmentation."""
    deg, gt = _paired_arrays()
    rng = np.random.default_rng(seed)

    deg_a, gt_a = augment_pair(deg, gt, rng)

    expected_gt = np.kron(deg_a, np.ones((2, 2), dtype=np.float32))
    assert np.array_equal(gt_a, expected_gt), (
        "degraded and GT received different transforms -- pairing is broken"
    )


@pytest.mark.parametrize("seed", range(8))
def test_shapes_are_preserved(seed):
    """Dihedral transforms of a square patch must not change its shape."""
    deg, gt = _paired_arrays()
    deg_a, gt_a = augment_pair(deg, gt, np.random.default_rng(seed))

    assert deg_a.shape == deg.shape
    assert gt_a.shape == gt.shape


@pytest.mark.parametrize("seed", range(8))
def test_output_is_contiguous(seed):
    """rot90/flip produce negative-stride views; torch requires contiguous."""
    deg, gt = _paired_arrays()
    deg_a, gt_a = augment_pair(deg, gt, np.random.default_rng(seed))

    assert deg_a.flags["C_CONTIGUOUS"]
    assert gt_a.flags["C_CONTIGUOUS"]


def test_augmentation_actually_varies():
    """Over many draws the transform must not be a constant identity."""
    deg, gt = _paired_arrays()
    rng = np.random.default_rng(0)

    seen = {augment_pair(deg, gt, rng)[0].tobytes() for _ in range(60)}
    assert len(seen) > 1, "augmentation produced only one distinct output"


def test_pixel_values_are_unchanged():
    """Geometric augmentation must permute pixels, never alter their values."""
    deg, gt = _paired_arrays()
    deg_a, _ = augment_pair(deg, gt, np.random.default_rng(3))

    assert np.array_equal(np.sort(deg_a.ravel()), np.sort(deg.ravel()))
