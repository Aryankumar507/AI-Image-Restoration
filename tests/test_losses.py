"""Tests for the training objective (defect W-02).

These guard two things the loss got wrong: a clamp that silently zeroed
gradients on exactly the pixels speckle noise produces, and a weighting that
under-optimized PSNR.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train import combined_loss, ssim  # noqa: E402


def test_gradient_reaches_out_of_range_pixels():
    """Pixels above 1.0 must still receive gradient (W-02).

    Speckle legally pushes values beyond the ground-truth range. The previous
    implementation computed SSIM on pred.clamp(0, 1); clamp has zero gradient
    outside its range, so those pixels learned nothing.
    """
    target = torch.rand(1, 1, 32, 32)
    # A prediction whose pixels sit ABOVE the valid range.
    pred = torch.full((1, 1, 32, 32), 1.8, requires_grad=True)

    loss, _, _ = combined_loss(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0, (
        "out-of-range pixels received zero gradient -- the clamp bug is back"
    )


def test_clamped_loss_would_have_killed_gradients():
    """Demonstrates the bug the fix removes, so the test above has meaning."""
    target = torch.rand(1, 1, 32, 32)
    pred = torch.full((1, 1, 32, 32), 1.8, requires_grad=True)

    # The OLD behaviour: SSIM on a clamped prediction.
    s = 1 - ssim(pred.clamp(0, 1), target.clamp(0, 1))
    s.backward()

    assert pred.grad.abs().sum() == 0, (
        "expected the clamped formulation to zero all gradients"
    )


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_alpha_controls_the_mix(alpha):
    """alpha must actually shift weight between the SSIM and L1 terms."""
    pred = torch.rand(1, 1, 32, 32, requires_grad=True)
    target = torch.rand(1, 1, 32, 32)

    total, l1v, ssimv = combined_loss(pred, target, alpha=alpha)
    expected = alpha * ssimv + (1 - alpha) * l1v

    assert total.item() == pytest.approx(expected, rel=1e-5)


def test_identical_images_give_near_zero_loss():
    """Sanity: a perfect prediction should score ~0 on both terms."""
    x = torch.rand(1, 1, 64, 64)
    total, l1v, ssimv = combined_loss(x, x.clone())

    assert l1v == pytest.approx(0.0, abs=1e-6)
    assert ssimv == pytest.approx(0.0, abs=1e-4)
    assert total.item() == pytest.approx(0.0, abs=1e-4)


def test_worse_prediction_scores_higher_loss():
    """The loss must be monotone in prediction quality."""
    target = torch.rand(1, 1, 64, 64)
    close = target + torch.randn_like(target) * 0.01
    far = target + torch.randn_like(target) * 0.30

    loss_close, _, _ = combined_loss(close, target)
    loss_far, _, _ = combined_loss(far, target)

    assert loss_close.item() < loss_far.item()


# ----------------------------------------- anomaly-weighted loss (Idea D) --
def test_anomaly_weight_emphasises_erased_defects():
    """The weight map must fire on a defect the model failed to reproduce.

    This is the exact failure the audit found: the model outputs the clean
    structure and drops the anomaly. The weight there must be far above 1.
    """
    import cv2
    import numpy as np
    from train import anomaly_weight_map
    from kla_restoration.data_generator import make_ground_truth

    for kind in ("texture", "dendrite", "linespace", "contact", "polygon"):
        clean = make_ground_truth(size=128, kind=kind, seed=5)
        defective = clean.copy()
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(defective, (64, 64), 5, 0.0, -1)      # a dark void
        cv2.circle(mask, (64, 64), 5, 1, -1)
        m = mask.astype(bool)

        target = torch.from_numpy(defective / 255.0).float()[None, None]
        pred = torch.from_numpy(clean / 255.0).float()[None, None]   # defect erased

        w = anomaly_weight_map(pred, target, gain=8.0)[0, 0].numpy()
        ratio = w[m].mean() / w[~m].mean()
        assert ratio > 50, (
            f"{kind}: weight ratio {ratio:.2f}x is too low -- an erased defect "
            "must be heavily up-weighted"
        )


def test_anomaly_weight_is_detached():
    """Weights must not be differentiable, or the model could game them."""
    from train import anomaly_weight_map

    pred = torch.rand(1, 1, 16, 16, requires_grad=True)
    target = torch.rand(1, 1, 16, 16)
    w = anomaly_weight_map(pred, target)
    assert not w.requires_grad


def test_defect_gain_zero_matches_plain_l1():
    """defect_gain=0 must be exactly the unweighted objective."""
    from train import combined_loss

    pred = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    a, _, _ = combined_loss(pred, target, defect_gain=0.0)
    b, _, _ = combined_loss(pred, target)
    assert a.item() == pytest.approx(b.item())


def test_weighted_loss_penalises_erased_defect_more():
    """Two predictions with equal mean error must not score equally when one
    concentrates its error on a defect."""
    import numpy as np
    from train import combined_loss

    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 30:34, 30:34] = 1.0                 # the "defect"

    erased = torch.zeros(1, 1, 64, 64)               # misses it entirely
    spread = torch.full((1, 1, 64, 64), float(target.mean()))  # same total error, diffuse

    l_erased, _, _ = combined_loss(erased, target, alpha=0.0, defect_gain=8.0)
    l_spread, _, _ = combined_loss(spread, target, alpha=0.0, defect_gain=8.0)
    assert l_erased.item() > l_spread.item(), (
        "the weighted loss must punish concentrating error on the defect"
    )


def test_weighted_loss_gradients_flow():
    """Sanity: the weighted objective must still train."""
    from train import combined_loss

    pred = torch.rand(2, 1, 32, 32, requires_grad=True)
    target = torch.rand(2, 1, 32, 32)
    loss, _, _ = combined_loss(pred, target, defect_gain=8.0)
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
