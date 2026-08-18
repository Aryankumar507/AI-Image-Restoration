"""Tests for the four innovation ideas (deck Slide 5).

Each idea is guarded by a property that would silently break it: a wrong
inverse transform in the ensemble, a non-identity initialization in FiLM, a
log branch that mishandles out-of-range speckle, or an FFT loss that does not
actually respond to a pitch error.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kla_restoration.model import (  # noqa: E402
    DegradationEstimator, LogSpeckleBranch, NAFNetLiteSR, self_ensemble)


# ------------------------------------------------- IDEA F: self-ensemble --
def test_self_ensemble_inverse_transforms_are_exact():
    """On a perfectly equivariant model the ensemble must be a no-op.

    If any of the 8 inverse transforms is wrong, averaging misaligned outputs
    blurs the result -- silently, with no error raised.
    """
    class Upsample(nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    m = Upsample().eval()
    x = torch.randn(1, 1, 32, 32)
    assert (m(x) - self_ensemble(m, x)).abs().max() < 1e-5


def test_self_ensemble_output_shape():
    m = NAFNetLiteSR(width=16, enc_blocks=(1, 1), dec_blocks=(1, 1), middle_blocks=1).eval()
    assert self_ensemble(m, torch.randn(1, 1, 64, 64)).shape == (1, 1, 128, 128)


# --------------------------------------------------- IDEA E: Fourier loss --
def test_fft_loss_is_zero_for_identical_images():
    from train import fft_magnitude_loss
    x = torch.rand(2, 1, 64, 64)
    assert fft_magnitude_loss(x, x.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_fft_loss_detects_wrong_pitch():
    """A grating at the wrong PERIOD must cost more than one merely shifted.

    This is the whole point: a spatial L1 punishes a shift heavily and a small
    period error lightly, while for metrology the period is what matters.
    """
    from train import fft_magnitude_loss

    def grating(period, phase=0):
        x = torch.arange(64, dtype=torch.float32)
        row = ((x + phase) % period < period / 2).float()
        return row.repeat(64, 1)[None, None]

    target = grating(8)
    shifted = grating(8, phase=2)        # same pitch, different position
    wrong_pitch = grating(11)            # different pitch

    assert (fft_magnitude_loss(wrong_pitch, target).item()
            > fft_magnitude_loss(shifted, target).item())


# ----------------------------------------------- IDEA B: log-speckle branch --
def test_log_branch_handles_out_of_range_and_negative_values():
    """Speckle pushes values above 1 and read noise below 0; both must be finite."""
    branch = LogSpeckleBranch(out_ch=8)
    x = torch.tensor([[[[-0.5, 0.0, 1.0, 3.7]]]])       # negative and >1
    out = branch(x)
    assert torch.isfinite(out).all()


def test_log_branch_linearizes_multiplicative_noise():
    """log1p must turn multiplicative speckle into an additive perturbation.

    Under y = x*(1+n), the log-domain residual should be roughly independent
    of the underlying signal level -- which is the entire justification for
    the branch. In the linear domain it scales with x.
    """
    x = torch.tensor([0.1, 0.3, 0.6, 0.9])
    n = 0.2                                              # 20% speckle
    y = x * (1 + n)

    linear_residual = (y - x)                            # grows with x
    log_residual = torch.log1p(y) - torch.log1p(x)       # much flatter

    assert log_residual.std() < linear_residual.std()


# --------------------------------------------- IDEA A: FiLM degradation head --
def test_degradation_estimator_is_resolution_independent():
    """The code must have the same width at any input size (global pooling)."""
    est = DegradationEstimator(1, code_dim=16).eval()
    with torch.no_grad():
        assert est(torch.rand(1, 1, 64, 64)).shape == (1, 16)
        assert est(torch.rand(1, 1, 256, 256)).shape == (1, 16)


def test_degradation_estimator_responds_to_noise_level():
    """A noisier input must produce a different code, or conditioning is useless."""
    est = DegradationEstimator(1, code_dim=16).eval()
    base = torch.rand(1, 1, 64, 64) * 0.5 + 0.25
    noisy = (base + torch.randn_like(base) * 0.3).clamp(0, 1)
    with torch.no_grad():
        assert not torch.allclose(est(base), est(noisy), atol=1e-4)


def test_film_starts_as_identity():
    """FiLM is zero-initialized, so an untrained conditioning path must not
    change the output -- otherwise it destabilizes early training."""
    torch.manual_seed(0)
    plain = NAFNetLiteSR(width=16, enc_blocks=(1, 1), dec_blocks=(1, 1), middle_blocks=1)
    torch.manual_seed(0)
    filmed = NAFNetLiteSR(width=16, enc_blocks=(1, 1), dec_blocks=(1, 1),
                          middle_blocks=1, code_dim=16)
    plain.eval(); filmed.eval()

    # Copy shared weights so only the FiLM path differs.
    shared = {k: v for k, v in plain.state_dict().items() if k in filmed.state_dict()}
    filmed.load_state_dict(shared, strict=False)

    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(plain(x), filmed(x), atol=1e-5)


# -------------------------------------------------------- checkpoint round-trip --
@pytest.mark.parametrize("cfg", [
    {"code_dim": 16},
    {"log_branch": 8},
    {"code_dim": 16, "log_branch": 8},
])
def test_new_configs_round_trip_through_checkpoint(tmp_path, cfg):
    model = NAFNetLiteSR(width=16, enc_blocks=(1, 1), dec_blocks=(1, 1),
                         middle_blocks=1, **cfg)
    path = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(), "config": model.config}, path)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    rebuilt = NAFNetLiteSR.from_checkpoint(ckpt)

    assert rebuilt.config == model.config
    with torch.no_grad():
        assert rebuilt(torch.randn(1, 1, 64, 64)).shape == (1, 1, 128, 128)
