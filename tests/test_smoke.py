"""Smoke tests: the submission must run without manual edits.

These guard the failure mode that matters most -- KLA's benchmarking team runs
evaluate_submission.py AS-IS on an H100, and an import error or a bad default
path means the submission is never scored at all.

Run with:  pytest
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "evaluate_submission.py"
CHECKPOINT = REPO_ROOT / "checkpoints" / "nafnet_lite.pt"


def test_package_imports():
    """The package must be importable (i.e. `pip install -e .` was run)."""
    from kla_restoration.model import NAFNetLiteSR  # noqa: F401
    from kla_restoration.data_generator import build_dataset  # noqa: F401


def test_model_output_is_2x_input():
    """The SR head must double spatial resolution, per the problem statement."""
    import torch
    from kla_restoration.model import NAFNetLiteSR

    model = NAFNetLiteSR(width=24, enc_blocks=(1, 1), dec_blocks=(1, 1), middle_blocks=1)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 64, 64))
    assert out.shape == (1, 1, 128, 128), f"expected 2x upsample, got {tuple(out.shape)}"


def test_checkpoint_exists_at_default_path():
    """The eval script's default --checkpoint must actually resolve (blocker B-01)."""
    assert CHECKPOINT.is_file(), f"checkpoint missing at default path: {CHECKPOINT}"


def test_checkpoint_loads_under_weights_only():
    """The checkpoint must load with weights_only=True (blocker B-04).

    That is the default from PyTorch 2.6 onward; a checkpoint that needs the
    unrestricted unpickler would fail outright on KLA's benchmark machine.
    """
    import torch
    from kla_restoration.model import NAFNetLiteSR

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    # Rebuild via the factory so this test never hardcodes an architecture --
    # that hardcoding is precisely the drift W-08 exists to prevent.
    model = NAFNetLiteSR.from_checkpoint(ckpt)
    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.parametrize("cwd_is_repo_root", [True, False])
def test_evaluate_submission_runs_from_any_cwd(tmp_path, cwd_is_repo_root):
    """The entry point must run regardless of the caller's working directory."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    rng = np.random.default_rng(0)
    cv2.imwrite(str(in_dir / "a.png"), rng.integers(0, 255, (128, 128)).astype(np.uint8))
    np.save(in_dir / "b.npy", rng.normal(120, 40, (128, 128)).astype(np.float32))

    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         "--input_dir", str(in_dir), "--output_dir", str(out_dir)],
        cwd=str(REPO_ROOT) if cwd_is_repo_root else str(tmp_path),
        capture_output=True, text=True,
    )

    assert result.returncode == 0, f"eval script failed:\n{result.stderr}"
    written = list(out_dir.glob("*.png"))
    assert len(written) == 2, f"expected 2 restored images, got {[p.name for p in written]}"


def test_output_filenames_match_input_stems_exactly(tmp_path):
    """Output stems must equal input stems (blocker B-02).

    Scorers commonly pair restored outputs to ground truth by filename stem;
    a suffix such as "_restored" makes every file fail to match.
    """
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    rng = np.random.default_rng(2)
    for name in ("wafer_001", "die_042"):
        cv2.imwrite(str(in_dir / f"{name}.png"),
                    rng.integers(0, 255, (128, 128)).astype(np.uint8))

    subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         "--input_dir", str(in_dir), "--output_dir", str(out_dir)],
        cwd=str(REPO_ROOT), check=True, capture_output=True,
    )

    produced = {p.stem for p in out_dir.glob("*.png")}
    assert produced == {"wafer_001", "die_042"}, (
        f"output stems must match input stems exactly, got {sorted(produced)}"
    )


def test_16bit_input_preserves_dynamic_range(tmp_path):
    """16-bit inputs must not be crushed to 8-bit on load (blocker B-03).

    cv2.IMREAD_GRAYSCALE force-converts to 8 bits, discarding most of the
    range of 16-bit inspection imagery. This asserts the loader sees the real
    values and normalizes them onto the training scale.
    """
    from evaluate_submission import load_image

    p = tmp_path / "deep.png"
    # A ramp spanning the full 16-bit range.
    img16 = np.linspace(0, 65535, 128 * 128).reshape(128, 128).astype(np.uint16)
    cv2.imwrite(str(p), img16)

    arr, scale = load_image(p)
    assert scale == 65535.0, f"expected 16-bit scale, got {scale}"
    assert arr.max() > 60000, (
        f"dynamic range was crushed on load (max={arr.max()}); "
        "IMREAD_GRAYSCALE was probably used instead of IMREAD_UNCHANGED"
    )
    # After normalization the image must land in [0,1], matching training.
    assert 0.9 < (arr / scale).max() <= 1.0


def test_speckle_values_above_range_are_not_clipped(tmp_path):
    """.npy inputs may legally exceed the GT range; the loader must not clip.

    The problem statement calls this out explicitly, and training preserved it
    (values >1.0 after normalization).
    """
    from evaluate_submission import load_image

    p = tmp_path / "hot.npy"
    arr_in = np.full((64, 64), 300.0, dtype=np.float32)  # deliberately >255
    np.save(p, arr_in)

    arr, scale = load_image(p)
    assert (arr / scale).max() > 1.0, "out-of-range speckle values were clipped away"


def test_checkpoint_carries_full_architecture_config(tmp_path):
    """A saved checkpoint must be reconstructable without hardcoded args (W-08).

    Guards the failure where scaling the model up in training silently breaks
    inference, because the eval script assumed a different topology.
    """
    import torch
    from kla_restoration.model import NAFNetLiteSR

    # A deliberately non-default, asymmetric architecture.
    model = NAFNetLiteSR(width=16, enc_blocks=(1, 2), dec_blocks=(2, 1), middle_blocks=2)
    path = tmp_path / "custom.pt"
    torch.save({"model": model.state_dict(), "config": model.config, "width": 16}, path)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    rebuilt = NAFNetLiteSR.from_checkpoint(ckpt)

    assert rebuilt.config == model.config
    with torch.no_grad():
        assert rebuilt(torch.randn(1, 1, 64, 64)).shape == (1, 1, 128, 128)


def test_legacy_checkpoint_without_config_still_loads(tmp_path):
    """Checkpoints predating the stored config must still load (W-08 back-compat).

    Builds a legacy-format checkpoint (weights + width only, no config) from a
    known architecture and checks that it is recovered from the weights alone.
    """
    import torch
    from kla_restoration.model import NAFNetLiteSR

    original = NAFNetLiteSR(width=24, enc_blocks=(1, 1), dec_blocks=(1, 1),
                            middle_blocks=1)
    path = tmp_path / "legacy.pt"
    torch.save({"model": original.state_dict(), "width": 24}, path)   # no "config"

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert "config" not in ckpt, "fixture must emulate the legacy format"

    rebuilt = NAFNetLiteSR.from_checkpoint(ckpt)
    assert rebuilt.config == original.config


@pytest.mark.parametrize("scale,expected", [
    (None, (256, 256)),   # default: 2x, per the problem statement
    ("2", (256, 256)),
    ("1", (128, 128)),    # denoise only, no upscale
    ("4", (512, 512)),
])
def test_output_resolution_honours_scale(tmp_path, scale, expected):
    """Output size must follow --scale, defaulting to 2x (blocker B-05)."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    rng = np.random.default_rng(1)
    cv2.imwrite(str(in_dir / "img.png"), rng.integers(0, 255, (128, 128)).astype(np.uint8))

    cmd = [sys.executable, str(EVAL_SCRIPT),
           "--input_dir", str(in_dir), "--output_dir", str(out_dir)]
    if scale is not None:
        cmd += ["--scale", scale]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, capture_output=True)

    written = list(out_dir.glob("*.png"))
    assert written, "no output written"
    restored = cv2.imread(str(written[0]), cv2.IMREAD_UNCHANGED)
    assert restored.shape[:2] == expected, (
        f"scale={scale}: expected {expected}, got {restored.shape[:2]}"
    )


def test_invalid_scale_is_rejected(tmp_path):
    """A non-positive --scale must fail loudly, not produce garbage."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    cv2.imwrite(str(in_dir / "img.png"), np.zeros((32, 32), dtype=np.uint8))

    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--input_dir", str(in_dir),
         "--output_dir", str(tmp_path / "out"), "--scale", "0"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode != 0, "--scale 0 should have been rejected"


def test_tlc_preserves_parameters_and_shape():
    """TLC must swap normalization without touching learned parameters (W-07)."""
    import torch
    import torch.nn as nn
    from kla_restoration.model import NAFNetLiteSR, apply_tlc, LocalGroupNorm

    model = NAFNetLiteSR(width=16, enc_blocks=(1, 1), dec_blocks=(1, 1), middle_blocks=1)
    model.eval()
    before = sum(p.numel() for p in model.parameters())

    apply_tlc(model, window=64)

    assert sum(p.numel() for p in model.parameters()) == before
    assert any(isinstance(m, LocalGroupNorm) for m in model.modules())
    assert not any(isinstance(m, nn.GroupNorm) for m in model.modules())
    with torch.no_grad():
        assert model(torch.randn(1, 1, 128, 128)).shape == (1, 1, 256, 256)


def test_eval_script_accepts_tlc_flag(tmp_path):
    """--tlc must run end-to-end and still produce correct output sizes."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    rng = np.random.default_rng(11)
    cv2.imwrite(str(in_dir / "x.png"), rng.integers(0, 255, (128, 128)).astype(np.uint8))

    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), "--input_dir", str(in_dir),
         "--output_dir", str(out_dir), "--tlc", "64"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    written = list(out_dir.glob("*.png"))
    assert written and cv2.imread(str(written[0]), cv2.IMREAD_UNCHANGED).shape[:2] == (256, 256)


def test_apply_tlc_is_device_safe():
    """apply_tlc must work before AND after .to(device) (regression).

    The swapped modules previously allocated their parameters on CPU, so
    calling apply_tlc after model.to("cuda") crashed with a device mismatch.
    """
    import torch
    from kla_restoration.model import NAFNetLiteSR, apply_tlc

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        # after .to(device)
        m = NAFNetLiteSR(width=16, enc_blocks=(1, 1), dec_blocks=(1, 1), middle_blocks=1)
        m.to(dev).eval()
        apply_tlc(m, window=32)
        with torch.no_grad():
            m(torch.randn(1, 1, 64, 64, device=dev))

        # before .to(device)
        m2 = NAFNetLiteSR(width=16, enc_blocks=(1, 1), dec_blocks=(1, 1), middle_blocks=1)
        apply_tlc(m2, window=32)
        m2.to(dev).eval()
        with torch.no_grad():
            m2(torch.randn(1, 1, 64, 64, device=dev))
