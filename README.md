# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**KLA / i4C Hackathon — Problem Statement PS01**

Restores microscopic semiconductor inspection images degraded by **speckle noise**, **Gaussian blur** and **2× spatial downsampling** — all three simultaneously, in a single forward pass.

The model is **NAFNet-Lite-SR**: a compact NAFNet U-Net backbone with a PixelShuffle super-resolution head. Denoising and deblurring happen in the low-resolution latent space (cheapest place computationally), and upsampling happens once at the end — chosen because the submission is scored on **both** restoration quality and H100 inference time.

---

## Quick start

```bash
git clone <your-repo-url>
cd AI-Image-Restoration

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

> **The `pip install -e .` step is required.** Library code lives under `src/`, and this makes `kla_restoration` importable. Without it the evaluation script raises `ModuleNotFoundError`.

### Run inference

```bash
python evaluate_submission.py \
    --input_dir  path/to/degraded_test_images \
    --output_dir outputs/restored
```

That is the complete inference path. The trained weights at `checkpoints/nafnet_lite.pt` are found automatically — no editing, no extra arguments.

### Regenerating the restored outputs

`outputs/` is generated, not tracked. To produce the full set of restored
images (mandatory submission content #5) run inference into it:

```bash
python evaluate_submission.py     --input_dir  path/to/degraded_test_images     --output_dir outputs/restored
```

Only a few before/after comparison panels and the measured result tables are
committed, as evidence for reviewers reading rather than running the code.

**Verify the install works** before relying on it:

```bash
pytest
```

All tests should pass. They assert the evaluation script runs end-to-end from any working directory, the checkpoint loads, and output resolutions are correct.

---

## Evaluation script reference

`evaluate_submission.py` is the script KLA's benchmarking team runs as-is.

| Argument | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input_dir` | **yes** | — | Directory of degraded test images |
| `--output_dir` | **yes** | — | Directory to write restored images (created if absent) |
| `--checkpoint` | no | `checkpoints/nafnet_lite.pt` | Model weights; resolved relative to the script, so it works from any CWD |
| `--device` | no | `cuda` if available, else `cpu` | Inference device |
| `--scale` | no | `2` | Output size relative to input. `2` matches the problem statement (256→512, 128→256). `1` denoises at native resolution without upscaling. |
| `--self_ensemble` | no | off | ×8 geometric self-ensemble. **+0.27 dB** at 6.6× cost (23.9 → 157 ms/image) — a quality/speed dial. |
| `--batch_size` | no | `1` | Images per forward pass. Measured fastest at 1 on a 4 GB card (23 ms/image vs 202 ms at 8, where activations spill VRAM). Raise on an H100. |
| `--tlc` | no | `0` (off) | Test-time Local Converter window. Measured neutral on this model (21.73 vs 21.71 dB); available for larger inputs. |

**Input formats:** `.png`, `.jpg`, `.jpeg`, `.npy` — grayscale, 8-bit or 16-bit. Files are read at **native bit depth**; 16-bit inputs are not down-converted. Pixel values above the ground-truth range (expected from speckle noise, per the spec) are passed through to the model rather than clipped away.

**Output format:** 8-bit grayscale `.png`, written with the **exact same base filename** as the input (`wafer_001.png` → `wafer_001.png`), at `--scale` × the input resolution.

---

## Repository layout

```
├── evaluate_submission.py       # ← submission entry point (run this)
├── pyproject.toml               # package definition
├── requirements.txt
├── src/kla_restoration/         # library code
│   ├── model.py                 #   NAFNet-Lite-SR architecture
│   └── data_generator.py        #   synthetic paired-data generator
├── scripts/
│   ├── train.py                 #   training loop
│   ├── generate_data.py         #   build the synthetic dataset
│   ├── report_metrics.py        #   PSNR/SSIM/LPIPS vs. a bicubic baseline
│   ├── defect_preservation.py   #   defect-erasure / hallucination audit
│   └── idea_ablation.py         #   ablation over the innovation ideas
├── checkpoints/nafnet_lite.pt   # trained weights
├── outputs/                     # generated; only evidence files are tracked
│   ├── comparisons/             #   labelled before/after panels
│   ├── defect_audit/            #   defect-preservation results
│   └── idea_ablation.txt        #   ablation over the innovation ideas
├── tests/                       # smoke tests for the submission path
└── docs/
    ├── PROBLEM_STATEMENT.md     #   the official problem statement
    ├── ASSESSMENT.md            #   internal pre-submission audit
    ├── RESEARCH_SUMMARY.md      #   approach, literature, results
    └── submission/              #   presentation deck
```

---

## Reproducing training

```bash
# 1. Build the dataset
python scripts/generate_data.py --n_train 300 --size 256

# 2. Train
python scripts/train.py --epochs 40 --width 48 \
    --enc_blocks 2 2 --dec_blocks 2 2 --middle_blocks 2

# 3. Evaluate against a bicubic baseline
python scripts/report_metrics.py
```

Paths default to the repository root, so these run from anywhere. `scripts/train.py --help` lists all hyperparameters.

> ### ⚠️ The training data is synthetic
>
> `scripts/generate_data.py` produces **procedurally generated stand-in images**, not real semiconductor data, because KLA's dataset is distributed during the hackathon. Metrics measured on it describe the pipeline, **not** real-world accuracy.
>
> When the real dataset arrives, replace the loader and retrain. See [`docs/ASSESSMENT.md`](docs/ASSESSMENT.md) (finding W-01) for the full analysis.

---

## Architecture

**NAFNet-Lite-SR** — single-stage U-Net, **1.163M parameters** at `width=48`, `enc/dec_blocks=(2,2)`, `middle_blocks=2`, plus a log-domain speckle branch.

**Homomorphic speckle handling.** Speckle is *multiplicative* — `y = x·(1+n)` — which convolutional networks and L1/SSIM losses handle poorly, since the noise magnitude scales with local brightness. A `log1p` branch runs alongside the linear input and is concatenated with it: `log(x·(1+n)) = log(x) + log(1+n)` turns the corruption *additive*, the form standard denoisers are built for. This is the classical basis of SAR despeckling applied to inspection imagery, and it also explains why degraded pixels legally exceed the ground-truth range. Costs 80 parameters, measured **+0.41 dB**.

- **NAFBlocks** (LayerNorm → depthwise conv → SimpleGate → simplified channel attention) with **no nonlinear activations**, following NAFNet's core finding that they are unnecessary for state-of-the-art restoration.
- **PixelShuffle SR head** appended after the backbone for 2× upsampling.
- Denoising and deblurring run at low resolution; upsampling happens once at the end.

**Loss:** `0.5 · (1 − SSIM) + 0.5 · L1` (`--alpha`) — L1 rather than L2 because speckle is multiplicative and heavy-tailed, plus a differentiable SSIM term that directly optimizes one of the three graded metrics.

Full rationale and literature review: [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md).

---

## Reference environment

| | |
| :--- | :--- |
| Python | 3.13.5 |
| PyTorch | 2.13.0 |
| torchvision | 0.28.0 |
| NumPy | 2.5.2 |
| OpenCV | 5.0.0.93 |
| scikit-image | 0.26.0 |
| OS | Windows 11 |

`requirements.txt` uses lower bounds rather than exact pins so the benchmark machine can resolve a CUDA build matching its own driver.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'kla_restoration'`**
Run `pip install -e .` from the repository root, with your virtual environment active.

**`FileNotFoundError` on the checkpoint**
Confirm `checkpoints/nafnet_lite.pt` exists. If cloned via Git LFS, run `git lfs pull`.

**Slow inference**
~24 ms/image on an RTX 2050. Pass `--device cuda` on a GPU machine. Leave `--batch_size` at 1 unless you have high-VRAM hardware — see the flag's note above.

---

## License & references

Built for the KLA / i4C hackathon.

- Chen, Chu, Zhang, Sun. *Simple Baselines for Image Restoration* (NAFNet), ECCV 2022. [arXiv:2204.04676](https://arxiv.org/abs/2204.04676)
- Zamir et al. *Restormer: Efficient Transformer for High-Resolution Image Restoration*, CVPR 2022. [arXiv:2111.09881](https://arxiv.org/abs/2111.09881)
- Wang et al. *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*, ICCV Workshops 2021. [arXiv:2107.10833](https://arxiv.org/abs/2107.10833)
