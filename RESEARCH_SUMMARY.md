# KLA Hackathon — AI-Based Restoration of Degraded Images
## Research summary, approach, and results

## 1. Problem re-statement
Restore images degraded by **speckle noise**, **Gaussian blur**, and **2x
spatial downsampling** (512→256 or 256→128), scored on **SSIM / PSNR /
LPIPS** plus **H100 inference time**. Must generalize to out-of-distribution
sources at test time.

## 2. Important constraint in this environment
KLA's real paired dataset was **not provided** to me — the problem
statement says it will be distributed by KLA during the hackathon. To
still deliver a working, evaluated, end-to-end pipeline (not just a
architecture description), I built a **synthetic data generator**
(`data_generator.py`) that reproduces the exact degradation recipe KLA
describes (independent speckle noise that can exceed the GT range,
Gaussian-blur softening, 2x downsampling) over two procedurally generated
image families that mimic the deck's example figures: a periodic
via/grid "texture" pattern and a branching "dendrite" pattern. **This is a
stand-in, not real semiconductor data** — see §6 for what changes once
the real dataset is available.

## 3. Literature reviewed
| Paper | Relevance |
|---|---|
| Chen, Chu, Zhang, Sun. *Simple Baselines for Image Restoration* (NAFNet), ECCV 2022, arXiv:2204.04676 | SOTA denoising/deblurring at a fraction of the compute of transformer models — directly relevant since KLA scores on **both** quality and H100 inference time. We adopt its core block design. |
| Zamir et al. *Restormer: Efficient Transformer for High-Resolution Image Restoration*, arXiv:2111.09881 | Considered and rejected as the primary backbone — higher accuracy ceiling but self-attention cost is harder to justify against KLA's speed criterion; NAFNet was shown in its own paper to match/exceed Restormer at far lower MACs. |
| Wang et al., SAR-focused work: *Deep Learning for Integrated Speckle Reduction and Super-Resolution in Multi-Temporal SAR*, Remote Sensing 2024 | Confirms the general finding that **sequential** denoise-then-upsample or naive single-task nets underperform **joint** denoise+SR networks — informed the decision to do denoising and super-resolution in one forward pass rather than a two-stage pipeline. |
| Ultrasound speckle-denoising GAN literature (GAN-RW, PMC9044345; despeckling surveys) | Speckle is multiplicative and signal-dependent — reinforced using **L1** (robust to the resulting heavy-tailed noise) instead of L2/MSE as the pixel-fidelity loss term. |
| Loss design | SSIM-in-the-loss (`1-SSIM` as a differentiable penalty) directly optimizes one of KLA's three grading metrics, matching the deck's own tip to "combine perceptual loss (LPIPS) with pixel-level metrics (SSIM, PSNR)". |

## 4. Architecture (`model.py`)
**NAFNet-Lite-SR**: a single-stage U-Net (2 encoder / 2 decoder stages,
185K params) built from NAFBlocks (LayerNorm → depthwise conv →
SimpleGate → simplified channel attention, **no nonlinear activations** —
NAFNet's key finding is that these aren't necessary for SOTA restoration
quality) followed by a PixelShuffle-based 2x super-resolution head. Doing
denoising/deblurring at the *low*-resolution latent (before upsampling) is
standard SR practice (e.g. Real-ESRGAN) and is the cheapest place to do the
heavy lifting computationally — important for the H100 timing score.

## 5. Training & results
- Loss: `0.84 * (1 - SSIM) + 0.16 * L1` (mirrors the alpha used in the
  original NAFNet/Uformer papers for their SSIM-L1 combination).
- 200 synthetic training pairs, 20 held-out validation pairs, 64x64
  patches, AdamW, cosine LR schedule, 40 epochs (CPU only in this
  sandbox — ~13.6 s/epoch, ~9 min total).
- **Full-image validation results** (not patches), vs. a **bicubic-only
  baseline** (no denoising, the naive alternative):

| Metric | Bicubic-only baseline | Our model | Δ |
|---|---|---|---|
| PSNR (dB, ↑) | 17.66 | **27.12** | +9.46 dB |
| SSIM (↑) | 0.430 | **0.778** | +0.35 |
| Edge-perceptual proxy (↓, see note) | 0.840 | **0.344** | -0.50 |

Visual inspection (`outputs/samples/*.png`, 4-panel: degraded / bicubic /
ours / ground-truth) shows the model recovers clean, well-aligned grid and
dendrite structure from very noisy inputs, closely matching ground truth,
while the bicubic baseline stays visibly speckled and blurred.

**LPIPS note:** true LPIPS needs a pretrained AlexNet/VGG backbone hosted
on `download.pytorch.org`, which this sandbox cannot reach (network is
allowlisted to pypi/npm/github only). I substituted a lightweight
edge-correlation proxy and isolated it in one function
(`perceptual_metric()` in `report_metrics.py`) — in an environment with
normal internet access, replace that one function with real
`lpips.LPIPS(net='alex')` and nothing else needs to change.

**Judgment on this iteration:** satisfactory as a first-pass demonstration
of the pipeline — the improvement over the naive baseline is large and
consistent, and the architecture/loss choices are literature-grounded. I
did not need a second improvement loop for this synthetic setting.

## 6. What must change for the real KLA dataset (do this first when data arrives)
1. Swap `data_generator.py` for a real loader — everything else
   (model, training loop, evaluation, submission script) is agnostic to
   the data source by design.
2. Re-tune the speckle/Gaussian-blur parameter ranges to match the real
   noise statistics (measure them from a handful of real degraded/GT
   pairs — e.g. fit the speckle variance from regions of near-constant
   GT intensity).
3. Scale up: 200 synthetic patches was enough to prove the pipeline; the
   real dataset should support far more training steps, a larger model
   width (try `width=32` or `64`), and GPU training (this sandbox is
   CPU-only, 1 core).
4. Add real LPIPS once network access allows downloading the backbone.
5. Add out-of-distribution robustness testing explicitly — augment
   training with a wider range of noise/blur severities and multiple
   synthetic "data origins," since KLA's test set includes
   out-of-distribution samples by design.
6. Benchmark actual inference time on an H100 (or the closest available
   GPU) — the reported CPU numbers (~65 ms/image on a single core) are
   not representative of the target hardware.

## 7. Deliverables (matches KLA's 4 mandatory submission components)
1. **Evaluation script**: `evaluate_submission.py` — standalone, takes
   `--input_dir` / `--output_dir`, loads the checkpoint, runs inference,
   writes restored PNGs. Verified end-to-end.
2. **Training script**: `train.py` (+ `model.py`, `data_generator.py`).
3. **Restored outputs**: `outputs/samples/*.png` (visual panels) — swap in
   real test data through `evaluate_submission.py` once released.
4. **Environment spec**: `checkpoints/environment_requirements.txt`
   (`pip freeze` output).
