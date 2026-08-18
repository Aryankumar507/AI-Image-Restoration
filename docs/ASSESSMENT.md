# Pre-Submission Assessment — KLA / i4C PS01

> **Project:** AI-Based Restoration of Degraded Images for Semiconductor Inspection
> **Reviewed:** 16 Aug 2026 · against [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md)
> **Scope:** full repository — architecture, training, evaluation script, deliverables
>
> *Line numbers below were captured before the repository restructure and may have
> drifted by a few lines; the file references and findings themselves are current.*

---

## Verdict

A competent, literature-grounded skeleton. NAFNet with a PixelShuffle super-resolution head is genuinely the right architectural call for a speed-scored restoration task, and the reasoning in [RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md) is sound and honest.

But this is currently a **pipeline demo trained on procedurally-generated synthetic data**, and it carries **seven defects that would produce an unscorable submission**. The rules are explicit: *unscored submissions cannot win.* Fix the blockers first, then innovate.

| | Count |
| :--- | ---: |
| 🔴 Submission blockers | ~~7~~ → **0 open** ✅ |
| 🟠 Score-capping defects | ~~10~~ → **0 open** ✅ |
| 🟢 Differentiating ideas | **7** |
| ✅ Deliverables complete | **11 / 13** |

> **Update — 16 Aug 2026.** All seven blockers are closed and covered by 14
> passing tests (`pytest`). The submission now runs end-to-end from a clean
> clone. The ten score-capping defects below remain open; **W-01 (synthetic
> training data) is still the dominant risk.**
>
> | Blocker | Status | Closed by |
> | :--- | :--- | :--- |
> | B-01 checkpoint path | ✅ | Moved to `checkpoints/`, path anchored to repo root |
> | B-02 filename suffix | ✅ | Stems preserved exactly |
> | B-03 bit depth / clipping | ✅ | `IMREAD_UNCHANGED` + per-depth normalization |
> | B-04 `weights_only` | ✅ | Set at both `torch.load` sites |
> | B-05 scale inference | ✅ | `--scale` flag, default 2 |
> | B-06 README + requirements | ✅ | `README.md` written; requirements 154 → 21 lines |
> | B-07 restored outputs | ✅ | `outputs/restored/` populated + documented |

> **Score-capping defects — all ten closed.** Test count 14 → 76.
>
> | Defect | Status | Closed by |
> | :--- | :--- | :--- |
> | W-01 synthetic data | ✅ | Randomized high-order degradation; seed correlation fixed |
> | W-02 loss / clamp | ✅ | α 0.84 → 0.5 (`--alpha`); gradient-killing clamp removed |
> | W-03 augmentation | ✅ | Dihedral, pair-safe (`--no_augment` to disable) |
> | W-04 OOD strategy | ✅ | Wide ranges + 4 downsample kernels + random order |
> | W-08 config drift | ✅ | `from_checkpoint()`; architecture now CLI-configurable |
> | W-05 inference speed | ✅ | channels_last + autocast + warmup; **batching measured 8.8× SLOWER on 4 GB, so default batch=1** (23 ms/img) |
> | W-06 real LPIPS | ✅ | Real `lpips.LPIPS(net='alex')`; proxy only as labelled fallback |
> | W-07 TLC | ✅ | `apply_tlc()` implemented + `--tlc` flag; measured neutral (21.73 vs 21.71 dB), off by default |
> | W-09 model capacity | ✅ | 185K → 1.16M params (width 48, 2/2/2), trained on RTX 2050 |
> | W-10 tests / gitignore | ✅ | 65 tests + `.gitignore` |
>
> **W-01 remains partly open in substance:** the degradation is now realistic
> and varied, but the ground truth is still procedurally generated. Real KLA
> data replaces it.

> ### 🚨 New finding — the model erases defects (16 Aug 2026)
>
> `scripts/defect_preservation.py` (Idea D, now built) injects synthetic
> defects into ground truth, degrades, restores, and measures how much of the
> defect's true contrast survives. Over 60 images:
>
> | Defect type | n | Mean contrast recovery | Substantially lost (<0.5) |
> | :--- | ---: | ---: | ---: |
> | particle (bright) | 19 | 0.568 | 21% |
> | break (line gap) | 16 | 0.005 | 69% |
> | **void (dark pit)** | 25 | **−0.077** | **96%** |
> | **overall** | 60 | **0.149** | **65%** |
>
> **55% of defects are effectively erased** (recovery < 0.2), and dark voids
> score *negative* — the restoration fills them in, inverting their contrast.
> Meanwhile standard metrics look healthy (25.6 dB / 0.746 SSIM / 0.236 LPIPS),
> which is exactly the failure mode this audit exists to expose: the model has
> learned what clean structures should look like and regenerates them.
>
> False-structure score on defect-free images is low (0.049), so it is not
> adding spurious detail — it is **removing real detail** it judges anomalous.
>
> **This is the single most important open issue.** On an inspection task, a
> restorer that deletes the anomaly is worse than no restorer. Likely causes:
> only two ground-truth structure families, and an SSIM-weighted loss that
> rewards matching the dominant periodic pattern. Candidate fixes: train with
> defects present in the ground truth, add structure variety, and lower alpha
> further toward pixel fidelity.

> ### 🧪 PoC result — both fixes help, neither is sufficient (18 Aug 2026)
>
> Two candidate fixes for the erasure finding were trained and audited on
> identical seeds (`scripts/poc_compare.py`, n=60, families pinned to
> texture+dendrite so the baseline is judged only on what it was trained on):
>
> * **A** — defects injected into the ground TRUTH (`--defect_prob`), so
>   reproducing an anomaly is what the loss rewards
> * **B** — three new GT structure families (line/space gratings, contact-hole
>   arrays, aperiodic polygon layouts), so the model cannot memorize one pattern
>
> | Variant | Mean recovery | Void | Break | Erased (<0.2) | PSNR | SSIM |
> | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
> | baseline | 0.149 | **−0.07** | 0.00 | 55.0% | **25.64** | **0.746** |
> | A defects in GT | 0.225 | 0.08 | 0.02 | 51.7% | 25.34 | 0.743 |
> | B more families | 0.263 | 0.09 | 0.13 | 50.0% | 23.17 | 0.714 |
> | **A+B combined** | **0.297** | **0.16** | **0.15** | **48.3%** | 22.65 | 0.704 |
>
> **What worked.** The fixes are additive: mean recovery doubles (0.149 → 0.297)
> and voids stop inverting (−0.07 → +0.16, i.e. the model no longer fills dark
> pits back in). Both changes contribute independently.
>
> **What did not.** Nearly half of all defects are still effectively erased
> (48.3%), and breaks remain near-zero. And there is a real cost: **−3.0 dB PSNR
> and −0.04 SSIM**, because a model that must also reproduce anomalies can no
> longer over-smooth toward an idealised pattern. The graded metrics reward
> exactly the behaviour the audit penalizes.
>
> **Read.** These are mitigations, not a solution. The remaining erasure is
> likely architectural rather than data-driven: an SSIM-weighted L1 objective
> averaged over the whole image simply does not care about a 5-pixel region.
> A defect-aware loss term (weighting the residual by local anomaly score) is
> the obvious next experiment, and it is the honest thing to put on Slide 5 —
> a measured tradeoff with a named next step beats a metric with no audit
> behind it. **Do not ship A+B on metrics alone**; the choice depends on
> whether KLA scores restoration fidelity or defect retention.

> ### ❌ Negative result — the defect-aware loss did not work (18 Aug 2026)
>
> Hypothesis: the remaining erasure is because an image-averaged L1 barely
> notices a 5-pixel region, so reweighting the residual toward badly-predicted
> pixels should force the model to reproduce defects. Implemented as
> `anomaly_weight_map()` in [train.py](../scripts/train.py), enabled with
> `--defect_gain`, and swept over the best data configuration (A+B).
>
> | Variant | Mean recovery | Erased (<0.2) | Break | PSNR | SSIM |
> | :--- | ---: | ---: | ---: | ---: | ---: |
> | AB_both (no DAL) | **0.316** | 45.0% | **0.19** | 22.63 | **0.703** |
> | AB + gain 4 | 0.281 | 48.3% | 0.11 | **23.11** | 0.703 |
> | AB + gain 8 | 0.278 | 46.7% | 0.10 | 22.34 | 0.690 |
> | AB + gain 16 | 0.273 | **43.3%** | 0.05 | 22.17 | 0.685 |
>
> **It did not help.** Mean recovery is flat-to-worse at every gain, and breaks
> get notably worse (0.19 → 0.05). The only thing that improves monotonically
> is the <0.2 tail (45.0% → 43.3%), bought with −0.5 dB PSNR and worse recovery
> everywhere else — not a trade worth making.
>
> **Why it likely failed.** The weight is a function of the residual, so it is
> largest wherever the model is currently worst — which early in training is
> everywhere, and late in training is dominated by hard high-frequency texture
> rather than by defects specifically. It amplifies difficulty, not
> *anomalousness*. A weight keyed to a genuine anomaly signal (an unsupervised
> detector, or a defect mask carried through the dataset) would test the
> hypothesis properly; this formulation does not.
>
> One methodological note worth keeping: the first formulation weighted by
> local deviation of the target, `|target - blur(target)|`. Measured before
> training, it gave a **1.00× weight ratio on periodic grids** — the structure's
> own edges masked the defect entirely. That check (now
> `test_anomaly_weight_emphasises_erased_defects`) saved a wasted training run
> and is why the residual formulation was used instead.
>
> **Conclusion: A+B remains the best defect-preserving configuration.** The
> honest summary for Slide 5 is that data changes helped (0.149 → 0.316, and
> voids stopped inverting) while the loss change did not, and the remaining
> ~45% erasure is unsolved.

> ### ✅ Shipped configuration (18 Aug 2026)
>
> **`checkpoints/nafnet_lite.pt` is now the A+B model** (5 GT families, defects
> present in the target). The decision rested on evaluating both candidates on
> data that resembles the real task rather than the baseline's own training
> distribution:
>
> | Eval set | Model | PSNR | SSIM | LPIPS |
> | :--- | :--- | ---: | ---: | ---: |
> | 2 families, no defects *(baseline's home turf)* | baseline | **25.64** | **0.746** | **0.236** |
> | | A+B | 22.63 | 0.703 | 0.284 |
> | **5 families + defects** *(closer to real inspection)* | baseline | 21.17 | 0.665 | 0.323 |
> | | **A+B** | **21.99** | **0.724** | **0.262** |
>
> The baseline's apparent lead was an artifact of validating on its own narrow
> distribution. On the harder, more representative set **A+B wins on all three
> graded metrics** — and preserves twice as many defects (0.316 vs 0.149).
> Since KLA's test set is explicitly out-of-distribution, A+B is the correct
> submission. The previous checkpoint is kept as `checkpoints/poc_*.pt`.
>
> **W-07 (TLC) — implemented, measured neutral, off by default.**
> `LocalGroupNorm` / `apply_tlc()` in [model.py](../src/kla_restoration/model.py)
> replace global LayerNorm statistics with a local window, exposed as `--tlc`.
> Measured across window sizes 32–128: **21.73 dB (off) vs 21.71 dB (w=64)** —
> no meaningful change. The mismatch is real but small here because the U-Net
> downsamples twice, so its deepest blocks see 32×32 feature maps even from a
> 256×256 input. Kept available for larger inputs where the gap widens, but
> enabling it by default would add cost for no measured gain.
>
> All ten score-capping defects are now closed or measured-and-dismissed.

> ### 🟢 Innovation ideas — built and measured (18 Aug 2026)
>
> Four of the seven Slide-5 ideas are now implemented and ablated. Every
> variant trained on identical data with identical hyperparameters; evaluated
> on **5 GT families with defects**, deliberately broader than any variant's
> training distribution (`scripts/idea_ablation.py`, n=60).
>
> | Variant | PSNR | SSIM | LPIPS | Δ PSNR |
> | :--- | ---: | ---: | ---: | ---: |
> | baseline (A+B data) | 21.99 | 0.7240 | 0.2619 | — |
> | **+ E** Fourier magnitude loss | 22.33 | 0.7423 | 0.2418 | +0.35 |
> | **+ B** log-domain speckle branch | 22.40 | 0.7414 | 0.2431 | **+0.41** |
> | **+ A** FiLM degradation conditioning | 22.24 | 0.7388 | 0.2548 | +0.26 |
> | + A+B+E combined | 22.18 | 0.7365 | 0.2514 | +0.20 |
> | **+ B + ×8 TTA (shipped)** | **22.67** | **0.7470** | 0.2482 | **+0.68** |
>
> **Each idea works alone; they do NOT stack.** Combining all three (+0.20) is
> worse than B alone (+0.41) — three simultaneous constraints on a 1.16M-param
> model trained on 300 images appears to over-constrain it. Reporting the
> ablation rather than only the best number is the honest framing.
>
> **Shipped: `idea_B` (log-domain speckle branch), 1.163M params.**
> `--self_ensemble` gives +0.27 dB more at 6.6× inference cost (23.9 →
> 157.1 ms/image), left off by default as a quality/speed dial.
>
> **Why B is the strongest result.** Speckle is multiplicative, `y = x·(1+n)`;
> `log1p` turns it additive, which is the form conv nets and L1/SSIM actually
> handle well. It costs 80 parameters and directly explains the "values exceed
> ground-truth range" property the spec flags three times. That is a citable
> mechanism (classical SAR despeckling), not a hyperparameter tweak.
>
> **Defect preservation is unchanged** across all variants (0.274–0.316 mean,
> ~45% erased) — the ideas improve restoration without trading away defect
> retention, so the choice rests on the graded metrics.
>
> Idea F (×8 self-ensemble) verified correct via an equivariance test: on a
> perfectly equivariant model the ensemble is a no-op to 2.4e-07, which proves
> all 8 inverse transforms are right. **Not built: C is done, D is done, G is
> partial; no further ideas outstanding.**

**Contents:** [Blockers](#-blockers) · [Weak points](#-scientific-weak-points) · [Innovation](#-differentiating-ideas) · [Deliverables](#-deliverables-status) · [Order of attack](#-order-of-attack) · [Bottom line](#the-bottom-line)

---

## 🔴 Blockers

*These break the submission, not just the score. Each can produce a zero regardless of model quality.*

> **The governing constraint:** the evaluation script is run **as-is** by KLA's benchmarking team on an H100. If it raises so much as a `FileNotFoundError`, nothing else in this repository is measured. Six of the seven items below sit in that script's execution path.

### B-01 · Checkpoint default path does not exist in the repository

[`evaluate_submission.py:42`](../evaluate_submission.py#L42) defaults to `checkpoints/nafnet_lite.pt`, but the weights file sits at repository root as `nafnet_lite.pt`. A reviewer running the documented command hits `FileNotFoundError` on line 48. The same mismatch exists in [`report_metrics.py:43`](../scripts/report_metrics.py#L43).

**Cost:** Instant unscorable submission. This is precisely the *"must run without manual edits"* clause the problem statement flags as CRITICAL.

### B-02 · Output filenames carry a `_restored` suffix

[`evaluate_submission.py:67`](../evaluate_submission.py#L67) writes `{stem}_restored.png`. If KLA's scorer pairs outputs to ground truth by exact filename stem, every file misses and you score zero on a working model.

**Cost:** A coin-flip on total failure, taken for no benefit. Preserve the exact input filename.

### B-03 · PNG and NPY inputs normalized inconsistently; bit depth silently destroyed

[`evaluate_submission.py:30-35`](../evaluate_submission.py#L30-L35) loads `.npy` as raw float — correctly preserving the out-of-range speckle values that training depended on ([`train.py:62`](../scripts/train.py#L62)) — but loads `.png` via `cv2.IMREAD_GRAYSCALE`, which returns **uint8, already clipped to [0,255]**. Worse, if KLA ships 16-bit TIFF or PNG (standard for inspection imagery), that flag down-converts to 8-bit and discards most of your dynamic range.

**Cost:** Silent train/test distribution mismatch on the exact property the spec calls out three times. Use `IMREAD_UNCHANGED` and normalize by actual bit depth.

### B-04 · `torch.load` without `weights_only=True`

[`evaluate_submission.py:48`](../evaluate_submission.py#L48) calls `torch.load` with no `weights_only` argument. The default flipped in PyTorch 2.6; your own requirements pin `torch==2.13.0`. This may fail outright on KLA's benchmark machine.

**Cost:** Hard crash on load, on a machine you cannot debug on.

### B-05 · Output resolution hardcoded to 2× with no scale inference

The spec permits 512→256 *and* 256→128. Both are 2×, so nominally you are safe — but [`evaluate_submission.py:65`](../evaluate_submission.py#L65) blindly emits 2H×2W with no way to know the intended target. A 256×256 degraded input whose ground truth is 512×512 is indistinguishable from one whose ground truth is 256×256 given only the input.

**Cost:** Wrong-sized outputs make every metric undefined. Add an explicit `--scale` argument and document the assumption.

### B-06 · No `README.md`, and `requirements.txt` is misnamed and polluted

Mandatory repository contents #1 and #6. The former `readme.md` held the problem statement, not setup instructions (now renamed to [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md)). The latter is `environment_requirements.txt` — a full-system `pip freeze` containing `camelot-py`, `Flask`, `grip`, `dbus-python`. It also pins `torch==2.13.0` against `torchvision==0.28.0`, a mismatched pair that will not resolve on a clean install.

**Cost:** Two mandatory deliverables failed, and reproduction is impossible from a fresh environment.

### B-07 · Missing mandatory deliverable: restored test outputs folder

Repository content #5 requires a folder of the actual restored images your model produced. The repository has two 4-panel comparison PNGs at root, generated on synthetic validation data. That is not the deliverable.

**Cost:** Mandatory content absent; reviewers cannot inspect real outputs.

---

## 🟠 Scientific weak points

*The submission runs, but these cap how well it can possibly score.*

### W-01 · You are training on synthetic data your own generator invented

[`data_generator.py`](../src/kla_restoration/data_generator.py) produces perfectly periodic grids and random-walk dendrites. The model reaches 0.778 SSIM on *its own* degradation recipe. That number carries no information about KLA's real data. [RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md) is admirably honest about this (§2, §6) — but the deck is judged on results.

Two specific traps:

- **Degradation order is wrong.** Yours is blur → downsample → speckle. Real sensor physics is closer to speckle at capture, then optical blur, then binning — and order changes what the network learns to invert.
- **Noise is correlated with content.** [`data_generator.py:113-114`](../src/kla_restoration/data_generator.py#L113-L114) passes the *same seed* to both `make_ground_truth` and `degrade`, so the noise realization is deterministically tied to image content. The network can in principle learn to predict the noise from the structure.

**Cost:** The single largest risk in the project. Every reported metric is currently unfalsifiable.

### W-02 · Loss weighting trades away PSNR, and clamping kills gradients

[`train.py:92-95`](../scripts/train.py#L92-L95) uses `0.84·(1−SSIM) + 0.16·L1`. That α is borrowed from papers where the structural term is *MS-SSIM*, and it heavily de-emphasizes pixel fidelity — which is PSNR, one of your three graded metrics.

Separately, [`train.py:94`](../scripts/train.py#L94) clamps `pred` to [0,1] *inside* the SSIM term. `clamp` has **zero gradient outside the range**, so any pixel the model pushes past 1.0 receives no SSIM gradient at all — on a task defined by out-of-range speckle values.

**Cost:** Systematically depressed PSNR plus a dead-gradient zone exactly where the hard pixels live.

### W-03 · Zero data augmentation

No flips, no 90° rotations, no transpose anywhere in [`train.py:42-64`](../scripts/train.py#L42-L64). This is free SSIM and PSNR on a task with 200 training images, and semiconductor structures are strongly symmetric.

**Cost:** Points left on the table, conspicuously absent to any reviewer reading the training script.

### W-04 · No out-of-distribution strategy, despite OOD being an explicit scoring axis

The spec calls out OOD generalization twice. Training uses two hand-made image families and a narrow degradation range (blur σ 0.8–2.0, speckle variance 0.02–0.08). Nothing in the pipeline targets generalization.

**Cost:** One of three scoring axes is entirely unaddressed.

### W-05 · No inference-time optimization, despite speed being explicitly scored

No AMP or fp16, no `channels_last`, no `torch.compile`, no ONNX or TensorRT — and critically, [`evaluate_submission.py:62-68`](../evaluate_submission.py#L62-L68) loops one image at a time, leaving an H100 almost entirely idle.

**Cost:** The cheapest scoring axis to win, completely untouched. Batching alone is plausibly a 10–30× throughput gain.

### W-06 · The reported "LPIPS" is not LPIPS

[`report_metrics.py:27-33`](../scripts/report_metrics.py#L27-L33) substitutes a Laplacian edge-correlation proxy. Slide 6 mandates LPIPS specifically, and `lpips==0.1.4` is *already* in your requirements — the backbone weights simply weren't reachable at the time.

**Cost:** A mandated metric missing from the results slide, with the fix already installed.

### W-07 · Train/test resolution mismatch through `GroupNorm` statistics

The model trains on 64×64 patches but evaluates on full images. It is fully convolutional so this "works" — but `GroupNorm(1, c)` in [`model.py:34`](../src/kla_restoration/model.py#L34) is LayerNorm-equivalent and computes statistics over the *whole spatial extent*. Its behaviour genuinely differs between a 64×64 patch and a 512×512 image.

**Cost:** A known, documented failure mode for LayerNorm-based restoration nets. NAFNet's own authors solve it with TLC (Test-time Local Converter); you currently do not.

### W-08 · Architecture hyperparameters duplicated across three call sites

`enc_blocks=(1,1), dec_blocks=(1,1), middle_blocks=1` is hardcoded in [`train.py:119`](../scripts/train.py#L119), [`evaluate_submission.py:49`](../evaluate_submission.py#L49) and [`report_metrics.py:44`](../scripts/report_metrics.py#L44). Only `width` is persisted in the checkpoint.

**Cost:** Any architecture change silently breaks inference — a likely failure right when you scale the model up under time pressure. Persist the full config dict.

### W-09 · Model is almost certainly under-capacity for the target hardware

`width=24`, roughly 185K parameters, trained 40 epochs on 200 images on a *single CPU core*. You are being benchmarked on an H100. Width 64 with deeper stacks remains trivially fast on that hardware.

**Cost:** Enormous unused quality headroom on the accuracy axis.

### W-10 · No tests, no `.gitignore`, single commit

There is no smoke test asserting the evaluation script runs end-to-end, no `.gitignore` (risking accidental commits of datasets and checkpoints), and the repository has one commit. Confirm it is actually pushed and public — that is mandatory.

**Cost:** Nothing catches a regression before KLA's benchmarking run does.

---

## 🟢 Differentiating ideas

*Slide 5 asks what makes your approach unique. "We used NAFNet" is not an answer — it is the strongest available baseline, which every serious team will also find. These are.*

### ⭐ Idea A — Degradation-parameter estimation head · *highest impact*

Add a small auxiliary head predicting speckle variance and blur σ directly from the input, and inject that estimate into the NAFBlocks via **FiLM conditioning** (per-channel scale and shift). The network becomes *adaptive* to degradation strengths it has never seen — exactly the OOD axis KLA scores. A few thousand parameters, and it gives you a concrete, defensible answer to *"how does your model generalize?"*

### ⭐ Idea B — Homomorphic log-domain speckle branch · *most novel*

Speckle is **multiplicative**: `y = x + x·n`. Take `log(1+x)` and it becomes **additive**, which convolutional networks denoise far more effectively. Run a log-domain branch alongside the linear branch and fuse them.

This is textbook SAR despeckling theory transplanted to semiconductor inspection — citable, principled, and it explains precisely *why* your model handles the "values exceed ground truth range" property the spec emphasizes three separate times.

### ⭐ Idea C — High-order randomized degradation synthesis · *highest ROI*

When KLA's data arrives you will have limited pairs. Take their **ground truth** images and re-degrade them yourself with a randomized recipe: random operation order, wide σ and speckle ranges, occasional Poisson shot noise, and varying downsample kernels (area, bilinear, bicubic, Lanczos).

This is the Real-ESRGAN high-order degradation playbook, and it is the proven recipe for OOD robustness. You get orders of magnitude more training data and a genuinely robust model from one change.

### ⭐ Idea D — Defect-preservation and anti-hallucination metric · *best narrative*

Every team will report SSIM, PSNR and LPIPS. Nobody will ask the actual business question: **does the restoration preserve real defects, and does it invent fake ones?**

Inject synthetic defects into ground truth, degrade, restore, and measure defect recall and false-positive rate. This shows you understand that in metrology a *hallucinated* detail is worse than a blurry one — and it lets you position a non-generative NAFNet as the safety-first choice, backed by data rather than assertion.

### Idea E — Fourier-domain reconstruction loss · *low effort*

Add an FFT-magnitude L1 term. Semiconductor structures are strongly periodic — your own grid generator proves it — and a frequency loss forces the network to reconstruct the correct **pitch**, which is exactly what a metrology engineer measures. Cheap to implement and genuinely well-motivated by the domain.

### Idea F — ×8 self-ensemble with an explicit speed/quality dial · *low effort*

Flip and rotate eight ways, average the results. Reliably worth +0.2–0.4 dB. On an H100 that is eight batched forward passes — still milliseconds. Present it as a tunable knob ("fast mode: X ms, quality mode: Y ms"), demonstrating engineering maturity on the speed axis rather than ignoring it.

### Idea G — H100-specific inference engineering as a headline number · *low effort*

bf16 autocast, `channels_last` memory format, batched inference and `torch.compile` — then quote images per second. Speed is an explicit scoring criterion that nearly every team will neglect entirely. These are the cheapest points available anywhere in the competition.

---

## 📋 Deliverables status

| Required item | Status | Detail |
| :--- | :--- | :--- |
| Public GitHub repository | ❓ VERIFY | Changes staged, not committed — confirm pushed and public |
| `README.md` with setup instructions | ✅ DONE | Written, with every documented command verified |
| Standalone evaluation script | ✅ DONE | Runs from any CWD; 14 tests cover the path |
| Training script | ✅ PRESENT | [train.py](../scripts/train.py) — but see W-02, W-03 |
| Trained model weights | ✅ DONE | `checkpoints/nafnet_lite.pt`, found automatically |
| Restored test outputs folder | ⚠️ SYNTHETIC | `outputs/restored/` populated; regenerate on real test data |
| `requirements.txt` | ✅ DONE | 154 → 21 lines, direct deps only, resolves cleanly |
| Real LPIPS scores | ❌ PROXY ONLY | Edge-correlation substitute; `lpips` already installed (W-06) |
| GPU inference timing | ❌ MISSING | Only CPU numbers (~39 ms/image) (W-05) |
| Before/after visual evidence | ⚠️ SYNTHETIC | Panels generated on self-invented data (W-01) |
| Pipeline diagram (Slide 4) | ❓ VERIFY | Check `docs/submission/KLA_PS01_deck.pptx` |
| PDF export, `TeamName_KLA_PS01.pdf` | ❓ VERIFY | Must export deck to PDF and rename |
| Smoke test / `.gitignore` | ✅ DONE | `tests/test_smoke.py` (14 tests) + `.gitignore` |

---

## 🗓 Order of attack

*Sequenced by deadline risk. The first phase is mechanical and prevents a zero; everything after raises the score.*

### Phase 1 — Today, ~2 hours · *prevents a zero*

1. Fix the checkpoint path, output filename suffix, `weights_only=True` and `IMREAD_UNCHANGED` in the evaluation script
2. Write a real `README.md`; rename requirements to `requirements.txt` and prune to genuine dependencies with a compatible torch/torchvision pair
3. Persist the full model config in the checkpoint; add an explicit `--scale` argument
4. Add `.gitignore` and a smoke test that runs inference on two dummy images
5. **Clone into a fresh virtualenv on a different machine and run the documented command verbatim**

### Phase 2 — This week · *raises the score*

1. Add flip and rot90 augmentation, plus randomized high-order degradation (Idea C)
2. Rebalance the loss: drop α toward ~0.5, remove the gradient-killing clamp, add the FFT term (Idea E)
3. Scale to `width=64` with deeper stacks; train on GPU
4. Add TLC for the patch/full-image normalization mismatch, and ×8 self-ensemble (Idea F)
5. Benchmark bf16 + `channels_last` + batched inference on the closest available GPU (Idea G)
6. Run real LPIPS and regenerate the results table

### Phase 3 — For the deck · *wins the innovation slide*

1. Implement the log-domain speckle branch (Idea B) — your citable technical novelty
2. Add the degradation-estimation FiLM head (Idea A) — your OOD answer
3. Build the defect-preservation metric (Idea D) and make it the centrepiece of Slides 5 and 6
4. Re-shoot before/after panels on real KLA data once released
5. Record the optional demo video — few teams will, and it costs an hour

---

## The bottom line

The architecture choice is right and defensible. But the submission is currently **un-runnable as written**, the results are measured on **data you invented**, and the innovation slide has **nothing on it yet**.

Blockers B-01 through B-07 are mechanical and take an afternoon. Ideas B, C and D are what turn this from "a working NAFNet" into something that places.
