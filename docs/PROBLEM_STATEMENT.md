# AI-Based Restoration of Degraded Images for Semiconductor Inspection

> **Problem Statement:** KLA / i4C Hackathon — PS01
> **Source:** Official problem statement, reproduced and reformatted for readability. Content is unchanged.

---

## Table of Contents

1. [Background](#1-background)
2. [Description](#2-description)
3. [Training Data — What You Get](#3-training-data--what-you-get)
4. [Test Data — What Comes Later](#4-test-data--what-comes-later)
5. [Submission Requirements](#5-submission-requirements)
   - [Component 1 — PPT/PDF Submission](#component-1--pptpdf-submission)
   - [Component 2 — GitHub Repository](#component-2--github-repository-mandatory)
6. [Requirements Checklist](#6-requirements-checklist)

---

## 1. Background

In semiconductor manufacturing, microscopic inspection images are used to measure and verify chip quality at every stage of production. These images must be extremely sharp and clean, because **a single pixel of noise or a small loss of detail can hide a defect that causes a chip to fail.**

In practice, inspection images are degraded by two types of signal loss:

**Speckle Noise**
Random pixel-level noise that makes the image look "grainy." This noise can actually push pixel values *beyond the true image range*, meaning some pixels appear brighter or darker than they actually are in reality.

**Spatial Resolution Reduction**
The image has been "shrunk" (downsampled), losing fine detail. A 512×512 pixel ground truth image becomes a blurry 256×256 pixel image, or a 256×256 becomes 128×128. Details that were visible at full resolution are gone.

Currently, engineers work with these degraded images and make do with the noise and lost detail. AI-powered restoration can recover information that appears lost — removing noise while sharpening detail back to the original resolution.

---

## 2. Description

You will receive a training dataset of **paired images**: for each sample, you get a degraded image (noisy + low resolution) and the corresponding ground truth image (clean + full resolution).

Your job is to train an AI model that learns to **reverse the degradation** — taking a bad image as input and producing a restored image that matches the ground truth as closely as possible.

### 2.1 Degradation Types Your Model Must Handle

| Degradation Type | What It Looks Like | What Your Model Must Fix |
| :--- | :--- | :--- |
| **Speckle Noise** | Random pixel-level noise, image looks grainy. Some pixel values are pushed beyond the true range of the image. | Remove the grain while preserving the real image details underneath. **Do not blur the image to remove noise** — that destroys useful information. |
| **Gaussian Noise** | Image appears soft and hazy; edges and fine structures lose sharpness. | Restore edge sharpness and contrast **without introducing artificial patterns or ringing.** |
| **Spatial Resolution Reduction** (Super-Resolution) | Image has been downsampled: 512×512 → 256×256, or 256×256 → 128×128. Fine details are lost. | Upscale the image back to the original resolution (512×512 or 256×256) while **reconstructing the fine details** lost during downsampling. |

### 2.2 Hard Constraints

> **Simultaneous degradation.** Your model must handle ALL degradation types simultaneously — a single image may have speckle noise AND reduced resolution at the same time.

> **Out-of-distribution generalization.** The test set will include images from **different sources** than the training data. Your model must generalize, not just memorize the training examples.

> **Speed matters.** Your model will be benchmarked on inference time. A model that produces great results but takes 10 minutes per image is less useful than one that produces good results in 10 seconds.

---

## 3. Training Data — What You Get

KLA will provide a paired training dataset. For each sample, you receive:

| Item | Resolution | Description |
| :--- | :--- | :--- |
| **Ground Truth Image** (clean, full resolution) | 512×512 or 256×256 px | The "correct answer" — the image as it should look. High resolution, high signal-to-noise ratio. This is what your model's output should match. |
| **Degraded Image** (noisy, low resolution) | 256×256 or 128×128 px | The input to your model — noisy and downsampled. Your model takes this as input and must produce an output that matches the ground truth. |

### 3.1 Important Data Notes

- **Intensity range may exceed ground truth.** The degraded image intensity range may EXCEED the ground truth range. This is expected behaviour, caused by speckle noise pushing pixel values beyond the original signal. *Your model must handle this.*
- **Diverse data origins.** The images come from different types of semiconductor structures. Your model should generalize across these variations, not overfit to one type.
- **Grayscale only.** Images are single channel. Colour images are NOT part of this challenge.

---

## 4. Test Data — What Comes Later

After the training phase, KLA will release a test dataset containing:

- **In-distribution samples** — images similar to what you trained on. *Tests accuracy.*
- **Out-of-distribution samples** — images from different sources than the training data. *Tests generalization and robustness* — whether your model can handle image types it has never seen.

---

## 5. Submission Requirements

All submissions are made through the **i4C hackathon portal**. You must submit a PPT/PDF using the provided Idea Submission Template **AND** a GitHub repository link.

### Component 1 — PPT/PDF Submission

Use the provided Hackathon Idea Submission Template. Fill in each slide as follows:

| Slide | What to Fill In |
| :--- | :--- |
| **1. Team Details** | Team name, member names, roles, college name, contact details. |
| **2. Problem Statement Addressed** | Select "AI-Based Restoration of Degraded Images." Describe in your own words why this problem matters in semiconductor manufacturing. |
| **3. Idea Description** | Your key concept and approach — what type of AI model did you choose? Why? How does it address all 3 degradation types (speckle, Gaussian, super-resolution)? |
| **4. Proposed Solution** | Detailed solution — model architecture, training strategy, loss function design, data augmentation approach. **Include a system/pipeline diagram.** |
| **5. Innovation & Uniqueness** | What makes your approach different? Did you design a novel loss function? A unique data augmentation strategy? A faster inference pipeline? |
| **6. Results** | SSIM, PSNR, LPIPS scores on your test split. Before/after image comparisons (degraded input → your restored output → ground truth). Confusion-free visual evidence that your model works. |
| **7. Technology & Feasibility** | Tech stack used (PyTorch/TensorFlow/other), hardware used for training (GPU type, cloud platform), training time, model size, inference time per image. |
| **8. GitHub & Video Link** | GitHub repository link (**mandatory**). Video link showing your model running (optional but recommended). |
| **9. References** | Research papers, datasets, tools referenced. |

**Format rules:**
- Save as **PDF** before uploading.
- File naming convention: `TeamName_KLA_PS01` (e.g. `VisionForge_KLA_PS01.pdf`).
- Maximum **8–9 slides**.
- Remove the instruction slide.

### Component 2 — GitHub Repository (Mandatory)

Your repository must be **public** and must contain:

| # | Content | Details |
| :-- | :--- | :--- |
| 1 | **README.md** | Complete setup instructions. *A reviewer must be able to clone your repo and run inference from the README without contacting you.* |
| 2 | **Evaluation Script** (standalone `.py`) | A Python script (**NOT** a Jupyter notebook) that accepts (a) path to test images directory, (b) path to output directory. It loads your trained model, runs inference on all input images, and writes restored outputs to the specified directory. **Must run without manual edits.** |
| 3 | **Training Script** | Python script or Jupyter notebook that reproduces your training process from scratch. |
| 4 | **Trained Model Weights** | Your final trained model file, in a format your evaluation script can load (`.pt`, `.onnx`, `.h5`, etc.). Must be downloadable — use Git LFS, or link to Google Drive / HuggingFace if the file is large. |
| 5 | **Restored Test Outputs** | A folder containing your model's output on the test set — the actual restored images your model produced. |
| 6 | **requirements.txt** | Complete `pip freeze` output from your training environment. Required for reproducibility. |

---

> ### ⚠️ CRITICAL
>
> **The evaluation script is the most important file in your repository.** It will be used **AS-IS** by KLA's benchmarking team to measure your model's quality scores and inference time on the **H100 GPU**.
>
> If your script does not run without manual edits, your submission **cannot be benchmarked** — and unscored submissions cannot win.
>
> **Test your script on a fresh machine before submitting.**

---

## 6. Requirements Checklist

Extracted from the above for tracking purposes.

**Model requirements**
- [ ] Handles speckle noise (including out-of-range pixel values)
- [ ] Handles Gaussian blur / softening
- [ ] Handles 2× super-resolution (512→256 and 256→128 cases)
- [ ] Handles all degradations *simultaneously* in one pass
- [ ] Generalizes to out-of-distribution sources
- [ ] Optimized for inference speed on H100
- [ ] Grayscale, single channel

**Deliverables — repository**
- [ ] Public GitHub repository
- [ ] `README.md` with complete, self-sufficient setup instructions
- [ ] Standalone evaluation script accepting input dir + output dir
- [ ] Evaluation script runs with **zero** manual edits
- [ ] Training script
- [ ] Trained model weights (downloadable)
- [ ] Folder of restored test outputs
- [ ] `requirements.txt`

**Deliverables — presentation**
- [ ] 8–9 slides, instruction slide removed
- [ ] Exported as PDF, named `TeamName_KLA_PS01.pdf`
- [ ] System/pipeline diagram included (Slide 4)
- [ ] SSIM, PSNR **and** LPIPS reported (Slide 6)
- [ ] Before/after visual comparisons (Slide 6)
- [ ] Hardware, training time, model size, inference time (Slide 7)
- [ ] GitHub link (Slide 8)
- [ ] References (Slide 9)

**Scoring axes**
- [ ] Accuracy on in-distribution samples
- [ ] Robustness on out-of-distribution samples
- [ ] Inference time on H100
