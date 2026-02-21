# NanoLILY: Ultra-Lightweight CNN-FFT Fusion for Low-Light Image Luminosity Yield

> A highly efficient dual-branch CNN-FFT architecture for restoring structural integrity and perceptual quality in low-light images with minimal parameter overhead.

![Hero Image Placeholder](results/hero_comparison.png) 
*(Before / Model Output / Ground Truth comparison image here is yet to be placed)*

## Overview
This repository contains an independent deep learning experiment focused on architectural efficiency in computer vision. **NanoLILY** tackles the problem of low-light image enhancement not by scaling up massive transformer blocks, but by combining a microscopic spatial CNN with a globally learnable Fast Fourier Transform (FFT) mask. 

Operating at just **129K trainable parameters**, this architecture bypasses the massive memory costs of traditional self-attention mechanisms while achieving competitive structural recovery on low-light datasets.

### Key Features
* **Extreme Efficiency:** Only 129K parameters, making it highly suitable for resource-constrained edge devices.
* **Dual-Branch Architecture:** Utilizes a shallow spatial CNN branch to capture local pixel textures, running in parallel with an FFT-based frequency branch (`torch.fft.rfft2`) to capture global illumination context.
* **Perceptual Optimization:** Trained using a composite loss function blending VGG-19 Perceptual Loss, Structural Similarity (SSIM), and L1 Pixel loss to ensure outputs are both structurally sound and visually pleasing.

---

## 🧠 Training Strategy: Two-Stage Pipeline
To maximize the model's ability to generalize across different low-light conditions, the training was executed in a two-phase pipeline:

1. **Phase 1: Baseline Pre-Training (LoL v1 Dataset)**
   * **Notebook:** `yet to be added`
   * The model was trained from scratch for 200 epochs on the standard LoL v1 dataset to learn the fundamental mapping between low-light degradation and target illumination.
   * **Dynamic Loss Strategy:** SSIM weight was intentionally kept low (`0.01`) for the first 50 epochs to focus strictly on pixel-level illumination recovery (PSNR/L1). At epoch 51, the weight was scaled up 20x to `0.20` to refine structural edges and focus on perceptual quality once the baseline brightness was achieved.

2. **Phase 2: Fine-Tuning & Adaptation (LoL v2 Dataset)**
   * **Notebook:** `yet to be added`
   * The pre-trained baseline was fine-tuned on the LoL v2 dataset to adapt to its specific noise profiles and distribution shifts.
   * **Refined Optimization:** The SSIM weight was strictly maintained at `0.2` from epoch 0. The learning rate was managed via a `ReduceLROnPlateau` scheduler, and training was monitored with an Early Stopping callback (patience=10, min_delta=0.00001) targeting maximum `val_ssim`.

---

## Performance Metrics
Baseline evaluation yields highly efficient structural recovery relative to the minimal parameter count (Fine-tuning run, Early Stopping triggered at Epoch 16):

* **Validation SSIM:** ~0.797
* **Validation PSNR:** ~18.64 dB
* **Validation Loss:** ~0.168

*(Note: See the `results/` folder for extended visual comparisons across various lighting conditions).*

---

## Repository Structure

```text
NanoLILY/
├── dataset/            # Training and validation datasets (LoL v1 & LoL v2 Real_captured)
├── notebook/           # Core code: Pre-training and fine-tuning Jupyter notebooks
├── model/              # Saved PyTorch checkpoints (.pth)
├── results/            # Visual comparisons (Before/After/Ground Truth)
├── other docs/         # Deep-dive documentation (Architecture maps, training logs)
├── README.md           # Project overview and quickstart
└── requirements.txt    # Python dependencies