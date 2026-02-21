# NanoLILY: Ultra-Lightweight Model for Low-Light Image Luminosity Yield

> A highly efficient dual-branch CNN-FFT architecture for restoring structural integrity and perceptual quality in low-light images with minimal parameter overhead.

<img width="976" height="506" alt="image" src="https://github.com/user-attachments/assets/5b057cf9-d2eb-4d52-99f4-69f95bbddd39" />


## Overview
This repository contains an independent deep learning experiment focused on architectural efficiency in computer vision. **NanoLILY** tackles the problem of low-light image enhancement not by scaling up massive transformer blocks, but by combining a microscopic spatial CNN with a globally learnable Fast Fourier Transform (FFT) mask. 

Operating at just **129K trainable parameters**, this architecture bypasses the massive memory costs of traditional self-attention mechanisms while achieving competitive structural recovery on low-light datasets.

### Key Features
* **Extreme Efficiency:** Only 129K parameters, making it highly suitable for resource-constrained edge devices.
* **Dual-Branch Architecture:** Utilizes a shallow spatial CNN branch to capture local pixel textures, running in parallel with an FFT-based frequency branch (`torch.fft.rfft2`) to capture global illumination context.
* **Perceptual Optimization:** Trained using a composite loss function blending VGG-19 Perceptual Loss, Structural Similarity (SSIM), and L1 Pixel loss to ensure outputs are both structurally sound and visually pleasing.

---

## 📂 Repository Structure

```
NanoLILY/
├── dataset/                     # Training and validation datasets (LoL v1 & LoL v2)
├── notebooks/                   # Core code: Pre-training, fine-tuning, and inference
├── model/                       # Saved PyTorch Lightning checkpoints (.ckpt / .pth)
├── results/                     # Visual evaluations and metrics
│   ├── sample-wise-evaluation/  # 40+ generated Before/After/Target grids
│   └── all-results.md           # Scrollable markdown gallery of all test samples
├── docs/                        # Deep-dive documentation (Architecture maps, logs)
├── README.md                    # Project overview and quickstart
└── requirements.txt             # Python dependencies
```

---
## 🧠 Training Strategy: Two-Stage Pipeline
To maximize the model's ability to generalize across different low-light conditions, the training was executed in a two-phase pipeline:

1. **Phase 1: Baseline Pre-Training (LoL v1 Dataset)**
   * **Notebook:** [NanoLILY.ipynb](https://github.com/Sanjeet2835/NanoLILY/blob/main/notebook/NanoLILY%20baseline.ipynb)
   * The model was trained from scratch for 200 epochs on the standard LoL v1 dataset to learn the fundamental mapping between low-light degradation and target illumination.
   * **Dynamic Loss Strategy:** SSIM weight was intentionally kept low (`0.01`) for the first 50 epochs to focus strictly on pixel-level illumination recovery (PSNR/L1). At epoch 51, the weight was scaled up 20x to `0.20` to refine structural edges and focus on perceptual quality once the baseline brightness was achieved.

2. **Phase 2: Fine-Tuning & Adaptation (LoL v2 Dataset)**
   * **Notebook:** [Fine tuning NanoLILY.ipynb](https://github.com/Sanjeet2835/NanoLILY/blob/main/notebook/Fine%20tuning%20NanoLILY.ipynb)
   * The pre-trained baseline was fine-tuned on the LoL v2 dataset (Real_captured) to adapt to its specific noise profiles and distribution shifts.
   * **Refined Optimization:** The SSIM weight was strictly maintained at `0.2` from epoch 0. The learning rate was managed via a `ReduceLROnPlateau` scheduler, and training was monitored with an Early Stopping callback (patience=10, min_delta=0.00001) targeting maximum `val_ssim`.

---

## Performance Metrics
Evaluation yields highly efficient structural recovery relative to the minimal parameter count:

* **Validation SSIM:** ~0.797
* **Validation PSNR:** ~18.64 dB
* **Validation Loss:** ~0.168
  
---

## Visual Results
Below are a few sample outputs demonstrating the model's ability to recover illumination and structure.


### low00736

![eval_low00736.png](results/sample-wise-evaluation/eval_low00736.png)

---

### low00742

![eval_low00742.png](results/sample-wise-evaluation/eval_low00742.png)

---

### low00748

![eval_low00748.png](results/sample-wise-evaluation/eval_low00748.png)

---

### low00756

![eval_low00756.png](results/sample-wise-evaluation/eval_low00756.png)

---

### low00758

![eval_low00758.png](results/sample-wise-evaluation/eval_low00758.png)

---



[➡️ Click Here to View the Full Gallery of 40+ Evaluation Samples](https://github.com/Sanjeet2835/NanoLILY/blob/main/results/all-results.md)
