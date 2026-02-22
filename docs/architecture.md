# Architecture & Tensor Flow Details

This document outlines the specific mathematical and structural flows of the NanoLILY core model and the LILY Bloom inference engine.

---

## I. NanoLILY Core (The Neural Network)
NanoLILY is a dual-branch architecture designed to decouple local texture extraction from global illumination mapping. It operates strictly on `224x224` spatial dimensions during training.

### 1. The Spatial CNN Branch (Local Textures)
This branch consists of lightweight convolutions that maintain the spatial resolution while extracting pixel-level features (edges, boundaries, and noise patterns).
* **Input:** `(B, 3, 224, 224)` 
* **Process:** Depthwise separable convolutions (to maintain the 129K parameter constraint) with LeakyReLU activations.
* **Output:** `(B, C, 224, 224)` where `C` is the latent channel dimension.

### 2. The Global FFT Branch (Illumination Mapping)
This branch shifts the image into the frequency domain to adjust the global lighting context without requiring memory-heavy self-attention matrices.
* **Input:** `(B, 3, 224, 224)`
* **Transform:** `torch.fft.rfft2` converts the spatial tensor into complex frequency representations.
* **Process:** A learnable complex-valued weight matrix scales the amplitude of the low-frequency components (the DC component representing baseline brightness) while preserving high-frequency phase data.
* **Inverse Transform:** `torch.fft.irfft2` brings the manipulated frequencies back to the spatial domain.
* **Output:** `(B, C, 224, 224)`

### 3. Fusion & Residual Reconstruction
* The outputs of both branches are concatenated along the channel dimension: `(B, 6, 224, 224)` (3 channels from spatial + 3 from frequency).
* A `1x1` convolution fuses these features back to standard RGB space.
* A `Tanh()` activation bounds this output to `[-1, 1]`, representing the **residual illumination map**.
* **Final Output:** The residual map is added directly to the original input tensor (`identity + residual`). This residual connection allows the network to learn only the necessary lighting adjustments, preserving the original structural details without having to reconstruct the entire image from scratch.

---

## II. LILY Bloom Engine (The Inference Pipeline)
Because the learnable weights in the FFT branch are mathematically locked to the `224x224` grid, NanoLILY cannot natively process 1080p or 4K images. LILY Bloom handles this spatial scaling.

### 1. Sliding Window Extraction
Given an arbitrary high-resolution input `(1, 3, H, W)`:
* The image is zero-padded symmetrically to ensure the dimensions are perfectly divisible by the `stride` (e.g., `112` pixels).
* The engine extracts overlapping `224x224` patches. With a stride of 112, every pixel is processed by the NanoLILY core approximately 4 times, ensuring robust local context.

### 2. Seamless 2D-Hanning Blending
To eliminate the "checkerboard" grid lines caused by slight global illumination differences between adjacent patches:
* A 2D Hanning Window is generated: `(1, 3, 224, 224)`.
* This mathematical bell curve acts as an alpha mask. The center of every processed patch retains 100% opacity, while the edges gradually fade to 0%.
* As the patches are stitched back into the empty `(1, 3, H, W)` canvas, they are multiplied by this window and accumulated into a global weight map.

### 3. Normalization & Output
* The accumulated canvas is divided by the overlapping weight map to normalize pixel intensities: `Output = Accumulated_Pixels / Weight_Map`.
* The padded edges are cropped, resulting in the final, full-resolution tensor `(1, 3, H, W)` with zero visible seam lines.