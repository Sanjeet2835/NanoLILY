# Architecture & Tensor Flow Details

This document outlines the specific mathematical and structural flows of the NanoLILY core model and the LILY Bloom inference engine.

---

## I. NanoLILY Core (The Neural Network)
NanoLILY is a dual-branch architecture designed to decouple local texture extraction from global illumination mapping. It operates strictly on `224x224` spatial dimensions during training and maintains an ultra-lightweight 129K parameter footprint.

### 1. The Spatial CNN Branch (Local Textures & Edges)
This branch acts as a spatial magnifying glass. It hunts for the ultra-faint edge gradients and textures that are technically present in the dark pixels but invisible to the human eye.
* **Input:** `(B, 3, 224, 224)` — Note: PyTorch normalizes these image tensors to a strict `[0.0, 1.0]` scale (where 0.0 is pitch black and 1.0 is pure white).
* **Process:** Standard 2D spatial convolutions coupled with Instance Normalization and LeakyReLU activations. The features are processed through a shallow encoder-decoder step with a skip connection (`up1 + s1`) to prevent spatial degradation.
* **Output:** A final convolution collapses the latent features back down to exactly 3 RGB channels: `(B, 3, 224, 224)`.

### 2. The Global FFT Branch (Illumination Mapping)
This branch shifts the image into the frequency domain to adjust the global lighting context, completely bypassing the need for memory-heavy self-attention matrices.
* **Input:** `(B, 3, 224, 224)`
* **Transform:** `torch.fft.rfft2` converts the spatial tensor into complex frequency representations.
* **Process:** A learnable, real-valued weight matrix `(1, 3, 224, 113)` initialized at `0.01` acts as an amplitude scalar. Because it is a real-valued scalar applied to complex numbers, it adjusts global brightness and contrast magnitudes while strictly preserving the high-frequency structural phase data.
* **Inverse Transform:** `torch.fft.irfft2` brings the manipulated frequencies back to the spatial domain.
* **Output:** `(B, 3, 224, 224)` (3 RGB channels).

### 3. Fusion & Residual Reconstruction (The Enhancement Map)
This is where the local textures and global lighting shifts are combined into a final mathematical mask.
* **Concatenation:** The outputs of both branches are concatenated along the channel dimension: `(B, 6, 224, 224)` (3 channels from spatial + 3 from frequency).
* **Fusion:** A `1x1` convolution fuses these 6 features back to standard 3-channel RGB space.
* **Activation:** A `Tanh()` activation bounds this output to strictly `[-1.0, 1.0]`. This tensor is not just a lighting mask; it is a highly complex **Enhancement Map** containing both the global illumination shifts (from the FFT) and the amplified high-frequency textures (from the CNN).
* **Final Output (`identity + residual`):** The Enhancement Map is added directly to the original dark input tensor. 
  * *The Scale Math:* Because the original input is normalized to `[0.0, 1.0]`, a residual in the range of `[-1.0, 1.0]` has massive mathematical power. If a dark pixel is `0.05` and the network outputs a residual of `+0.85`, that pixel instantly shifts to a brightly illuminated `0.90`. 
  * *The Benefit:* This residual connection injects the recovered textures and lighting directly into the image, preventing the network from having to destructively reconstruct the baseline structural details from scratch.

*(Note: To see the step-by-step tensor visualization and mathematical proof of this branch separation, see the [Deep Dive Documentation](deep_dive.md)).*

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