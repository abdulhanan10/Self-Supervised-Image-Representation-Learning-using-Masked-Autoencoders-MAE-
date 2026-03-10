# 🎭 Masked Autoencoder (MAE) — Self-Supervised Image Representation Learning

> **Course:** Generative AI (AI4009) — Spring 2026
> **University:** National University of Computer and Emerging Sciences (FAST-NUCES)
> **Assignment:** No. 2

---

## 📌 Overview

This repository implements a **Masked Autoencoder (MAE)** from scratch using pure PyTorch — no external MAE libraries. The model learns rich visual representations by reconstructing images where **75% of patches have been randomly masked**, following the architecture proposed by He et al. (2022).

The system uses an **asymmetric encoder-decoder transformer design**:
- A **large ViT-Base encoder** that processes only the visible 25% of patches
- A **lightweight ViT-Small decoder** that reconstructs the full image from visible tokens + learnable mask tokens

---

## 🏗️ Architecture

```
Input Image (64×64)
      │
      ▼
 Patchify → 64 patches of 8×8
      │
      ▼
 Random Masking → keep 25% (16 patches)
      │
      ▼
 ┌─────────────────────────┐
 │  MAE Encoder (ViT-Base) │   ~86M params
 │  12 Transformer Blocks  │
 │  Hidden Dim: 768        │
 │  Heads: 12              │
 └─────────────────────────┘
      │  latent tokens (visible only)
      ▼
 Project 768 → 384 + Append Mask Tokens
      │
      ▼
 ┌────────────────────────────┐
 │  MAE Decoder (ViT-Small)   │   ~22M params
 │  6 Transformer Blocks      │
 │  Hidden Dim: 384           │
 │  Heads: 6                  │
 └────────────────────────────┘
      │
      ▼
 Pixel Prediction Head → Reconstructed Image (64×64)
```

---

## 📂 Repository Structure

```
├── MAE_Assignment_Kaggle.py        # Complete implementation (16 cells)
├── README.md                       # This file
└── outputs/
    ├── best_mae.pth                # Best model checkpoint
    ├── training_curves.png         # Loss & LR schedule plots
    ├── reconstruction_samples.png  # 5 qualitative reconstruction samples
    └── per_sample_metrics.png      # Per-sample PSNR & SSIM chart
```

---

## ⚙️ Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 64 × 64 |
| Patch Size | 8 × 8 |
| Total Patches | 64 |
| Mask Ratio | 75% |
| Visible Patches | 16 (25%) |
| Encoder Dim | 768 |
| Encoder Layers | 12 |
| Encoder Heads | 12 |
| Decoder Dim | 384 |
| Decoder Layers | 6 |
| Decoder Heads | 6 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Learning Rate | 1.5e-4 |
| Scheduler | Cosine Annealing + Warmup |
| Epochs | 5 |
| Batch Size | 64 (128 on dual GPU) |
| Mixed Precision | ✅ torch.cuda.amp |
| Gradient Clipping | 1.0 |

---

## 🚀 Running on Kaggle

### Step 1 — Open a new Kaggle Notebook
Go to [kaggle.com](https://kaggle.com) → **New Notebook**

### Step 2 — Enable GPU
Right panel → **Session Options** → **Accelerator** → Select **GPU T4 x1** or **GPU T4 x2**

### Step 3 — Copy cells into the notebook
Paste each `# CELL N` block from `MAE_Assignment_Kaggle.py` into a separate notebook cell.

### Step 4 — Run All
Click **Run All**. The full pipeline runs automatically:
```
Cell 1  → Install dependencies
Cell 2  → Imports & GPU setup
Cell 3  → Configuration
Cell 4  → Dataset download & preparation  (~10 sec)
Cell 5  → DataLoaders
Cell 6  → Building blocks (Attention, MLP, TransformerBlock)
Cell 7  → MAE Encoder (ViT-Base)
Cell 8  → MAE Decoder (ViT-Small)
Cell 9  → Full MAE model + loss
Cell 10 → Optimizer & LR scheduler
Cell 11 → Train/validate functions
Cell 12 → Training loop (5 epochs)
Cell 13 → Training curves plot
Cell 14 → Qualitative visualisation (5 samples)
Cell 15 → PSNR & SSIM evaluation
Cell 16 → Gradio interactive app
```

---

## 📊 Loss Function

MSE is computed **exclusively on masked patches** — not visible ones:

```python
loss = (pred - target) ** 2           # element-wise squared error
loss = loss.mean(dim=-1)              # mean over patch pixel dimension
loss = (loss * mask).sum() / mask.sum()  # mean over masked patches only
```

This forces the model to genuinely reconstruct missing content rather than copying visible patches.

---

## 📈 Evaluation

| Metric | Description |
|--------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) — higher is better |
| **SSIM** | Structural Similarity Index (0–1) — higher is better |

Metrics compare the model's full reconstruction against the original ground truth image.

---

## 🖼️ Qualitative Results

Each sample shows three panels:

| Masked Input (75%) | Model Reconstruction | Ground Truth |
|:------------------:|:-------------------:|:------------:|
| 75% patches zeroed | Predicted pixels | Original image |

---

## 🌐 Gradio Demo

Cell 16 launches an interactive web app:
- Upload any image
- Adjust masking ratio (10% – 90%) via slider
- See masked input, reconstruction, and original side by side
- Public shareable link via `share=True` (valid 72 hours)

---

## 🧰 Dependencies

```bash
pip install einops gradio scikit-image
# torch, torchvision, numpy, matplotlib, kagglehub already on Kaggle
```

---

## 📚 Dataset

**TinyImageNet** — 200 classes, 64×64 RGB images
- Train: 100,000 images | Val: 10,000 images
- [Kaggle Dataset — akash2sharma/tiny-imagenet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)

---

## 📖 References

1. He et al. (2022). **Masked Autoencoders Are Scalable Vision Learners.** CVPR 2022. [[arXiv]](https://arxiv.org/abs/2111.06377)
2. Dosovitskiy et al. (2021). **An Image is Worth 16×16 Words.** ICLR 2021. [[arXiv]](https://arxiv.org/abs/2010.11929)
3. Vaswani et al. (2017). **Attention Is All You Need.** NeurIPS 2017. [[arXiv]](https://arxiv.org/abs/1706.03762)

---

## 👥 Authors

| Name | Roll No |
|------|---------|
| [Your Name] | XXXX |
| [Partner Name] | XXXX |

*FAST-NUCES · Generative AI (AI4009) · Spring 2026*
