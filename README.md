# ACB-TriNet: Malware Classification using Deep Learning

[![Conference](https://img.shields.io/badge/ICETCS%202025-Best%20Technical%20Paper-gold)](https://github.com/RezwanulHaqueRizu/ACB_Trinet)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🏆 Award
**Best Technical Paper Award** at the **International Conference on Emerging Trends in Cybersecurity (ICETCS 2025, UK)**

---

## 📋 Overview

Deep learning architecture for malware classification using multi-channel image preprocessing and advanced attention mechanisms. Achieves **98.82%** accuracy on the MalImg benchmark dataset with 25 malware families.

### Key Features
- **Multi-Channel Preprocessing**: Grayscale, Entropy, and Sobel edge detection
- **Asymmetric Convolutional Blocks (ACB)**: Efficient feature extraction with 3×3 + 1×3 + 3×1 convolutions
- **Triplet Attention Mechanism**: Cross-dimensional spatial-channel attention
- **Dual-Branch Architecture**: ResNet-inspired and VGG-inspired pathways with feature fusion
- **Class-Balanced Focal Loss**: Handles class imbalance with deferred re-weighting

---

## 🗂️ Dataset

**Malimg Dataset**
- **Total Samples**: 9,339 malware images
- **Families**: 25 malware families
- **Image Size**: 32×32 pixels
- **Split**: 8,405 training / 934 validation (90/10 stratified split)

**Top 5 Families by Sample Count:**
- Allaple.A: 2,949 samples
- Allaple.L: 1,591 samples
- Yuner.A: 800 samples
- Instantaccess: 431 samples
- VB.AT: 408 samples

---

## 🏗️ Architecture

### Multi-Channel Preprocessing
Each malware binary image is transformed into three complementary channels:

1. **Grayscale**: Raw binary visualization
2. **Entropy**: Local information content (5×5 window) to capture encryption/packing
3. **Sobel Edges**: Structural boundaries and code section transitions

### Network Architecture

```
Input (32×32×3)
    ↓
Dual Branches:
├─ ResNet Branch: [64 → 128 → 160 channels]
│  └─ ACB + Triplet Attention in each block
│
└─ VGG Branch: [64 → 128 → 128 channels]
   └─ ACB + Triplet Attention in each block
    ↓
Feature Fusion (1×1 Conv, 192 channels)
    ↓
Global Attention Block (Channel + Spatial)
    ↓
Global Average Pooling
    ↓
Dense Layer (25 classes, logits)
```

**Key Components:**
- **Asymmetric Convolution Blocks**: Combines 3×3, 1×3, and 3×1 convolutions
- **Triplet Attention**: Processes spatial and channel dimensions with cross-dimensional interaction
- **Global Attention Block**: SE-like channel attention + spatial attention gates

---

## 🚀 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Loss Function** | Class-Balanced Focal Loss (β=0.9999, γ=1.5) |
| **Optimizer** | AdamW (lr=3e-4, weight_decay=1e-4) |
| **LR Schedule** | 5-epoch warmup + Cosine annealing |
| **Batch Size** | 32 |
| **Total Epochs** | 50 |
| **Early Stopping** | Patience = 8 epochs |
| **Augmentation** | MixUp (α=0.4), CutMix (α=1.0), Random Erasing |
| **Hardware** | Tesla P100 GPU |

**Data Augmentation Pipeline:**
- MixUp: Blends two images with β-distribution sampling
- CutMix: Patches from one image replace regions in another
- Random Erasing: Random rectangular regions zeroed out (p=0.25)
- Crop Jittering: Mild random crops for robustness

---

## 📊 Results

### Overall Performance (Validation Set)

| Metric | Score |
|--------|-------|
| **Accuracy** | **98.82%** |
| **Macro F1** | **97.67%** |
| **Weighted F1** | 98.82% |
| **Micro F1** | 98.82% |
| **Top-5 Accuracy** | 100.00% |
| **ROC-AUC (macro)** | 99.95% |

### Sample Per-Class Results

| Malware Family | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Allaple.A | 1.0000 | 0.9932 | 0.9966 | 295 |
| VB.AT | 0.9762 | 1.0000 | 0.9880 | 41 |
| Yuner.A | 1.0000 | 1.0000 | 1.0000 | 80 |
| Instantaccess | 1.0000 | 1.0000 | 1.0000 | 43 |
| Lolyda.AA1 | 0.9130 | 1.0000 | 0.9545 | 21 |

---

## 🛠️ Installation & Usage

### Prerequisites
```bash
Python >= 3.8
TensorFlow >= 2.10
```

### Install Dependencies
```bash
git clone https://github.com/RezwanulHaqueRizu/ACB_Trinet.git
cd ACB_Trinet
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook ACB_TriNet.ipynb
```

### Load Trained Model
```python
from tensorflow import keras

# Load model weights
model = build_teacher_same_as_student(NUM_CLASSES=25)
model.load_weights("teacher_same.weights.h5")
```

---

## 📦 Repository Contents

```
ACB_Trinet/
├── ACB_TriNet.ipynb      # Main implementation notebook
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── LICENSE               # MIT License
```

---

## 📈 Visualizations

The notebook includes comprehensive visualizations:

- **Multi-Channel Preprocessing**: Grayscale, Entropy, and Sobel visualizations
- **Data Augmentation Examples**: MixUp and CutMix transformations
- **Training Curves**: Loss, accuracy, and macro F1 progression
- **Confusion Matrices**: Per-class error analysis
- **Grad-CAM**: Attention heatmaps for model interpretability
- **t-SNE Embeddings**: Feature space visualization
- **ROC & PR Curves**: Per-class and macro-averaged performance

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{acbtrinet2025,
  title={ACB-TriNet: Malware Classification using Multi-Channel Deep Learning},
  booktitle={International Conference on Emerging Trends in Cybersecurity (ICETCS)},
  year={2025},
  location={United Kingdom},
  note={Best Technical Paper Award}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Repository**: [github.com/RezwanulHaqueRizu/ACB_Trinet](https://github.com/RezwanulHaqueRizu/ACB_Trinet)
- **Dataset**: [Malimg Dataset (Kaggle)](https://www.kaggle.com/datasets)

---

<div align="center">

**⭐ Star this repository if you find it useful! ⭐**

</div>
