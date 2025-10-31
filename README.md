# ACB-TriNet: Advanced Malware Classification using Multi-Channel Deep Learning

[![Conference](https://img.shields.io/badge/ICETCS%202025-Best%20Technical%20Paper-gold)](https://github.com/RezwanulHaqueRizu/ACB_Trinet)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🏆 Award
**Best Technical Paper Award** at the **International Conference on Emerging Trends in Cybersecurity (ICETCS 2025, UK)**

---

## 📋 Overview

This repository contains the implementation of **ACB-TriNet**, a novel deep learning architecture for malware classification that achieves state-of-the-art performance on the MalImg benchmark dataset. Our approach combines multiple innovative techniques including:

- **Asymmetric Convolutional Blocks (ACB)** for enhanced feature extraction
- **Triplet Attention Mechanism** for multi-dimensional spatial-channel attention
- **Multi-Channel Image Representation** (Gray, Entropy, Sobel edges)
- **Dual-Branch Architecture** (ResNet-inspired + VGG-inspired pathways)
- **Class-Balanced Focal Loss with Deferred Re-Weighting (DRW)**
- **Knowledge Distillation** for model compression and deployment

### Key Results
- **98.82%** Validation Accuracy
- **97.67%** Macro F1-Score
- **99.95%** ROC-AUC (macro, OVR)
- **100%** Top-5 Accuracy
- **25 malware families** classified

---

## 🎯 Research Contributions

### 1. **Novel Multi-Channel Preprocessing**
We transform malware binaries into three complementary image channels:
- **Grayscale**: Raw pixel representation of binary executables
- **Entropy**: Local information content to capture encryption/packing
- **Sobel Edges**: Structural boundaries and code section transitions

### 2. **ACB-TriNet Architecture**
Our architecture features:
- **Asymmetric Convolution Blocks**: Decomposed convolutions (3×3 + 1×3 + 3×1) for efficient feature learning
- **Triplet Attention**: Captures cross-dimensional dependencies across spatial and channel dimensions
- **Dual-Branch Fusion**: Combines ResNet-style residual learning and VGG-style hierarchical features
- **Global Attention Block (GAB)**: Channel and spatial attention for refined feature selection

### 3. **Advanced Training Strategy**
- **Class-Balanced Focal Loss**: Addresses class imbalance with effective number weighting (β=0.9999, γ=1.5)
- **Deferred Re-Weighting (DRW)**: Delayed class weight activation for better convergence
- **Strong Augmentation Pipeline**: MixUp, CutMix, Random Erasing, and crop jittering
- **Warm-up Cosine Annealing**: Smooth learning rate scheduling

### 4. **Knowledge Distillation for Deployment**
- Teacher model: Full ACB-TriNet (~20M parameters)
- Student models: Nano (<1.5M parameters) and Micro (<0.9M parameters)
- Achieves >95% of teacher performance with <10% of parameters

---

## 🗂️ Dataset

**MalImg Dataset**
- Source: [Vision Research Lab](https://www.kaggle.com/datasets/keerthicheepurupalli/malimg-dataset)
- 9,339 malware samples across 25 families
- Images: 32×32 pixels (resized from variable-size originals)
- Families include: Allaple, Yuner, Lolyda, C2LOP, Rbot, VB.AT, and more

| Largest Families | Sample Count |
|-----------------|--------------|
| Allaple.A | 2,949 |
| Allaple.L | 1,591 |
| Yuner.A | 800 |
| Instantaccess | 431 |
| VB.AT | 408 |

---

## 🏗️ Architecture Details

### Teacher Model (ACB-TriNet)
```
Input (32×32×3) → Dual Branches:
├─ ResNet Branch: ACB-ResBlocks [64→128→160 channels]
└─ VGG Branch: ACB-VGGBlocks [64→128→128 channels]
→ Feature Fusion (1×1 Conv, 192 channels)
→ Global Attention Block
→ Global Average Pooling
→ Dense (25 classes, logits)
```

### Training Configuration
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Schedule**: Warm-up (5 epochs) + Cosine Annealing (45 epochs)
- **Batch Size**: 32
- **Augmentation**: MixUp (α=0.4), CutMix (α=1.0, p=0.5), Random Erase (p=0.25)
- **Loss**: Class-Balanced Focal Loss (β=0.9999, γ=1.5)
- **Hardware**: 2× Tesla P100 GPUs

---

## 📊 Performance Metrics

### Overall Performance
| Metric | Score |
|--------|-------|
| Accuracy | 98.82% |
| Macro F1 | 97.67% |
| Weighted F1 | 98.82% |
| Micro F1 | 98.82% |
| Top-5 Accuracy | 100.00% |
| ROC-AUC (macro) | 99.95% |

### Per-Class Performance (Selected)
| Family | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Allaple.A | 1.0000 | 0.9932 | 0.9966 | 295 |
| Yuner.A | 1.0000 | 1.0000 | 1.0000 | 80 |
| VB.AT | 0.9762 | 1.0000 | 0.9880 | 41 |
| Instantaccess | 1.0000 | 1.0000 | 1.0000 | 43 |
| Lolyda.AA1 | 0.9130 | 1.0000 | 0.9545 | 21 |

---

## 🚀 Quick Start

### Prerequisites
```bash
Python >= 3.8
TensorFlow >= 2.10
NumPy >= 1.21
Matplotlib >= 3.5
scikit-learn >= 1.0
Pandas >= 1.3
```

### Installation
```bash
git clone https://github.com/RezwanulHaqueRizu/ACB_Trinet.git
cd ACB_Trinet
pip install -r requirements.txt
```

### Training the Model
```python
# Open and run the Jupyter notebook
jupyter notebook ACB_TriNet.ipynb

# Or convert to Python script and run
jupyter nbconvert --to script ACB_TriNet.ipynb
python ACB_TriNet.py
```

### Loading Pre-trained Weights
```python
from tensorflow import keras

# Load teacher model
teacher = build_teacher_same_as_student(NUM_CLASSES=25)
teacher.load_weights("teacher_same.weights.h5")

# Load student model (for deployment)
student = build_student_mckdff_nano(NUM_CLASSES=25)
student.load_weights("best_student_kd32.weights.h5")
```

---

## 📈 Visualization Examples

The notebook includes comprehensive visualizations:

1. **Multi-Channel Preprocessing**: Gray, Entropy, and Sobel channels
2. **Data Augmentation**: MixUp and CutMix examples
3. **Training Curves**: Loss, accuracy, and macro F1 progression
4. **Confusion Matrices**: Both count and normalized versions
5. **Per-Class Analysis**: F1 scores, support, and error patterns
6. **Grad-CAM**: Attention heatmaps for model interpretability
7. **t-SNE Embeddings**: Feature space visualization
8. **ROC & PR Curves**: Per-class and macro-averaged

---

## 🔬 Ablation Studies

| Configuration | Val Macro F1 |
|---------------|--------------|
| **Full Model (Ours)** | **98.9%** |
| Gray only | 98.2% |
| Gray + Entropy | 98.7% |
| Without ACB | 98.4% |
| Without Triplet Attention | 98.5% |
| Without GAB | 98.6% |
| Cross-Entropy Loss | 98.4% |
| Focal Loss | 98.7% |

Key Findings:
- Multi-channel representation improves F1 by **0.7%**
- Triplet Attention contributes **0.4%** improvement
- Class-Balanced Focal + DRW adds **0.2%** over standard Focal Loss

---

## 🧪 Reproducing Results

### Dataset Preparation
1. Download MalImg dataset from [Kaggle](https://www.kaggle.com/datasets/keerthicheepurupalli/malimg-dataset)
2. Extract to `malimg_paper_dataset_imgs/`
3. Update `BASE_DIR` in the notebook

### Training Pipeline
1. **Preprocessing**: Automatic 3-channel generation (Gray, Entropy, Sobel)
2. **Augmentation**: MixUp/CutMix applied on-the-fly
3. **Training**: 50 epochs with early stopping (patience=8)
4. **Evaluation**: Macro F1 computed after each epoch
5. **Checkpointing**: Best model saved based on val_macro_f1

### Expected Training Time
- **Teacher Model**: ~11 hours (50 epochs on 2× Tesla P100)
- **Student Distillation**: ~6 hours (40 epochs)

---

## 🛠️ Model Deployment

### Exporting for Production
```python
# Export teacher
teacher.save("acb_trinet_teacher.keras", include_optimizer=False)

# Export student (lightweight)
student.save("acb_trinet_student_nano.keras", include_optimizer=False)
```

### Inference Example
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("acb_trinet_teacher.keras")

# Preprocess image
img = preprocess_malware_image(path)  # See notebook for full function

# Predict
logits = model(img[None], training=False)
probs = tf.nn.softmax(logits).numpy()[0]
pred_class = class_names[probs.argmax()]
confidence = probs.max()

print(f"Predicted: {pred_class} (confidence: {confidence:.2%})")
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{acbtrinet2025,
  title={ACB-TriNet: Advanced Malware Classification using Multi-Channel Deep Learning with Asymmetric Convolutions and Triplet Attention},
  author={Mohamed Shafeek T and Rezwanul Haque Rizu},
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

## 👥 Authors

- **Mohamed Shafeek T** - Principal Researcher
- **Rezwanul Haque Rizu** - Co-Author

---

## 🙏 Acknowledgments

- **Vision Research Lab** for the MalImg dataset
- **Visual Geometry Group (VGG)**, University of Oxford for architectural inspiration
- **ICETCS 2025** organizing committee for recognizing this work

---

## 🔗 Related Work

- **VGG16**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- **Triplet Attention**: [Rotate to Attend: Convolutional Triplet Attention Module](https://arxiv.org/abs/2010.03045)
- **Knowledge Distillation**: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

---

## 📧 Contact

For questions or collaborations:
- GitHub: [@RezwanulHaqueRizu](https://github.com/RezwanulHaqueRizu)
- Repository: [ACB_Trinet](https://github.com/RezwanulHaqueRizu/ACB_Trinet)

---

## 🔄 Updates

- **[2025-01]**: Paper accepted at ICETCS 2025 (UK) ✅
- **[2025-01]**: Awarded Best Technical Paper 🏆
- **[2025-01]**: Initial code release

---

<div align="center">

**⭐ Star this repository if you find it useful! ⭐**

Made with ❤️ for Cybersecurity Research

</div>

