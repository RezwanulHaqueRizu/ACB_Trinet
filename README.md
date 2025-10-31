# ACB-TriNet: Malware Classification using Deep Learning

**Best Technical Paper Award - ICETCS 2025, UK**

*By Mohamed Shafeek T*

---

## Overview

This project implements a deep learning approach for classifying malware images from the MalImg dataset. The architecture uses multi-channel preprocessing (Grayscale, Entropy, Sobel) combined with Asymmetric Convolutional Blocks (ACB), Triplet Attention, and dual-branch feature fusion.

## Dataset

**MalImg Dataset**
- 9,339 malware samples across 25 families
- Images resized to 32×32 pixels
- Source: Vision Research Lab

**Dataset Split:**
- Training: 8,405 samples (90%)
- Validation: 934 samples (10%)

## Architecture

### Multi-Channel Preprocessing
The model uses three complementary channels:
1. **Grayscale**: Raw binary visualization
2. **Entropy**: Local information content (5×5 window)
3. **Sobel**: Edge detection for structural features

### Model Components
- **Asymmetric Convolution Blocks (ACB)**: 3×3 + 1×3 + 3×1 convolutions
- **Triplet Attention**: Cross-dimensional spatial-channel attention
- **Dual-Branch Architecture**:
  - ResNet-inspired branch (64→128→160 channels)
  - VGG-inspired branch (64→128→128 channels)
- **Global Attention Block (GAB)**: Channel and spatial attention
- **Feature Fusion**: 1×1 convolution (192 channels)

## Training Configuration

- **Loss Function**: Class-Balanced Focal Loss (β=0.9999, γ=1.5)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Learning Rate Schedule**: 5-epoch warmup + Cosine annealing
- **Augmentation**: MixUp, CutMix, Random Erasing, Crop Jittering
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping, patience=8)
- **Hardware**: Tesla P100 GPU

## Results

Based on validation set (934 samples):

| Metric | Score |
|--------|-------|
| Accuracy | 98.82% |
| Macro F1 | 97.67% |
| Weighted F1 | 98.82% |
| Top-5 Accuracy | 100.00% |
| ROC-AUC (macro) | 99.95% |

## Requirements

```bash
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
opencv-python>=4.5.0
jupyter>=1.0.0
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/RezwanulHaqueRizu/ACB_Trinet.git
cd ACB_Trinet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook:
```bash
jupyter notebook ACB_TriNet.ipynb
```

## Files

- `ACB_TriNet.ipynb` - Main implementation notebook
- `requirements.txt` - Python dependencies
- `LICENSE` - MIT License

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{acbtrinet2025,
  title={Malware Classification using Multi-Channel Deep Learning with ACB and Triplet Attention},
  author={Mohamed Shafeek T},
  booktitle={International Conference on Emerging Trends in Cybersecurity (ICETCS)},
  year={2025},
  location={United Kingdom},
  note={Best Technical Paper Award}
}
```

## License

MIT License - see LICENSE file for details.

## Author

Mohamed Shafeek T

Repository: https://github.com/RezwanulHaqueRizu/ACB_Trinet
