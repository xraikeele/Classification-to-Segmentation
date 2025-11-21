# Classification-to-Segmentation: Class Activation Mapping for Zero-Shot Skin Lesion Segmentation


[![Paper](https://img.shields.io/badge/Paper-AIIH%202025-blue)](https://doi.org/10.1007/978-3-032-00656-1_24)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Zero-shot skin lesion segmentation using Class Activation Maps (CAM) to guide Segment Anything Model (SAM).**
---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [1. Train Classification Models](#1-train-classification-models)
  - [2. Generate CAMs](#2-generate-class-activation-maps)
  - [3. CAM-Guided Segmentation](#3-cam-guided-segmentation)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Overview

This repository implements a **zero-shot segmentation pipeline** for skin lesions that combines:

1. **Classification Models** (CNNs & Vision Transformers) trained on binary melanoma classification
2. **Class Activation Maps (CAM)** for visual explanations highlighting lesion regions
3. **Bounding Box Generation** from CAM heatmap thresholding
4. **MedSAM** (medical Segment Anything Model) guided by dual prompts: CAM-derived points + bounding box

**Key Innovation:** Leveraging classification models to achieve segmentation **without pixel-level annotations**, enabling cost-effective medical image analysis.

> **Interactive Demo:** Check out `notebooks/Zero_Shot_Segmentation_Demo.ipynb` for a fully documented, step-by-step walkthrough using **ImageNet pre-trained weights only** (no domain-specific training) to demonstrate the truly zero-shot capability of this approach.

---

## Key Features

- **Zero-Shot Segmentation**: No segmentation masks required during training
- **Multiple Model Architectures**: 
  - CNNs: ResNet50, MobileNetV2, EfficientNet-B0
  - Transformers: Vision Transformer (ViT), Swin Transformer
- **CAM Methods**: GradCAM, GradCAM++, EigenCAM, AblationCAM
- **Comprehensive Evaluation**: Precision, Recall, F1-score, IoU metrics
- **Reproducible Results**: Fixed random seeds, detailed logging
- **ISIC 2017 Dataset**: Standardized benchmark for skin lesion analysis

---

## Framework

![Classification-to-Segmentation Framework](classification-to-segmentation.png)

**Pipeline Workflow:**
1. **Classification**: Train binary classifier (melanoma vs. benign) on ISIC 2017  
   *(Demo uses ImageNet pre-trained weights for truly zero-shot demonstration)*
2. **CAM Extraction**: Generate activation maps highlighting discriminative regions
3. **Dual Prompt Generation**:
   - **Bounding Box**: Threshold heatmap at 0.5, extract ROI, add 10% padding
   - **Point Sampling**: Probabilistic sampling from CAM peaks (power=30 amplification)
4. **MedSAM Segmentation** *(Full implementation in `results-vis.py`)*:
   - Encode image at 1024x1024 resolution
   - Generate 10 masks with perturbed prompts
   - Select best mask via IoU with ground truth (or confidence score)
5. **Post-processing**: Morphological operations (opening + closing)

---

## Repository Structure

```
Classification-to-Segmentation/
├── classification/
│   ├── train.py                    # Training script with early stopping
│   └── test.py                     # Model evaluation
│
├── explainers/
│   ├── CAM.py                      # Custom CAM implementation
│   ├── pytorch_CAM.py              # PyTorch GradCAM integration
│   ├── LRP_heatmap.py              # Layer-wise Relevance Propagation
│   └── test.py                     # CAM testing utilities
│
├── models/
│   ├── load_models.py              # Model loading utilities
│   └── nest.py                     # NeST architecture (optional)
│
├── data/
│   └── dataloader.py               # ISIC 2017 dataset loader
│
├── CAM-SAM/
│   ├── CAM-SAM.py                  # Basic zero-shot pipeline (points only)
│   ├── CAM-MedSAM.py               # Medical SAM integration (points only)
│   ├── results-vis.py              # **Full implementation: bbox + points, IoU-based mask selection**
│   ├── SAM_baseline.py             # SAM-only baseline
│   └── sample_points.py            # Point sampling strategies
│
├── visualisations/
│   ├── CAM-SAM-performance.py      # Performance plots
│   ├── CAM-performance_vis.py      # CAM quality analysis
│   ├── classification_vis.py       # Classification metrics
│   └── json_results.py             # JSON result parser
│
├── MedSAM/                         # MedSAM library (https://github.com/bowang-lab/MedSAM)
│   ├── segment_anything/          # SAM model architecture
│   ├── MedSAM_Inference.py        # Inference utilities
│   └── ...
│
├── model_checkpoints/              # Trained model weights
│
├── requirements.txt                # Python dependencies
├── classification-to-segmentation.png  # Framework diagram
└── README.md                           # This file
```

---

## Installation

### Prerequisites
- Python 3.12+
- CUDA 12.4+ (for GPU acceleration)
- 16GB+ GPU memory (recommended for SAM)

### 1. Clone Repository
```bash
git clone https://github.com/xraikeele/Classification-to-Segmentation.git
cd Classification-to-Segmentation/Zero-Shot_SkinLesion_Segmentation
```

### 2. Create Environment
```bash
conda create -n zero-shot-seg python=3.12
conda activate zero-shot-seg
```

### 3. Install Dependencies
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install segment-anything opencv-python scikit-learn scikit-image
pip install pytorch-grad-cam timm matplotlib pandas tqdm
pip install grad-cam torchcam ttach
```

**Note:** The MedSAM library code is included in this repository. Original source: [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)

### 4. Download ISIC 2017 Dataset
```bash
# Download from: https://challenge.isic-archive.com/data/#2017
# Expected structure:
# ISIC_2017/
# ├── train/
# │   ├── ISIC-2017_Training_Data/         # Images
# │   ├── ISIC-2017_Training_Part1_GroundTruth/  # Segmentation masks
# │   └── ISIC-2017_Training_Part3_GroundTruth.csv  # Labels
# ├── val/
# │   ├── ISIC-2017_Validation_Data/
# │   ├── ISIC-2017_Validation_Part1_GroundTruth/
# │   └── ISIC-2017_Validation_Part3_GroundTruth.csv
# └── test/
#     ├── ISIC-2017_Test_v2_Data/
#     ├── ISIC-2017_Test_v2_Part1_GroundTruth/
#     └── ISIC-2017_Test_v2_Part3_GroundTruth.csv
```

### 5. Download SAM Checkpoint
```bash
# Download ViT-H SAM model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Quick Start

### End-to-End Pipeline (3 Steps)

```bash
# 1. Train classification model
python classification/train.py

# 2. Generate CAMs for test set
python explainers/CAM.py

# 3. Run CAM-guided SAM segmentation
python CAM-SAM/CAM-SAM.py
```

---

## Usage

### 1. Train Classification Models

Train binary melanoma classifiers on ISIC 2017:

```bash
cd classification
python train.py
```

**Configuration:**
- Models: `resnet50`, `mobilenet_v2`, `efficientnet_b0`, `ViT`, `swin`
- Batch size: 32
- Learning rate: 1e-4
- Early stopping patience: 50 epochs
- Optimizer: Adam

**Outputs:**
- `results/classification/{model_name}/`
  - `{model_name}_best_model.pth` - Best checkpoint
  - `log.txt` - Training metrics per epoch
  - `{model_name}_accuracy_loss_plots.png` - Learning curves

**Example Training Logs:**
```
Epoch 1/200
Train Loss: 0.6234, Train Accuracy: 0.6890
Validation Loss: 0.5821, Validation Accuracy: 0.7150
Best model saved with validation accuracy: 0.7150

Epoch 50/200
Train Loss: 0.2145, Train Accuracy: 0.9120
Validation Loss: 0.3421, Validation Accuracy: 0.8560
Early stopping triggered at epoch 50
```

---

### 2. Generate Class Activation Maps

Extract visual explanations from trained classifiers:

```bash
cd explainers
python CAM.py
```

**Key Parameters:**
```python
# explainers/CAM.py

model_configs = [
    {'name': 'resnet50', 'type': 'CNN'},
    {'name': 'mobilenet_v2', 'type': 'CNN'},
    {'name': 'efficientnet_b0', 'type': 'CNN'},
    {'name': 'ViT', 'type': 'ViT'},
    {'name': 'swin', 'type': 'swin'},
]

# Target layers for CAM extraction
def layer_selection(model, layer):
    if layer == "resnet50":
        return [model.layer4[-1]]
    elif layer == "ViT":
        return [model.encoder.layers[-1].ln_1]
    # ... etc
```

**Outputs:**
- `results/CAM_zero-shot/{model_name}/`
  - `cam_0.png`, `cam_1.png`, ... - CAM overlays

**CAM Quality Metrics:**
- Faithfulness: Deletion/insertion curves
- Localization: Pointing game accuracy
- Visual coherence: Normalized heatmaps

---

### 3. CAM-Guided Segmentation

Run zero-shot segmentation using CAM-guided MedSAM:

```bash
cd CAM-SAM
# Basic implementation (points only)
python CAM-SAM.py

# OR: Full implementation with bounding box + points (recommended)
python results-vis.py
```

> **Note:** `results-vis.py` contains the complete implementation used in the paper with:
> - Bounding box generation from CAM heatmap
> - Dual prompts (box + points)
> - Multiple mask generation with perturbations
> - IoU-based best mask selection

**Pipeline Steps:**

1. **Load Models:**
```python
# Classification model
model = load_model(model_name, num_classes=1)
model.load_state_dict(torch.load(checkpoint_path))

# SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
```

2. **Generate CAM:**
```python
# Forward pass with gradient tracking
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()

# Extract activation maps and gradients
feature_map = feature_maps[-1]
gradient = gradients[-1]

# Compute weighted CAM
weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
cam = torch.sum(weights * feature_map, dim=1)
cam = torch.relu(cam)  # ReLU to remove negative activations
```

3. **Sample Points:**
```python
def sample_points(heatmap, num_points=10):
    # Amplify high-confidence regions
    flat_heatmap = np.power(heatmap.flatten(), 30)
    
    # Probabilistic sampling
    probabilities = flat_heatmap / flat_heatmap.sum()
    indices = np.random.choice(
        len(flat_heatmap), 
        size=num_points, 
        p=probabilities,
        replace=False
    )
    
    # Convert to 2D coordinates
    points = [(idx % width, idx // width) for idx in indices]
    return points
```

4. **SAM Segmentation:**
```python
# Set image in predictor
predictor.set_image(image)

# Generate mask from points
masks, scores, logits = predictor.predict(
    point_coords=np.array(points),
    point_labels=np.ones(len(points)),  # All positive prompts
    multimask_output=False
)
```

**Outputs:**
- `results/experiment_results.json` - Per-image metrics
- `results/single-image-analysis/` - Visual results

**Evaluation Metrics:**
```json
{
    "ISIC_0012375.jpg": {
        "Precision": 0.6357,
        "Recall": 0.9639,
        "F1": 0.7661,
        "IoU": 0.6209
    }
}
```

---

## Results

### Quantitative Performance (ISIC 2017 Test Set)

#### Best Method Performance (Adaptive CAM Selection per Image)

| Model | Training | Precision | Recall | F1-Score | IoU | Dice |
|-------|----------|-----------|--------|----------|-----|------|
| **MobileNetV2** | **Finetuned** | **0.764** | **0.547** | **0.510** | **0.366** | **0.510** |
| **Swin** | **Finetuned** | **0.750** | **0.534** | **0.498** | **0.357** | **0.498** |
| ResNet50 | Finetuned | 0.739 | 0.496 | 0.475 | 0.337 | 0.475 |
| MobileNetV2 | Zero-shot | 0.740 | 0.468 | 0.470 | 0.337 | 0.470 |
| Swin | Zero-shot | 0.743 | 0.427 | 0.446 | 0.315 | 0.446 |

**Note:** "Best Method" adaptively selects the optimal CAM method (GradCAM, GradCAM++, AblationCAM, or ScoreCAM) for each individual image, simulating clinician selection from multiple mask options.

#### Baseline Comparisons

| Method | IoU | Dice | Notes |
|--------|-----|------|-------|
| SAM (manual prompts) | 0.672 | 0.805 | Ground-truth guided prompting |
| MedSAM (manual prompts) | 0.671 | 0.782 | Ground-truth guided prompting |
| **MobileNetV2 + Best CAM** | **0.366** | **0.510** | **Automated CAM-guided prompting (finetuned)** |
| MobileNetV2 + Best CAM | 0.337 | 0.470 | Automated CAM-guided prompting (zero-shot) |

#### Individual CAM Method Performance

**Highest performing combinations:**
- **Zero-shot**: MobileNetV2 + ScoreCAM (IoU: 0.184)
- **Finetuned**: MobileNetV2 + AblationCAM (IoU: 0.282)

**CAM Baseline (Finetuned Swin Transformer):**

| CAM Method | Precision | Recall | F1 | IoU | Dice |
|------------|-----------|--------|----|----|------|
| GradCAM | 0.487 | 0.400 | 0.345 | 0.236 | 0.345 |
| AblationCAM | 0.294 | 0.329 | 0.268 | 0.180 | 0.268 |
| ScoreCAM | 0.365 | 0.301 | 0.263 | 0.177 | 0.263 |
| GradCAM++ | 0.254 | 0.165 | 0.155 | 0.096 | 0.155 |
| RandomCAM | 0.286 | 0.329 | 0.225 | 0.148 | 0.225 |

### Key Findings

- **Best Overall**: Finetuned MobileNetV2 with adaptive CAM selection (IoU: 0.366, Dice: 0.510)
- **Domain Training Impact**: Fine-tuning improved performance across all models, with MobileNetV2 showing the most robust generalization
- **Zero-shot Capability**: MobileNetV2 achieved competitive performance (IoU: 0.337) even with ImageNet weights only
- **Adaptive Selection Benefit**: The "best method" approach reduced complete segmentation failures by ~98.75% (from ~400 to ~5 zero-IoU cases)
- **CAM Method Variability**: GradCAM consistently outperformed other methods for noisy mask generation (IoU: 0.236 vs. 0.096-0.180)
- **Architecture Insights**: 
  - CNNs (especially MobileNetV2) showed superior CAM-based localization compared to transformers
  - Transformer models (ViT, Swin) struggled with effective prompt generation despite reasonable classification accuracy
- **Performance Gap**: Automated CAM-guided prompting achieved ~54% of manual prompting performance (0.366 vs. 0.671 IoU), demonstrating promise for fully automated workflows while highlighting room for improvement

### Qualitative Examples

```
Input Image    →    CAM Heatmap    →    Sampled Points    →    SAM Segmentation
```

**Successful Cases:**
- Clear lesion boundaries with high CAM confidence
- Minimal background activations
- SAM correctly segments based on point prompts

**Failure Cases:**
- Diffuse lesions with unclear boundaries
- High background clutter (hair, skin markings)
- Multiple lesions in single image

---

## Reproducibility

### Fixed Random Seeds
```python
def set_seed(seed=905):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Logging and Checkpoints
All experiments automatically log:
- Training/validation metrics per epoch
- Best model checkpoints based on validation accuracy
- Segmentation results in JSON format
- Visualizations for qualitative analysis

### Hardware Requirements
**Minimum:**
- GPU: 8GB VRAM (e.g., RTX 3070)
- RAM: 16GB
- Storage: 10GB (dataset + models)

**Recommended:**
- GPU: 16GB+ VRAM (e.g., A100, V100)
- RAM: 32GB
- Storage: 50GB (for multiple experiments)

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{cockayne2025class2seg,
  title={Classification-to-Segmentation: Class Activation Mapping for Zero-Shot Skin Lesion Segmentation},
  author={Matthew John Cockayne, Marco Ortolani, Baidaa Al-Bander},
  booktitle={International Conference on AI in Healthcare},
  pages={327--340},
  year={2025},
  organization={Springer}
}
```

---

## Acknowledgments

- **ISIC 2017 Challenge** for providing the skin lesion dataset
- **Meta AI** for Segment Anything Model (SAM)
- **[MedSAM](https://github.com/bowang-lab/MedSAM)** ([Ma et al., 2024](https://www.nature.com/articles/s41467-024-44824-z)) - Medical image segmentation foundation model
- **PyTorch GradCAM** library for CAM implementations
- **Timm** library for pre-trained vision transformer models

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Matthew Cockayne**
- PhD Candidate, Keele University
- Research: Responsible AI, Medical Image processing, Explainability, Multi-modal Learning
- GitHub: https://matt-cockayne.github.io
- Email: m.j.cockayne@keele.ac.uk

---

## Related Projects

- [DermFormer](https://github.com/xraikeele/DermFormer) - Multi-modal vision transformer for robust skin cancer detection
---

## Issues & Contributions

Found a bug or have a feature request? Please open an issue on [GitHub](https://github.com/xraikeele/Classification-to-Segmentation/issues).

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Future Work

- [ ] Extension to multi-class segmentation
- [ ] Real-time inference optimization
- [ ] Web demo with Gradio/Streamlit
- [ ] Support for additional datasets (HAM10000, Derm7pt)
- [ ] Ensemble CAM methods for improved localization
- [ ] Uncertainty quantification for predictions

---

**Last Updated:** November 2025