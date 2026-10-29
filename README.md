# Hybrid YOLO + EfficientNet Road Damage Classifier

A deep-learning model that classifies road surface damage into four categories by **fusing features from two pretrained backbones** — a YOLOv8 feature extractor and an EfficientNet-B0 CNN. The combined (hybrid) representation is passed through a fusion head to predict the damage type, reaching ~98% accuracy on the validation set.

## Overview

Rather than relying on a single architecture, this project extracts features from **two complementary networks** and concatenates them:

- **YOLOv8n backbone** — the first 10 layers of a pretrained YOLOv8-nano model, used purely as a feature extractor (spatial features pooled to a vector).
- **EfficientNet-B0** — a pretrained `timm` model (classifier head removed) producing a 1280-dim feature vector.

The two feature vectors are fused and fed into a fully connected classification head with batch norm, ReLU, and dropout. The model is trained with label-smoothing cross-entropy, AdamW, and a cosine-annealing learning-rate schedule.

## Damage Classes

| ID | Class |
|----|-------|
| 0 | Pothole |
| 1 | Alligator Crack |
| 2 | Longitudinal Crack |
| 3 | Lateral Crack |

## Dataset

[Road Damage dataset](https://www.kaggle.com/datasets/alvarobasily/road-damage) by *alvarobasily*, downloaded via `kagglehub`. Images (`.jpeg`) are paired with YOLO-format label files (`.txt`); the class of the first annotation in each file is used as the image-level label. Data is split 80/20 into train/validation.

## Architecture

```
Input image (3 × 640 × 640)
        │
        ├── YOLOv8n backbone ──► adaptive avg pool ──► 256-d vector
        │
        └── EfficientNet-B0 ───────────────────────► 1280-d vector
                                   │
                        concat (1536-d)
                                   │
          Linear(1536 → 512) → BatchNorm → ReLU → Dropout(0.5)
                                   │
                           Linear(512 → 4)
                                   │
                            class logits
```

## Requirements

- Python 3.8+
- PyTorch (with CUDA recommended)
- [ultralytics](https://pypi.org/project/ultralytics/) (YOLOv8)
- [timm](https://pypi.org/project/timm/) (EfficientNet)
- kagglehub, scikit-learn, albumentations, opencv-python, numpy, matplotlib, seaborn

Install:

```bash
pip install ultralytics kagglehub scikit-learn matplotlib seaborn albumentations timm torch opencv-python
```

## Usage

The project is a single Jupyter notebook designed to run end-to-end (originally on Google Colab with a GPU).

1. Open the notebook:

   ```bash
   jupyter notebook "Pretrained yolo with hybrid EfficientNet CNN layers 98%.ipynb"
   ```

   Or upload it to [Google Colab](https://colab.research.google.com/) and select a GPU runtime.

2. Run the cells in order. The notebook will:
   - install dependencies and download the dataset,
   - split the data and build augmented data loaders,
   - construct the hybrid model,
   - train for 40 epochs,
   - evaluate and plot the results.

> **Kaggle access:** `kagglehub` may prompt for Kaggle API credentials to download the dataset.

## Training

- **Loss:** Label-smoothing cross-entropy (`eps=0.1`)
- **Optimizer:** AdamW (`lr=3e-4`)
- **Scheduler:** Cosine annealing (`T_max=40`)
- **Epochs:** 40
- **Batch size:** 16
- **Augmentation (train only):** horizontal flip, random brightness/contrast, shift-scale-rotate, Gaussian blur (via Albumentations)

## Evaluation

The notebook reports a full suite of classification metrics and visualizations:

- Global **accuracy**, **weighted F1**, **ROC-AUC** (one-vs-rest), and **log loss**
- **Per-class F1** scores
- **Confusion matrix** heatmap
- **ROC curves** per class
- **Precision–recall curves** per class
- **F1-score bar chart** per class

## Notes

- The YOLO backbone is used as a **feature extractor for classification**, not for object detection — bounding boxes are not predicted.
- Running on CPU is possible but very slow; a GPU is strongly recommended.
- Results depend on the random train/val split and may vary slightly between runs.

## License

Released under the [MIT License](LICENSE). Model weights from Ultralytics and `timm` are subject to their respective licenses.
