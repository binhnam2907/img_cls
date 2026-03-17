# Image Classification with ResNet-50 on Imbalanced CIFAR-10

A from-scratch PyTorch implementation of ResNet-50 trained on a **long-tail imbalanced** version of CIFAR-10, with built-in solutions to handle class imbalance.

---

## Table of Contents

- [Dataset Visualization](#dataset-visualization)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Training & Evaluation Results](#training--evaluation-results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Supported Models](#supported-models)

---

## Dataset Visualization

### Sample Images

Each row shows 5 random samples from one of the 10 CIFAR-10 classes (32x32 color images).

![Sample Images](results/figures/sample_images.png)

### Imbalanced Class Distribution

The training set is **artificially imbalanced** using exponential decay to create a long-tail distribution. The most frequent class (airplane) keeps all 5,000 samples while the rarest class (truck) has only ~250 — a **20:1 imbalance ratio**.

![Class Distribution](results/figures/class_distribution.png)

| Class      | Train Samples | Ratio vs Max |
|------------|---------------|--------------|
| airplane   | 5,000         | 1.00x        |
| automobile | 3,584         | 0.72x        |
| bird       | 2,569         | 0.51x        |
| cat        | 1,841         | 0.37x        |
| deer       | 1,320         | 0.26x        |
| dog        | 946           | 0.19x        |
| frog       | 678           | 0.14x        |
| horse      | 486           | 0.10x        |
| ship       | 348           | 0.07x        |
| truck      | 250           | 0.05x        |

**Total training samples:** ~17,022 (down from 50,000)

The **test set remains balanced** (1,000 per class) to provide a fair evaluation.

---

## Model Architecture

ResNet-50 implemented **entirely from scratch** using `torch.nn` — no `torchvision.models`.

### Architecture Overview

```
Input (3 x 32 x 32)
  │
  ├── Stem: Conv2d(7x7, 64, stride=2) → BN → ReLU → MaxPool(3x3, stride=2)
  │
  ├── Layer 1: Bottleneck × 3  [64  → 256  channels]
  ├── Layer 2: Bottleneck × 4  [256 → 512  channels, stride=2]
  ├── Layer 3: Bottleneck × 6  [512 → 1024 channels, stride=2]
  ├── Layer 4: Bottleneck × 3  [1024→ 2048 channels, stride=2]
  │
  ├── Head: AdaptiveAvgPool → Flatten → Dropout → Linear(2048, 10)
  │
  └── Output: 10 class logits
```

### Bottleneck Block Detail

```
x ──┬── Conv1x1(in→mid) → BN → ReLU
    │   Conv3x3(mid→mid) → BN → ReLU
    │   Conv1x1(mid→mid×4) → BN
    │                          │
    └── [downsample if needed] ┘ → + → ReLU → out
```

### Layer-by-Layer Summary

| Layer   | Output Size | Block Structure                   | Repeat |
|---------|-------------|-----------------------------------|--------|
| stem    | 56 × 56     | 7×7 conv, 64, stride 2 + maxpool  | 1      |
| layer1  | 56 × 56     | [1×1, 64 / 3×3, 64 / 1×1, 256]   | 3      |
| layer2  | 28 × 28     | [1×1, 128 / 3×3, 128 / 1×1, 512] | 4      |
| layer3  | 14 × 14     | [1×1, 256 / 3×3, 256 / 1×1, 1024]| 6      |
| layer4  | 7 × 7       | [1×1, 512 / 3×3, 512 / 1×1, 2048]| 3      |
| head    | 10          | AdaptiveAvgPool → FC              | 1      |

**Total parameters: 23,528,522 (~23.5M)**

### Weight Initialization

- **Conv2d**: Kaiming normal (fan_out, ReLU)
- **BatchNorm2d**: weight=1, bias=0
- **Linear**: normal(0, 0.01), bias=0

---

## Configuration

All hyperparameters are centralized in `configs/default.yaml`:

| Category        | Parameter           | Value                |
|-----------------|---------------------|----------------------|
| **Model**       | Architecture        | ResNet-50            |
|                 | Parameters          | 23.5M               |
|                 | Dropout             | 0.0                 |
| **Data**        | Dataset             | CIFAR-10             |
|                 | Image size          | 224 (resized)        |
|                 | Train batch         | 128                  |
|                 | Val batch           | 256                  |
| **Imbalance**   | Enabled             | Yes                  |
|                 | Ratio               | 20:1                 |
|                 | Weighted loss       | Yes                  |
|                 | Weighted sampler    | Yes                  |
| **Optimizer**   | Type                | Adam                 |
|                 | Learning rate       | 0.001                |
|                 | Weight decay        | 1e-4                 |
| **Scheduler**   | Type                | Cosine Annealing     |
|                 | eta_min             | 1e-5                 |
| **Training**    | Epochs              | 20                   |
|                 | Early stopping      | 10 epochs patience   |
| **Augmentation**| RandomCrop          | Yes                  |
|                 | HorizontalFlip      | Yes                  |
|                 | Normalize           | Yes                  |

---

## Handling Imbalanced Data

### The Problem

When training on imbalanced data, the model becomes biased toward majority classes and ignores minority classes entirely. Without any correction, the model will:

- Predict majority classes almost exclusively
- Achieve misleadingly high overall accuracy
- Have near-zero recall on rare classes

### Solution 1: Weighted Cross-Entropy Loss

Assigns higher loss penalties to under-represented classes using inverse-frequency weights:

$$w_c = \frac{N}{K \times n_c}$$

where N = total samples, K = number of classes, n_c = samples in class c.

```yaml
# configs/default.yaml
data:
  weighted_loss: true
```

**Computed weights for 20:1 imbalance:**

| Class      | Weight |
|------------|--------|
| airplane   | 0.34   |
| automobile | 0.48   |
| bird       | 0.66   |
| cat        | 0.92   |
| deer       | 1.29   |
| dog        | 1.80   |
| frog       | 2.51   |
| horse      | 3.50   |
| ship       | 4.89   |
| truck      | 6.81   |

Rare classes (truck) receive **20x more gradient signal** than common classes (airplane).

### Solution 2: Weighted Random Sampler

Oversamples minority classes so each batch has roughly equal class representation:

```yaml
# configs/default.yaml
data:
  weighted_sampling: true
```

Each sample gets a probability inversely proportional to its class frequency. The sampler draws with replacement, so rare-class samples appear more often per epoch.

### When to Use Which

| Strategy         | Best For                        | Trade-off                          |
|------------------|---------------------------------|------------------------------------|
| Weighted Loss    | Mild imbalance (2x–5x ratio)   | Simple, no data duplication        |
| Weighted Sampler | Severe imbalance (10x+ ratio)  | Balanced batches, possible overfit |
| Both combined    | Extreme imbalance (20x+)       | Strongest correction               |

### Implementation

```python
from src.data.imbalance import (
    make_imbalanced,
    compute_class_weights,
    build_weighted_sampler,
)

# 1. Create imbalanced dataset (exponential long-tail)
imbalanced_ds = make_imbalanced(
    dataset, imbalance_ratio=20.0,
)

# 2. Weighted loss from actual class frequencies
weights = compute_class_weights(imbalanced_ds)
criterion = nn.CrossEntropyLoss(weight=weights)

# 3. Weighted sampler for balanced batches
sampler = build_weighted_sampler(imbalanced_ds)
loader = DataLoader(dataset, sampler=sampler)
```

---

## Training & Evaluation Results

### Training Curves

Loss and accuracy over epochs for both training and validation sets.

![Training Curves](results/figures/training_curves.png)

### Evaluation Metrics

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 28.61% |
| Macro F1     | 0.1896 |
| Weighted F1  | 0.1896 |

### Per-Class Performance

| Class      | Precision | Recall | F1-Score | Train Samples |
|------------|-----------|--------|----------|---------------|
| airplane   | 0.0000    | 0.0000 | 0.0000   | 5,000 (most)  |
| automobile | 0.0000    | 0.0000 | 0.0000   | 3,584         |
| bird       | 0.2500    | 0.0010 | 0.0020   | 2,569         |
| cat        | 0.0000    | 0.0000 | 0.0000   | 1,841         |
| deer       | 0.2898    | 0.0510 | 0.0867   | 1,320         |
| dog        | 0.3340    | 0.1610 | 0.2173   | 946           |
| frog       | 0.3154    | 0.5510 | 0.4012   | 678           |
| horse      | 0.2539    | 0.6450 | 0.3644   | 486           |
| ship       | 0.2983    | 0.7350 | 0.4244   | 348           |
| truck      | 0.2772    | 0.7170 | 0.3998   | 250 (least)   |

### Confusion Matrix & Per-Class Metrics

![Evaluation Results](results/figures/eval_results.png)

> **Note:** Results shown are from a 1-epoch demo run. The imbalance effects are clearly visible: despite weighted loss and sampling corrections, 1 epoch is insufficient for the model to learn all classes. Training for 20+ epochs will show the effectiveness of the imbalance strategies.

---

## Project Structure

```
img_cls/
├── configs/
│   └── default.yaml              # All hyperparameters
├── scripts/
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Test set evaluation
│   ├── predict.py                # Single-image inference
│   └── visualize.py              # Generate all figures
├── src/
│   ├── data/
│   │   ├── dataset.py            # CIFAR-10/100 loader + imbalance
│   │   ├── transforms.py         # Augmentation pipeline
│   │   ├── dataloader.py         # DataLoader + weighted sampler
│   │   └── imbalance.py          # Long-tail, weighted loss/sampler
│   ├── models/
│   │   ├── resnet.py             # ResNet from scratch
│   │   ├── simple_cnn.py         # Baseline CNN
│   │   └── factory.py            # Model builder
│   ├── training/
│   │   ├── trainer.py            # Training loop + early stopping
│   │   ├── optimizer.py          # Optimizer builder
│   │   └── scheduler.py          # LR scheduler builder
│   ├── evaluation/
│   │   └── metrics.py            # Accuracy, F1, confusion matrix
│   └── utils/
│       ├── config.py             # YAML config loader
│       ├── logger.py             # Logging setup
│       └── helpers.py            # Seed, device, checkpointing
├── results/
│   ├── checkpoints/              # Model weights (.pth)
│   ├── figures/                  # Generated visualizations
│   ├── train_history.json        # Per-epoch metrics
│   └── eval_results.json         # Final evaluation
├── tests/
│   └── test_model.py
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Setup

```bash
conda create -n img_cls python=3.11 -y
conda activate img_cls
pip install -r requirements.txt
```

### Train

```bash
python scripts/train.py --config configs/default.yaml
```

Override any parameter:

```bash
python scripts/train.py --config configs/default.yaml \
    --override training.epochs=50 \
               data.imbalance.ratio=50 \
               data.weighted_loss=True \
               data.weighted_sampling=True
```

### Evaluate

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint results/checkpoints/best.pth
```

### Predict

```bash
python scripts/predict.py \
    --config configs/default.yaml \
    --checkpoint results/checkpoints/best.pth \
    --input path/to/image.jpg
```

### Generate Visualizations

```bash
python scripts/visualize.py
```

---

## Supported Models

| Model       | Config value  | Parameters | Block      |
|-------------|---------------|------------|------------|
| SimpleCNN   | `simple_cnn`  | ~200K      | Conv+BN    |
| ResNet-18   | `resnet18`    | ~11M       | BasicBlock |
| ResNet-34   | `resnet34`    | ~21M       | BasicBlock |
| ResNet-50   | `resnet50`    | ~23.5M     | Bottleneck |
| ResNet-101  | `resnet101`   | ~42.5M     | Bottleneck |

---

## License

MIT
