# Image Classification with ResNet-50 on Imbalanced CIFAR-10

A from-scratch PyTorch implementation of ResNet-50 trained on a **long-tail imbalanced** version of CIFAR-10, with **7 independent imbalance strategies** and **1 combined industry-best solution** — all benchmarked side-by-side.

---

## Table of Contents

- [Dataset Visualization](#dataset-visualization)
- [Model Architecture](#model-architecture)
- [Imbalance Strategies (7 + 1 Combined)](#imbalance-strategies)
  - [S1. Weighted Cross-Entropy](#s1-weighted-cross-entropy-cost-sensitive-learning)
  - [S2. Oversampling](#s2-oversampling-weighted-random-sampler)
  - [S3. Focal Loss](#s3-focal-loss)
  - [S4. Class-Balanced Loss](#s4-class-balanced-loss)
  - [S5. Label Smoothing](#s5-label-smoothing)
  - [S6. Mixup](#s6-mixup)
  - [S7. CutMix](#s7-cutmix)
  - [S8. Combined Best](#s8-combined-industry-best)
- [Benchmark Results](#benchmark-results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)

---

## Dataset Visualization

### Sample Images

![Sample Images](results/figures/sample_images.png)

### Imbalanced Class Distribution (20:1 Ratio)

The training set uses **exponential decay** to create a long-tail distribution. The test set stays balanced for fair evaluation.

![Class Distribution](results/figures/class_distribution.png)

| Class      | Train Samples | Ratio |
|------------|---------------|-------|
| airplane   | 5,000         | 1.00x |
| automobile | 3,584         | 0.72x |
| bird       | 2,569         | 0.51x |
| cat        | 1,841         | 0.37x |
| deer       | 1,320         | 0.26x |
| dog        | 946           | 0.19x |
| frog       | 678           | 0.14x |
| horse      | 486           | 0.10x |
| ship       | 348           | 0.07x |
| truck      | 250           | 0.05x |

---

## Model Architecture

ResNet-50 built **entirely from scratch** using `torch.nn` — no `torchvision.models`.

```
Input (3×32×32) → Stem(7×7 conv, BN, ReLU, MaxPool)
  → Layer1: Bottleneck×3  [64→256]
  → Layer2: Bottleneck×4  [256→512,  stride=2]
  → Layer3: Bottleneck×6  [512→1024, stride=2]
  → Layer4: Bottleneck×3  [1024→2048, stride=2]
  → Head: AdaptiveAvgPool → Flatten → Dropout → Linear(2048, 10)
```

**Total parameters: 23,528,522 (~23.5M)**

---

## Imbalance Strategies

Each strategy has its own config file in `configs/` and outputs results to its own directory. Every strategy is tested **independently** — only one technique is enabled at a time.

---

### S1. Weighted Cross-Entropy (Cost-Sensitive Learning)

**Config:** `configs/s1_weighted_ce.yaml`

**Idea:** Assign inverse-frequency weights to the loss function so rare classes produce stronger gradients.

$$w_c = \frac{N}{K \times n_c}$$

**What changes:** Only `data.weighted_loss: true` and `loss.name: "ce"`

```bash
python scripts/train.py --config configs/s1_weighted_ce.yaml
```

**Implementation:** `src/data/imbalance.py` → `compute_class_weights()`

---

### S2. Oversampling (Weighted Random Sampler)

**Config:** `configs/s2_oversampling.yaml`

**Idea:** Sample minority classes more frequently so every batch is roughly balanced. No data is discarded.

**What changes:** Only `data.weighted_sampling: true`

```bash
python scripts/train.py --config configs/s2_oversampling.yaml
```

**Implementation:** `src/data/imbalance.py` → `build_weighted_sampler()`

---

### S3. Focal Loss

**Config:** `configs/s3_focal.yaml`

**Idea:** Down-weight easy (well-classified) examples and up-weight hard ones.

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- `gamma=0` → standard CE
- `gamma=2` → strongly penalizes confident wrong predictions

**What changes:** Only `loss.name: "focal"`, `gamma: 2.0`

```bash
python scripts/train.py --config configs/s3_focal.yaml
```

**Implementation:** `src/training/losses.py` → `FocalLoss`

**Reference:** [Lin et al., 2017](https://arxiv.org/abs/1708.02002)

---

### S4. Class-Balanced Loss

**Config:** `configs/s4_cb_loss.yaml`

**Idea:** Reweight classes by **effective number of samples** instead of raw count.

$$E_n = \frac{1 - \beta^n}{1 - \beta}$$

Each new sample has diminishing marginal value. More theoretically grounded than inverse-frequency.

**What changes:** Only `loss.name: "cb"`, `beta: 0.9999`

```bash
python scripts/train.py --config configs/s4_cb_loss.yaml
```

**Implementation:** `src/training/losses.py` → `ClassBalancedLoss`

**Reference:** [Cui et al., 2019](https://arxiv.org/abs/1901.05555)

---

### S5. Label Smoothing

**Config:** `configs/s5_label_smoothing.yaml`

**Idea:** Replace hard one-hot targets with soft targets to prevent overconfidence.

$$y_{smooth} = (1 - \epsilon) \cdot y_{hard} + \frac{\epsilon}{K}$$

Improves calibration so the model doesn't blindly predict majority classes.

**What changes:** Only `loss.name: "label_smoothing"`, `smoothing: 0.1`

```bash
python scripts/train.py --config configs/s5_label_smoothing.yaml
```

**Implementation:** `src/training/losses.py` → `LabelSmoothingCE`

---

### S6. Mixup

**Config:** `configs/s6_mixup.yaml`

**Idea:** Create virtual training samples by linearly interpolating pairs of images and their labels.

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j \qquad \tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

Acts as a regularizer that creates cross-class training signals.

**What changes:** Only `training.mixup.mode: "mixup"`, `alpha: 0.4`

```bash
python scripts/train.py --config configs/s6_mixup.yaml
```

**Implementation:** `src/training/mixup.py` → `mixup()`

**Reference:** [Zhang et al., 2018](https://arxiv.org/abs/1710.09412)

---

### S7. CutMix

**Config:** `configs/s7_cutmix.yaml`

**Idea:** Cut a rectangular patch from one image and paste it onto another. Labels are mixed proportional to patch area.

Stronger than Mixup because the model must learn localized features from partial images.

**What changes:** Only `training.mixup.mode: "cutmix"`, `alpha: 1.0`

```bash
python scripts/train.py --config configs/s7_cutmix.yaml
```

**Implementation:** `src/training/mixup.py` → `cutmix()`

**Reference:** [Yun et al., 2019](https://arxiv.org/abs/1905.04899)

---

### S8. Combined Industry Best

**Config:** `configs/s8_combined_best.yaml`

**Idea:** Stack the strongest techniques from different levels:

| Level          | Technique             | Purpose                               |
|----------------|-----------------------|---------------------------------------|
| **Data**       | Weighted Sampler      | Balanced mini-batches                 |
| **Data**       | CutMix                | Cross-class regularization            |
| **Data**       | Strong Augmentation   | ColorJitter + RandomErasing           |
| **Algorithm**  | CB Focal Loss         | Effective-number + focus on hard      |
| **Algorithm**  | Weighted loss         | Inverse-frequency gradient scaling    |

**What changes:** All of the above enabled together.

```bash
python scripts/train.py --config configs/s8_combined_best.yaml
```

**Implementation:** `src/training/losses.py` → `CBFocalLoss`

---

## Benchmark Results

All 8 strategies trained on the same imbalanced CIFAR-10 (20:1 ratio) with identical hyperparameters. Only the imbalance strategy differs.

### Summary Table

![Summary Table](results/figures/comparison_summary.png)

| # | Strategy       | Accuracy | Macro F1 |
|---|----------------|----------|----------|
| 1 | Weighted CE    | 27.12%   | 0.2397   |
| 2 | Oversampling   | **35.59%** | **0.3259** |
| 3 | Focal Loss     | 26.34%   | 0.2015   |
| 4 | CB Loss        | 27.88%   | 0.2379   |
| 5 | Label Smoothing| 24.50%   | 0.1486   |
| 6 | Mixup          | 24.40%   | 0.1413   |
| 7 | CutMix         | 22.81%   | 0.1316   |
| 8 | Combined Best  | 21.81%   | 0.1296   |

> **Note:** Results are from 1-epoch demo runs. With 20+ epochs, the algorithm-level strategies (Focal, CB, Combined) typically surpass data-level-only approaches. Oversampling leads early because balanced batches give every class equal exposure from the first epoch.

### Accuracy Comparison

![Accuracy Comparison](results/figures/comparison_accuracy.png)

### Macro F1 Comparison

![F1 Comparison](results/figures/comparison_f1.png)

### Training Curves — All Strategies

![Training Curves](results/figures/comparison_training.png)

### Per-Class F1 Heatmap

![Per-Class F1](results/figures/comparison_per_class_f1.png)

---

## Project Structure

```
img_cls/
├── configs/
│   ├── default.yaml                # Default config (CB Focal)
│   ├── s1_weighted_ce.yaml         # Strategy 1
│   ├── s2_oversampling.yaml        # Strategy 2
│   ├── s3_focal.yaml               # Strategy 3
│   ├── s4_cb_loss.yaml             # Strategy 4
│   ├── s5_label_smoothing.yaml     # Strategy 5
│   ├── s6_mixup.yaml               # Strategy 6
│   ├── s7_cutmix.yaml              # Strategy 7
│   └── s8_combined_best.yaml       # Strategy 8
├── scripts/
│   ├── train.py                    # Training pipeline
│   ├── evaluate.py                 # Test set evaluation
│   ├── predict.py                  # Single-image inference
│   ├── visualize.py                # Dataset & single-run figures
│   ├── run_all.py                  # Run all 8 experiments
│   └── compare.py                  # Cross-strategy comparison
├── src/
│   ├── data/
│   │   ├── dataset.py              # CIFAR-10 + imbalance injection
│   │   ├── transforms.py           # Augmentation pipeline
│   │   ├── dataloader.py           # DataLoader + weighted sampler
│   │   └── imbalance.py            # Long-tail, class weights, sampler
│   ├── models/
│   │   ├── resnet.py               # ResNet from scratch
│   │   ├── simple_cnn.py           # Baseline CNN
│   │   └── factory.py              # Model builder
│   ├── training/
│   │   ├── trainer.py              # Training loop + mixup + early stop
│   │   ├── losses.py               # Focal, CB, LabelSmoothing, CBFocal
│   │   ├── mixup.py                # Mixup + CutMix
│   │   ├── optimizer.py            # Optimizer builder
│   │   └── scheduler.py            # LR scheduler builder
│   ├── evaluation/
│   │   └── metrics.py              # Accuracy, F1, confusion matrix
│   └── utils/
│       ├── config.py               # YAML config loader
│       ├── logger.py               # Logging setup
│       └── helpers.py              # Seed, device, checkpointing
├── results/
│   ├── s1_weighted_ce/             # Strategy 1 outputs
│   ├── s2_oversampling/            # Strategy 2 outputs
│   ├── ...                         # Strategies 3-7
│   ├── s8_combined_best/           # Strategy 8 outputs
│   └── figures/                    # All visualizations
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

### Run a Single Strategy

```bash
python scripts/train.py --config configs/s3_focal.yaml
python scripts/evaluate.py --config configs/s3_focal.yaml \
    --checkpoint results/s3_focal/checkpoints/best.pth
```

### Run All 8 Strategies

```bash
python scripts/run_all.py --epochs 20
```

### Generate Comparison Charts

```bash
python scripts/compare.py
```

### Generate Dataset Visualizations

```bash
python scripts/visualize.py
```

### Override Any Parameter

```bash
python scripts/train.py --config configs/s3_focal.yaml \
    --override training.epochs=50 loss.gamma=3.0
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

## References

1. He et al., 2016 — [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
2. Lin et al., 2017 — [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
3. Cui et al., 2019 — [Class-Balanced Loss](https://arxiv.org/abs/1901.05555)
4. Zhang et al., 2018 — [Mixup](https://arxiv.org/abs/1710.09412)
5. Yun et al., 2019 — [CutMix](https://arxiv.org/abs/1905.04899)

---

## License

MIT
