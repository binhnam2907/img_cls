# Image Classification with ResNet-50 on Imbalanced CIFAR-10

A from-scratch PyTorch implementation of ResNet-50 trained on a **long-tail imbalanced** version of CIFAR-10, with systematic benchmarking of imbalance-handling techniques organized by the taxonomy from [Gao et al., 2025](https://doi.org/10.1007/s11704-025-50274-7).

---

## Table of Contents

- [Why Class Imbalance Matters](#why-class-imbalance-matters)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Formal Problem Statement](#formal-problem-statement)
- [How We Handle Imbalance — 5 Categories of Solutions](#how-we-handle-imbalance--5-categories-of-solutions)
  - [Category 1: Data Re-balancing](#category-1-data-re-balancing--fix-the-data)
  - [Category 2: Cost-Sensitive Learning](#category-2-cost-sensitive-learning--fix-the-loss)
  - [Category 3: Training Strategy](#category-3-training-strategy--fix-the-training-loop)
  - [Category 4: Logit Adjustment](#category-4-logit-adjustment--fix-the-predictions)
  - [Category 5: Combined Best Practice](#category-5-combined--stack-the-best-of-each)
- [Benchmark Results & Analysis](#benchmark-results--analysis)
- [Imbalanced Data Toolkit](#imbalanced-data-toolkit)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [References](#references)

---

## Why Class Imbalance Matters

In real-world datasets, some classes naturally have far fewer samples than others. A medical dataset might have 10,000 healthy scans but only 50 disease cases. A fraud detection system sees millions of legitimate transactions for every fraudulent one.

When a model trains on such data, it learns a simple shortcut: **always predict the majority class**. If 95% of samples are "healthy", predicting "healthy" every time gives 95% accuracy — while missing every single disease case.

This project deliberately creates this problem on CIFAR-10, then systematically tests solutions from five different angles:

```
The Problem:  airplane has 5,000 images, but truck has only 250.
              The model learns to ignore trucks entirely.

The Goal:     Force the model to learn ALL classes equally well,
              even when some have 20x fewer training images.
```

We organize solutions using the taxonomy from Gao et al. (2025), a comprehensive survey covering 200+ papers:

```
Where do we intervene?

  [Data] ──> [Model] ──> [Loss Function] ──> [Training Loop] ──> [Prediction]
    │              │              │                  │                  │
    ▼              ▼              ▼                  ▼                  ▼
  Cat.1          Cat.5         Cat.2              Cat.3              Cat.4
  Re-balance    Ensemble     Cost-Sensitive     Training           Logit
  the data      models       learning           Strategy           Adjustment
```

---

## Dataset

### Sample Images

![Sample Images](results/figures/sample_images.png)

### Imbalanced Class Distribution (20:1 Ratio)

We start with CIFAR-10 (50,000 balanced training images) and apply **exponential decay** to simulate a real-world long-tail distribution. The majority class keeps all 5,000 images; the rarest class is subsampled down to just 250. The test set stays balanced so evaluation is fair.

![Class Distribution](results/figures/class_distribution.png)

The subsampling follows an exponential decay function:

\[ n_c = n_{\max} \cdot \rho^{\frac{c}{K-1}}, \quad \rho = \frac{n_{\min}}{n_{\max}} = \frac{1}{20} \]

where \(c\) is the class index (0 to K-1), producing the distribution:

| Class      | Train Samples | Ratio | Imbalance factor \(\rho^{c/9}\) |
|------------|---------------|-------|--------------------------------|
| airplane   | 5,000         | 1.00x | 1.000 |
| automobile | 3,584         | 0.72x | 0.717 |
| bird       | 2,569         | 0.51x | 0.514 |
| cat        | 1,841         | 0.37x | 0.368 |
| deer       | 1,320         | 0.26x | 0.264 |
| dog        | 946           | 0.19x | 0.189 |
| frog       | 678           | 0.14x | 0.136 |
| horse      | 486           | 0.10x | 0.097 |
| ship       | 348           | 0.07x | 0.070 |
| truck      | 250           | 0.05x | 0.050 |

**Total training samples:** N = 17,022 (down from 50,000 balanced).

---

## Model Architecture

ResNet-50 built **entirely from scratch** using `torch.nn` — no `torchvision.models`.

```
Input (3x32x32) -> Stem(7x7 conv, BN, ReLU, MaxPool)
  -> Layer1: Bottleneck x3  [64 -> 256]
  -> Layer2: Bottleneck x4  [256 -> 512,  stride=2]
  -> Layer3: Bottleneck x6  [512 -> 1024, stride=2]
  -> Layer4: Bottleneck x3  [1024 -> 2048, stride=2]
  -> Head: AdaptiveAvgPool -> Flatten -> Dropout -> Linear(2048, 10)
```

Each **Bottleneck** block computes:

\[ \mathbf{y} = \sigma\!\big(\,\text{BN}(\text{Conv}_{1\times1} \to \text{Conv}_{3\times3} \to \text{Conv}_{1\times1}(\mathbf{x})) + \mathbf{x}\,\big) \]

The residual shortcut \(+\mathbf{x}\) allows gradients to flow directly through the identity path, preventing vanishing gradients in deep networks. The bottleneck design reduces intermediate channels (256 -> 64 -> 64 -> 256) so each block has 3x fewer FLOPs than a naive 3x3 stack.

**Parameters:** ~23.5M. Deliberately over-parameterized for CIFAR-10 to study how imbalance affects a high-capacity model.

---

## Formal Problem Statement

Let \(\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}\) be a training set with \(K\) classes, where class \(c\) has \(n_c\) samples. The distribution is **long-tailed** if the class frequencies are highly skewed:

\[ n_1 \gg n_2 \gg \cdots \gg n_K, \quad \text{with imbalance ratio } \rho = \frac{n_1}{n_K} \gg 1 \]

A classifier \(f_\theta : \mathbb{R}^{d} \to \mathbb{R}^{K}\) produces logits \(\mathbf{z} = f_\theta(\mathbf{x})\). The standard empirical risk minimization (ERM) objective is:

\[ \min_\theta \; \frac{1}{N} \sum_{i=1}^{N} \ell(f_\theta(\mathbf{x}_i), y_i) \]

Under imbalance, the \(1/N\) average is dominated by majority classes. Each gradient step receives approximately \(n_c / N\) contribution from class \(c\), meaning minority classes contribute negligibly to parameter updates. This causes two pathologies:

1. **Decision boundary bias:** The classifier hyperplane shifts toward minority regions, making the model predict majority classes even for minority inputs.
2. **Representation collapse:** The backbone underallocates feature capacity to minority classes, producing poorly separable embeddings for rare classes.

The methods below attack these pathologies at different points in the pipeline.

---

## How We Handle Imbalance — 5 Categories of Solutions

Each method intervenes at a different point in the machine learning pipeline. We test each one **independently** (only one technique enabled at a time) so we can measure its isolated effect.

### Category 1: Data Re-balancing — *Fix the Data*

> **Core idea:** Modify the sampling distribution so the model receives balanced class exposure, changing the effective training distribution \(\tilde{P}\) without modifying \(\ell\) or \(f_\theta\).

---

#### S1. Random Oversampling (WeightedRandomSampler)

**The problem it solves:** Under standard uniform sampling, the probability of drawing a sample from class \(c\) in any batch is \(P(c) = n_c / N\). For truck: \(P(\text{truck}) = 250/17022 = 1.5\%\). The model sees truck examples so rarely that their gradients are washed out by majority classes.

**Mathematical formulation.** We assign each sample \((\mathbf{x}_i, y_i)\) a sampling weight:

\[ w_i = \frac{1}{n_{y_i}} \]

where \(n_{y_i}\) is the count of class \(y_i\). The probability of drawing sample \(i\) becomes:

\[ P(\text{draw } i) = \frac{w_i}{\sum_{j=1}^{N} w_j} = \frac{1/n_{y_i}}{\sum_{c=1}^{K} n_c \cdot (1/n_c)} = \frac{1}{n_{y_i} \cdot K} \]

This means every **class** has equal probability \(1/K\) of being represented, regardless of its size. Within each class, samples are drawn uniformly.

**Concrete example (our data):**

```
Without sampler (uniform draw from 17,022 samples):
  P(airplane sample)  = 5000/17022 = 29.4%  per draw
  P(truck sample)     = 250/17022  =  1.5%  per draw

With WeightedRandomSampler:
  P(any airplane sample) = 1/(5000 × 10) = 0.002%  per draw
  P(any truck sample)    = 1/(250 × 10)  = 0.040%  per draw
  P(airplane CLASS)      = 5000 × 0.002% = 10%     ← balanced!
  P(truck CLASS)         = 250 × 0.040%  = 10%     ← balanced!
```

**Gradient impact.** Under balanced sampling, the expected gradient per step becomes:

\[ \mathbb{E}_{\tilde{P}}[\nabla_\theta \ell] = \frac{1}{K} \sum_{c=1}^{K} \mathbb{E}_{\mathbf{x} \sim \mathcal{D}_c}[\nabla_\theta \ell(f_\theta(\mathbf{x}), c)] \]

Every class contributes equally, compared to the standard ERM gradient where majority classes dominate.

**Failure mode:** Each truck image is sampled ~20x more often than each airplane image. After enough epochs, the model memorizes the 250 truck images. The training loss on truck drops to near-zero, but test accuracy plateaus — the model has overfit to the specific pixel patterns of those 250 images rather than learning the general concept of "truck".

**Config:** `data.weighted_sampling: true`

---

#### S2. SMOTE (Synthetic Minority Over-sampling Technique)

**The problem it solves:** Oversampling via duplication gives no new information — the model sees identical copies. The gradient for duplicate samples is identical, providing no additional learning signal. SMOTE creates genuinely new points in the feature space.

**Algorithm (Chawla et al., 2002):**

```
Algorithm: SMOTE
Input:  Minority class samples X_min = {x_1, ..., x_m}, k neighbors, N_syn
Output: Synthetic samples S

1. For each x_i ∈ X_min:
   a. Find k nearest neighbors of x_i within X_min using L2 distance
   b. Repeat floor(N_syn / m) times:
      i.   Randomly select neighbor x_nn from the k neighbors
      ii.  Sample λ ~ Uniform(0, 1)
      iii. x_new = x_i + λ · (x_nn - x_i)
      iv.  S ← S ∪ {x_new}
2. Return S
```

**Mathematical formulation.** Given a minority sample \(\mathbf{x}_i\) and its k-nearest neighbor \(\mathbf{x}_{nn}\) (both from the same class), the synthetic sample is:

\[ \mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_{nn} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}(0, 1) \]

Geometrically, \(\mathbf{x}_{\text{new}}\) lies on the **line segment** between \(\mathbf{x}_i\) and \(\mathbf{x}_{nn}\) in pixel space. Since both endpoints belong to the same class, the convex combination is likely to be a valid representative — assuming the class manifold is locally convex.

**Concrete example (our data):**

```
truck has 250 samples, airplane has 5000.
To reach target_ratio = 1.0: need 5000 - 250 = 4750 synthetic truck images.

For each of the 250 truck images:
  Find k=5 nearest truck neighbors
  Generate floor(4750/250) = 19 synthetics each

Synthetic image:
  truck_42  = [0.3, 0.5, 0.2, ...]     (flattened 32×32×3 = 3072-dim vector)
  truck_nn  = [0.4, 0.4, 0.3, ...]     (nearest neighbor)
  λ = 0.7
  new_truck = 0.7 × truck_42 + 0.3 × truck_nn
            = [0.33, 0.47, 0.23, ...]   ← blend of two real trucks
```

**Why pixel-space interpolation has limitations.** For high-dimensional images, the line between two images in pixel space may pass through regions that don't look like valid images. A blend of two trucks can produce a ghostly double-exposure. However, for CIFAR-10's tiny 32x32 resolution, pixel blending is a reasonable approximation — the images are low-resolution enough that interpolation mostly produces "blurry but plausible" variants.

**Complexity:** \(O(m^2 \cdot d)\) for the k-NN step (where \(m\) is the minority class size, \(d\) is dimensionality). For 250 samples at 3072 dimensions, this is trivial.

**Config:** `data.oversampling.method: "smote"`, `data.oversampling.k_neighbors: 5`

---

#### S3. ADASYN (Adaptive Synthetic Sampling)

**The problem it solves:** SMOTE distributes synthetic samples uniformly across all minority instances. But samples deep inside their class cluster are already well-classified — generating more of them wastes capacity. Samples near the decision boundary (surrounded by majority-class neighbors) are the ones the model struggles with.

**Algorithm (He et al., 2008):**

```
Algorithm: ADASYN
Input:  Minority class X_min, majority class X_maj, k, β ∈ (0,1]
Output: Synthetic samples S

1. Compute G = (|X_maj| - |X_min|) × β  (total synthetics needed)
2. For each x_i ∈ X_min:
   a. Find k nearest neighbors of x_i in the FULL dataset (both classes)
   b. Compute difficulty ratio:
        Γ_i = (# neighbors from majority class) / k
   c. Normalize: r_i = Γ_i / Σ_j Γ_j
3. For each x_i:
   a. Compute g_i = r_i × G  (number of synthetics for x_i)
   b. Generate g_i synthetic samples using SMOTE interpolation
4. Return S
```

**Key distinction from SMOTE.** The difficulty ratio \(\Gamma_i\) measures how "borderline" each minority sample is:

\[ \Gamma_i = \frac{|\{\mathbf{x} \in \text{kNN}(\mathbf{x}_i) : \text{class}(\mathbf{x}) \neq \text{class}(\mathbf{x}_i)\}|}{k} \]

- \(\Gamma_i \approx 0\): sample is surrounded by same-class neighbors (safe, easy) -> few synthetics
- \(\Gamma_i \approx 1\): sample is surrounded by other-class neighbors (borderline, hard) -> many synthetics

This creates a **density-adaptive** augmentation that concentrates new samples near the decision boundary, exactly where the classifier needs the most help.

**Concrete example:**

```
truck_42 has 5 nearest neighbors: [truck, airplane, airplane, truck, airplane]
  Γ_42 = 3/5 = 0.6  (hard — surrounded by airplanes)
  → gets many synthetics

truck_100 has 5 nearest neighbors: [truck, truck, truck, truck, truck]
  Γ_100 = 0/5 = 0.0  (safe — well-separated from other classes)
  → gets zero synthetics
```

**Risk.** If a minority sample near the boundary is actually an outlier or mislabeled, ADASYN amplifies that noise by generating many synthetic variants around it. This can push the decision boundary in the wrong direction.

**Config:** `data.oversampling.method: "adasyn"`

---

### Category 2: Cost-Sensitive Learning — *Fix the Loss*

> **Core idea:** Modify the loss function \(\ell\) so that errors on minority classes produce disproportionately large gradients. The training data remains unchanged; only the penalty structure changes.

---

#### S4. Weighted Cross-Entropy

**The problem it solves:** Standard cross-entropy loss is:

\[ \mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \log p_{y_i} = -\frac{1}{N} \sum_{c=1}^{K} \sum_{i: y_i = c} \log p_c(\mathbf{x}_i) \]

The contribution of class \(c\) to the total gradient is proportional to \(n_c / N\). For truck: \(250 / 17022 = 1.5\%\) of the total gradient. The optimizer barely notices when it misclassifies a truck.

**Mathematical formulation.** Assign weight \(w_c\) to each class:

\[ \mathcal{L}_{\text{WCE}} = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \cdot \log p_{y_i} \]

with inverse-frequency weights:

\[ w_c = \frac{N}{K \cdot n_c} \]

**Gradient derivation.** The gradient of WCE with respect to the logit \(z_c\) for a sample with true class \(y\) is:

\[ \frac{\partial \mathcal{L}_{\text{WCE}}}{\partial z_c} = w_y \cdot (p_c - \mathbb{1}[c = y]) \]

For a truck sample (\(w_{\text{truck}} = 6.81\)), every gradient is amplified by 6.81x compared to an airplane sample (\(w_{\text{airplane}} = 0.34\)). This compensates for truck appearing in \(1/20\)th of the batches.

**Worked example with our data (\(N = 17,022\), \(K = 10\)):**

```
Class weights w_c = N / (K × n_c):

  airplane:    17022 / (10 × 5000) = 0.340
  automobile:  17022 / (10 × 3584) = 0.475
  bird:        17022 / (10 × 2569) = 0.663
  cat:         17022 / (10 × 1841) = 0.925
  deer:        17022 / (10 × 1320) = 1.290
  dog:         17022 / (10 × 946)  = 1.799
  frog:        17022 / (10 × 678)  = 2.511
  horse:       17022 / (10 × 486)  = 3.502
  ship:        17022 / (10 × 348)  = 4.891
  truck:       17022 / (10 × 250)  = 6.809

Ratio truck/airplane = 6.809 / 0.340 = 20.0x  (exactly the imbalance ratio)
```

**Limitation.** The weights are fixed before training. If the model learns truck well early on (perhaps truck is visually distinctive), it keeps receiving 6.81x gradients for truck even when it already classifies truck perfectly. This can cause oscillation or push the decision boundary too far toward truck's territory, hurting neighboring classes.

**Config:** `data.weighted_loss: true`, `loss.name: "ce"`

---

#### S5. Focal Loss

**The problem it solves:** In a typical training batch, most samples are already correctly classified with high confidence. These "easy" examples still contribute gradient, wasting optimization budget. In imbalanced settings, the majority of easy examples come from majority classes — so the model keeps refining its already-good majority predictions instead of improving on the rare, hard minority examples.

**Mathematical formulation (Lin et al., 2017).** Standard CE for a sample with predicted probability \(p_t\) (probability assigned to the true class):

\[ \mathcal{L}_{\text{CE}} = -\log(p_t) \]

Focal Loss adds a modulating factor:

\[ \mathcal{L}_{\text{FL}} = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t) \]

where \(\gamma \geq 0\) is the focusing parameter and \(\alpha_t\) is an optional class-balancing weight.

**Gradient analysis.** The gradient with respect to \(p_t\) is:

\[ \frac{\partial \mathcal{L}_{\text{FL}}}{\partial p_t} = -\alpha_t \left[ \gamma (1 - p_t)^{\gamma - 1} \log(p_t) + \frac{(1 - p_t)^\gamma}{p_t} \right] \]

The key term is \((1 - p_t)^\gamma\):

| \(p_t\) (confidence) | \(\gamma=0\) (CE) | \(\gamma=1\) | \(\gamma=2\) | \(\gamma=5\) |
|---|---|---|---|---|
| 0.10 (hard) | 1.000 | 0.900 | 0.810 | 0.590 |
| 0.50 | 1.000 | 0.500 | 0.250 | 0.031 |
| 0.90 (easy) | 1.000 | 0.100 | 0.010 | 0.00001 |
| 0.99 (trivial) | 1.000 | 0.010 | 0.0001 | 10\(^{-10}\) |

With \(\gamma = 2\), a sample classified at 90% confidence receives only **1% of the loss** it would receive under standard CE. A sample at 10% confidence receives **81% of its CE loss** — almost full strength. The model naturally concentrates its learning on whatever it's currently getting wrong.

**Why this helps imbalance.** In an imbalanced setting, the model quickly becomes confident on majority-class examples (easy examples). Focal Loss effectively silences these, leaving almost all gradient signal coming from hard examples — which tend to be minority-class samples. This achieves a similar effect to class weighting, but **adaptively**: the weights change as the model learns, rather than being fixed before training.

**Hyperparameter sensitivity:**
- \(\gamma = 0\): identical to CE (no focusing)
- \(\gamma = 2\): standard choice; suppresses samples above ~80% confidence
- \(\gamma \geq 3\): aggressive focusing; can destabilize training if too many samples are suppressed simultaneously

**Config:** `loss.name: "focal"`, `loss.gamma: 2.0`

---

#### S6. Class-Balanced Loss

**The problem it solves:** Inverse-frequency weighting (S4) assigns \(w_c \propto 1/n_c\). This assumes that doubling the number of samples doubles the information. In reality, there are diminishing returns: the first 100 airplane images cover diverse viewpoints, but the 5000th airplane image is likely very similar to images already seen.

**Theoretical foundation (Cui et al., 2019).** Define the **effective number** of samples as the expected volume of feature space covered by \(n\) random samples drawn from a class whose feature space has total volume 1:

\[ E_n = \frac{1 - \beta^n}{1 - \beta}, \quad \beta \in [0, 1) \]

The parameter \(\beta\) controls how much overlap exists between samples:
- \(\beta \to 0\): no overlap, every sample is unique: \(E_n = n\) (linear, same as inverse-frequency)
- \(\beta \to 1\): high overlap (many redundant samples): \(E_n \to n\) very slowly

The class-balanced weight is:

\[ w_c = \frac{1}{E_{n_c}} = \frac{1 - \beta}{1 - \beta^{n_c}} \]

and the loss becomes:

\[ \mathcal{L}_{\text{CB}} = -\frac{1}{N}\sum_{i=1}^{N} \frac{1 - \beta}{1 - \beta^{n_{y_i}}} \cdot \ell(f_\theta(\mathbf{x}_i), y_i) \]

where \(\ell\) can be CE or Focal Loss.

**Worked example (\(\beta = 0.9999\)):**

```
                     n_c     E_n = (1 - 0.9999^n) / 0.0001    w = 1/E_n     vs. 1/n_c

  airplane          5000     E = 3935.4                        w = 0.000254   vs. 0.000200
  truck              250     E = 247.0                         w = 0.004049   vs. 0.004000

  Weight ratio truck/airplane:
    CB:               0.004049 / 0.000254 = 15.9x
    Inverse-freq:     0.004000 / 0.000200 = 20.0x
```

The CB weight ratio (15.9x) is **less extreme** than the naive inverse-frequency ratio (20.0x). This is because the effective number for airplane (3935) is lower than the raw count (5000) — many of those 5000 airplanes are redundant. For truck, \(E_{250} \approx 247\), very close to the raw count — nearly every truck image is unique and informative.

**When paired with Focal Loss (CB-Focal):**

\[ \mathcal{L}_{\text{CB-FL}} = -\frac{1 - \beta}{1 - \beta^{n_{y_i}}} \cdot (1 - p_t)^\gamma \cdot \log(p_t) \]

This combines the static class reweighting of CB with the dynamic example-weighting of Focal, addressing both the class-level and example-level imbalance simultaneously.

**Config:** `loss.name: "cb"`, `loss.beta: 0.9999`

---

### Category 3: Training Strategy — *Fix the Training Loop*

> **Core idea:** Modify the training procedure — label representation, data augmentation, or the training schedule — without changing the loss formula itself. These techniques are general regularizers that provide secondary benefits for imbalanced learning.

---

#### S7. Label Smoothing

**The problem it solves:** With hard one-hot targets, the model is pushed toward infinite logit magnitude for the true class. The optimal logits under standard CE are \(z_y \to +\infty\) and \(z_c \to -\infty\) for \(c \neq y\). On imbalanced data, this causes the model to produce extremely high logits for majority classes and extremely negative logits for minority classes, making it nearly impossible for a minority class to "win" the argmax.

**Mathematical formulation.** Replace the hard target \(\mathbf{y}_{\text{hard}} = \mathbf{e}_c\) (one-hot vector) with:

\[ \mathbf{y}_{\text{smooth}} = (1 - \varepsilon) \cdot \mathbf{e}_c + \frac{\varepsilon}{K} \cdot \mathbf{1} \]

The loss becomes:

\[ \mathcal{L}_{\text{LS}} = (1 - \varepsilon) \cdot \mathcal{H}({\mathbf{e}_c}, \mathbf{p}) + \varepsilon \cdot \mathcal{H}(\mathbf{u}, \mathbf{p}) \]

where \(\mathcal{H}(\mathbf{q}, \mathbf{p}) = -\sum_c q_c \log p_c\) is cross-entropy, and \(\mathbf{u} = \frac{1}{K}\mathbf{1}\) is the uniform distribution.

**Concrete example (\(\varepsilon = 0.1\), \(K = 10\)):**

```
Hard target for "cat" (class 3):
  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

Smoothed target:
  Non-cat classes: ε/K = 0.1/10 = 0.01
  Cat class:       1 - ε + ε/K = 0.9 + 0.01 = 0.91

  [0.01, 0.01, 0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
```

**Effect on logit magnitude.** The optimal logits under label smoothing satisfy:

\[ p_c^* = (1-\varepsilon)\cdot\mathbb{1}[c=y] + \varepsilon/K \]

This means the model targets \(p_y = 0.91\) instead of \(p_y = 1.0\). The logits don't need to go to infinity — they converge to a finite gap. This prevents the extreme logit magnitudes that cause minority classes to receive near-zero probability.

**Connection to imbalance.** Label smoothing acts as a **calibration regularizer**: it keeps the model's probability estimates closer to the true predictive uncertainty. On imbalanced data, a well-calibrated model is less likely to assign probability 0.0 to minority classes (which would require \(z_{\text{minority}} \to -\infty\)).

**Config:** `loss.name: "label_smoothing"`, `loss.smoothing: 0.1`

---

#### S8. Mixup

**The problem it solves:** The model only sees individual training images with hard labels. It learns decision boundaries that pass through gaps between training points — fragile boundaries that don't generalize, especially where minority-class data is sparse.

**Mathematical formulation (Zhang et al., 2018).** For a random pair \((\mathbf{x}_i, y_i)\) and \((\mathbf{x}_j, y_j)\):

\[ \tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1 - \lambda) \mathbf{x}_j \]
\[ \tilde{\mathcal{L}} = \lambda \cdot \ell(f_\theta(\tilde{\mathbf{x}}), y_i) + (1 - \lambda) \cdot \ell(f_\theta(\tilde{\mathbf{x}}), y_j) \]

where \(\lambda \sim \text{Beta}(\alpha, \alpha)\).

**The Beta distribution and \(\alpha\).**

```
α = 0.1: Beta(0.1, 0.1) → U-shaped, λ near 0 or 1 (weak mixing)
α = 0.4: Beta(0.4, 0.4) → moderately U-shaped (default)
α = 1.0: Beta(1.0, 1.0) = Uniform(0, 1) (any mix ratio equally likely)
α = 2.0: Beta(2.0, 2.0) → bell-shaped around 0.5 (strong mixing)
```

**Theoretical justification.** Mixup implements vicinal risk minimization (VRM). Standard ERM places all probability mass on the observed data points. VRM smears probability mass into a **vicinity** around each point. The Mixup vicinity is the set of convex combinations of training pairs.

The VRM objective is:

\[ R_{\text{VRM}}(\theta) = \int \ell(f_\theta(\mathbf{x}), y) \, d\tilde{P}(\mathbf{x}, y) \]

where \(\tilde{P}\) has support on the convex hull of the training set. This smooths the decision boundary — the classifier can't fit arbitrary sharp boundaries because it must maintain smooth predictions across the interpolation path.

**Gradient analysis for imbalance.** Consider a batch where sample \(i\) is truck (minority) and sample \(j\) is airplane (majority), with \(\lambda = 0.3\):

\[ \tilde{\mathcal{L}} = 0.3 \cdot \ell(f_\theta(\tilde{\mathbf{x}}), \text{truck}) + 0.7 \cdot \ell(f_\theta(\tilde{\mathbf{x}}), \text{airplane}) \]

The gradient with respect to truck is scaled by 0.3 — the minority class contributes weakly. Since random pairing with the imbalanced distribution means truck is almost always the minority member of the pair, Mixup can actually **suppress** minority gradients relative to pure CE on the same samples.

**Config:** `training.mixup.mode: "mixup"`, `training.mixup.alpha: 0.4`

---

#### S9. CutMix

**The problem it solves:** Mixup blends entire images via linear interpolation, producing ghostly overlays. The resulting images lose spatial coherence — the model can't learn localized features. CutMix preserves spatial structure by keeping intact regions from each image.

**Mathematical formulation (Yun et al., 2019).** Instead of pixel-wise blending, CutMix creates a binary mask \(\mathbf{M} \in \{0, 1\}^{H \times W}\) defining a rectangular region:

\[ \tilde{\mathbf{x}} = \mathbf{M} \odot \mathbf{x}_i + (1 - \mathbf{M}) \odot \mathbf{x}_j \]

The mask \(\mathbf{M}\) is 0 inside a randomly placed rectangle and 1 outside. The mixing ratio is determined by the area:

\[ \lambda = 1 - \frac{(\text{box width}) \times (\text{box height})}{W \times H} \]

**Box generation.** Given \(\lambda \sim \text{Beta}(\alpha, \alpha)\):

```
1. cut_ratio = sqrt(1 - λ)
2. cut_h = H × cut_ratio,  cut_w = W × cut_ratio
3. Center (cx, cy) ~ Uniform([0,H]) × Uniform([0,W])
4. Box = clip([cx - cut_h/2, cy - cut_w/2, cx + cut_h/2, cy + cut_w/2])
5. Recompute λ = 1 - (actual box area) / (H × W)
```

**Why spatial coherence matters.** Consider a cat image with a dog patch pasted in. The model sees a realistic cat body + a realistic dog face. It must learn to:
1. Recognize "cat" from the partial cat regions
2. Recognize "dog" from the patch region
3. Weight its predictions by the area ratio

This forces the model to be robust to occlusion and to attend to local features rather than relying on global statistics. For imbalanced data, this is valuable because minority classes often have to be recognized from limited visual cues.

**Advantage over Mixup.** CutMix images contain regions of **unmodified, realistic** pixels. The features extracted from uncut regions are identical to features from the original image. Only the cut boundary introduces artifacts. This is a more information-preserving augmentation than the global blending of Mixup.

**Config:** `training.mixup.mode: "cutmix"`, `training.mixup.alpha: 1.0`

---

#### S10. Remix — *Imbalance-Aware Mixup*

**The problem it solves:** Standard Mixup uses the same \(\lambda\) for both image blending and label blending. This is mathematically elegant but imbalance-blind: when a minority sample gets a small \(\lambda\) (say 0.2), its label contribution is also only 20%, further suppressing the minority gradient. Remix decouples the two lambdas to explicitly boost minority representation.

**Mathematical formulation (Chou et al., 2020).** Given samples \((\mathbf{x}_a, y_a)\) and \((\mathbf{x}_b, y_b)\) with class counts \(n_{y_a}\) and \(n_{y_b}\):

**Feature mixing** (unchanged from Mixup):

\[ \tilde{\mathbf{x}} = \lambda_f \cdot \mathbf{x}_a + (1 - \lambda_f) \cdot \mathbf{x}_b, \quad \lambda_f \sim \text{Beta}(\alpha, \alpha) \]

**Label mixing** (biased toward minority):

\[ \lambda_l = \begin{cases}
\max(\lambda_f, \kappa) & \text{if } \lambda_f < \tau \text{ and } n_{y_a} \leq n_{y_b} \\
\min(\lambda_f, 1 - \kappa) & \text{if } \lambda_f < \tau \text{ and } n_{y_a} > n_{y_b} \\
\lambda_f & \text{otherwise}
\end{cases} \]

The loss is:

\[ \tilde{\mathcal{L}} = \lambda_l \cdot \ell(f_\theta(\tilde{\mathbf{x}}), y_a) + (1 - \lambda_l) \cdot \ell(f_\theta(\tilde{\mathbf{x}}), y_b) \]

**How the parameters interact:**

| Parameter | Role | Default |
|-----------|------|---------|
| \(\tau\) | Activation threshold: only bias when \(\lambda_f < \tau\) (minority has small feature weight) | 0.5 |
| \(\kappa\) | Minority label floor: minority class gets at least \(\kappa\) label weight | 0.9 |

**Worked example:**

```
Pair: (truck, airplane), λ_f = 0.3
  n_truck = 250, n_airplane = 5000 → truck is the minority

Standard Mixup:
  Image: 0.3 × truck + 0.7 × airplane
  Loss:  0.3 × L(pred, truck) + 0.7 × L(pred, airplane)
         ^^^                      ^^^
         truck gets only 30%      airplane gets 70%

Remix (τ=0.5, κ=0.9):
  λ_f = 0.3 < τ = 0.5 → activate!
  truck is minority (250 < 5000) → λ_l = max(0.3, 0.9) = 0.9
  Image: 0.3 × truck + 0.7 × airplane  (SAME image)
  Loss:  0.9 × L(pred, truck) + 0.1 × L(pred, airplane)
         ^^^                      ^^^
         truck gets 90% label!    airplane only 10%!
```

**Gradient analysis.** The expected gradient contribution from truck in a Remix pair is:

\[ \mathbb{E}[\nabla_\theta \ell_{\text{truck}}] \propto \mathbb{E}[\lambda_l] \gg \mathbb{E}[\lambda_f] \]

For \(\kappa = 0.9\), whenever \(\lambda_f < 0.5\) (half the time with Beta(1,1)), the truck label weight jumps to 0.9. This makes the expected \(\lambda_l\) for the minority class approximately 0.7 (vs. 0.5 under standard Mixup), a 40% boost in expected gradient magnitude.

**Config:** `training.mixup.mode: "remix"`, `training.mixup.remix.tau: 0.5`, `training.mixup.remix.kappa: 0.9`

---

#### S11. Decoupled Training (cRT)

**The problem it solves:** Deep networks can be decomposed into a **feature extractor** \(\phi(\mathbf{x}; \theta_{\text{backbone}})\) and a **linear classifier** \(W\mathbf{h} + \mathbf{b}\). Kang et al. (2020) demonstrated a surprising finding: **features learned on imbalanced data are actually high-quality**. The representation space separates classes well — it's only the linear classifier that's biased toward majority classes. Retraining the classifier on balanced data fixes the bias without harming the features.

**Two-stage algorithm:**

```
Algorithm: cRT (Classifier Re-Training)

Stage 1 — Representation learning (standard ERM on imbalanced data):
  1. Train full model f_θ = W · φ(x; θ_backbone) on imbalanced D
  2. Use standard CE loss, no class balancing
  3. The backbone learns rich, discriminative features for ALL classes
     (even minority classes get some representation because their
      features are useful for separating nearby majority classes)

Stage 2 — Classifier re-training:
  4. FREEZE θ_backbone (no gradient flows to backbone)
  5. Re-initialize the classifier head W, b
  6. Train ONLY W, b using class-balanced sampling/loss
  7. Use higher learning rate (e.g., 0.01 vs. 0.001) since we're
     only learning K × d_embed parameters (10 × 2048 = 20K params)
```

**Why imbalanced features are good.** Consider two minority classes (truck, ship) that are visually distinct. Even though they appear rarely, the backbone must learn to distinguish their features from nearby majority classes. The feature extractor doesn't directly produce class predictions — it extracts useful visual primitives (edges, textures, shapes) that are shared across classes. The imbalanced distribution provides enough signal for the backbone to learn a well-structured embedding space.

The problem is in the last linear layer \(W \in \mathbb{R}^{K \times d}\). Under imbalanced training, the weight vector \(\mathbf{w}_c\) for a majority class gets much larger in norm than \(\mathbf{w}_c\) for a minority class:

\[ \|\mathbf{w}_{\text{airplane}}\| \gg \|\mathbf{w}_{\text{truck}}\| \]

This is because the airplane weight vector receives 20x more gradient updates, pushing it to larger magnitudes. The classifier is biased simply because majority weight vectors are **bigger**, not because the features are bad.

**Stage 2 fixes this.** By retraining the classifier with balanced sampling, each \(\mathbf{w}_c\) receives equal gradient, and the norms equalize. The decision boundaries shift to their unbiased positions.

**Our implementation.** In Stage 2, we freeze all parameters except those containing `"head"` or `"fc"` in their name. We use SGD with lr=0.01 (10x the Stage 1 lr) and momentum=0.9. The high learning rate is appropriate because we're only optimizing 20K parameters (vs. 23.5M in the full model), and we want to quickly converge to the balanced solution.

**Config:** `training.crt.enabled: true`, `training.crt.classifier_epochs: 5`, `training.crt.lr: 0.01`

---

### Category 4: Logit Adjustment — *Fix the Predictions*

> **Core idea:** The model's output logits are biased because training on \(P_{\text{train}}(y)\) produces predictions that approximate \(P_{\text{train}}(y|\mathbf{x})\) rather than the desired balanced posterior \(P_{\text{balanced}}(y|\mathbf{x})\). We can correct for this bias mathematically by adjusting the logits.

**Bayesian derivation.** By Bayes' theorem:

\[ P_{\text{train}}(y|\mathbf{x}) = \frac{P(\mathbf{x}|y) \cdot P_{\text{train}}(y)}{P(\mathbf{x})} \]

Under a balanced distribution \(P_{\text{bal}}(y) = 1/K\):

\[ P_{\text{bal}}(y|\mathbf{x}) = \frac{P(\mathbf{x}|y) \cdot (1/K)}{P(\mathbf{x})} = \frac{P_{\text{train}}(y|\mathbf{x})}{P_{\text{train}}(y) \cdot K} \]

Taking the log and noting that \(z_c = \log P_{\text{train}}(c|\mathbf{x})\) (up to a constant):

\[ z_c^{\text{adjusted}} = z_c - \log P_{\text{train}}(c) = z_c - \log(n_c / N) \]

This is equivalent to **adding** \(\log(n_c / N)\) to the logits during training (since the negative sign cancels with the CE loss direction), which is what Balanced Softmax does.

---

#### S12. Balanced Softmax

**Mathematical formulation (Ren et al., 2020).** The standard softmax probability is:

\[ p_c = \frac{\exp(z_c)}{\sum_{j=1}^{K} \exp(z_j)} \]

Balanced Softmax adjusts the logits by the log class prior:

\[ p_c^{\text{BS}} = \frac{n_c \cdot \exp(z_c)}{\sum_{j=1}^{K} n_j \cdot \exp(z_j)} = \frac{\exp(z_c + \log n_c)}{\sum_{j=1}^{K} \exp(z_j + \log n_j)} \]

The loss is standard CE applied to the adjusted probabilities:

\[ \mathcal{L}_{\text{BS}} = -\log p_{y}^{\text{BS}} = -\log \frac{\exp(z_y + \log n_y)}{\sum_{j} \exp(z_j + \log n_j)} \]

**Gradient analysis.** The gradient with respect to logit \(z_c\) is:

\[ \frac{\partial \mathcal{L}_{\text{BS}}}{\partial z_c} = p_c^{\text{BS}} - \mathbb{1}[c = y] \]

The adjusted probability \(p_c^{\text{BS}}\) shifts probability mass away from majority classes (which have large \(n_c\) but the effect is absorbed into the normalization), creating a more balanced posterior estimate.

**Worked example:**

```
Suppose model produces logits z = [2.0, 1.5] for classes airplane (n=5000) and truck (n=250).

Standard softmax:
  P(airplane) = exp(2.0) / (exp(2.0) + exp(1.5))
              = 7.39 / (7.39 + 4.48) = 62.2%
  P(truck)    = 37.8%
  → Predicts airplane

Balanced Softmax (add log prior):
  z_airplane + log(5000) = 2.0 + 8.52 = 10.52
  z_truck + log(250)     = 1.5 + 5.52 = 7.02
  P_BS(airplane) = exp(10.52) / (exp(10.52) + exp(7.02))
                 = 37,163 / (37,163 + 1,122) = 97.1%

Wait — that made airplane win MORE? The confusion is that during TRAINING,
we minimize -log P_BS(y|x). The log(n_c) terms shift the loss landscape
so that the OPTIMAL logits z* satisfy:

  z*_c ∝ log P(x|c)  (class-conditional likelihood)

rather than the biased:

  z*_c ∝ log P(x|c) + log P_train(c)

In other words, BS removes the prior bias from the learned logits.
At TEST time, the raw logits z are used directly (without the log n_c shift),
and they produce balanced predictions because they've been trained to
encode likelihoods rather than posteriors.
```

**Key insight.** Balanced Softmax doesn't change the prediction rule at test time. It changes the **training objective** so that the model learns logits proportional to the class-conditional likelihood \(P(\mathbf{x}|c)\) rather than the biased posterior \(P_{\text{train}}(c|\mathbf{x})\). At test time, these unbiased logits produce balanced predictions.

**Config:** `loss.name: "balanced_softmax"`

---

#### S13. Logit Adjustment (Post-hoc)

**Mathematical formulation (Menon et al., 2021).** Logit Adjustment generalizes Balanced Softmax with a temperature parameter \(\tau\):

\[ \mathcal{L}_{\text{LA}} = -\log \frac{\exp(z_y + \tau \cdot \log \pi_y)}{\sum_{j} \exp(z_j + \tau \cdot \log \pi_j)} \]

where \(\pi_c = n_c / N\) is the class prior.

**The role of \(\tau\):**

| \(\tau\) | Effect | Interpretation |
|------|--------|----------------|
| 0.0 | No adjustment | Standard CE — ignores imbalance |
| 1.0 | Full correction | Equivalent to Balanced Softmax — Bayes-optimal |
| < 1.0 | Partial correction | Conservative — trusts the model's natural calibration |
| > 1.0 | Over-correction | Aggressively favors minority — can hurt majority accuracy |

**Fisher-consistent property.** Menon et al. (2021) proved that the minimizer of the Logit Adjustment loss with \(\tau = 1\) satisfies:

\[ f^*(\mathbf{x}) = \arg\max_c \; P(\mathbf{x}|c) \]

This is the **Bayes-optimal** classifier under balanced class priors, regardless of the training distribution. This theoretical guarantee makes Logit Adjustment one of the most principled methods for handling imbalance.

**Relationship between methods:**

```
tau = 0.0  →  Standard CE (no correction)
tau = 1.0  →  Balanced Softmax (Ren et al., 2020)
tau = 1.0  →  Logit Adjustment (Menon et al., 2021)
              (identical at tau=1; differs in that LA supports tau ≠ 1)
```

**When to use \(\tau \neq 1\).** If the model is poorly calibrated (e.g., early in training, or with a small model), the Bayesian correction may be too aggressive. Using \(\tau < 1\) provides a conservative correction. The optimal \(\tau\) can be found by validation.

**Config:** `loss.name: "logit_adjust"`, `loss.tau: 1.0`

---

### Category 5: Combined — *Stack the Best of Each*

> **Core idea:** Each category addresses a different symptom of imbalance. Combining orthogonal interventions can provide additive benefits. The key is choosing techniques that don't **conflict** (e.g., don't combine two loss reweighting schemes that fight over gradient magnitudes).

**Config:** `configs/s8_combined_best.yaml`

We stack techniques from different intervention points:

| Layer | Technique | Mechanism | Interaction |
|-------|-----------|-----------|-------------|
| **Sampling** | WeightedRandomSampler | Equalizes class frequency in mini-batches | Ensures every class gets gradient signal every step |
| **Augmentation** | ColorJitter + RandomErasing | Expands visual diversity of minority samples | Compensates for limited minority image variety |
| **Augmentation** | Remix | Minority-biased label mixing | Amplifies minority gradient during mixed training |
| **Loss** | Balanced Softmax | Log-prior logit correction | Removes prior bias from learned logits |
| **Loss** | Class weights | Inverse-frequency scaling | Amplifies minority loss magnitude |
| **Training** | cRT (Stage 2) | Freeze backbone, retrain classifier | Fixes classifier bias without hurting features |

**Why certain combinations work:**
1. Weighted Sampler + Balanced Softmax are complementary: the sampler fixes the **data exposure**, while Balanced Softmax fixes the **output bias**. Neither modifies the loss magnitude.
2. Remix + cRT are complementary: Remix improves Stage 1 feature learning by boosting minority gradients during augmentation; cRT independently recalibrates the classifier in Stage 2.
3. Strong augmentation + any method: augmentation increases effective sample diversity, which benefits every other technique.

**Why certain combinations conflict:**
- Class-Weighted CE + CB Loss: both scale the loss by class-dependent factors. The combined weight would be \(w_c^{\text{CE}} \times w_c^{\text{CB}}\), which can produce extreme values for rare classes.
- Mixup + CutMix: both modify the input images. Running both simultaneously produces doubly-augmented, unrealistic inputs.
- Oversampling + Remix: oversampling already balances the data, so Remix's minority-biasing label adjustment over-corrects.

---

## Benchmark Results & Analysis

All 12 strategies trained on the same imbalanced CIFAR-10 (20:1 ratio) with identical hyperparameters (Adam optimizer, lr=0.001, cosine scheduler). Only the imbalance strategy differs. Results below are from **1-epoch** runs.

### Summary Table

![Summary Table](results/figures/comparison_summary.png)

| # | Strategy | Accuracy | Macro F1 | Category | Why this result? |
|---|----------|----------|----------|----------|------------------|
| 1 | Weighted CE | 27.12% | 0.2397 | Cost-Sensitive | Class weights help but static weights cause instability in early epochs |
| 2 | **Oversampling** | **35.59%** | **0.3259** | **Data** | **Wins at 1 epoch**: balanced batches give every class equal exposure from the first batch |
| 3 | Focal Loss | 26.34% | 0.2015 | Cost-Sensitive | Needs multiple epochs to identify "hard" examples — at epoch 1, everything is hard |
| 4 | CB Loss | 27.88% | 0.2379 | Cost-Sensitive | Similar to Weighted CE; effective-number weights are better calibrated but need more training |
| 5 | Label Smoothing | 24.50% | 0.1486 | Training | A regularizer, not a direct fix — helps more in later epochs when the model starts overfitting |
| 6 | Mixup | 24.40% | 0.1413 | Training | Never sees a "pure" image — at epoch 1 it hasn't learned to decompose mixed signals |
| 7 | CutMix | 22.81% | 0.1316 | Training | Same as Mixup but harder — partial images are more difficult to learn from initially |
| 8 | Combined Best | 16.59% | 0.0868 | Combined | Multiple simultaneous regularizers + cRT overhead overwhelm a random-init model in 1 epoch |
| 9 | Remix | 10.27% | 0.0234 | Training | Aggressive label biasing (kappa=0.9) confuses the model when features aren't yet learned |
| 10 | Balanced Softmax | 28.31% | 0.2542 | Logit Adj. | Log-prior correction provides immediate benefit — no training needed to learn the bias |
| 11 | Logit Adjustment | 28.31% | 0.2542 | Logit Adj. | Same as Balanced Softmax (tau=1.0). Among the best single-technique results |
| 12 | Decoupled cRT | 24.52% | 0.1420 | Training | cRT phase helps but 1 epoch of feature learning is too little for the backbone |

### Analysis by Category

**Convergence rate ordering (from fastest to slowest):**

```
Speed of convergence at 1 epoch:

  Data-level ▸▸▸▸▸▸▸▸▸▸  (immediate — changes distribution before training)
  Logit Adj  ▸▸▸▸▸▸▸▸    (immediate — mathematical correction, no learning needed)
  Cost-Sens  ▸▸▸▸▸▸      (fast — reweights gradients but needs a few epochs to settle)
  Regulariz  ▸▸▸          (slow — prevents overfitting, which isn't the problem yet)
  Combined   ▸            (slowest — many interacting components need time to stabilize)
```

**Why Oversampling wins at 1 epoch.** WeightedRandomSampler changes the **data distribution** seen by the model. From the very first batch, every class has ~10% representation. The model receives balanced gradient signal from step 1. No other technique provides this immediate correction — all others still sample from the imbalanced distribution and try to correct downstream.

**Why Balanced Softmax / Logit Adjustment are second.** These methods add a constant \(\log(n_c)\) shift to logits. This shift is computed from class counts (known before training) and doesn't need to be learned. From the first gradient step, the loss landscape is adjusted so that the model must produce higher logits for minority classes to minimize the loss. The correction is instant but indirect — the model still sees imbalanced batches.

**Why Focal Loss underperforms early.** Focal Loss modulates by \((1-p_t)^\gamma\). At epoch 1, \(p_t \approx 0.1\) for all classes (random initialization). The modulating factor \((1-0.1)^2 = 0.81\) is nearly identical for all samples. Focal Loss provides almost no differentiation between easy and hard examples because **nothing is easy yet**. It only becomes useful once the model has learned to classify some samples confidently.

**Why Combined backfires at 1 epoch.** The combined strategy (S8) applies Weighted Sampler + Strong Augmentation + Remix + Balanced Softmax + cRT. Each component adds noise or regularization that helps with long training. At 1 epoch: (a) the model has barely learned anything, (b) Remix aggressively biases labels before features are meaningful, (c) cRT wastes epochs retraining a random classifier. The signal-to-noise ratio is terrible.

**Expected behavior with more epochs (20+):** The ranking typically inverts. Combined/cRT methods pull ahead once features stabilize. Oversampling plateaus due to overfitting on repeated minority samples. Cost-sensitive methods find their equilibrium. Regularizers (Mixup, CutMix, Label Smoothing) prevent late-training overfitting.

### Accuracy Comparison

![Accuracy Comparison](results/figures/comparison_accuracy.png)

### Macro F1 Comparison

![F1 Comparison](results/figures/comparison_f1.png)

### Training Curves — All Strategies

![Training Curves](results/figures/comparison_training.png)

### Per-Class F1 Heatmap

![Per-Class F1](results/figures/comparison_per_class_f1.png)

The heatmap reveals the core challenge: minority classes (right columns: frog, horse, ship, truck) have near-zero F1 across most strategies after just 1 epoch. Oversampling is the only strategy that achieves meaningful recall on these classes this early because it physically forces them into every training batch. Balanced Softmax / Logit Adjustment also show non-trivial minority recall because they mathematically adjust the decision boundary.

---

## Imbalanced Data Toolkit

Beyond the strategies used in the benchmark above, this project includes a standalone modular library (`imbalance_toolkit/`) implementing the full taxonomy from [Gao et al., 2025](https://doi.org/10.1007/s11704-025-50274-7). Every component is accessible via a **Factory Pattern**:

```python
import imbalance_toolkit

sampler  = imbalance_toolkit.create({"method": "SMOTE", "k_neighbors": 5})
loss_fn  = imbalance_toolkit.create({"loss": "FocalLoss", "gamma": 2.0})
trainer  = imbalance_toolkit.create({"trainer": "cRT", "model": model})
```

### All Implemented Methods

| Category | Class | Paper | What it does |
|----------|-------|-------|--------------|
| **Data Re-balancing** | `SMOTE` | Chawla et al., 2002 | k-NN interpolation to synthesize minority samples |
| | `ADASYN` | He et al., 2008 | Adaptive SMOTE: more synthetics near hard decision boundaries |
| | `VAESampler` | Wan et al., 2017 | VAE-based deep generative model for minority oversampling |
| | `Remix` | Chou et al., 2020 | Mixup with separate feature/label lambdas biased toward minority |
| | `BalancedMixup` | Galdran et al., 2021 | Dual-sampler Mixup pairing imbalanced and balanced batches |
| **Feature Representation** | `TripletLoss` | Schroff et al., 2015 | Pulls same-class embeddings together, pushes different-class apart |
| | `ContrastiveLoss` | — | Pair-wise loss with a margin to separate class clusters |
| | `SupConLoss` | Khosla et al., 2020 | Supervised contrastive learning with balanced positive set caps |
| **Training Strategy** | `DecoupledTrainer` (cRT) | Kang et al., 2020 | Stage 1: train everything; Stage 2: freeze backbone, retrain head with balanced sampling |
| | `BalancedSoftmaxLoss` | Ren et al., 2020 | Adds log(class_prior) to logits before softmax — implicit minority boost |
| | `LogitAdjustmentLoss` | Menon et al., 2021 | Parameterized logit shift with tunable strength (tau) |
| | `ClassWeightedCE` | — | Inverse-frequency class weights on cross-entropy |
| | `FocalLoss` | Lin et al., 2017 | (1-p_t)^gamma modulation: ignore easy, focus on hard |
| | `ClassBalancedLoss` | Cui et al., 2019 | Effective-number reweighting (CE or Focal variant) |
| **Ensemble** | `ImbalancedBagging` | Wang & Yao, 2009 | Train multiple models on balanced subsets, aggregate by vote |
| | `ImbalancedBoosting` | Chawla et al., 2003 | Iterative reweighting with amplified minority misclassification cost |
| **Evaluation** | `macro_f1` | — | Class-averaged F1 (every class counts equally) |
| | `g_mean` | — | Geometric mean of per-class recalls (0 if any class is completely missed) |
| | `pr_auc` | — | Precision-Recall AUC (better than ROC-AUC for imbalanced data) |
| | `balanced_accuracy` | — | Mean of per-class recalls |
| | `mcc_score` | — | Matthews Correlation Coefficient (uses all 4 quadrants of confusion matrix) |

---

## Configuration Reference

Every strategy is toggled through YAML config — no code changes needed:

| Technique | Config Key | Values | Default |
|-----------|------------|--------|---------|
| Long-tail creation | `data.imbalance.enabled` | `true` / `false` | `true` |
| Imbalance ratio | `data.imbalance.ratio` | float | `20` |
| Weighted Sampler | `data.weighted_sampling` | `true` / `false` | `true` |
| Weighted Loss | `data.weighted_loss` | `true` / `false` | `true` |
| SMOTE / ADASYN | `data.oversampling.method` | `none`, `smote`, `adasyn` | `none` |
| Oversampling ratio | `data.oversampling.target_ratio` | float | `1.0` |
| SMOTE neighbors | `data.oversampling.k_neighbors` | int | `5` |
| Loss function | `loss.name` | `ce`, `focal`, `cb`, `label_smoothing`, `cb_focal`, `balanced_softmax`, `logit_adjust` | `cb_focal` |
| Focal gamma | `loss.gamma` | float | `2.0` |
| CB beta | `loss.beta` | float | `0.9999` |
| Smoothing epsilon | `loss.smoothing` | float | `0.1` |
| Logit Adjust tau | `loss.tau` | float | `1.0` |
| Mixup / CutMix / Remix | `training.mixup.mode` | `none`, `mixup`, `cutmix`, `remix` | `none` |
| Mixup alpha | `training.mixup.alpha` | float | `0.4` |
| Remix tau | `training.mixup.remix.tau` | float | `0.5` |
| Remix kappa | `training.mixup.remix.kappa` | float | `0.9` |
| cRT enabled | `training.crt.enabled` | `true` / `false` | `false` |
| cRT classifier epochs | `training.crt.classifier_epochs` | int | `5` |
| cRT learning rate | `training.crt.lr` | float | `0.01` |

---

## Project Structure

```
img_cls/
├── imbalance_toolkit/              # Standalone modular library
│   ├── factory.py                  # Config-driven Factory Pattern
│   ├── sampling/                   # Data re-balancing (Sec. 2.1)
│   │   ├── smote.py                # SMOTE + ADASYN
│   │   ├── vae_sampler.py          # VAE deep generation
│   │   └── remix.py                # Remix + Balanced-Mixup
│   ├── representation/             # Feature representation (Sec. 2.2)
│   │   ├── metric_learning.py      # Triplet + Contrastive Loss
│   │   └── supcon.py               # SupCon Loss
│   ├── strategies/                 # Training strategies (Sec. 2.3)
│   │   ├── decoupled.py            # cRT two-stage
│   │   ├── balanced_softmax.py     # Balanced Softmax + Logit Adj.
│   │   └── cost_sensitive.py       # Weighted CE, Focal, CB Loss
│   ├── ensembles/                  # Ensemble methods (Sec. 2.4)
│   │   └── wrapper.py              # Bagging + Boosting
│   └── evaluation/                 # Metrics (Sec. 4)
│       └── metrics.py              # Macro-F1, G-Mean, PR-AUC, MCC
├── configs/
│   ├── default.yaml                # Default config (CB Focal)
│   └── s1..s12_*.yaml              # Per-strategy configs
├── knowledge/
│   └── s11704-025-50274-7.pdf      # Survey paper (Gao et al., 2025)
├── scripts/
│   ├── train.py                    # Training pipeline
│   ├── evaluate.py                 # Test set evaluation
│   ├── predict.py                  # Single-image inference
│   ├── visualize.py                # Dataset & single-run figures
│   ├── run_all.py                  # Run all 12 experiments
│   └── compare.py                  # Cross-strategy comparison
├── src/
│   ├── data/
│   │   ├── dataset.py              # CIFAR-10 + imbalance + SMOTE/ADASYN
│   │   ├── transforms.py           # Augmentation pipeline
│   │   ├── dataloader.py           # DataLoader + weighted sampler
│   │   ├── imbalance.py            # Long-tail, class weights, sampler
│   │   └── smote.py                # SMOTE + ADASYN (pipeline-integrated)
│   ├── models/
│   │   ├── resnet.py               # ResNet from scratch
│   │   ├── simple_cnn.py           # Baseline CNN
│   │   └── factory.py              # Model builder
│   ├── training/
│   │   ├── trainer.py              # Training loop + mixup/remix + cRT + early stop
│   │   ├── losses.py               # Focal, CB, LabelSmoothing, CBFocal, BalSoftmax, LogitAdj
│   │   ├── mixup.py                # Mixup + CutMix + Remix
│   │   ├── optimizer.py            # Optimizer builder
│   │   └── scheduler.py            # LR scheduler builder
│   ├── evaluation/
│   │   └── metrics.py              # Accuracy, F1, confusion matrix
│   └── utils/
│       ├── config.py               # YAML config loader
│       ├── logger.py               # Logging setup
│       └── helpers.py              # Seed, device, checkpointing
├── results/
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
# Train with Focal Loss
python scripts/train.py --config configs/s3_focal.yaml

# Evaluate the trained model
python scripts/evaluate.py --config configs/s3_focal.yaml \
    --checkpoint results/s3_focal/checkpoints/best.pth
```

### Run All 12 Strategies

```bash
python scripts/run_all.py --epochs 20
```

### Generate Comparison Charts

```bash
python scripts/compare.py
```

### Toggle SMOTE/ADASYN Preprocessing

```bash
python scripts/train.py --config configs/default.yaml \
    --override data.oversampling.method=smote \
               data.oversampling.target_ratio=1.0
```

### Override Any Parameter from CLI

```bash
python scripts/train.py --config configs/s3_focal.yaml \
    --override training.epochs=50 loss.gamma=3.0
```

---

## Supported Models

| Model | Config value | Parameters | Block |
|-------|-------------|------------|-------|
| SimpleCNN | `simple_cnn` | ~200K | Conv+BN |
| ResNet-18 | `resnet18` | ~11M | BasicBlock |
| ResNet-34 | `resnet34` | ~21M | BasicBlock |
| ResNet-50 | `resnet50` | ~23.5M | Bottleneck |
| ResNet-101 | `resnet101` | ~42.5M | Bottleneck |

---

## References

1. He et al., 2016 — [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
2. Chawla et al., 2002 — [SMOTE](https://arxiv.org/abs/1106.1813)
3. He et al., 2008 — [ADASYN](https://doi.org/10.1109/IJCNN.2008.4633969)
4. Wan et al., 2017 — [VAE for imbalanced learning](https://doi.org/10.1109/SSCI.2017.8285168)
5. Chou et al., 2020 — [Remix: Rebalanced Mixup](https://arxiv.org/abs/2007.03943)
6. Galdran et al., 2021 — [Balanced-MixUp](https://arxiv.org/abs/2109.09350)
7. Lin et al., 2017 — [Focal Loss](https://arxiv.org/abs/1708.02002)
8. Cui et al., 2019 — [Class-Balanced Loss](https://arxiv.org/abs/1901.05555)
9. Schroff et al., 2015 — [FaceNet / Triplet Loss](https://arxiv.org/abs/1503.03832)
10. Khosla et al., 2020 — [SupCon](https://arxiv.org/abs/2004.11362)
11. Kang et al., 2020 — [Decoupled Training (cRT)](https://arxiv.org/abs/1910.09217)
12. Ren et al., 2020 — [Balanced Softmax](https://arxiv.org/abs/2007.10740)
13. Menon et al., 2021 — [Logit Adjustment](https://arxiv.org/abs/2007.07314)
14. Zhang et al., 2018 — [Mixup](https://arxiv.org/abs/1710.09412)
15. Yun et al., 2019 — [CutMix](https://arxiv.org/abs/1905.04899)
16. **Gao et al., 2025** — [A comprehensive survey on imbalanced data learning](https://doi.org/10.1007/s11704-025-50274-7)

---

## License

MIT
