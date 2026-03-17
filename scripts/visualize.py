"""Generate all visualizations for the README.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent),
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from torchvision import datasets  # noqa: E402

from src.data import build_dataset  # noqa: E402
from src.utils import load_config  # noqa: E402

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat",
    "deer", "dog", "frog", "horse", "ship", "truck",
]

FIGURE_DIR = Path("results/figures")


def _save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    plt.close(fig)
    print(f"  Saved {path}")


def _add_bar_labels(ax, bars, fontsize=8):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h,
            str(int(h)),
            ha="center", va="bottom",
            fontsize=fontsize,
        )


def plot_sample_images(data_dir: str = "data"):
    """Grid of 5 sample images per class."""
    ds = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
    )
    images = np.array(ds.data)
    labels = np.array(ds.targets)
    n_classes = len(CIFAR10_CLASSES)
    samples_per_class = 5

    fig, axes = plt.subplots(
        n_classes, samples_per_class,
        figsize=(10, 20),
    )
    fig.suptitle(
        "CIFAR-10 Sample Images",
        fontsize=16, fontweight="bold", y=0.995,
    )

    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        idxs = np.where(labels == cls_idx)[0]
        chosen = np.random.choice(
            idxs, samples_per_class, replace=False,
        )
        for j, idx in enumerate(chosen):
            ax = axes[cls_idx, j]
            ax.imshow(images[idx])
            ax.axis("off")
            if j == 0:
                ax.set_title(
                    cls_name, fontsize=10,
                    fontweight="bold", loc="left",
                )

    fig.subplots_adjust(
        hspace=0.3, wspace=0.05,
    )
    _save(fig, "sample_images.png")


def plot_class_distribution(cfg: dict):
    """Side-by-side: imbalanced train vs balanced test.
    """
    train_ds = build_dataset(cfg, split="train")
    test_ds = build_dataset(cfg, split="val")

    train_targets = (
        list(train_ds.targets)
        if hasattr(train_ds, "targets")
        else [train_ds[i][1] for i in range(len(train_ds))]
    )
    test_targets = (
        list(test_ds.targets)
        if hasattr(test_ds, "targets")
        else [test_ds[i][1] for i in range(len(test_ds))]
    )

    train_counts = Counter(train_targets)
    test_counts = Counter(test_targets)

    x = np.arange(len(CIFAR10_CLASSES))
    width = 0.35

    train_vals = [
        train_counts.get(i, 0) for i in range(len(x))
    ]
    test_vals = [
        test_counts.get(i, 0) for i in range(len(x))
    ]

    imb_cfg = cfg["data"].get("imbalance", {})
    ratio = imb_cfg.get("ratio", 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(
        x - width / 2, train_vals, width,
        label="Train (imbalanced)",
        color="#C44E52", alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2, test_vals, width,
        label="Test (balanced)",
        color="#4C72B0", alpha=0.85,
    )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title(
        f"CIFAR-10 Class Distribution "
        f"(Imbalance Ratio {ratio}:1)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        CIFAR10_CLASSES, rotation=45, ha="right",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _add_bar_labels(ax, bars1)
    _add_bar_labels(ax, bars2)

    total_train = sum(train_vals)
    total_orig = len(CIFAR10_CLASSES) * 5000
    ax.annotate(
        f"Total train: {total_train:,} "
        f"(from {total_orig:,})",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right", fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="wheat", alpha=0.5,
        ),
    )

    _save(fig, "class_distribution.png")


def plot_training_curves(
    history_path: str = "results/train_history.json",
):
    """Loss and accuracy curves from training."""
    path = Path(history_path)
    if not path.exists():
        print(f"  Skipped (no {path})")
        return

    with open(path) as f:
        history = json.load(f)

    epochs = range(
        1, len(history["train_loss"]) + 1,
    )
    has_val = len(history.get("val_loss", [])) > 0

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5),
    )

    ax1.plot(
        epochs, history["train_loss"],
        "o-", label="Train Loss",
        color="#4C72B0", markersize=3,
    )
    if has_val:
        ax1.plot(
            epochs, history["val_loss"],
            "s-", label="Val Loss",
            color="#DD8452", markersize=3,
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(
        "Training & Validation Loss",
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(
        epochs, history["train_acc"],
        "o-", label="Train Acc",
        color="#4C72B0", markersize=3,
    )
    if has_val:
        ax2.plot(
            epochs, history["val_acc"],
            "s-", label="Val Acc",
            color="#DD8452", markersize=3,
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(
        "Training & Validation Accuracy",
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Training Curves (Imbalanced CIFAR-10)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "training_curves.png")


def plot_eval_results(
    eval_path: str = "results/eval_results.json",
):
    """Confusion matrix heatmap + per-class F1."""
    path = Path(eval_path)
    if not path.exists():
        print(f"  Skipped (no {path})")
        return

    with open(path) as f:
        results = json.load(f)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 6),
    )

    cm = np.array(results["confusion_matrix"])
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks(range(len(CIFAR10_CLASSES)))
    ax1.set_yticks(range(len(CIFAR10_CLASSES)))
    ax1.set_xticklabels(
        CIFAR10_CLASSES, rotation=45, ha="right",
        fontsize=8,
    )
    ax1.set_yticklabels(
        CIFAR10_CLASSES, fontsize=8,
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title(
        "Confusion Matrix", fontweight="bold",
    )
    fig.colorbar(im, ax=ax1, shrink=0.8)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = (
                "white" if cm[i, j] > thresh
                else "black"
            )
            ax1.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=6, color=color,
            )

    if "per_class" in results:
        per_class = results["per_class"]
        classes = list(per_class.keys())
        f1_scores = [
            per_class[c]["f1"] for c in classes
        ]
        prec = [
            per_class[c]["precision"] for c in classes
        ]
        recall = [
            per_class[c]["recall"] for c in classes
        ]

        x = np.arange(len(classes))
        w = 0.25
        ax2.bar(
            x - w, prec, w,
            label="Precision", color="#4C72B0",
            alpha=0.85,
        )
        ax2.bar(
            x, recall, w,
            label="Recall", color="#DD8452",
            alpha=0.85,
        )
        ax2.bar(
            x + w, f1_scores, w,
            label="F1", color="#55A868",
            alpha=0.85,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            classes, rotation=45, ha="right",
            fontsize=8,
        )
        ax2.set_ylabel("Score")
        ax2.set_title(
            "Per-Class Metrics", fontweight="bold",
        )
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_ylim(0, 1.0)

    acc = results["accuracy"]
    f1 = results["macro_f1"]
    fig.suptitle(
        f"Evaluation Results  |  "
        f"Accuracy: {acc:.2f}%  |  "
        f"Macro F1: {f1:.4f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "eval_results.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/default.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg["data"]["data_dir"]

    np.random.seed(cfg["project"]["seed"])

    print("Generating visualizations...")

    print("[1/4] Sample images")
    plot_sample_images(data_dir)

    print("[2/4] Class distribution")
    plot_class_distribution(cfg)

    print("[3/4] Training curves")
    plot_training_curves()

    print("[4/4] Evaluation results")
    plot_eval_results()

    print("Done!")


if __name__ == "__main__":
    main()
