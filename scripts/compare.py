"""Compare results across all 8 imbalance strategies.

Usage:
    python scripts/compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent),
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STRATEGIES = [
    ("s1_weighted_ce", "Weighted CE"),
    ("s2_oversampling", "Oversampling"),
    ("s3_focal", "Focal Loss"),
    ("s4_cb_loss", "CB Loss"),
    ("s5_label_smoothing", "Label Smooth"),
    ("s6_mixup", "Mixup"),
    ("s7_cutmix", "CutMix"),
    ("s8_combined_best", "Combined"),
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat",
    "deer", "dog", "frog", "horse", "ship", "truck",
]

FIGURE_DIR = Path("results/figures")
COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def _save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(
        path, dpi=150, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    plt.close(fig)
    print(f"  Saved {path}")


def _load_eval(folder: str) -> dict | None:
    path = Path(f"results/{folder}/eval_results.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_history(folder: str) -> dict | None:
    path = Path(
        f"results/{folder}/train_history.json",
    )
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_accuracy_comparison():
    """Bar chart: accuracy across all strategies."""
    names, accs = [], []
    for folder, label in STRATEGIES:
        data = _load_eval(folder)
        if data is None:
            continue
        names.append(label)
        accs.append(data["accuracy"])

    if not names:
        print("  Skipped (no eval data)")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    bars = ax.bar(
        x, accs, color=COLORS[:len(names)],
        alpha=0.85, edgecolor="black",
        linewidth=0.5,
    )

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.1f}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Accuracy Comparison Across "
        "Imbalance Strategies",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        names, rotation=25, ha="right",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(accs) * 1.15)
    _save(fig, "comparison_accuracy.png")


def plot_f1_comparison():
    """Bar chart: macro F1 across all strategies."""
    names, f1s = [], []
    for folder, label in STRATEGIES:
        data = _load_eval(folder)
        if data is None:
            continue
        names.append(label)
        f1s.append(data["macro_f1"])

    if not names:
        print("  Skipped (no eval data)")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    bars = ax.bar(
        x, f1s, color=COLORS[:len(names)],
        alpha=0.85, edgecolor="black",
        linewidth=0.5,
    )

    for bar, f1 in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{f1:.4f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title(
        "Macro F1 Comparison Across "
        "Imbalance Strategies",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        names, rotation=25, ha="right",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(f1s) * 1.15)
    _save(fig, "comparison_f1.png")


def plot_training_curves_all():
    """Overlay training loss/accuracy for all."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 6),
    )

    found = False
    for i, (folder, label) in enumerate(STRATEGIES):
        hist = _load_history(folder)
        if hist is None:
            continue
        found = True
        epochs = range(1, len(hist["train_loss"]) + 1)
        color = COLORS[i % len(COLORS)]

        ax1.plot(
            epochs, hist["train_loss"],
            "o-", label=label, color=color,
            markersize=3, linewidth=1.5,
        )

        val_acc = hist.get("val_acc", [])
        if val_acc:
            ax2.plot(
                epochs, val_acc,
                "s-", label=label, color=color,
                markersize=3, linewidth=1.5,
            )

    if not found:
        print("  Skipped (no training data)")
        plt.close(fig)
        return

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title(
        "Training Loss", fontweight="bold",
    )
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Accuracy (%)")
    ax2.set_title(
        "Validation Accuracy", fontweight="bold",
    )
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Training Curves — All Strategies",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "comparison_training.png")


def plot_per_class_f1_heatmap():
    """Heatmap: per-class F1 for each strategy."""
    labels_list = []
    f1_matrix = []

    for folder, label in STRATEGIES:
        data = _load_eval(folder)
        if data is None or "per_class" not in data:
            continue
        labels_list.append(label)
        per_class = data["per_class"]
        row = [
            per_class.get(c, {}).get("f1", 0.0)
            for c in CIFAR10_CLASSES
        ]
        f1_matrix.append(row)

    if not labels_list:
        print("  Skipped (no per-class data)")
        return

    matrix = np.array(f1_matrix)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        matrix, cmap="YlOrRd", aspect="auto",
        vmin=0, vmax=max(0.5, matrix.max()),
    )

    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticks(range(len(labels_list)))
    ax.set_xticklabels(
        CIFAR10_CLASSES, rotation=45,
        ha="right", fontsize=10,
    )
    ax.set_yticklabels(
        labels_list, fontsize=10,
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = (
                "white" if val > matrix.max() * 0.6
                else "black"
            )
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=8, color=color,
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="F1")
    ax.set_title(
        "Per-Class F1 Score — All Strategies",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Strategy")
    fig.tight_layout()
    _save(fig, "comparison_per_class_f1.png")


def plot_eval_summary_table():
    """Summary table of all metrics."""
    rows = []
    for folder, label in STRATEGIES:
        data = _load_eval(folder)
        if data is None:
            continue
        rows.append({
            "strategy": label,
            "accuracy": data["accuracy"],
            "macro_f1": data["macro_f1"],
            "weighted_f1": data["weighted_f1"],
        })

    if not rows:
        print("  Skipped (no data)")
        return

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")

    col_labels = [
        "Strategy", "Accuracy (%)",
        "Macro F1", "Weighted F1",
    ]
    cell_text = [
        [
            r["strategy"],
            f"{r['accuracy']:.2f}",
            f"{r['macro_f1']:.4f}",
            f"{r['weighted_f1']:.4f}",
        ]
        for r in rows
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(
            color="white", fontweight="bold",
        )

    best_acc = max(r["accuracy"] for r in rows)
    best_f1 = max(r["macro_f1"] for r in rows)
    for i, r in enumerate(rows, 1):
        if r["accuracy"] == best_acc:
            table[i, 1].set_facecolor("#d4edda")
        if r["macro_f1"] == best_f1:
            table[i, 2].set_facecolor("#d4edda")

    ax.set_title(
        "Evaluation Summary — All Strategies",
        fontsize=14, fontweight="bold", pad=20,
    )
    fig.tight_layout()
    _save(fig, "comparison_summary.png")


def main():
    print("Generating comparison visualizations...")

    print("[1/5] Accuracy comparison")
    plot_accuracy_comparison()

    print("[2/5] F1 comparison")
    plot_f1_comparison()

    print("[3/5] Training curves")
    plot_training_curves_all()

    print("[4/5] Per-class F1 heatmap")
    plot_per_class_f1_heatmap()

    print("[5/5] Summary table")
    plot_eval_summary_table()

    print("Done!")


if __name__ == "__main__":
    main()
