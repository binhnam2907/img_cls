"""Evaluate a trained model on test set.

Usage:
    python scripts/evaluate.py \\
        --checkpoint results/checkpoints/best.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent),
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data import build_dataset  # noqa: E402
from src.evaluation import compute_metrics  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import (  # noqa: E402
    load_config, setup_logger, set_seed,
    get_device, load_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image classifier",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
    )
    parser.add_argument(
        "--split", type=str, default="val",
    )
    return parser.parse_args()


def _load_model(
    cfg: dict, ckpt_path: str, device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    ckpt = load_checkpoint(ckpt_path, device=device)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_preds, all_labels = [], []

    for images, labels in loader:
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def _log_metrics(
    logger, metrics: dict,
    class_names: list[str] | None,
) -> None:
    acc = metrics["accuracy"] * 100
    logger.info(f"Accuracy:    {acc:.2f}%")
    logger.info(
        f"Macro F1:    {metrics['macro_f1']:.4f}"
    )
    logger.info(
        f"Weighted F1: {metrics['weighted_f1']:.4f}"
    )
    cm = metrics["confusion_matrix"]
    logger.info(f"Confusion Matrix:\n{cm}")

    if not class_names:
        return

    logger.info("Per-class results:")
    report = metrics["classification_report"]
    for cls in class_names:
        s = report[cls]
        logger.info(
            f"  {cls:>12s}"
            f"  prec={s['precision']:.4f}"
            f"  rec={s['recall']:.4f}"
            f"  f1={s['f1-score']:.4f}"
        )


def _build_results_dict(
    args: argparse.Namespace,
    metrics: dict,
    class_names: list[str] | None,
) -> dict:
    results = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "accuracy": round(
            metrics["accuracy"] * 100, 4,
        ),
        "macro_f1": round(metrics["macro_f1"], 4),
        "weighted_f1": round(
            metrics["weighted_f1"], 4,
        ),
        "confusion_matrix": (
            metrics["confusion_matrix"].tolist()
        ),
    }

    if not class_names:
        return results

    report = metrics["classification_report"]
    results["per_class"] = {
        cls: {
            "precision": round(
                report[cls]["precision"], 4,
            ),
            "recall": round(
                report[cls]["recall"], 4,
            ),
            "f1": round(
                report[cls]["f1-score"], 4,
            ),
        }
        for cls in class_names
    }
    return results


def main():
    args = _parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=output_dir)

    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])

    logger.info(
        f"Loading checkpoint: {args.checkpoint}"
    )
    model, ckpt = _load_model(
        cfg, args.checkpoint, device,
    )
    epoch = ckpt.get("epoch", "?")
    logger.info(f"Checkpoint from epoch {epoch}")

    dataset = build_dataset(cfg, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["val"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    preds, gt = _collect_predictions(
        model, loader, device,
    )

    class_names = getattr(dataset, "classes", None)
    metrics = compute_metrics(
        preds, gt, class_names=class_names,
    )

    _log_metrics(logger, metrics, class_names)

    results = _build_results_dict(
        args, metrics, class_names,
    )
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(
        f"Evaluation results saved to {results_path}"
    )


if __name__ == "__main__":
    main()
