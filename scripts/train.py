"""Train an image classification model.

Usage:
    python scripts/train.py --config default.yaml
    python scripts/train.py --override epochs=50
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent),
)

from src.data import (  # noqa: E402
    build_dataloaders, compute_class_weights,
)
from src.data.imbalance import (  # noqa: E402
    _extract_targets,
)
from src.models import build_model  # noqa: E402
from src.training import (  # noqa: E402
    Trainer, build_optimizer, build_scheduler,
)
from src.training.losses import build_loss  # noqa: E402
from src.utils import (  # noqa: E402
    load_config, merge_configs, parse_cli_overrides,
    setup_logger, set_seed, get_device,
    count_parameters,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train image classifier",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--override", nargs="*",
        help="key=value config overrides",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    if args.override:
        overrides = parse_cli_overrides(args.override)
        cfg = merge_configs(cfg, overrides)
    return cfg


def _log_data_info(logger, loaders: dict) -> None:
    logger.info(
        f"Train batches: {len(loaders['train'])}"
    )
    if "val" in loaders:
        logger.info(
            f"Val batches:   {len(loaders['val'])}"
        )


def _get_class_info(dataset):
    """Extract per-class sample counts from dataset."""
    targets = _extract_targets(dataset)
    counts = Counter(targets)
    n_classes = len(counts)
    return [counts[i] for i in range(n_classes)]


def main():
    args = _parse_args()
    cfg = _build_config(args)

    output_dir = Path(cfg["project"]["output_dir"])
    logger = setup_logger(log_dir=output_dir)

    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])
    logger.info(f"Device: {device}")

    logger.info("Building data loaders...")
    loaders = build_dataloaders(cfg)
    _log_data_info(logger, loaders)

    model_name = cfg["model"]["name"]
    logger.info(f"Building model: {model_name}")
    model = build_model(cfg)
    n_params = count_parameters(model)
    logger.info(f"Parameters: {n_params:,}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    train_ds = loaders["train"].dataset
    samples_per_class = _get_class_info(train_ds)

    use_weighted = cfg["data"].get(
        "weighted_loss", False,
    )
    class_weights = (
        compute_class_weights(train_ds)
        if use_weighted
        else None
    )

    loss_name = cfg.get("loss", {}).get("name", "ce")
    criterion = build_loss(
        cfg, device,
        samples_per_class=samples_per_class,
        class_weights=class_weights,
    )
    logger.info(
        f"Loss: {loss_name}"
        f" | weighted={use_weighted}"
    )

    mix_mode = cfg["training"].get(
        "mixup", {},
    ).get("mode", "none")
    if mix_mode != "none":
        logger.info(f"Mixup: {mix_mode}")

    trainer = Trainer(
        model, optimizer, scheduler,
        criterion, device, cfg,
    )
    trainer.fit(loaders["train"], loaders.get("val"))

    logger.info("Training complete.")
    best = trainer.best_metric
    logger.info(f"Best val accuracy: {best:.2f}%")


if __name__ == "__main__":
    main()
