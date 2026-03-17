"""Train an image classification model.

Usage:
    python scripts/train.py --config default.yaml
    python scripts/train.py --override epochs=50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent),
)

import torch.nn as nn  # noqa: E402

from src.data import (  # noqa: E402
    build_dataloaders, compute_class_weights,
)
from src.models import build_model  # noqa: E402
from src.training import (  # noqa: E402
    Trainer, build_optimizer, build_scheduler,
)
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

    use_weighted_loss = cfg["data"].get(
        "weighted_loss", False,
    )
    if use_weighted_loss:
        train_ds = loaders["train"].dataset
        weights = compute_class_weights(train_ds)
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(
            weight=weights,
        )
        logger.info(
            "Using weighted loss for imbalance"
        )
        logger.info(f"Class weights: {weights}")
    else:
        criterion = nn.CrossEntropyLoss()

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
