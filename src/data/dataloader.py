from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset

from src.data.dataset import build_dataset
from src.data.imbalance import build_weighted_sampler


def _create_loader(
    dataset: Dataset,
    split_cfg: dict,
    global_cfg: dict,
    use_weighted_sampler: bool = False,
) -> DataLoader:
    shuffle = split_cfg.get("shuffle", False)
    sampler = None

    if use_weighted_sampler and shuffle:
        sampler = build_weighted_sampler(dataset)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=split_cfg.get("batch_size", 64),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=global_cfg.get("num_workers", 4),
        pin_memory=global_cfg.get("pin_memory", True),
        drop_last=split_cfg.get("drop_last", False),
    )


def build_dataloaders(
    cfg: dict[str, Any],
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    use_sampler = cfg["data"].get(
        "weighted_sampling", False,
    )

    for split in ("train", "val"):
        split_cfg = cfg["data"].get(split, {})

        try:
            dataset = build_dataset(cfg, split=split)
        except FileNotFoundError:
            if split == "val":
                continue
            raise

        is_train = split == "train"
        loaders[split] = _create_loader(
            dataset, split_cfg, cfg["data"],
            use_weighted_sampler=(
                use_sampler and is_train
            ),
        )

    return loaders
