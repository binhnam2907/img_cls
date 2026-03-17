from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torchvision import datasets

from src.data.imbalance import make_imbalanced
from src.data.transforms import build_transforms

_CIFAR_CLASSES = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}

_FOLDER_DATASETS = {"imagenet", "custom"}

SUPPORTED_DATASETS = list(_CIFAR_CLASSES) + list(
    _FOLDER_DATASETS
)


def _build_cifar(
    dataset_cls: type,
    data_dir: Path,
    split: str,
    transform,
) -> Dataset:
    return dataset_cls(
        root=str(data_dir),
        train=(split == "train"),
        download=True,
        transform=transform,
    )


def _build_imagefolder(
    data_dir: Path, split: str, transform,
) -> Dataset:
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Expected directory {split_dir} "
            "with class sub-folders."
        )
    return datasets.ImageFolder(
        root=str(split_dir), transform=transform,
    )


def _maybe_apply_imbalance(
    dataset: Dataset, cfg: dict[str, Any],
    is_train: bool,
) -> Dataset:
    """Apply long-tail imbalance to training set
    when configured."""
    imb_cfg = cfg["data"].get("imbalance", {})
    if not imb_cfg.get("enabled", False):
        return dataset
    if not is_train:
        return dataset

    ratio = imb_cfg.get("ratio", 20.0)
    seed = cfg["project"].get("seed", 42)
    return make_imbalanced(
        dataset, imbalance_ratio=ratio, seed=seed,
    )


def build_dataset(
    cfg: dict[str, Any], split: str = "train",
) -> Dataset:
    dataset_name = cfg["data"]["dataset"]
    data_dir = Path(cfg["data"]["data_dir"])
    is_train = split == "train"
    transform = build_transforms(
        cfg, is_train=is_train,
    )

    if dataset_name in _CIFAR_CLASSES:
        ds = _build_cifar(
            _CIFAR_CLASSES[dataset_name],
            data_dir, split, transform,
        )
    elif dataset_name in _FOLDER_DATASETS:
        ds = _build_imagefolder(
            data_dir, split, transform,
        )
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {SUPPORTED_DATASETS}"
        )

    return _maybe_apply_imbalance(
        ds, cfg, is_train,
    )
