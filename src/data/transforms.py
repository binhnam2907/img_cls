from __future__ import annotations

from typing import Any

from torchvision import transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_NORM_STATS = {
    "cifar": (CIFAR_MEAN, CIFAR_STD),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
}


def _is_cifar(name: str) -> bool:
    return name.startswith("cifar")


def _resolve_image_size(cfg: dict[str, Any]) -> int:
    if _is_cifar(cfg["data"]["dataset"]):
        return 32
    return cfg["data"].get("custom", {}).get(
        "image_size", 224
    )


def _get_norm_stats(
    name: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    key = "cifar" if _is_cifar(name) else "imagenet"
    return _NORM_STATS[key]


def _spatial_train(
    name: str, size: int, aug: dict,
) -> list:
    if aug.get("random_crop", False):
        if _is_cifar(name):
            return [transforms.RandomCrop(size, padding=4)]
        return [transforms.RandomResizedCrop(size)]

    if not _is_cifar(name):
        return [transforms.Resize((size, size))]

    return []


def _spatial_eval(name: str, size: int) -> list:
    if _is_cifar(name):
        return []
    resize_dim = int(size * 1.143)
    return [
        transforms.Resize(resize_dim),
        transforms.CenterCrop(size),
    ]


def _color_augmentations(aug: dict) -> list:
    tfms = []
    if aug.get("random_horizontal_flip", False):
        tfms.append(transforms.RandomHorizontalFlip())
    if aug.get("color_jitter", False):
        tfms.append(
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        )
    return tfms


def build_transforms(
    cfg: dict[str, Any], is_train: bool = True,
) -> transforms.Compose:
    name = cfg["data"]["dataset"]
    aug = cfg["data"]["augmentation"]
    size = _resolve_image_size(cfg)
    mean, std = _get_norm_stats(name)

    if is_train:
        pipeline = _spatial_train(name, size, aug)
        pipeline += _color_augmentations(aug)
    else:
        pipeline = _spatial_eval(name, size)

    pipeline.append(transforms.ToTensor())

    if aug.get("normalize", True):
        pipeline.append(transforms.Normalize(mean, std))

    if is_train and aug.get("random_erasing", False):
        pipeline.append(transforms.RandomErasing())

    return transforms.Compose(pipeline)
