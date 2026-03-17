"""Run inference on images.

Usage:
    python scripts/predict.py \\
        --checkpoint results/checkpoints/best.pth \\
        --input image.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent),
)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.transforms import (  # noqa: E402
    build_transforms,
)
from src.models import build_model  # noqa: E402
from src.utils import (  # noqa: E402
    load_config, get_device, load_checkpoint,
)

_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict with trained classifier",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Image file or directory",
    )
    parser.add_argument(
        "--class-names", type=str, default=None,
        help="Comma-separated class names",
    )
    return parser.parse_args()


def _load_model(
    cfg: dict,
    ckpt_path: str,
    device: torch.device,
) -> torch.nn.Module:
    ckpt = load_checkpoint(ckpt_path, device=device)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _resolve_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        )

    print(
        f"Error: '{path}' is not a valid "
        "file or directory."
    )
    sys.exit(1)


def _parse_class_names(
    raw: str | None,
) -> list[str] | None:
    if not raw:
        return None
    return [n.strip() for n in raw.split(",")]


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    transform,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    top_k = min(5, probs.size(1))
    top_p, top_i = probs.topk(top_k, dim=1)

    predictions = [
        {
            "class": (
                class_names[idx.item()]
                if class_names
                else str(idx.item())
            ),
            "confidence": prob.item(),
        }
        for prob, idx in zip(top_p[0], top_i[0])
    ]

    return {
        "file": str(image_path),
        "predictions": predictions,
    }


def main():
    args = _parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg["project"]["device"])

    model = _load_model(
        cfg, args.checkpoint, device,
    )
    transform = build_transforms(
        cfg, is_train=False,
    )
    class_names = _parse_class_names(args.class_names)
    images = _resolve_images(Path(args.input))

    for img_path in images:
        result = predict_image(
            model, img_path, transform,
            device, class_names,
        )
        print(f"\n{result['file']}:")
        for pred in result["predictions"]:
            cls = pred["class"]
            conf = pred["confidence"]
            print(f"  {cls:>20s}  {conf:.4f}")


if __name__ == "__main__":
    main()
