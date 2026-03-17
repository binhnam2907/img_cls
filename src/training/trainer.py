from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.evaluation.metrics import accuracy
from src.training.mixup import (
    mixup, cutmix, mixup_criterion, MixupOutput,
)
from src.utils.helpers import save_checkpoint
from src.utils.logger import get_logger


@dataclass
class EarlyStopTracker:
    enabled: bool = False
    patience: int = 15
    mode: str = "max"
    best_metric: float = field(init=False)
    epochs_no_improve: int = field(
        init=False, default=0,
    )

    def __post_init__(self):
        self.best_metric = (
            -float("inf")
            if self.mode == "max"
            else float("inf")
        )

    def step(self, metric: float) -> bool:
        improved = (
            self.mode == "max"
            and metric > self.best_metric
        ) or (
            self.mode == "min"
            and metric < self.best_metric
        )
        if improved:
            self.best_metric = metric
            self.epochs_no_improve = 0
            return True
        self.epochs_no_improve += 1
        return False

    @property
    def should_stop(self) -> bool:
        return (
            self.enabled
            and self.epochs_no_improve >= self.patience
        )


@dataclass
class EpochStats:
    loss: float = 0.0
    correct: float = 0.0
    total: int = 0

    def update(
        self,
        batch_loss: float,
        batch_acc: float,
        batch_size: int,
    ) -> None:
        self.loss += batch_loss * batch_size
        self.correct += batch_acc * batch_size / 100.0
        self.total += batch_size

    @property
    def avg_loss(self) -> float:
        return self.loss / self.total

    @property
    def avg_acc(self) -> float:
        return self.correct / self.total * 100.0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        criterion: nn.Module,
        device: torch.device,
        cfg: dict[str, Any],
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.cfg = cfg
        self.logger = get_logger()

        self.output_dir = Path(cfg["project"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_cfg = cfg["logging"]
        self.log_interval = log_cfg["log_interval"]
        self.save_interval = log_cfg["save_interval"]

        train_cfg = cfg["training"]
        self.grad_clip = train_cfg.get(
            "gradient_clip", 0.0,
        )

        mix_cfg = train_cfg.get("mixup", {})
        self.mixup_mode = mix_cfg.get("mode", "none")
        self.mixup_alpha = mix_cfg.get("alpha", 0.4)

        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stop = EarlyStopTracker(
            enabled=es_cfg.get("enabled", False),
            patience=es_cfg.get("patience", 15),
            mode=es_cfg.get("mode", "max"),
        )

    @property
    def best_metric(self) -> float:
        return self.early_stop.best_metric

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int | None = None,
    ) -> dict[str, list[float]]:
        num_epochs = (
            epochs or self.cfg["training"]["epochs"]
        )
        history = _empty_history()
        wall_start = time.time()
        last_epoch = 0

        for epoch in range(1, num_epochs + 1):
            last_epoch = epoch
            epoch_start = time.time()

            train_loss, train_acc = (
                self._train_one_epoch(train_loader, epoch)
            )
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            lr = self.optimizer.param_groups[0]["lr"]
            history["lr"].append(lr)

            val_loss, val_acc = self._maybe_validate(
                val_loader, history,
            )
            self._step_scheduler(val_acc)

            if val_acc is not None:
                if self._handle_early_stop(
                    epoch, val_acc,
                ):
                    break

            elapsed = time.time() - epoch_start
            self._log_epoch(
                epoch, num_epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                lr, elapsed,
            )

            if epoch % self.save_interval == 0:
                self._save_ckpt(
                    epoch, tag=f"epoch_{epoch}",
                )

        history["total_time_sec"] = (
            time.time() - wall_start
        )
        self._save_ckpt(last_epoch, tag="last")
        self._save_history(history)
        return history

    def _apply_mixup(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> MixupOutput | None:
        if self.mixup_mode == "mixup":
            return mixup(
                images, labels, self.mixup_alpha,
            )
        if self.mixup_mode == "cutmix":
            return cutmix(
                images, labels, self.mixup_alpha,
            )
        return None

    def _train_one_epoch(
        self, loader: DataLoader, epoch: int,
    ) -> tuple[float, float]:
        self.model.train()
        stats = EpochStats()

        for i, (images, labels) in enumerate(loader, 1):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            mix_out = self._apply_mixup(images, labels)
            if mix_out is not None:
                logits = self.model(mix_out.images)
                loss = mixup_criterion(
                    self.criterion, logits, mix_out,
                )
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip,
                )

            self.optimizer.step()

            batch_acc = accuracy(
                logits, labels, topk=(1,),
            )[0]
            stats.update(
                loss.item(), batch_acc, images.size(0),
            )

            if i % self.log_interval == 0:
                self.logger.debug(
                    f"  [{epoch}] batch {i}/{len(loader)}"
                    f" loss={loss.item():.4f}"
                    f" acc={batch_acc:.2f}%"
                )

        return stats.avg_loss, stats.avg_acc

    @torch.no_grad()
    def _validate(
        self, loader: DataLoader,
    ) -> tuple[float, float]:
        self.model.eval()
        stats = EpochStats()

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            batch_acc = accuracy(
                logits, labels, topk=(1,),
            )[0]
            stats.update(
                loss.item(), batch_acc, images.size(0),
            )

        return stats.avg_loss, stats.avg_acc

    def _maybe_validate(
        self,
        val_loader: DataLoader | None,
        history: dict,
    ) -> tuple[float | None, float | None]:
        if val_loader is None:
            return None, None
        val_loss, val_acc = self._validate(val_loader)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        return val_loss, val_acc

    def _step_scheduler(
        self, val_acc: float | None,
    ) -> None:
        if self.scheduler is None:
            return
        if isinstance(
            self.scheduler, ReduceLROnPlateau,
        ):
            if val_acc is not None:
                self.scheduler.step(val_acc)
        else:
            self.scheduler.step()

    def _handle_early_stop(
        self, epoch: int, val_metric: float,
    ) -> bool:
        improved = self.early_stop.step(val_metric)
        if improved:
            self._save_ckpt(epoch, tag="best")
        if self.early_stop.should_stop:
            self.logger.info(
                f"Early stopping at epoch {epoch}"
            )
            return True
        return False

    def _log_epoch(
        self,
        epoch: int, total: int,
        train_loss: float, train_acc: float,
        val_loss: float | None,
        val_acc: float | None,
        lr: float, elapsed: float,
    ) -> None:
        parts = [f"Epoch {epoch}/{total}"]
        parts.append(
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}%"
        )
        if val_loss is not None:
            parts.append(
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.2f}%"
            )
        parts.append(f"lr={lr:.6f}")
        parts.append(f"{elapsed:.1f}s")
        self.logger.info(" | ".join(parts))

    def _save_ckpt(
        self, epoch: int, tag: str = "latest",
    ) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": (
                self.model.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "best_metric": self.early_stop.best_metric,
            "config": self.cfg,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = (
                self.scheduler.state_dict()
            )
        save_checkpoint(
            state,
            self.output_dir / "checkpoints",
            f"{tag}.pth",
        )

    def _save_history(self, history: dict) -> None:
        path = self.output_dir / "train_history.json"
        serializable = {
            k: (
                [
                    round(x, 6)
                    if isinstance(x, float)
                    else x
                    for x in v
                ]
                if isinstance(v, list)
                else v
            )
            for k, v in history.items()
        }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        self.logger.info(
            f"Training history saved to {path}"
        )


def _empty_history() -> dict[str, list]:
    return {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }
