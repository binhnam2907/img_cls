"""Linear generative methods for minority oversampling.

Based on Sec. 2.1.1 of Gao et al., 2025:
    "SMOTE selects k pairs of nearest minority neighbors and
     performs linear interpolation to create new synthetic
     minority samples."
    "ADASYN adjusts sample generation based on density
     distribution."

References
----------
- Chawla et al., 2002 — SMOTE (Ref [5])
- He et al., 2008 — ADASYN (Ref [10])
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


class SMOTE:
    """Synthetic Minority Over-sampling Technique.

    Generates synthetic minority samples by interpolating
    between a sample and its k-nearest same-class neighbors
    (Algorithm 1 in Gao et al., 2025, Sec. 2.1.1).

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors for interpolation.
    target_ratio : float
        Desired minority/majority count ratio (1.0 = balanced).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        target_ratio: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.k = k_neighbors
        self.target_ratio = target_ratio
        self.rng = np.random.RandomState(seed)

    def fit_resample(
        self, dataset: Dataset,
    ) -> TensorDataset:
        """Augment *dataset* so every class reaches
        ``target_ratio * max_class_count`` samples.

        Returns a new :class:`TensorDataset` with
        ``.targets`` and ``.classes`` attributes preserved.
        """
        images, labels = _extract_all(dataset)
        counts = Counter(labels.tolist())
        target_n = int(
            max(counts.values()) * self.target_ratio,
        )

        img_parts: list[torch.Tensor] = [images]
        lbl_parts: list[torch.Tensor] = [labels]

        for cls, n_cls in sorted(counts.items()):
            n_syn = target_n - n_cls
            if n_syn <= 0:
                continue
            cls_imgs = images[labels == cls]
            syn = self._interpolate(cls_imgs, n_syn)
            img_parts.append(syn)
            lbl_parts.append(
                torch.full(
                    (n_syn,), cls, dtype=torch.long,
                ),
            )

        return _pack(
            torch.cat(img_parts),
            torch.cat(lbl_parts),
            dataset,
        )

    # ── internals ────────────────────────────────

    def _interpolate(
        self,
        cls_imgs: torch.Tensor,
        n_syn: int,
    ) -> torch.Tensor:
        n = len(cls_imgs)
        flat = cls_imgs.reshape(n, -1).numpy()
        nn_idx = _knn(flat, flat, self.k)
        out: list[torch.Tensor] = []
        for _ in range(n_syn):
            i = self.rng.randint(0, n)
            j = nn_idx[i, self.rng.randint(0, self.k)]
            lam = self.rng.uniform()
            out.append(
                cls_imgs[i] * lam
                + cls_imgs[j] * (1.0 - lam),
            )
        return torch.stack(out)


class ADASYN:
    """Adaptive Synthetic Sampling.

    Extends SMOTE by allocating more synthetic samples to
    minority instances that are harder to learn — those
    surrounded by many majority-class neighbors
    (Sec. 2.1.1, Ref [10]).

    Parameters
    ----------
    k_neighbors : int
        Neighborhood size for difficulty estimation.
    target_ratio : float
        Desired minority/majority count ratio.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        target_ratio: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.k = k_neighbors
        self.target_ratio = target_ratio
        self.rng = np.random.RandomState(seed)

    def fit_resample(
        self, dataset: Dataset,
    ) -> TensorDataset:
        """Augment *dataset* with adaptive allocation."""
        images, labels = _extract_all(dataset)
        counts = Counter(labels.tolist())
        target_n = int(
            max(counts.values()) * self.target_ratio,
        )

        flat_all = images.reshape(len(images), -1).numpy()
        lbl_np = labels.numpy()

        img_parts: list[torch.Tensor] = [images]
        lbl_parts: list[torch.Tensor] = [labels]

        for cls, n_cls in sorted(counts.items()):
            g_total = target_n - n_cls
            if g_total <= 0:
                continue

            mask = lbl_np == cls
            cls_flat = flat_all[mask]
            cls_imgs = images[mask]

            diff = _difficulty(
                cls_flat, flat_all, lbl_np, cls, self.k,
            )
            if diff.sum() < 1e-8:
                diff = np.ones_like(diff)
            diff /= diff.sum()

            per_sample = np.round(
                diff * g_total,
            ).astype(int)
            syn = self._generate(
                cls_imgs, cls_flat, per_sample,
            )
            if len(syn) > 0:
                img_parts.append(syn)
                lbl_parts.append(
                    torch.full(
                        (len(syn),), cls,
                        dtype=torch.long,
                    ),
                )

        return _pack(
            torch.cat(img_parts),
            torch.cat(lbl_parts),
            dataset,
        )

    def _generate(
        self,
        cls_imgs: torch.Tensor,
        cls_flat: np.ndarray,
        per_sample: np.ndarray,
    ) -> torch.Tensor:
        nn_idx = _knn(cls_flat, cls_flat, self.k)
        out: list[torch.Tensor] = []
        for i, count in enumerate(per_sample):
            for _ in range(count):
                j = nn_idx[
                    i, self.rng.randint(0, self.k),
                ]
                lam = self.rng.uniform()
                out.append(
                    cls_imgs[i] * lam
                    + cls_imgs[j] * (1.0 - lam),
                )
        if not out:
            return torch.empty(0)
        return torch.stack(out)


# ── shared helpers ──────────────────────────────────


def _extract_all(
    ds: Dataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = [ds[i][0] for i in range(len(ds))]
    lbls = [int(ds[i][1]) for i in range(len(ds))]
    return (
        torch.stack(imgs),
        torch.tensor(lbls, dtype=torch.long),
    )


def _knn(
    query: np.ndarray,
    pool: np.ndarray,
    k: int,
) -> np.ndarray:
    """Brute-force k-NN on flattened vectors (L2)."""
    k = min(k, len(pool) - 1)
    n = len(query)
    out = np.empty((n, k), dtype=int)
    for i in range(n):
        d = np.linalg.norm(pool - query[i], axis=1)
        d[i] = np.inf
        out[i] = np.argsort(d)[:k]
    return out


def _difficulty(
    cls_flat: np.ndarray,
    all_flat: np.ndarray,
    all_labels: np.ndarray,
    cls_idx: int,
    k: int,
) -> np.ndarray:
    """Ratio of majority neighbors for each minority sample."""
    n = len(cls_flat)
    k_eff = min(k, len(all_flat) - 1)
    scores = np.zeros(n)
    for i in range(n):
        d = np.linalg.norm(all_flat - cls_flat[i], axis=1)
        nn = np.argsort(d)[1: k_eff + 1]
        scores[i] = np.sum(
            all_labels[nn] != cls_idx,
        ) / k_eff
    return scores


def _pack(
    images: torch.Tensor,
    labels: torch.Tensor,
    original: Dataset,
) -> TensorDataset:
    ds = TensorDataset(images, labels)
    ds.targets = labels.tolist()          # type: ignore[attr-defined]
    if hasattr(original, "classes"):
        ds.classes = original.classes     # type: ignore[attr-defined]
    return ds
