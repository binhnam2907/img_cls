"""Run all 8 imbalance strategy experiments.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --epochs 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [
    "configs/s1_weighted_ce.yaml",
    "configs/s2_oversampling.yaml",
    "configs/s3_focal.yaml",
    "configs/s4_cb_loss.yaml",
    "configs/s5_label_smoothing.yaml",
    "configs/s6_mixup.yaml",
    "configs/s7_cutmix.yaml",
    "configs/s8_combined_best.yaml",
]

STRATEGY_NAMES = [
    "Weighted CE",
    "Oversampling",
    "Focal Loss",
    "CB Loss",
    "Label Smoothing",
    "Mixup",
    "CutMix",
    "Combined Best",
]

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _run_cmd(cmd: list[str], label: str) -> bool:
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=str(ROOT),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  FAILED: {label}")
        print(result.stderr[-500:])
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=1,
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
    )
    args = parser.parse_args()

    total_start = time.time()
    results = []

    for i, (cfg_path, name) in enumerate(
        zip(CONFIGS, STRATEGY_NAMES), 1,
    ):
        print(
            f"\n{'='*50}\n"
            f"[{i}/8] {name} ({cfg_path})\n"
            f"{'='*50}"
        )
        start = time.time()

        overrides = [
            f"training.epochs={args.epochs}",
            f"data.num_workers={args.num_workers}",
        ]
        train_ok = _run_cmd(
            [
                PYTHON, "scripts/train.py",
                "--config", cfg_path,
                "--override",
            ] + overrides,
            f"{name} training",
        )

        if not train_ok:
            results.append((name, "TRAIN FAILED"))
            continue

        cfg_dir = Path(cfg_path)
        out_dir = f"results/{cfg_dir.stem}"
        ckpt = f"{out_dir}/checkpoints/best.pth"

        eval_ok = _run_cmd(
            [
                PYTHON, "scripts/evaluate.py",
                "--config", cfg_path,
                "--checkpoint", ckpt,
                "--split", "val",
            ],
            f"{name} evaluation",
        )

        elapsed = time.time() - start
        status = "OK" if eval_ok else "EVAL FAILED"
        results.append((name, status, elapsed))
        print(f"  Done in {elapsed:.1f}s [{status}]")

    total = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"All experiments completed in {total:.1f}s")
    print(f"{'='*50}")
    for r in results:
        name = r[0]
        status = r[1]
        t = r[2] if len(r) > 2 else 0
        print(f"  {name:20s} {status:15s} {t:.1f}s")


if __name__ == "__main__":
    main()
