# -*- coding: utf-8 -*-
"""
baseline_local.py -- Pure local training baseline (floor reference)

Trains the exact same SimpleNet on a single node using a single client's
MNIST shard (even-index rows, matching cid=0 in the FL experiments).
No federation, no communication overhead.

Run:
    python baseline_local.py --rounds 60 --epochs-per-round 2 --output-dir ./fl_results_hfl

Results land in:
    <output-dir>/local_baseline_eval_loss.csv
    <output-dir>/local_baseline_eval_metrics.csv
    <output-dir>/local_baseline_fit_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

import hw_metrics
from hw_metrics import EnergyAccumulator, snapshot, delta
from clients import (
    SimpleNet, load_data, train_one_epoch, evaluate_model,
    build_metrics, model_size_bytes, BATCH_SIZE, MAX_LOCAL_EPOCHS,
)

# -- Config ---------------------------------------------------------------------
BASE_LR = 0.01
TARGET_TAU = 5.0


def run(rounds: int, epochs_per_round: int, output_dir: Path, cid: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    train_loader, test_loader = load_data(cid, data_workers=0, num_total_clients=100)

    print(f"[LocalBaseline] device={device}  hw={hw_metrics.DEVICE}")
    print(f"[LocalBaseline] {rounds} rounds x {epochs_per_round} epochs  "
          f"train_n={len(train_loader.dataset)}  test_n={len(test_loader.dataset)}")

    fit_rows:  List[Dict] = []
    eval_rows: List[Dict] = []
    loss_rows: List[Dict] = []

    for rnd in range(1, rounds + 1):
        tau = epochs_per_round
        # Match FL client LR formula + cosine decay
        decay = max(0.5 * (1.0 + math.cos(math.pi * rnd / rounds)), 0.1)
        eta = BASE_LR * (TARGET_TAU / max(tau, 1)) * decay
        optimizer = optim.Adam(model.parameters(), lr=eta)

        hw_before = snapshot()
        acc = EnergyAccumulator()
        acc.start()
        t0 = time.perf_counter()

        per_epoch = [
            train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            for _ in range(tau)
        ]

        training_time = time.perf_counter() - t0
        energy_j = acc.stop_and_get_joules()
        hw_after = snapshot()

        extra = {
            "compression_ratio_applied": 1.0,
            "local_epochs": float(tau),
            "learning_rate": float(eta),
            "server_round": float(rnd),
            "data_transfer_size_bytes": 0.0,
            "model_size_bytes": model_size_bytes(model),
            "fit_wall_time_s": float(training_time),
        }
        fit_m = build_metrics(hw_before, hw_after, per_epoch, training_time, energy_j, extra)
        fit_m["round"] = float(rnd)
        fit_rows.append(fit_m)

        eval_loss, eval_acc = evaluate_model(model, test_loader, criterion, device)
        hw_eval = snapshot()
        eval_m = {
            "round": rnd,
            "loss": eval_loss,
            "accuracy": eval_acc,
            "hw_cpu_util_pct": hw_eval.get("cpu_util_pct", 0.0),
            "hw_ram_util_pct": hw_eval.get("ram_util_pct", 0.0),
        }
        eval_rows.append(eval_m)
        loss_rows.append({"round": rnd, "eval_loss": eval_loss})

        print(f"  round {rnd:3d}/{rounds}  loss={eval_loss:.4f}  acc={eval_acc:.4f}"
              f"  energy={energy_j:.2f}J  lr={eta:.5f}")

    _save_csvs(fit_rows, eval_rows, loss_rows, output_dir)


def _save_csvs(fit_rows, eval_rows, loss_rows, output_dir: Path):
    import pandas as pd

    pd.DataFrame(fit_rows).sort_values("round").to_csv(
        output_dir / "local_baseline_fit_metrics.csv", index=False
    )
    pd.DataFrame(eval_rows).sort_values("round").to_csv(
        output_dir / "local_baseline_eval_metrics.csv", index=False
    )
    with open(output_dir / "local_baseline_eval_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "eval_loss"])
        for row in loss_rows:
            w.writerow([row["round"], row["eval_loss"]])

    print(f"\n[LocalBaseline] Results saved to {output_dir}/local_baseline_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure local training baseline")
    parser.add_argument("--rounds",           type=int, default=60)
    parser.add_argument("--epochs-per-round", type=int, default=MAX_LOCAL_EPOCHS)
    parser.add_argument("--cid",              type=int, default=0,
                        help="Which client shard to use (0=even, 1=odd indices)")
    parser.add_argument("--output-dir",       type=str, default="./fl_results_hfl")
    args = parser.parse_args()

    run(
        rounds=args.rounds,
        epochs_per_round=args.epochs_per_round,
        output_dir=Path(args.output_dir),
        cid=args.cid,
    )
