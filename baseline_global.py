# -*- coding: utf-8 -*-
"""
baseline_global.py -- Global pooled training baseline (ceiling reference)

Simulates the oracle upper bound: all client data is pooled on the server
and trained in a standard centralized fashion. No federation, no communication,
no data heterogeneity effects.

This establishes the accuracy/loss ceiling that the federated setup is measured
against. The gap between this and the FL results quantifies the cost of federation.

Run:
    python baseline_global.py --rounds 60 --epochs-per-round 2 --output-dir ./fl_results_hfl

Results land in:
    <output-dir>/global_baseline_eval_loss.csv
    <output-dir>/global_baseline_eval_metrics.csv
    <output-dir>/global_baseline_fit_metrics.csv
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
from torch.utils.data import ConcatDataset, DataLoader

import hw_metrics
from hw_metrics import EnergyAccumulator, snapshot
from clients import (
    SimpleNet, train_one_epoch, evaluate_model,
    build_metrics, model_size_bytes, BATCH_SIZE, MAX_LOCAL_EPOCHS,
    DATA_DIR,
)

# -- Config ---------------------------------------------------------------------
BASE_LR = 0.01
TARGET_TAU = 5.0
NUM_CLIENTS = 100


def _load_pooled() -> tuple:
    """Load full MNIST train split by concatenating all client shards."""
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_full = datasets.MNIST(DATA_DIR, train=True,  download=False, transform=transform)
        test_full  = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    except Exception as e:
        raise RuntimeError(
            f"MNIST not found at {DATA_DIR}: {e}\n"
            "Run: python -c \"from torchvision.datasets import MNIST; "
            f"MNIST('{DATA_DIR}', download=True)\""
        )

    # Re-create each client's shard and concatenate -- same split as FL experiment
    from torch.utils.data import Subset
    # Reconstruct the same sharding as the FL experiment so no data is excluded
    shards = [
        Subset(train_full, [i for i in range(len(train_full)) if i % NUM_CLIENTS == cid])
        for cid in range(NUM_CLIENTS)
    ]
    pooled_train = ConcatDataset(shards)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        pooled_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin
    )
    test_loader = DataLoader(
        test_full, batch_size=256, shuffle=False, pin_memory=pin
    )
    return train_loader, test_loader


def run(rounds: int, epochs_per_round: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    train_loader, test_loader = _load_pooled()

    print(f"[GlobalBaseline] device={device}  hw={hw_metrics.DEVICE}")
    print(f"[GlobalBaseline] Pooled {NUM_CLIENTS} client shards -> "
          f"{len(train_loader.dataset)} train / {len(test_loader.dataset)} test")
    print(f"[GlobalBaseline] {rounds} rounds x {epochs_per_round} epochs")

    fit_rows:  List[Dict] = []
    eval_rows: List[Dict] = []
    loss_rows: List[Dict] = []

    for rnd in range(1, rounds + 1):
        tau = epochs_per_round
        # Match FL LR formula + cosine decay for fair comparison
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
            "num_clients_pooled": float(NUM_CLIENTS),
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
        output_dir / "global_baseline_fit_metrics.csv", index=False
    )
    pd.DataFrame(eval_rows).sort_values("round").to_csv(
        output_dir / "global_baseline_eval_metrics.csv", index=False
    )
    with open(output_dir / "global_baseline_eval_loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "eval_loss"])
        for row in loss_rows:
            w.writerow([row["round"], row["eval_loss"]])

    print(f"\n[GlobalBaseline] Results saved to {output_dir}/global_baseline_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global pooled training baseline (ceiling)")
    parser.add_argument("--rounds",           type=int, default=60)
    parser.add_argument("--epochs-per-round", type=int, default=MAX_LOCAL_EPOCHS)
    parser.add_argument("--output-dir",       type=str, default="./fl_results_hfl")
    args = parser.parse_args()

    run(
        rounds=args.rounds,
        epochs_per_round=args.epochs_per_round,
        output_dir=Path(args.output_dir),
    )
