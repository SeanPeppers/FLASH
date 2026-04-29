# -*- coding: utf-8 -*-
"""
verify_flash.py -- Local sanity tests for the FLASH compression + aggregation pipeline.

Runs entirely on CPU with no Flower, no real devices.
Falls back to synthetic data if MNIST is not downloaded.

Tests
-----
1. Compression round-trip  -- compress_topk -> decompress_topk recovers the delta
2. Aggregation bug regression -- old (buggy) vs new (fixed) logic side by side
3. Mini FL training run  -- 5 rounds, 2 clients, loss must decrease and model must stay healthy

Usage:
    python verify_flash.py
"""

from __future__ import annotations

import sys
import copy
import math
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from clients import (
    SimpleNet,
    get_parameters,
    set_parameters,
    compress_topk,
    decompress_topk,
    train_one_epoch,
    evaluate_model,
    BATCH_SIZE,
)

DEVICE = torch.device("cpu")
PASS = "[PASS]"
FAIL = "[FAIL]"
SEP  = "-" * 60


def _norm(params: List[np.ndarray]) -> float:
    return float(np.sqrt(sum(np.sum(p.astype(np.float32) ** 2) for p in params)))


def _make_loader(n_samples: int = 1200, batch_size: int = BATCH_SIZE):
    """MNIST if available, synthetic fallback."""
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds = datasets.MNIST("./data", train=True, download=False, transform=transform)
        ds = torch.utils.data.Subset(ds, list(range(n_samples)))
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True), "MNIST"
    except Exception:
        x = torch.randn(n_samples, 1, 28, 28)
        y = torch.randint(0, 10, (n_samples,))
        ds = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True), "synthetic"


def _fedavg(client_params_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Weighted FedAvg (equal weights here)."""
    n = len(client_params_list)
    return [
        np.mean([c[i].astype(np.float32) for c in client_params_list], axis=0)
        for i in range(len(client_params_list[0]))
    ]


# ── Test 1: Compression round-trip ────────────────────────────────────────────
def test_compression_roundtrip():
    print(SEP)
    print("TEST 1: compress_topk -> decompress_topk round-trip")
    print(SEP)

    model = SimpleNet()
    params = get_parameters(model)
    original_norm = _norm(params)
    passed = True

    for r in [1.0, 0.75, 0.5, 0.25]:
        packed = compress_topk(params, r)

        if r >= 1.0:
            recovered = packed  # r=1.0 returns params unchanged
        else:
            recovered = decompress_topk(packed)

        rec_norm = _norm(recovered)

        # Shape must be preserved
        shapes_ok = all(a.shape == b.shape for a, b in zip(params, recovered))

        # For r=1.0: exact round-trip (no compression applied)
        # For r<1.0: top-k keeps the largest values; check we recover >= r of the energy
        if r >= 1.0:
            max_err = float(max(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))
                                for a, b in zip(params, recovered)))
            ok = shapes_ok and max_err == 0.0
            print(f"  r={r:.2f}  shapes_ok={shapes_ok}  max_err={max_err:.2e}  norm={rec_norm:.4f}  {PASS if ok else FAIL}")
        else:
            energy_ratio = (rec_norm ** 2) / max(original_norm ** 2, 1e-9)
            # Top-k at ratio r should retain at least r*0.9 of energy (magnitudes are biased high)
            ok = shapes_ok and energy_ratio >= r * 0.9
            print(f"  r={r:.2f}  shapes_ok={shapes_ok}  energy_retained={energy_ratio:.3f} (>={r*0.9:.3f})  {PASS if ok else FAIL}")

        if not ok:
            passed = False

    print(f"\n  Result: {PASS if passed else FAIL}\n")
    return passed


# ── Test 2: Aggregation bug regression ────────────────────────────────────────
def test_aggregation_bug_regression():
    print(SEP)
    print("TEST 2: Aggregation bug regression (old logic vs fixed logic)")
    print(SEP)

    global_model = SimpleNet()
    base = get_parameters(global_model)
    base_norm = _norm(base)
    print(f"  Global model norm (base): {base_norm:.4f}")

    # Simulate 2 clients training and computing deltas
    client_deltas = []
    for cid in range(2):
        client_model = SimpleNet()
        set_parameters(client_model, base)
        optimizer = optim.SGD(client_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        loader, _ = _make_loader(n_samples=256)
        train_one_epoch(client_model, loader, criterion, optimizer, DEVICE)
        trained = get_parameters(client_model)
        delta = [t.astype(np.float32) - b.astype(np.float32) for t, b in zip(trained, base)]
        client_deltas.append(delta)
        print(f"  Client {cid} delta_norm={_norm(delta):.4f}  trained_norm={_norm(trained):.4f}")

    # ── Old (buggy) logic: r=1.0 skips delta reconstruction, FedAvgs raw deltas ──
    old_aggregated = _fedavg(client_deltas)
    old_norm = _norm(old_aggregated)

    # ── New (fixed) logic: always add base to delta ──
    reconstructed = [_fedavg([d])[0] + b.astype(np.float32)
                     for d, b in zip(zip(*client_deltas), base)]
    new_aggregated = [
        (np.mean([client_deltas[c][i].astype(np.float32) for c in range(2)], axis=0)
         + base[i].astype(np.float32))
        for i in range(len(base))
    ]
    new_norm = _norm(new_aggregated)

    print(f"\n  OLD logic (bug): treated deltas as full params -> norm={old_norm:.4f}")
    print(f"  NEW logic (fix): base + averaged deltas        -> norm={new_norm:.4f}")
    print(f"  Base norm for reference:                          {base_norm:.4f}")

    # Old logic produces near-zero model (deltas are tiny); new logic stays close to base norm
    old_is_corrupted = old_norm < base_norm * 0.05
    new_is_healthy   = abs(new_norm - base_norm) / base_norm < 0.5

    print(f"\n  Old model corrupted (norm << base): {old_is_corrupted}  {PASS if old_is_corrupted else FAIL}")
    print(f"  New model healthy (norm ~ base):    {new_is_healthy}   {PASS if new_is_healthy else FAIL}")

    passed = old_is_corrupted and new_is_healthy
    print(f"\n  Result: {PASS if passed else FAIL}\n")
    return passed


# ── Test 3: Mini FL training run ───────────────────────────────────────────────
def test_mini_training_run():
    print(SEP)
    print("TEST 3: Mini FL training run (5 rounds, 2 clients, r=0.75)")
    print(SEP)

    loader, data_source = _make_loader(n_samples=1200)
    eval_loader, _ = _make_loader(n_samples=400)
    print(f"  Data source: {data_source}")

    global_model = SimpleNet()
    criterion = nn.CrossEntropyLoss()

    initial_loss, initial_acc = evaluate_model(global_model, eval_loader, criterion, DEVICE)
    print(f"  Round 0 (init): loss={initial_loss:.4f}  acc={initial_acc:.4f}")

    losses = [initial_loss]
    accs   = [initial_acc]

    num_rounds = 5
    num_clients = 2
    r = 0.75

    for rnd in range(1, num_rounds + 1):
        base = get_parameters(global_model)
        client_full_params = []

        for cid in range(num_clients):
            client_model = SimpleNet()
            set_parameters(client_model, base)

            decay = max(0.5 * (1.0 + math.cos(math.pi * rnd / num_rounds)), 0.1)
            lr = 0.01 * decay
            optimizer = optim.Adam(client_model.parameters(), lr=lr)

            # Use a different slice per client to simulate non-IID shards
            shard_size = 600
            offset = cid * shard_size
            shard = torch.utils.data.Subset(
                loader.dataset,
                list(range(offset, min(offset + shard_size, len(loader.dataset))))
            )
            shard_loader = torch.utils.data.DataLoader(shard, batch_size=BATCH_SIZE, shuffle=True)

            train_one_epoch(client_model, shard_loader, criterion, optimizer, DEVICE)
            trained = get_parameters(client_model)

            # FLASH delta + compress
            delta = [t.astype(np.float32) - b.astype(np.float32) for t, b in zip(trained, base)]
            packed = compress_topk(delta, r)

            # FIXED aggregation: always decompress + add base (r=1.0 path uses raw delta)
            d = decompress_topk(packed) if r < 1.0 else packed
            full = [b.astype(np.float32) + di for b, di in zip(base, d)]
            client_full_params.append(full)

        # FedAvg
        new_params = _fedavg(client_full_params)
        set_parameters(global_model, new_params)

        loss, acc = evaluate_model(global_model, eval_loader, criterion, DEVICE)
        model_norm = _norm(new_params)
        losses.append(loss)
        accs.append(acc)
        print(f"  Round {rnd}: loss={loss:.4f}  acc={acc:.4f}  model_norm={model_norm:.2f}")

    print()
    loss_decreased  = losses[-1] < losses[0]
    acc_above_random = accs[-1] > 0.12  # well above 10% random baseline
    model_not_corrupted = _norm(get_parameters(global_model)) > 1.0

    print(f"  Loss decreased  ({losses[0]:.4f} -> {losses[-1]:.4f}): {PASS if loss_decreased else FAIL}")
    print(f"  Accuracy above random ({accs[-1]:.4f} > 0.12):         {PASS if acc_above_random else FAIL}")
    print(f"  Model norm healthy (> 1.0):                            {PASS if model_not_corrupted else FAIL}")

    passed = loss_decreased and acc_above_random and model_not_corrupted
    print(f"\n  Result: {PASS if passed else FAIL}\n")
    return passed


# ── Runner ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FLASH Pipeline Verification")
    print("=" * 60 + "\n")

    t0 = time.perf_counter()
    results = {
        "Compression round-trip":       test_compression_roundtrip(),
        "Aggregation bug regression":   test_aggregation_bug_regression(),
        "Mini FL training run":         test_mini_training_run(),
    }
    elapsed = time.perf_counter() - t0

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Overall: {'ALL TESTS PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 60 + "\n")

    sys.exit(0 if all_passed else 1)
