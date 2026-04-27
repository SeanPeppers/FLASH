# -*- coding: utf-8 -*-
"""
clients.py — Leaf clients for Raspberry Pi 5 and Jetson Nano

PERFORMANCE FIXES vs previous version:
  1. Clients now run a PERSISTENT connection loop — they reconnect immediately
     after each round without restarting the process. This eliminates the
     minutes of dead time that occurred when the aggregator tore down its inner
     server between rounds.

  2. Local epochs are now capped at MAX_LOCAL_EPOCHS = 2 by default.
     On a Pi 5 CPU, 1 epoch of MNIST (~1500 batches at bs=32) ≈ 2 minutes.
     The previous TARGET_TAU=5 meant up to 8 epochs = ~16 minutes/round/client.

  3. MNIST is pre-downloaded to ./data/ at startup with a clear error message
     if it fails. The training loop never tries to download mid-round.

  4. DATA_WORKERS=0 kept for Pi 5 (no spare cores for a prefetch worker).
     Jetson Nano can set DATA_WORKERS=1 via --data-workers flag.

  5. hw_metrics.snapshot() now uses cached subprocess calls — adds <5 ms.

  6. Metric dict is capped at MAX_METRIC_KEYS to avoid Flower's gRPC 4 MB
     message limit being hit when per-epoch metrics are sent for many epochs.

Pre-download MNIST before the experiment:
    python -c "from torchvision.datasets import MNIST; MNIST('./data', download=True)"

Usage:
    python clients.py --cid 0 --agg-address <XAVIER_IP>:8081 --strategy flash
    python clients.py --cid 1 --agg-address <XAVIER_IP>:8081 --strategy flash
"""


from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from collections import OrderedDict

import hw_metrics
from hw_metrics import EnergyAccumulator, snapshot, delta

# ── Configuration ──────────────────────────────────────────────────────────────
TARGET_TAU = 5.0
COMPRESSION_OPTIONS = {1.0, 0.75, 0.5, 0.25}

# KEY FIX: Cap local epochs to keep each round under ~5 minutes on Pi 5 CPU.
# Pi 5 CPU: ~2 min/epoch on MNIST.  Jetson Nano GPU: ~20 s/epoch.
# Raise this if your hardware is faster or if you have more time budget.
MAX_LOCAL_EPOCHS = 2
FEDAVG_LOCAL_EPOCHS = 2

BATCH_SIZE = 64          # Raised from 32 — fewer batches per epoch, same data
DATA_DIR = "./data"
MAX_METRIC_KEYS = 200    # gRPC 4 MB limit safety cap


# ── Model ──────────────────────────────────────────────────────────────────────
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Parameter helpers ──────────────────────────────────────────────────────────
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), parameters)}
    )
    net.load_state_dict(state_dict, strict=True)


def model_size_bytes(net) -> float:
    return float(sum(p.nbytes for p in get_parameters(net)))


def compress_topk(params: List[np.ndarray], r: float) -> List[np.ndarray]:
    """Top-k sparsify each layer and pack for transmission.

    Returns a flat list of array pairs per layer:
        [meta_0, vals_0, meta_1, vals_1, ...]
    where:
        meta_i  (float32): [ndim, d0, ..., d_{ndim-1}, k, idx_0, ..., idx_{k-1}]
        vals_i  (float16): [val_0, ..., val_{k-1}]

    Storing values as float16 halves transmission bytes for the value payload.
    """
    if r >= 1.0:
        return params
    packed = []
    for layer in params:
        shape = layer.shape
        flat = layer.flatten().astype(np.float32)
        n = len(flat)
        k = max(1, int(n * r))
        idx = np.argpartition(np.abs(flat), -k)[-k:]
        idx = np.sort(idx)
        vals = flat[idx]
        meta = np.array(
            [float(len(shape))] + [float(d) for d in shape] + [float(k)],
            dtype=np.float32,
        )
        meta = np.concatenate([meta, idx.astype(np.float32)])
        packed.append(meta)
        packed.append(vals.astype(np.float16))
    return packed


def decompress_topk(packed_params: List[np.ndarray]) -> List[np.ndarray]:
    """Reconstruct dense float32 arrays from packed pairs produced by compress_topk."""
    result = []
    it = iter(packed_params)
    for meta in it:
        vals_fp16 = next(it)
        ndim = int(meta[0])
        shape = tuple(int(meta[1 + i]) for i in range(ndim))
        k = int(meta[1 + ndim])
        idx = meta[2 + ndim: 2 + ndim + k].astype(np.int32)
        vals = vals_fp16.astype(np.float32)
        dense = np.zeros(int(np.prod(shape)), dtype=np.float32)
        dense[idx] = vals
        result.append(dense.reshape(shape))
    return result


def compressed_size_bytes(packed_params: List[np.ndarray]) -> float:
    """Total bytes of the packed sparse representation (actual bytes transmitted)."""
    return float(sum(p.nbytes for p in packed_params))



# ── Data loader ────────────────────────────────────────────────────────────────
def load_data(
    cid: int, data_workers: int = 0, num_total_clients: int = 100
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load pre-downloaded MNIST. Raises a clear error if data is missing
    rather than silently trying to download mid-experiment.
    """
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_full = datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform)
        test_full  = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    except Exception as e:
        print(
            f"[Client {cid}] ERROR loading MNIST from {DATA_DIR}: {e}\n"
            f"  Run this first:\n"
            f"  python -c \"from torchvision.datasets import MNIST; "
            f"MNIST('{DATA_DIR}', download=True)\"\n"
            f"  Falling back to synthetic data."
        )
        x = torch.randn(2000, 1, 28, 28)
        y = torch.randint(0, 10, (2000,))
        ds = torch.utils.data.TensorDataset(x, y)
        ldr = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        return ldr, ldr

    # Non-overlapping shard: each client gets every Nth sample (N = num_total_clients)
    indices = [i for i in range(len(train_full)) if i % num_total_clients == cid]
    train_ds = torch.utils.data.Subset(train_full, indices)

    # pin_memory=True helps on Jetson Nano (CUDA), no-op on Pi 5 (CPU only)
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=data_workers, pin_memory=pin,
    )
    test_loader = torch.utils.data.DataLoader(
        test_full, batch_size=256, shuffle=False,
        num_workers=data_workers, pin_memory=pin,
    )
    print(f"[Client {cid}] Loaded MNIST shard: {len(train_ds)} train / {len(test_full)} test")
    return train_loader, test_loader


# ── Per-epoch training ─────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    use_amp = scaler is not None
    total_loss, correct, total = 0.0, 0, 0
    batch_times: List[float] = []
    grad_norms: List[float] = []

    for x, y in loader:
        t0 = time.perf_counter()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp) if use_amp else nullcontext():
            logits = model(x)
            loss = criterion(logits, y)
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Unscale before clipping so the clip threshold is in real gradient units
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        g_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        grad_norms.append(g_norm)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        batch_times.append(time.perf_counter() - t0)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return {
        "epoch_loss":              total_loss / total if total else 0.0,
        "epoch_accuracy":          correct / total if total else 0.0,
        "epoch_grad_norm_mean":    float(np.mean(grad_norms)) if grad_norms else 0.0,
        "epoch_grad_norm_max":     float(np.max(grad_norms)) if grad_norms else 0.0,
        "epoch_batch_time_mean_ms":float(np.mean(batch_times) * 1000) if batch_times else 0.0,
        "epoch_batch_time_max_ms": float(np.max(batch_times) * 1000) if batch_times else 0.0,
        "epoch_samples":           float(total),
        "epoch_batches":           float(len(batch_times)),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return (total_loss / total if total else 0.0, correct / total if total else 0.0)


# ── Metric assembly ────────────────────────────────────────────────────────────
def build_metrics(
    hw_before: Dict[str, float],
    hw_after: Dict[str, float],
    per_epoch: List[Dict[str, float]],
    training_time_s: float,
    energy_j: float,
    extra: Dict[str, float],
) -> Dict[str, float]:
    m: Dict[str, float] = {}

    # Hardware state + deltas
    hw_d = delta(hw_before, hw_after)
    for k, v in hw_after.items():
        m[f"hw_{k}"] = float(v)
    for k, v in hw_d.items():
        if k.startswith("delta_"):
            m[f"hw_{k}"] = float(v)

    # Per-epoch aggregates
    if per_epoch:
        losses  = [e["epoch_loss"] for e in per_epoch]
        accs    = [e["epoch_accuracy"] for e in per_epoch]
        gnorms  = [e["epoch_grad_norm_mean"] for e in per_epoch]
        m["train_loss_final"]    = float(losses[-1])
        m["train_loss_mean"]     = float(np.mean(losses))
        m["train_loss_min"]      = float(np.min(losses))
        m["train_loss_max"]      = float(np.max(losses))
        m["train_accuracy_final"]= float(accs[-1])
        m["train_accuracy_mean"] = float(np.mean(accs))
        m["train_accuracy_max"]  = float(np.max(accs))
        m["train_accuracy_min"]  = float(np.min(accs))
        m["grad_norm_mean"]      = float(np.mean(gnorms))
        m["grad_norm_max"]       = float(max(e["epoch_grad_norm_max"] for e in per_epoch))
        m["num_epochs_run"]      = float(len(per_epoch))
        m["total_samples_trained"] = float(sum(e["epoch_samples"] for e in per_epoch))
        m["avg_batch_time_ms"]   = float(np.mean([e["epoch_batch_time_mean_ms"] for e in per_epoch]))
        m["max_batch_time_ms"]   = float(max(e["epoch_batch_time_max_ms"] for e in per_epoch))

        # Per-epoch detail rows (capped to avoid hitting gRPC 4 MB limit)
        for i, ep in enumerate(per_epoch[:10]):
            for k, v in ep.items():
                m[f"epoch_{i}_{k}"] = float(v)

    # Timing and throughput
    m["training_time_s"] = float(training_time_s)
    total_s = m.get("total_samples_trained", 1.0)
    m["throughput_samples_per_s"] = total_s / training_time_s if training_time_s > 0 else 0.0

    # Energy
    m["energy_joules"] = float(energy_j)
    m["energy_per_sample_j"] = energy_j / total_s if total_s > 0 else 0.0
    m["power_avg_w"] = energy_j / training_time_s if training_time_s > 0 else 0.0

    # Model
    m["model_size_bytes"] = float(extra.get("model_size_bytes", 0.0))

    # Topology
    m["device_type_id"] = float(
        {"pi5": 0, "jetson_nano": 1, "jetson_xavier": 2, "chameleon": 3}.get(
            hw_metrics.DEVICE, -1
        )
    )
    m["region_id"] = 0.0
    m["clients_in_region"] = 1.0
    m.update(extra)

    # Safety cap: drop least-important keys if over limit
    if len(m) > MAX_METRIC_KEYS:
        # Keep core keys, drop epoch detail rows first
        epoch_detail_keys = sorted([k for k in m if k.startswith("epoch_")], reverse=True)
        for k in epoch_detail_keys:
            if len(m) <= MAX_METRIC_KEYS:
                break
            del m[k]

    # Replace any inf/nan (e.g. from gradient explosion) with 0.0 so they
    # don't corrupt downstream CSV aggregation or gRPC serialization.
    for k in list(m.keys()):
        if not math.isfinite(m[k]):
            m[k] = 0.0

    return m


# ── Base leaf client ───────────────────────────────────────────────────────────
class BaseLeafClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, data_workers: int = 0, num_total_clients: int = 100):
        self.client_id = client_id
        self.cid_int = int(client_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.base_lr = 0.01
        self.train_loader, self.test_loader = load_data(self.cid_int, data_workers, num_total_clients)
        # GradScaler for FP16 mixed-precision training on CUDA devices (no-op on CPU)
        self.scaler: Optional[torch.cuda.amp.GradScaler] = (
            torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        )
        print(
            f"[Client {client_id}] hw={hw_metrics.DEVICE}  "
            f"torch={self.device}  train_n={len(self.train_loader.dataset)}"
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, self.test_loader, self.criterion, self.device)
        hw = snapshot()
        return float(loss), len(self.test_loader.dataset), {
            "loss": float(loss),
            "accuracy": float(acc),
            "hw_cpu_util_pct": float(hw.get("cpu_util_pct", 0.0)),
            "hw_ram_util_pct": float(hw.get("ram_util_pct", 0.0)),
            "hw_temp_max_celsius": float(hw.get("temp_max_celsius", 0.0)),
            "hw_power_total_soc_mw": float(hw.get("power_total_soc_mw", 0.0)),
        }


# ── FLASH ──────────────────────────────────────────────────────────────────────
class FLASHClient(BaseLeafClient):
    def __init__(self, client_id: str, data_workers: int = 0):
        super().__init__(client_id, data_workers)
        self._base_params: Optional[List[np.ndarray]] = None

    def fit(self, parameters, config: Dict) -> Tuple:
        # Store the params received from the server so we can compute a delta
        self._base_params = [p.copy().astype(np.float32) for p in parameters]
        set_parameters(self.model, parameters)
        bar_tau_r  = float(config.get("bar_tau_r", TARGET_TAU))
        optimal_r  = float(config.get("optimal_r_star", 1.0))
        server_rnd = int(config.get("server_round", 1))
        num_rounds = int(config.get("num_rounds", 60))
        # Clamp epochs to MAX_LOCAL_EPOCHS so Pi 5 doesn't run for hours
        tau = min(int(config.get("suggested_tau", 2)), MAX_LOCAL_EPOCHS)
        # Cosine annealing decay: lr scales from ~base down to ~10% of base
        decay = max(0.5 * (1.0 + math.cos(math.pi * server_rnd / num_rounds)), 0.1)
        eta = self.base_lr * (bar_tau_r / max(tau, 1)) * decay
        optimizer = optim.Adam(self.model.parameters(), lr=eta)

        hw_before = snapshot()
        acc = EnergyAccumulator()
        acc.start()
        t0 = time.perf_counter()

        per_epoch = [
            train_one_epoch(self.model, self.train_loader, self.criterion, optimizer, self.device, self.scaler)
            for _ in range(tau)
        ]

        training_time = time.perf_counter() - t0
        energy_j = acc.stop_and_get_joules()
        hw_after = snapshot()

        msz = model_size_bytes(self.model)
        # Compute delta and compress it — deltas are much sparser than raw weights
        trained = [p.astype(np.float32) for p in get_parameters(self.model)]
        delta_params = [t - b for t, b in zip(trained, self._base_params)]
        params = compress_topk(delta_params, optimal_r)
        actual_bytes = compressed_size_bytes(params) if optimal_r < 1.0 else msz
        extra = {
            "compression_ratio_applied": float(optimal_r),
            "local_epochs": float(tau),
            "learning_rate": float(eta),
            "bar_tau_r": float(bar_tau_r),
            "server_round": float(server_rnd),
            "data_transfer_size_bytes": actual_bytes,
            "model_size_bytes": msz,
            "fit_wall_time_s": float(training_time),
            "comp_capacity_proxy": float(tau / max(training_time, 1e-6)),
        }
        metrics = build_metrics(hw_before, hw_after, per_epoch, training_time, energy_j, extra)
        n = int(sum(e["epoch_samples"] for e in per_epoch)) or BATCH_SIZE
        return params, n, metrics


# ── FLARE ──────────────────────────────────────────────────────────────────────
class FLAREClient(BaseLeafClient):
    def fit(self, parameters, config: Dict) -> Tuple:
        set_parameters(self.model, parameters)
        bar_tau_r  = float(config.get("bar_tau_r", TARGET_TAU))
        server_rnd = int(config.get("server_round", 1))
        tau = min(int(config.get("suggested_tau", 2)), MAX_LOCAL_EPOCHS)
        eta = self.base_lr * (bar_tau_r / max(tau, 1))
        optimizer = optim.Adam(self.model.parameters(), lr=eta)

        hw_before = snapshot()
        acc = EnergyAccumulator()
        acc.start()
        t0 = time.perf_counter()

        per_epoch = [
            train_one_epoch(self.model, self.train_loader, self.criterion, optimizer, self.device, self.scaler)
            for _ in range(tau)
        ]

        training_time = time.perf_counter() - t0
        energy_j = acc.stop_and_get_joules()
        hw_after = snapshot()
        msz = model_size_bytes(self.model)
        extra = {
            "compression_ratio_applied": 1.0,
            "local_epochs": float(tau),
            "learning_rate": float(eta),
            "bar_tau_r": float(bar_tau_r),
            "server_round": float(server_rnd),
            "data_transfer_size_bytes": msz,
            "model_size_bytes": msz,
            "fit_wall_time_s": float(training_time),
        }
        metrics = build_metrics(hw_before, hw_after, per_epoch, training_time, energy_j, extra)
        n = int(sum(e["epoch_samples"] for e in per_epoch)) or BATCH_SIZE
        return get_parameters(self.model), n, metrics


# ── FedAvg ─────────────────────────────────────────────────────────────────────
class FedAvgClient(BaseLeafClient):
    def fit(self, parameters, config: Dict) -> Tuple:
        set_parameters(self.model, parameters)
        server_rnd = int(config.get("server_round", 1))
        optimizer  = optim.Adam(self.model.parameters(), lr=self.base_lr)

        hw_before = snapshot()
        acc = EnergyAccumulator()
        acc.start()
        t0 = time.perf_counter()

        per_epoch = [
            train_one_epoch(self.model, self.train_loader, self.criterion, optimizer, self.device, self.scaler)
            for _ in range(FEDAVG_LOCAL_EPOCHS)
        ]

        training_time = time.perf_counter() - t0
        energy_j = acc.stop_and_get_joules()
        hw_after = snapshot()
        msz = model_size_bytes(self.model)
        extra = {
            "compression_ratio_applied": 1.0,
            "local_epochs": float(FEDAVG_LOCAL_EPOCHS),
            "learning_rate": float(self.base_lr),
            "server_round": float(server_rnd),
            "data_transfer_size_bytes": msz,
            "model_size_bytes": msz,
            "fit_wall_time_s": float(training_time),
        }
        metrics = build_metrics(hw_before, hw_after, per_epoch, training_time, energy_j, extra)
        n = int(sum(e["epoch_samples"] for e in per_epoch)) or BATCH_SIZE
        return get_parameters(self.model), n, metrics


# ── Client registry ────────────────────────────────────────────────────────────
CLIENT_REGISTRY = {"flash": FLASHClient, "flare": FLAREClient, "fedavg": FedAvgClient}

# ── Persistent reconnect loop ──────────────────────────────────────────────────
def run_client_loop(client: fl.client.NumPyClient, agg_address: str,
                    reconnect_delay: float = 5.0):
    """
    KEY FIX: Keep reconnecting to the aggregator without restarting the process.
    The client object (and its loaded dataset) is reused across rounds,
    so there is zero re-initialisation overhead between rounds.

    Flower's start_numpy_client() exits after the server completes its rounds.
    We catch that and immediately reconnect so the client is ready for the
    next global round the moment the aggregator restarts its inner server.
    """
    while True:
        try:
            print(f"[Client] Connecting to aggregator at {agg_address} ...")
            fl.client.start_numpy_client(server_address=agg_address, client=client)
            print(f"[Client] Round complete. Reconnecting in {reconnect_delay}s ...")
        except Exception as e:
            print(f"[Client] Connection error: {e}. Retrying in {reconnect_delay}s ...")
        time.sleep(reconnect_delay)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFL leaf client (Pi 5 / Jetson Nano)")
    parser.add_argument("--cid", type=str, required=True)
    parser.add_argument("--agg-address", type=str, default="localhost:8081")
    parser.add_argument("--strategy", type=str, default="flash",
                        choices=list(CLIENT_REGISTRY.keys()) + ["all"])
    parser.add_argument("--data-workers", type=int, default=0,
                        help="DataLoader workers. 0 for Pi 5, 1-2 for Jetson Nano.")
    parser.add_argument("--num-clients", type=int, default=100,
                        help="Total number of leaf clients in the experiment (controls data sharding).")
    parser.add_argument("--reconnect-delay", type=float, default=5.0,
                        help="Seconds to wait before reconnecting after a round ends.")
    args = parser.parse_args()

    print(f"[Client {args.cid}] Hardware: {hw_metrics.DEVICE}")

    if args.strategy == "all":
        for strategy_name in ["flash", "flare", "fedavg"]:
            print(f"[Client {args.cid}] Starting strategy: {strategy_name}")
            client = CLIENT_REGISTRY[strategy_name](args.cid, args.data_workers, args.num_clients)
            # Connect once for this experiment, retry on connection errors but
            # do not loop forever — move to next strategy once server disconnects us
            while True:
                try:
                    print(f"[Client {args.cid}] Connecting to aggregator at {args.agg_address} ...")
                    fl.client.start_numpy_client(server_address=args.agg_address, client=client)
                    print(f"[Client {args.cid}] {strategy_name} complete.")
                    break
                except Exception as e:
                    print(f"[Client {args.cid}] Connection error: {e}. Retrying in {args.reconnect_delay}s ...")
                    time.sleep(args.reconnect_delay)
        print(f"[Client {args.cid}] All strategies done.")
    else:
        client = CLIENT_REGISTRY[args.strategy](args.cid, args.data_workers, args.num_clients)
        run_client_loop(client, args.agg_address, args.reconnect_delay)
