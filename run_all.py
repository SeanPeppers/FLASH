# -*- coding: utf-8 -*-
"""
run_all.py — Full automated experiment runner

Runs in this order on a single machine (or the Chameleon server):
  1. FL experiment — all 3 strategies (FLASH → FLARE → FedAvg)
     Starts: 1 server + 5 aggregators + 100 clients as local subprocesses
  2. Global simulation baseline (ceiling) — pooled centralised training
  3. Local baseline (floor) — single-client local training

All subprocess logs are written to ./logs/<role>_<id>.log.
Results land in ./fl_results_hfl/ as usual.

Usage:
    python run_all.py                          # full run, all defaults
    python run_all.py --rounds 60 --skip-fl    # baselines only (FL already done)
    python run_all.py --skip-baselines         # FL only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── Config ─────────────────────────────────────────────────────────────────────
NUM_AGGREGATORS   = 5
NUM_LEAF_CLIENTS  = 20   # per aggregator
NUM_TOTAL_CLIENTS = NUM_AGGREGATORS * NUM_LEAF_CLIENTS  # 100
SERVER_PORT       = 8080
AGG_BASE_PORT     = 8081  # aggregator 0 → 8081, agg 1 → 8082, … agg 4 → 8085
SERVER_HOST       = "127.0.0.1"
ROUNDS            = 60
OUTPUT_DIR        = "./fl_results_hfl"
LOG_DIR           = "./logs"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _python() -> str:
    return sys.executable


def _log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", buffering=1)


def _launch(args: List[str], log_path: Path, env=None) -> subprocess.Popen:
    log = _log(log_path)
    print(f"  [launch] {' '.join(args)}")
    print(f"           → {log_path}")
    return subprocess.Popen(
        args,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env or os.environ.copy(),
    )


def _kill_all(procs: List[subprocess.Popen]):
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(2)
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass


def _wait_for_server(timeout: float = 60.0) -> bool:
    """Poll until the server's gRPC port is open."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


# ── Phase 1: FL experiment ─────────────────────────────────────────────────────

def run_fl(rounds: int, output_dir: str, log_dir: Path):
    print("\n" + "=" * 64)
    print("  PHASE 1 — Federated Learning (FLASH → FLARE → FedAvg)")
    print("=" * 64)

    procs: List[subprocess.Popen] = []

    try:
        # 1. Server
        server_proc = _launch(
            [_python(), "server.py",
             "--rounds", str(rounds),
             "--port", str(SERVER_PORT),
             "--experiment", "all",
             "--no-wait",
             "--aggregators", str(NUM_AGGREGATORS),
             "--output-dir", output_dir],
            log_dir / "server.log",
        )
        procs.append(server_proc)

        print(f"\n  Waiting for server to open port {SERVER_PORT} ...")
        if not _wait_for_server(timeout=60):
            raise RuntimeError("Server did not open port within 60 s")
        print("  Server ready.")

        # 2. Aggregators (one per port)
        agg_procs = []
        for agg_id in range(NUM_AGGREGATORS):
            agg_port = AGG_BASE_PORT + agg_id
            p = _launch(
                [_python(), "aggregator.py",
                 "--strategy", "all",
                 "--rounds", str(rounds),
                 "--leaf-clients", str(NUM_LEAF_CLIENTS),
                 "--agg-port", str(agg_port),
                 "--server-address", f"{SERVER_HOST}:{SERVER_PORT}"],
                log_dir / f"aggregator_{agg_id}.log",
            )
            agg_procs.append(p)
            procs.append(p)

        # Wait for all aggregator inner servers to open their ports
        print(f"\n  Waiting for {NUM_AGGREGATORS} aggregator inner servers ...")
        for agg_id in range(NUM_AGGREGATORS):
            agg_port = AGG_BASE_PORT + agg_id
            if not _wait_for_port(SERVER_HOST, agg_port, timeout=60):
                raise RuntimeError(f"Aggregator {agg_id} port {agg_port} never opened")
        print("  All aggregators ready.")

        # 3. Clients — 20 per aggregator, cids are globally unique (0-99)
        print(f"\n  Starting {NUM_TOTAL_CLIENTS} leaf clients ...")
        for agg_id in range(NUM_AGGREGATORS):
            agg_port = AGG_BASE_PORT + agg_id
            agg_addr = f"{SERVER_HOST}:{agg_port}"
            for local_idx in range(NUM_LEAF_CLIENTS):
                cid = agg_id * NUM_LEAF_CLIENTS + local_idx
                p = _launch(
                    [_python(), "clients.py",
                     "--cid", str(cid),
                     "--num-clients", str(NUM_TOTAL_CLIENTS),
                     "--strategy", "all",
                     "--agg-address", agg_addr,
                     "--reconnect-delay", "3"],
                    log_dir / f"client_{cid:03d}.log",
                )
                procs.append(p)

        # 4. Wait for server to finish all 3 experiments
        print(f"\n  Waiting for server to complete all {rounds} rounds × 3 strategies ...")
        print("  (This will take a while — tail ./logs/server.log to monitor)\n")
        server_proc.wait()
        rc = server_proc.returncode
        if rc != 0:
            print(f"  WARNING: server exited with code {rc} — check logs/server.log")
        else:
            print("  Server finished successfully.")

    finally:
        print("\n  Stopping aggregators and clients ...")
        _kill_all(procs[1:])   # leave server proc already exited

    print("  Phase 1 complete.\n")


# ── Phase 2: Global simulation baseline ───────────────────────────────────────

def run_global_baseline(rounds: int, output_dir: str, log_dir: Path):
    print("\n" + "=" * 64)
    print("  PHASE 2 — Global simulation baseline (ceiling)")
    print("=" * 64)
    log_path = log_dir / "baseline_global.log"
    p = _launch(
        [_python(), "baseline_global.py",
         "--rounds", str(rounds),
         "--output-dir", output_dir],
        log_path,
    )
    p.wait()
    if p.returncode != 0:
        print(f"  WARNING: baseline_global exited with code {p.returncode}")
    else:
        print("  Global baseline complete.")


# ── Phase 3: Local baseline ────────────────────────────────────────────────────

def run_local_baseline(rounds: int, output_dir: str, log_dir: Path):
    print("\n" + "=" * 64)
    print("  PHASE 3 — Local baseline (floor)")
    print("=" * 64)
    log_path = log_dir / "baseline_local.log"
    p = _launch(
        [_python(), "baseline_local.py",
         "--rounds", str(rounds),
         "--output-dir", output_dir],
        log_path,
    )
    p.wait()
    if p.returncode != 0:
        print(f"  WARNING: baseline_local exited with code {p.returncode}")
    else:
        print("  Local baseline complete.")


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(output_dir: str):
    print("\n" + "=" * 64)
    print("  ALL DONE — Results summary")
    print("=" * 64)
    out = Path(output_dir)
    csvs = sorted(out.glob("*.csv"))
    if csvs:
        print(f"\n  CSVs written to {out}/")
        for f in csvs:
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<45} {size_kb:6.1f} KB")
    else:
        print(f"  No CSVs found in {out}/ — check logs/ for errors.")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full automated experiment runner")
    parser.add_argument("--rounds",          type=int,  default=ROUNDS)
    parser.add_argument("--output-dir",      type=str,  default=OUTPUT_DIR)
    parser.add_argument("--log-dir",         type=str,  default=LOG_DIR)
    parser.add_argument("--skip-fl",         action="store_true",
                        help="Skip the FL experiment (run baselines only)")
    parser.add_argument("--skip-baselines",  action="store_true",
                        help="Skip baselines (run FL experiment only)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFLASH Full Experiment Runner")
    print(f"  Rounds:      {args.rounds}")
    print(f"  Aggregators: {NUM_AGGREGATORS}  ×  {NUM_LEAF_CLIENTS} clients = {NUM_TOTAL_CLIENTS} total")
    print(f"  Output:      {args.output_dir}")
    print(f"  Logs:        {args.log_dir}")

    t_start = time.time()

    if not args.skip_fl:
        run_fl(args.rounds, args.output_dir, log_dir)

    if not args.skip_baselines:
        run_global_baseline(args.rounds, args.output_dir, log_dir)
        run_local_baseline(args.rounds, args.output_dir, log_dir)

    print_summary(args.output_dir)
    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"  Total wall time: {h}h {m}m {s}s")
