# -*- coding: utf-8 -*-
"""
run_noniid_sim.py -- Non-IID simulation sweep for paper revision.

Runs all four strategies (flash, fixedcompress, fedavg, adamc) under
Dirichlet-partitioned non-IID data at alpha in {0.5, 0.1}.

Topology mirrors the hardware testbed: 1 aggregator, 2 leaf clients, 60 rounds.
This makes the non-IID numbers directly comparable to the IID hardware results
in the paper.

Usage:
    python run_noniid_sim.py                        # alpha 0.5 and 0.1, 60 rounds
    python run_noniid_sim.py --alphas 0.5           # single alpha
    python run_noniid_sim.py --rounds 30 --quick    # quick smoke test

Output:
    fl_results_noniid/alpha_0.50/   -- CSVs for alpha=0.5
    fl_results_noniid/alpha_0.10/   -- CSVs for alpha=0.1
    logs_noniid/                    -- per-process logs

Summary table (Loss@R60, Acc@R60) is printed when all runs complete.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ── Topology (matches hardware testbed) ────────────────────────────────────────
N_AGGS          = 1
CLIENTS_PER_AGG = 2
N_TOTAL_CLIENTS = N_AGGS * CLIENTS_PER_AGG

SERVER_PORT     = 8080
AGG_BASE_PORT   = 8081
ROUNDS          = 60
STRATEGIES      = ["flash", "fixedcompress", "fedavg", "adamc"]

OUTPUT_BASE     = "./fl_results_noniid"
LOG_BASE        = "./logs_noniid"


# ── Subprocess helpers ─────────────────────────────────────────────────────────

def _py() -> str:
    return sys.executable


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", buffering=1)


def _launch(cmd: List[str], log_path: Path) -> subprocess.Popen:
    print(f"  [launch] {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=_open_log(log_path), stderr=subprocess.STDOUT)


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


def _wait_port(host: str, port: int, timeout: float = 120.0) -> bool:
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


# ── Single run ─────────────────────────────────────────────────────────────────

def _run_one(alpha: float, rounds: int, output_dir: str, log_dir: Path) -> int:
    """
    Spin up server + 1 aggregator + 2 clients for all strategies.
    Returns server exit code.
    """
    procs: List[subprocess.Popen] = []
    tag = f"alpha{alpha:.2f}"

    server_cmd = [
        _py(), "server.py",
        "--rounds", str(rounds),
        "--port", str(SERVER_PORT),
        "--aggregators", str(N_AGGS),
        "--experiment", "all",
        "--no-wait",
        "--output-dir", output_dir,
    ]

    try:
        server_proc = _launch(server_cmd, log_dir / f"{tag}_server.log")
        procs.append(server_proc)

        if not _wait_port("127.0.0.1", SERVER_PORT, timeout=120):
            raise RuntimeError(f"Server never opened — check {tag}_server.log")
        print("  Server ready.")

        agg_cmd = [
            _py(), "aggregator.py",
            "--strategy", "all",
            "--rounds", str(rounds),
            "--leaf-clients", str(CLIENTS_PER_AGG),
            "--agg-port", str(AGG_BASE_PORT),
            "--server-address", f"127.0.0.1:{SERVER_PORT}",
        ]
        procs.append(_launch(agg_cmd, log_dir / f"{tag}_agg0.log"))

        if not _wait_port("127.0.0.1", AGG_BASE_PORT, timeout=120):
            raise RuntimeError(f"Aggregator never opened — check {tag}_agg0.log")
        print("  Aggregator ready.")

        for cid in range(N_TOTAL_CLIENTS):
            procs.append(_launch(
                [
                    _py(), "clients.py",
                    "--cid", str(cid),
                    "--num-clients", str(N_TOTAL_CLIENTS),
                    "--strategy", "all",
                    "--agg-address", f"127.0.0.1:{AGG_BASE_PORT}",
                    "--reconnect-delay", "3",
                    "--dirichlet-alpha", str(alpha),
                ],
                log_dir / f"{tag}_client{cid:02d}.log",
            ))

        print(f"  {N_TOTAL_CLIENTS} clients started. Running {rounds} rounds ...")
        print(f"  Monitor: tail -f {log_dir}/{tag}_server.log")
        server_proc.wait()
        rc = server_proc.returncode
        print(f"  Server finished (exit {rc}).")
        return rc

    finally:
        _kill_all(procs[1:])


# ── Summary table ──────────────────────────────────────────────────────────────

def _print_summary(alphas: List[float]):
    """Read Loss@R60 and Acc@R60 from CSVs and print a comparison table."""
    print("\n" + "=" * 70)
    print("  NON-IID SIMULATION RESULTS SUMMARY")
    print("=" * 70)

    rows = []
    for alpha in alphas:
        out = Path(OUTPUT_BASE) / f"alpha_{alpha:.2f}"
        loss_csv = out / "flash_HFL_eval_loss.csv"
        metrics_csv = out / "flash_HFL_eval_metrics.csv"

        if not loss_csv.exists():
            print(f"  [alpha={alpha}] No results found at {out}/ — did the run finish?")
            continue

        try:
            loss_df = pd.read_csv(loss_csv)
            metrics_df = pd.read_csv(metrics_csv)
        except Exception as e:
            print(f"  [alpha={alpha}] Could not read CSVs: {e}")
            continue

        # Each CSV has one row per round per strategy; strategy is a column
        for strategy in STRATEGIES:
            try:
                s_loss = loss_df[loss_df["strategy"] == strategy]
                s_met  = metrics_df[metrics_df["strategy"] == strategy]
                if s_loss.empty:
                    continue
                final_loss = s_loss["loss"].iloc[-1]
                final_acc  = s_met["accuracy"].iloc[-1] if not s_met.empty else float("nan")
                rows.append({
                    "alpha": alpha,
                    "strategy": strategy,
                    "Loss@R60": round(final_loss, 4),
                    "Acc@R60": f"{final_acc * 100:.2f}%",
                })
            except Exception:
                continue

    if not rows:
        print("  No data to display.")
        return

    df = pd.DataFrame(rows)
    # Pivot for readable layout: rows = strategy, cols = alpha
    for col in ["Loss@R60", "Acc@R60"]:
        pivot = df.pivot(index="strategy", columns="alpha", values=col)
        print(f"\n  {col}")
        print(pivot.to_string())

    print()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-IID simulation sweep")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.1],
                        help="Dirichlet alpha values to sweep (default: 0.5 0.1)")
    parser.add_argument("--rounds", type=int, default=ROUNDS,
                        help=f"Rounds per run (default: {ROUNDS})")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "ucihar"],
                        help="Dataset (default: mnist)")
    args = parser.parse_args()

    log_dir = Path(LOG_BASE)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFLASH Non-IID Simulation")
    print(f"  Alphas   : {args.alphas}")
    print(f"  Rounds   : {args.rounds}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Topology : {N_AGGS} agg x {CLIENTS_PER_AGG} clients")
    print(f"  Strategies: {STRATEGIES}")
    print(f"  Output   : {OUTPUT_BASE}/alpha_<a>/")

    t_start = time.time()

    for alpha in args.alphas:
        out = str(Path(OUTPUT_BASE) / f"alpha_{alpha:.2f}")
        Path(out).mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 64}")
        print(f"  Run: alpha={alpha}  ({N_TOTAL_CLIENTS} clients, {args.rounds} rounds)")
        print(f"  Output: {out}")
        print("=" * 64)

        rc = _run_one(alpha, args.rounds, out, log_dir)
        if rc != 0:
            print(f"  WARNING: run for alpha={alpha} exited non-zero")

        # Brief pause so OS fully releases ports before next run
        if alpha != args.alphas[-1]:
            print("  Pausing 10s before next run ...")
            time.sleep(10)

    _print_summary(args.alphas)

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"  Total wall time: {h}h {m}m {s}s")
