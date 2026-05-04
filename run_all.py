# -*- coding: utf-8 -*-
"""
run_all.py -- Full automated experiment runner

TWO MODES:

  LOCAL (default) -- everything runs on one machine via localhost.
  Good for development and the simulation phases.

      python run_all.py --mode local --rounds 60

  DISTRIBUTED -- Phase 1 runs on real hardware (Chameleon + Xavier + Pi5/Nano).
  Phases 2-5 (large-scale sim, ablation, baselines) always run locally on
  Chameleon after the hardware phase completes.

      python run_all.py --mode distributed \\
          --server-private-ip 172.28.77.1 \\
          --xavier-ips 192.168.1.10 \\
          --ssh-user cc --pem-key ~/FLASH_v2.pem \\
          --rounds 60

Phases (all run by default):
  1  FL comparison      FLASH / FLARE / FedAvg
                        distributed: 1 Xavier + 2 leaf clients (real hw)
                        local:        5 aggs  + 100 clients (simulated)
  2  Large-scale sim    FLASH / FLARE / FedAvg, 5 aggs x 40 = 200 clients
  3  Ablation           FLASH fixed r in {1.0, 0.75, 0.5, 0.25}, 5 aggs x 40
  4  Global baseline    Centralised upper-bound (clients.py --strategy global)
  5  Local baseline     No-communication lower-bound

Skip flags:
  --skip-fl              skip phase 1
  --skip-large-scale     skip phase 2
  --skip-ablation        skip phase 3
  --skip-baselines       skip phases 4-5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Topology constants
# ---------------------------------------------------------------------------
NUM_AGGREGATORS          = 1    # real-hardware phase (1 Xavier)
NUM_LEAF_CLIENTS         = 2    # real-hardware phase (Pi5 + Nano)

LARGE_SCALE_AGGS         = 5
LARGE_SCALE_CLIENTS_PER_AGG = 40   # 5 × 40 = 200 total

ABLATION_RATIOS          = [1.0, 0.75, 0.5, 0.25]
ABLATION_RATIO_DIRS      = {1.0: "r100", 0.75: "r075", 0.5: "r050", 0.25: "r025"}
ABLATION_LABELS          = ["r=1.0 (none)", "r=0.75", "r=0.5", "r=0.25"]

SERVER_PORT              = 8080
AGG_BASE_PORT            = 8081   # agg 0→8081, agg 1→8082, …
ROUNDS                   = 60
OUTPUT_DIR               = "./fl_results_hfl"
LOG_DIR                  = "./logs"


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _py() -> str:
    return sys.executable


def _open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", buffering=1)


def _launch(cmd: List[str], log_path: Path) -> subprocess.Popen:
    print(f"  [launch] {' '.join(cmd)}")
    print(f"           -> {log_path}")
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


def _ssh_launch(user: str, host: str, remote_cmd: str, log_path: Path,
                pem_key: Optional[str] = None) -> subprocess.Popen:
    key_args = ["-i", pem_key] if pem_key else []
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no"] + key_args + [f"{user}@{host}", remote_cmd]
    print(f"  [ssh {host}] {remote_cmd}")
    print(f"               -> {log_path}")
    return subprocess.Popen(cmd, stdout=_open_log(log_path), stderr=subprocess.STDOUT)


def _agg_remote_cmd(strategy: str, rounds: int, agg_port: int,
                    server_ip: str, leaf_clients: int,
                    fixed_r: Optional[float] = None) -> str:
    cmd = (
        f"cd ~/FLASH && python aggregator.py"
        f" --strategy {strategy}"
        f" --rounds {rounds}"
        f" --leaf-clients {leaf_clients}"
        f" --agg-port {agg_port}"
        f" --server-address {server_ip}:{SERVER_PORT}"
    )
    if fixed_r is not None:
        cmd += f" --fixed-r {fixed_r}"
    return cmd


# ---------------------------------------------------------------------------
# Generic local topology runner
# ---------------------------------------------------------------------------

def _run_local_topology(
    rounds: int,
    output_dir: str,
    log_dir: Path,
    n_aggs: int,
    clients_per_agg: int,
    strategy: str = "all",
    fixed_r: Optional[float] = None,
    log_prefix: str = "run",
) -> int:
    """
    Start a server + n_aggs aggregators + n_aggs*clients_per_agg clients on
    localhost.  Blocks until the server exits.  Returns the server exit code.
    """
    n_total = n_aggs * clients_per_agg
    procs: List[subprocess.Popen] = []

    server_cmd = [
        _py(), "server.py",
        "--rounds", str(rounds),
        "--port", str(SERVER_PORT),
        "--aggregators", str(n_aggs),
        "--experiment", "flash" if (strategy == "flash" or fixed_r is not None) else "all",
        "--no-wait",
        "--output-dir", output_dir,
    ]
    if fixed_r is not None:
        server_cmd += ["--fixed-r", str(fixed_r)]

    try:
        server_proc = _launch(server_cmd, log_dir / f"{log_prefix}_server.log")
        procs.append(server_proc)

        if not _wait_port("127.0.0.1", SERVER_PORT):
            raise RuntimeError(f"Server never opened — check {log_prefix}_server.log")
        print("  Server ready.")

        for agg_id in range(n_aggs):
            agg_port = AGG_BASE_PORT + agg_id
            agg_cmd = [
                _py(), "aggregator.py",
                "--strategy", strategy,
                "--rounds", str(rounds),
                "--leaf-clients", str(clients_per_agg),
                "--agg-port", str(agg_port),
                "--server-address", f"127.0.0.1:{SERVER_PORT}",
            ]
            if fixed_r is not None:
                agg_cmd += ["--fixed-r", str(fixed_r)]
            procs.append(_launch(agg_cmd, log_dir / f"{log_prefix}_agg{agg_id}.log"))

        for agg_id in range(n_aggs):
            if not _wait_port("127.0.0.1", AGG_BASE_PORT + agg_id):
                raise RuntimeError(f"Aggregator {agg_id} never opened")
        print(f"  All {n_aggs} aggregators ready.")

        print(f"  Starting {n_total} clients ...")
        for agg_id in range(n_aggs):
            for local_idx in range(clients_per_agg):
                cid = agg_id * clients_per_agg + local_idx
                procs.append(_launch(
                    [_py(), "clients.py",
                     "--cid", str(cid),
                     "--num-clients", str(n_total),
                     "--strategy", strategy,
                     "--agg-address", f"127.0.0.1:{AGG_BASE_PORT + agg_id}",
                     "--reconnect-delay", "3"],
                    log_dir / f"{log_prefix}_client{cid:03d}.log",
                ))

        print(f"  Running ...  monitor: tail -f {log_dir}/{log_prefix}_server.log")
        server_proc.wait()
        rc = server_proc.returncode
        print(f"  Server finished (exit {rc}).")
        if rc != 0:
            print(f"  WARNING: non-zero exit — check {log_prefix}_server.log")
        return rc

    finally:
        _kill_all(procs[1:])


# ---------------------------------------------------------------------------
# Phase 1 — FL comparison (local)
# ---------------------------------------------------------------------------

def run_fl_local(rounds: int, output_dir: str, log_dir: Path):
    print("\n" + "=" * 64)
    print("  PHASE 1 -- FL comparison  [LOCAL]")
    print(f"  {LARGE_SCALE_AGGS} aggs x {LARGE_SCALE_CLIENTS_PER_AGG} clients = "
          f"{LARGE_SCALE_AGGS * LARGE_SCALE_CLIENTS_PER_AGG} total")
    print("=" * 64)
    _run_local_topology(
        rounds, output_dir, log_dir,
        n_aggs=LARGE_SCALE_AGGS,
        clients_per_agg=LARGE_SCALE_CLIENTS_PER_AGG,
        strategy="all",
        log_prefix="phase1",
    )
    print("  Phase 1 complete.")


# ---------------------------------------------------------------------------
# Phase 1 — FL comparison (distributed)
# ---------------------------------------------------------------------------

def run_fl_distributed(
    rounds: int, output_dir: str, log_dir: Path,
    server_private_ip: str, xavier_ips: List[str],
    ssh_user: str, pem_key: Optional[str], no_ssh: bool,
):
    print("\n" + "=" * 64)
    print("  PHASE 1 -- FL comparison  [DISTRIBUTED / real hardware]")
    print(f"  Server private IP : {server_private_ip}:{SERVER_PORT}")
    for i, ip in enumerate(xavier_ips):
        start_cid = i * NUM_LEAF_CLIENTS
        print(f"  Xavier {i}: {ip}:{AGG_BASE_PORT + i}  "
              f"(clients {start_cid}–{start_cid + NUM_LEAF_CLIENTS - 1})")
    print("=" * 64)

    procs: List[subprocess.Popen] = []
    try:
        server_proc = _launch(
            [_py(), "server.py",
             "--rounds", str(rounds),
             "--port", str(SERVER_PORT),
             "--aggregators", str(len(xavier_ips)),
             "--experiment", "all",
             "--no-wait",
             "--output-dir", output_dir],
            log_dir / "phase1_server.log",
        )
        procs.append(server_proc)

        if not _wait_port("127.0.0.1", SERVER_PORT):
            raise RuntimeError("Server never opened — check logs/phase1_server.log")
        print("  Server ready.\n")

        for agg_id, xavier_ip in enumerate(xavier_ips):
            remote_cmd = _agg_remote_cmd(
                "all", rounds, AGG_BASE_PORT + agg_id,
                server_private_ip, NUM_LEAF_CLIENTS,
            )
            if no_ssh:
                print(f"  Run on Xavier {agg_id} ({xavier_ip}):\n    {remote_cmd}\n")
            else:
                procs.append(_ssh_launch(
                    ssh_user, xavier_ip, remote_cmd,
                    log_dir / f"phase1_agg{agg_id}.log",
                    pem_key=pem_key,
                ))

        print("  -- Client commands (run on each Pi5 / Nano) --")
        for agg_id, xavier_ip in enumerate(xavier_ips):
            for local_idx in range(NUM_LEAF_CLIENTS):
                cid = agg_id * NUM_LEAF_CLIENTS + local_idx
                print(
                    f"  python clients.py --cid {cid}"
                    f" --num-clients {len(xavier_ips) * NUM_LEAF_CLIENTS}"
                    f" --strategy all"
                    f" --agg-address {xavier_ip}:{AGG_BASE_PORT + agg_id}"
                )

        if not no_ssh:
            print(f"\n  Waiting for {len(xavier_ips)} aggregator port(s) ...")
            for agg_id, xavier_ip in enumerate(xavier_ips):
                if not _wait_port(xavier_ip, AGG_BASE_PORT + agg_id, timeout=180):
                    raise RuntimeError(f"Aggregator {agg_id} on {xavier_ip} never opened")
            print("  All aggregators ready.")

        print("\n  Start the clients above, then press ENTER to continue ...")
        input()

        print("  Running ...  monitor: tail -f logs/phase1_server.log\n")
        server_proc.wait()
        print(f"  Server finished (exit {server_proc.returncode}).")

    finally:
        _kill_all(procs)

    print("  Phase 1 complete.")


# ---------------------------------------------------------------------------
# Phase 2 — Large-scale simulation (200 clients, always local)
# ---------------------------------------------------------------------------

def run_large_scale(rounds: int, output_dir: str, log_dir: Path):
    n_total = LARGE_SCALE_AGGS * LARGE_SCALE_CLIENTS_PER_AGG
    out = str(Path(output_dir) / "large_scale")
    print("\n" + "=" * 64)
    print(f"  PHASE 2 -- Large-scale simulation  [LOCAL]")
    print(f"  {LARGE_SCALE_AGGS} aggs x {LARGE_SCALE_CLIENTS_PER_AGG} clients = {n_total} total")
    print(f"  Output: {out}")
    print("=" * 64)
    _run_local_topology(
        rounds, out, log_dir,
        n_aggs=LARGE_SCALE_AGGS,
        clients_per_agg=LARGE_SCALE_CLIENTS_PER_AGG,
        strategy="all",
        log_prefix="phase2",
    )
    print("  Phase 2 complete.")


# ---------------------------------------------------------------------------
# Phase 3 — Ablation (fixed-r FLASH, always local)
# ---------------------------------------------------------------------------

def run_ablation(rounds: int, output_dir: str, log_dir: Path):
    n_total = LARGE_SCALE_AGGS * LARGE_SCALE_CLIENTS_PER_AGG
    base = Path(output_dir) / "ablation"
    print("\n" + "=" * 64)
    print(f"  PHASE 3 -- Top-k ablation  [LOCAL]")
    print(f"  Ratios: {ABLATION_RATIOS}  |  {LARGE_SCALE_AGGS} aggs x {LARGE_SCALE_CLIENTS_PER_AGG} = {n_total} clients")
    print("=" * 64)

    completed_dirs: List[Path] = []

    for r in ABLATION_RATIOS:
        suffix = ABLATION_RATIO_DIRS[r]
        out = str(base / suffix)
        print(f"\n  --- Ablation r={r} ({suffix}) ---")
        rc = _run_local_topology(
            rounds, out, log_dir,
            n_aggs=LARGE_SCALE_AGGS,
            clients_per_agg=LARGE_SCALE_CLIENTS_PER_AGG,
            strategy="flash",
            fixed_r=r,
            log_prefix=f"phase3_{suffix}",
        )
        if rc == 0:
            completed_dirs.append(base / suffix)
        else:
            print(f"  WARNING: r={r} run exited non-zero — included in analysis anyway")
            completed_dirs.append(base / suffix)

        # Brief pause between runs so ports fully release
        time.sleep(5)

    # Auto-run ablation analysis over all completed runs
    if completed_dirs:
        print("\n  Running ablation analysis ...")
        ablation_out = str(base / "figures")
        dirs_args  = [str(d) for d in completed_dirs]
        labels_args = ABLATION_LABELS[:len(completed_dirs)]
        cmd = [
            _py(), "ablation_topk.py",
            "--results-dirs", *dirs_args,
            "--labels", *labels_args,
            "--output-dir", ablation_out,
        ]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print("  WARNING: ablation_topk.py exited non-zero — check output above")

    print("  Phase 3 complete.")


# ---------------------------------------------------------------------------
# Phases 4 & 5 — Baselines
# ---------------------------------------------------------------------------

def run_global_baseline(rounds: int, output_dir: str, log_dir: Path):
    print("\n" + "=" * 64)
    print("  PHASE 4 -- Global simulation baseline (ceiling)")
    print("=" * 64)
    p = _launch(
        [_py(), "baseline_global.py", "--rounds", str(rounds), "--output-dir", output_dir],
        log_dir / "phase4_baseline_global.log",
    )
    p.wait()
    if p.returncode != 0:
        print("  WARNING: exited non-zero — check logs/phase4_baseline_global.log")
    else:
        print("  Phase 4 complete.")


def run_local_baseline(rounds: int, output_dir: str, log_dir: Path):
    print("\n" + "=" * 64)
    print("  PHASE 5 -- Local baseline (floor)")
    print("=" * 64)
    p = _launch(
        [_py(), "baseline_local.py", "--rounds", str(rounds), "--output-dir", output_dir],
        log_dir / "phase5_baseline_local.log",
    )
    p.wait()
    if p.returncode != 0:
        print("  WARNING: exited non-zero — check logs/phase5_baseline_local.log")
    else:
        print("  Phase 5 complete.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(output_dir: str):
    print("\n" + "=" * 64)
    print("  ALL DONE -- Results")
    print("=" * 64)
    out = Path(output_dir)
    csvs = sorted(out.rglob("*.csv"))
    if csvs:
        print(f"\n  {out}/")
        for f in csvs:
            rel = f.relative_to(out)
            print(f"    {str(rel):<60} {f.stat().st_size / 1024:6.1f} KB")
    else:
        print(f"  No CSVs found in {out}/ — check logs/ for errors.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FLASH full experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local simulation — all 5 phases on one machine
  python run_all.py --mode local --rounds 60

  # Real hardware — SSH into Xavier automatically
  python run_all.py --mode distributed \\
      --server-private-ip 172.28.77.1 \\
      --xavier-ips 192.168.1.10 \\
      --ssh-user cc --pem-key ~/FLASH_v2.pem

  # Real hardware — print all commands, run yourself
  python run_all.py --mode distributed --no-ssh \\
      --server-private-ip 172.28.77.1 \\
      --xavier-ips 192.168.1.10

  # Skip the hardware phase, run only sim + ablation + baselines
  python run_all.py --mode distributed --skip-fl \\
      --server-private-ip 172.28.77.1 --xavier-ips 192.168.1.10

  # Ablation only
  python run_all.py --skip-fl --skip-large-scale --skip-baselines --rounds 60
        """,
    )

    parser.add_argument("--mode", choices=["local", "distributed"], default="local")

    # Distributed IPs
    parser.add_argument("--server-private-ip", type=str, default=None)
    parser.add_argument("--xavier-ips", nargs="+", metavar="IP", default=None)
    parser.add_argument("--ssh-user", type=str, default="cc")
    parser.add_argument("--pem-key", type=str, default=None)
    parser.add_argument("--no-ssh", action="store_true",
                        help="Print aggregator SSH commands instead of running them")

    # Common
    parser.add_argument("--rounds",     type=int, default=ROUNDS)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--log-dir",    type=str, default=LOG_DIR)

    # Skip flags
    parser.add_argument("--skip-fl",          action="store_true", help="Skip phase 1 (FL comparison)")
    parser.add_argument("--skip-large-scale", action="store_true", help="Skip phase 2 (200-client sim)")
    parser.add_argument("--skip-ablation",    action="store_true", help="Skip phase 3 (top-k ablation)")
    parser.add_argument("--skip-baselines",   action="store_true", help="Skip phases 4-5 (baselines)")

    args = parser.parse_args()

    if args.mode == "distributed":
        if not args.server_private_ip:
            parser.error("--mode distributed requires --server-private-ip")
        if not args.xavier_ips:
            parser.error("--mode distributed requires --xavier-ips")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFLASH Experiment Runner")
    print(f"  Mode:       {args.mode}")
    print(f"  Rounds:     {args.rounds}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Logs:       {args.log_dir}")
    print(f"  Phases:     "
          f"{'[1-FL] ' if not args.skip_fl else ''}"
          f"{'[2-LargeScale] ' if not args.skip_large_scale else ''}"
          f"{'[3-Ablation] ' if not args.skip_ablation else ''}"
          f"{'[4-5-Baselines]' if not args.skip_baselines else ''}")
    if args.mode == "distributed":
        print(f"  Server IP:  {args.server_private_ip}")
        print(f"  Xavier IPs: {', '.join(args.xavier_ips)}")

    t_start = time.time()

    # Phase 1 — FL comparison
    if not args.skip_fl:
        if args.mode == "local":
            run_fl_local(args.rounds, args.output_dir, log_dir)
        else:
            run_fl_distributed(
                args.rounds, args.output_dir, log_dir,
                server_private_ip=args.server_private_ip,
                xavier_ips=args.xavier_ips,
                ssh_user=args.ssh_user,
                pem_key=args.pem_key,
                no_ssh=args.no_ssh,
            )

    # Phase 2 — Large-scale simulation (always local)
    if not args.skip_large_scale:
        run_large_scale(args.rounds, args.output_dir, log_dir)

    # Phase 3 — Ablation (always local)
    if not args.skip_ablation:
        run_ablation(args.rounds, args.output_dir, log_dir)

    # Phases 4-5 — Baselines
    if not args.skip_baselines:
        run_global_baseline(args.rounds, args.output_dir, log_dir)
        run_local_baseline(args.rounds, args.output_dir, log_dir)

    print_summary(args.output_dir)

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"  Total wall time: {h}h {m}m {s}s")
