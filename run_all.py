# -*- coding: utf-8 -*-
"""
run_all.py -- Full automated experiment runner

TWO MODES:

  LOCAL (default) -- everything runs on one machine via localhost.
  Good for testing / development. Hardware energy metrics won't reflect
  real edge devices but training/accuracy results are valid.

      python run_all.py --mode local --rounds 60

  DISTRIBUTED -- runs on real hardware (Chameleon + Xaviers + Pi 5s).
  Run this script on the Chameleon server. Provide the real IPs.
  It starts the server locally, SSHs into each Xavier to start an
  aggregator, and prints the client commands for the Pi 5s / Nanos.

      python run_all.py --mode distributed \\
          --server-ip   <CHAMELEON_PUBLIC_IP> \\
          --xavier-ips  192.168.1.10 192.168.1.11 192.168.1.12 192.168.1.13 192.168.1.14 \\
          --ssh-user    ubuntu \\
          --rounds 60

  In both modes the script then runs the global and local baselines
  locally on the Chameleon server (they need no other machines).

Flags:
  --skip-fl          Skip the FL experiment (baselines only)
  --skip-baselines   Skip baselines (FL only)
  --no-ssh           Print aggregator SSH commands instead of running them
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# -- Fixed topology -------------------------------------------------------------
NUM_AGGREGATORS   = 5
NUM_LEAF_CLIENTS  = 20          # per aggregator
NUM_TOTAL_CLIENTS = NUM_AGGREGATORS * NUM_LEAF_CLIENTS   # 100
SERVER_PORT       = 8080
AGG_BASE_PORT     = 8081        # agg 0->8081, agg 1->8082, … agg 4->8085
ROUNDS            = 60
OUTPUT_DIR        = "./fl_results_hfl"
LOG_DIR           = "./logs"


# -- Subprocess helpers ---------------------------------------------------------

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


def _wait_port(host: str, port: int, timeout: float = 90.0) -> bool:
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


# -- SSH helper -----------------------------------------------------------------

def _ssh_launch(user: str, host: str, remote_cmd: str, log_path: Path,
                pem_key: Optional[str] = None) -> subprocess.Popen:
    """Launch a command on a remote machine via SSH, log output locally."""
    key_args = ["-i", pem_key] if pem_key else []
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no"] + key_args + [f"{user}@{host}", remote_cmd]
    print(f"  [ssh {host}] {remote_cmd}")
    print(f"               -> {log_path}")
    return subprocess.Popen(cmd, stdout=_open_log(log_path), stderr=subprocess.STDOUT)


def _agg_remote_cmd(strategy: str, rounds: int, agg_port: int, server_private_ip: str) -> str:
    return (
        f"cd ~/FLASH && python aggregator.py"
        f" --strategy {strategy}"
        f" --rounds {rounds}"
        f" --leaf-clients {NUM_LEAF_CLIENTS}"
        f" --agg-port {agg_port}"
        f" --server-address {server_private_ip}:{SERVER_PORT}"
    )


# -- Phase 1a: FL -- LOCAL mode --------------------------------------------------

def run_fl_local(rounds: int, output_dir: str, log_dir: Path):
    print("\n" + "=" * 64)
    print("  PHASE 1 -- FL experiment  [LOCAL / localhost]")
    print(f"  {NUM_AGGREGATORS} aggregators x {NUM_LEAF_CLIENTS} clients = {NUM_TOTAL_CLIENTS} total")
    print("=" * 64)

    procs: List[subprocess.Popen] = []
    try:
        # Server
        server_proc = _launch(
            [_py(), "server.py",
             "--rounds", str(rounds),
             "--port", str(SERVER_PORT),
             "--aggregators", str(NUM_AGGREGATORS),
             "--experiment", "all",
             "--no-wait",
             "--output-dir", output_dir],
            log_dir / "server.log",
        )
        procs.append(server_proc)

        print(f"\n  Waiting for server on 127.0.0.1:{SERVER_PORT} ...")
        if not _wait_port("127.0.0.1", SERVER_PORT):
            raise RuntimeError("Server never opened -- check logs/server.log")
        print("  Server ready.")

        # Aggregators
        for agg_id in range(NUM_AGGREGATORS):
            agg_port = AGG_BASE_PORT + agg_id
            p = _launch(
                [_py(), "aggregator.py",
                 "--strategy", "all",
                 "--rounds", str(rounds),
                 "--leaf-clients", str(NUM_LEAF_CLIENTS),
                 "--agg-port", str(agg_port),
                 "--server-address", f"127.0.0.1:{SERVER_PORT}"],
                log_dir / f"aggregator_{agg_id}.log",
            )
            procs.append(p)

        print(f"\n  Waiting for {NUM_AGGREGATORS} aggregator ports ...")
        for agg_id in range(NUM_AGGREGATORS):
            if not _wait_port("127.0.0.1", AGG_BASE_PORT + agg_id):
                raise RuntimeError(f"Aggregator {agg_id} never opened port {AGG_BASE_PORT + agg_id}")
        print("  All aggregators ready.")

        # Clients -- 20 per aggregator, globally unique cids 0-99
        print(f"\n  Starting {NUM_TOTAL_CLIENTS} clients ...")
        for agg_id in range(NUM_AGGREGATORS):
            agg_port = AGG_BASE_PORT + agg_id
            for local_idx in range(NUM_LEAF_CLIENTS):
                cid = agg_id * NUM_LEAF_CLIENTS + local_idx
                p = _launch(
                    [_py(), "clients.py",
                     "--cid", str(cid),
                     "--num-clients", str(NUM_TOTAL_CLIENTS),
                     "--strategy", "all",
                     "--agg-address", f"127.0.0.1:{agg_port}",
                     "--reconnect-delay", "3"],
                    log_dir / f"client_{cid:03d}.log",
                )
                procs.append(p)

        print(f"\n  Running {rounds} rounds x 3 strategies ...")
        print("  Monitor progress:  tail -f logs/server.log\n")
        server_proc.wait()
        rc = server_proc.returncode
        print(f"  Server finished (exit code {rc}).")
        if rc != 0:
            print("  WARNING: non-zero exit -- check logs/server.log")

    finally:
        print("\n  Stopping aggregators and clients ...")
        _kill_all(procs[1:])

    print("  Phase 1 complete.")


# -- Phase 1b: FL -- DISTRIBUTED mode -------------------------------------------

def run_fl_distributed(
    rounds: int, output_dir: str, log_dir: Path,
    server_private_ip: str, xavier_ips: List[str],
    ssh_user: str, pem_key: Optional[str], no_ssh: bool,
):
    if len(xavier_ips) != NUM_AGGREGATORS:
        raise ValueError(
            f"Need exactly {NUM_AGGREGATORS} --xavier-ips, got {len(xavier_ips)}"
        )

    print("\n" + "=" * 64)
    print("  PHASE 1 -- FL experiment  [DISTRIBUTED / real hardware]")
    print(f"  Server private IP:  {server_private_ip}:{SERVER_PORT}  (aggregators connect here)")
    for i, ip in enumerate(xavier_ips):
        print(f"  Xavier {i}:  {ip}:{AGG_BASE_PORT + i}  (clients {i*NUM_LEAF_CLIENTS}-{(i+1)*NUM_LEAF_CLIENTS-1})")
    print("=" * 64)

    procs: List[subprocess.Popen] = []
    try:
        # Start server locally -- this script runs on the Chameleon box
        server_proc = _launch(
            [_py(), "server.py",
             "--rounds", str(rounds),
             "--port", str(SERVER_PORT),
             "--aggregators", str(NUM_AGGREGATORS),
             "--experiment", "all",
             "--no-wait",
             "--output-dir", output_dir],
            log_dir / "server.log",
        )
        procs.append(server_proc)

        print(f"\n  Waiting for server to open port {SERVER_PORT} ...")
        if not _wait_port("127.0.0.1", SERVER_PORT):
            raise RuntimeError("Server never opened -- check logs/server.log")
        print("  Server ready.\n")

        # Aggregators -- SSH to each Xavier or print commands
        for agg_id, xavier_ip in enumerate(xavier_ips):
            agg_port = AGG_BASE_PORT + agg_id
            # Aggregators connect back to server using its private 172.28.x.x IP
            remote_cmd = _agg_remote_cmd("all", rounds, agg_port, server_private_ip)

            if no_ssh:
                print(f"  Run on Xavier {agg_id} ({xavier_ip}):")
                print(f"    {remote_cmd}\n")
            else:
                p = _ssh_launch(
                    ssh_user, xavier_ip, remote_cmd,
                    log_dir / f"aggregator_{agg_id}.log",
                    pem_key=pem_key,
                )
                procs.append(p)

        # Always print client commands -- clients run on Pi 5s which aren't SSH'd
        print("  -- Client commands -- run on each Pi 5 / Jetson Nano --")
        for agg_id, xavier_ip in enumerate(xavier_ips):
            agg_port = AGG_BASE_PORT + agg_id
            print(f"\n  Clients for Xavier {agg_id} ({xavier_ip}:{agg_port}):")
            for local_idx in range(NUM_LEAF_CLIENTS):
                cid = agg_id * NUM_LEAF_CLIENTS + local_idx
                print(
                    f"    python clients.py --cid {cid} --num-clients {NUM_TOTAL_CLIENTS}"
                    f" --strategy all --agg-address {xavier_ip}:{agg_port}"
                )

        if not no_ssh:
            # Wait for aggregator ports before telling user to connect clients
            print(f"\n  Waiting for {NUM_AGGREGATORS} aggregator ports to open ...")
            for agg_id, xavier_ip in enumerate(xavier_ips):
                agg_port = AGG_BASE_PORT + agg_id
                if not _wait_port(xavier_ip, agg_port, timeout=120):
                    raise RuntimeError(f"Aggregator {agg_id} on {xavier_ip}:{agg_port} never opened")
            print("  All aggregators ready.")

        print("\n  Start the clients above, then press ENTER here to continue ...")
        input()

        print(f"\n  Running {rounds} rounds x 3 strategies ...")
        print("  Monitor:  tail -f logs/server.log\n")
        server_proc.wait()
        print(f"  Server finished (exit code {server_proc.returncode}).")

    finally:
        print("\n  Stopping local server and any SSH-launched aggregators ...")
        _kill_all(procs)

    print("  Phase 1 complete.")


# -- Phase 2: Global baseline ---------------------------------------------------

def run_global_baseline(rounds: int, output_dir: str, log_dir: Path):
    print("\n" + "=" * 64)
    print("  PHASE 2 -- Global simulation baseline (ceiling)")
    print("=" * 64)
    p = _launch(
        [_py(), "baseline_global.py", "--rounds", str(rounds), "--output-dir", output_dir],
        log_dir / "baseline_global.log",
    )
    p.wait()
    if p.returncode != 0:
        print(f"  WARNING: exited with code {p.returncode} -- check logs/baseline_global.log")
    else:
        print("  Global baseline complete.")


# -- Phase 3: Local baseline ----------------------------------------------------

def run_local_baseline(rounds: int, output_dir: str, log_dir: Path):
    print("\n" + "=" * 64)
    print("  PHASE 3 -- Local baseline (floor)")
    print("=" * 64)
    p = _launch(
        [_py(), "baseline_local.py", "--rounds", str(rounds), "--output-dir", output_dir],
        log_dir / "baseline_local.log",
    )
    p.wait()
    if p.returncode != 0:
        print(f"  WARNING: exited with code {p.returncode} -- check logs/baseline_local.log")
    else:
        print("  Local baseline complete.")


# -- Summary --------------------------------------------------------------------

def print_summary(output_dir: str):
    print("\n" + "=" * 64)
    print("  ALL DONE -- Results")
    print("=" * 64)
    out = Path(output_dir)
    csvs = sorted(out.glob("*.csv"))
    if csvs:
        print(f"\n  {out}/")
        for f in csvs:
            print(f"    {f.name:<48} {f.stat().st_size/1024:6.1f} KB")
    else:
        print(f"  No CSVs found in {out}/ -- check logs/ for errors.")
    print()


# -- Entry point ----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FLASH full experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local simulation (one machine, localhost)
  python run_all.py --mode local --rounds 60

  # Real hardware -- SSH into Xaviers automatically
  # --server-private-ip: Chameleon's 172.28.x.x IP (what aggregators connect back to)
  # --xavier-ips:        private IPs of the 5 Xaviers  (what clients connect to)
  python run_all.py --mode distributed \\
      --server-private-ip 172.28.77.1 \\
      --xavier-ips 172.28.77.10 172.28.77.11 172.28.77.12 172.28.77.13 172.28.77.14 \\
      --ssh-user cc --pem-key ~/FLASH_v2.pem

  # Real hardware -- just print all commands, run them yourself
  python run_all.py --mode distributed --no-ssh \\
      --server-private-ip 172.28.77.1 \\
      --xavier-ips 172.28.77.10 172.28.77.11 172.28.77.12 172.28.77.13 172.28.77.14
        """,
    )
    parser.add_argument("--mode", choices=["local", "distributed"], default="local",
                        help="'local' = everything on localhost; 'distributed' = real hardware")

    # Distributed-mode IPs
    parser.add_argument("--server-private-ip", type=str, default=None,
                        help="[distributed] Chameleon's private 172.28.x.x IP -- "
                             "what aggregators use to connect back to the server")
    parser.add_argument("--xavier-ips",  nargs=NUM_AGGREGATORS, metavar="IP", default=None,
                        help=f"[distributed] Private IPs of the {NUM_AGGREGATORS} Jetson Xaviers "
                             "(clients connect to these)")
    parser.add_argument("--ssh-user",    type=str, default="cc",
                        help="[distributed] SSH username for Xaviers (default: cc)")
    parser.add_argument("--pem-key",     type=str, default=None,
                        help="[distributed] Path to PEM key file, e.g. ~/FLASH_v2.pem")
    parser.add_argument("--no-ssh",      action="store_true",
                        help="[distributed] Print all commands instead of SSH-ing into Xaviers")

    # Common
    parser.add_argument("--rounds",         type=int,  default=ROUNDS)
    parser.add_argument("--output-dir",     type=str,  default=OUTPUT_DIR)
    parser.add_argument("--log-dir",        type=str,  default=LOG_DIR)
    parser.add_argument("--skip-fl",        action="store_true",
                        help="Skip FL experiment (baselines only)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baselines (FL only)")
    args = parser.parse_args()

    # Validate distributed args
    if args.mode == "distributed":
        if not args.server_private_ip:
            parser.error("--mode distributed requires --server-private-ip "
                         "(Chameleon's 172.28.x.x address)")
        if not args.xavier_ips:
            parser.error("--mode distributed requires --xavier-ips (5 IPs)")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFLASH Experiment Runner")
    print(f"  Mode:        {args.mode}")
    print(f"  Rounds:      {args.rounds}")
    print(f"  Aggregators: {NUM_AGGREGATORS} x {NUM_LEAF_CLIENTS} clients = {NUM_TOTAL_CLIENTS} total")
    print(f"  Output:      {args.output_dir}")
    print(f"  Logs:        {args.log_dir}")
    if args.mode == "distributed":
        print(f"  Server private IP: {args.server_private_ip}")
        print(f"  Xavier IPs:        {', '.join(args.xavier_ips)}")
        if args.pem_key:
            print(f"  PEM key:           {args.pem_key}")

    t_start = time.time()

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

    if not args.skip_baselines:
        run_global_baseline(args.rounds, args.output_dir, log_dir)
        run_local_baseline(args.rounds, args.output_dir, log_dir)

    print_summary(args.output_dir)
    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"  Total wall time: {h}h {m}m {s}s")
