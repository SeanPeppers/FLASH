# -*- coding: utf-8 -*-
"""
server.py — Global coordinator on Chameleon Cloud

No changes to the core FL logic — this file is updated to:
  - Remove the interactive input() prompt between experiments
  - Accept --no-wait flag for running a single experiment non-interactively
  - Add a startup banner showing all connection addresses
  - Save all metrics with consistent column naming

Usage (single experiment, no interactive prompts):
    python server.py --experiment flash --rounds 60 --port 8080 --no-wait

Start order:
    1. python server.py --experiment flash --rounds 60
    2. python aggregator.py --strategy flash --rounds 60 \\
             --server-address <CHAMELEON_IP>:8080
    3. python clients.py --cid 0 --agg-address <XAVIER_IP>:8081 --strategy flash
    4. python clients.py --cid 1 --agg-address <XAVIER_IP>:8081 --strategy flash
"""


import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import FitIns, Scalar
try:
    from flwr.common import ndarrays_to_parameters
except ImportError:
    from flwr.common import weights_to_parameters as ndarrays_to_parameters
from flwr.server.strategy import FedAvg

import hw_metrics
from hw_metrics import EnergyAccumulator, snapshot, delta
from clients import SimpleNet, get_parameters, TARGET_TAU, COMPRESSION_OPTIONS

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_ROUNDS = 60
LATENCY_THRESHOLD = 0.5
ROLLING_WINDOW = 5
NUM_AGGREGATORS = 1


# ── Metric aggregation ─────────────────────────────────────────────────────────
def _global_agg(all_metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    weighted = {
        "accuracy", "loss", "train_accuracy_final", "train_loss_final",
        "train_accuracy_mean", "train_loss_mean",
    }
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    total_n = 0
    n_agg = 0

    for num_ex, m in all_metrics:
        total_n += num_ex
        n_agg += 1
        for k, v in m.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            sums.setdefault(k, 0.0)
            counts.setdefault(k, 0)
            sums[k] += fv * num_ex if k in weighted else fv
            counts[k] += 1

    out: Dict[str, Scalar] = {}
    for k in sums:
        if k in weighted and total_n > 0:
            out[k] = sums[k] / total_n
        elif counts[k] > 0:
            out[k] = sums[k] / counts[k]
    out["num_aggregators"] = float(n_agg)
    return out


# ── Per-round server hardware logger ──────────────────────────────────────────
class RoundHWLogger:
    """Wraps a strategy to capture Chameleon-side hw metrics per round."""

    def __init__(self, strategy: FedAvg, output_dir: Path, name: str):
        self._strat = strategy
        self._output_dir = output_dir
        self._name = name
        self._rows: List[Dict] = []
        self._hw_before: Optional[Dict[str, float]] = None
        self._energy_acc: Optional[EnergyAccumulator] = None
        self._t0: float = 0.0

        # Monkey-patch aggregate_fit
        orig = strategy.aggregate_fit

        def patched(server_round, results, failures):
            self._t0 = time.perf_counter()
            self._hw_before = snapshot()
            self._energy_acc = EnergyAccumulator()
            self._energy_acc.start()

            result = orig(server_round, results, failures)

            duration = time.perf_counter() - self._t0
            energy_j = self._energy_acc.stop_and_get_joules()
            hw_now = snapshot()
            hw_d = delta(self._hw_before or {}, hw_now)

            row: Dict[str, Any] = {"round": server_round}
            for k, v in hw_now.items():
                row[f"server_hw_{k}"] = float(v)
            for k, v in hw_d.items():
                if k.startswith("delta_"):
                    row[f"server_hw_{k}"] = float(v)
            row["server_round_duration_s"] = float(duration)
            row["server_energy_joules"]    = float(energy_j)
            row["server_power_avg_w"]      = energy_j / duration if duration > 0 else 0.0

            # Also grab the fed metrics from the result
            if result and len(result) > 1 and result[1]:
                for k, v in (result[1] or {}).items():
                    try:
                        row[f"fed_{k}"] = float(v)
                    except (TypeError, ValueError):
                        pass

            self._rows.append(row)
            return result

        strategy.aggregate_fit = patched

    def save(self):
        if not self._rows:
            return
        df = pd.DataFrame(self._rows)
        path = self._output_dir / f"{self._name}_server_hw.csv"
        df.to_csv(path, index=False)
        print(f"  saved server hw metrics → {path}")


# ── FLASH global strategy ──────────────────────────────────────────────────────
class FLASHGlobalStrategy(FedAvg):
    def __init__(self, bar_tau_r: float, t_thr: float,
                 compression_options: Set[float], **kwargs):
        super().__init__(**kwargs)
        self.bar_tau_r = bar_tau_r
        self.t_thr = t_thr
        self.compression_options = sorted(compression_options, reverse=True)
        self._agg_history: Dict[str, Dict] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        aggs = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )
        out = []
        for agg in aggs:
            last = self._agg_history.get(agg.cid, {})
            lat = last.get("simulated_latency_seconds", 0.0)
            r_hint = 1.0
            if lat > self.t_thr:
                candidates = [r for r in self.compression_options if r < 1.0]
                r_hint = candidates[0] if candidates else 0.25
            cfg = {
                "server_round": server_round,
                "bar_tau_r": self.bar_tau_r,
                "optimal_r_star": r_hint,
                "t_thr": self.t_thr,
            }
            out.append((agg, FitIns(parameters, cfg)))
        return out

    def aggregate_fit(self, server_round, results, failures):
        for agg, fit_res in results:
            self._agg_history[agg.cid] = fit_res.metrics
        return super().aggregate_fit(server_round, results, failures)


# ── FLARE global strategy ──────────────────────────────────────────────────────
class FLAREGlobalStrategy(FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        aggs = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )
        cfg = {"server_round": server_round, "bar_tau_r": TARGET_TAU}
        return [(agg, FitIns(parameters, cfg)) for agg in aggs]


# ── CSV / plot helpers ─────────────────────────────────────────────────────────
def save_csvs(history, name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    if history.losses_distributed:
        p = output_dir / f"{name}_eval_loss.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "eval_loss"])
            for rnd, val in history.losses_distributed:
                w.writerow([int(rnd), float(val)])
        print(f"  saved {p}")

    if history.metrics_distributed_fit:
        rows: Dict[int, Dict] = {}
        for key, pairs in history.metrics_distributed_fit.items():
            for rnd, val in pairs:
                r = int(rnd)
                rows.setdefault(r, {"round": r})[key] = float(val)
        if rows:
            df = pd.DataFrame(list(rows.values())).sort_values("round")
            p = output_dir / f"{name}_fit_metrics.csv"
            df.to_csv(p, index=False)
            print(f"  saved {p}")

    if history.metrics_distributed:
        rows2: Dict[int, Dict] = {}
        for key, pairs in history.metrics_distributed.items():
            for rnd, val in pairs:
                r = int(rnd)
                rows2.setdefault(r, {"round": r})[key] = float(val)
        if rows2:
            df2 = pd.DataFrame(list(rows2.values())).sort_values("round")
            p = output_dir / f"{name}_eval_metrics.csv"
            df2.to_csv(p, index=False)
            print(f"  saved {p}")


PLOT_SPECS = [
    # (metric_key, source_attr, y_label, title)
    (None,                        "losses_distributed",     "Loss",        "Evaluation loss"),
    ("accuracy",                  "metrics_distributed_fit","Accuracy",    "Training accuracy"),
    ("train_loss_final",          "metrics_distributed_fit","Loss",        "Training loss (final epoch)"),
    ("train_loss_mean",           "metrics_distributed_fit","Loss",        "Training loss (mean)"),
    ("grad_norm_mean",            "metrics_distributed_fit","L2 norm",     "Gradient norm (mean)"),
    ("energy_joules",             "metrics_distributed_fit","Joules",      "Total energy per round"),
    ("agg_energy_joules",         "metrics_distributed_fit","Joules",      "Xavier energy per round"),
    ("leaf_energy_joules",        "metrics_distributed_fit","Joules",      "Leaf energy per round"),
    ("power_avg_w",               "metrics_distributed_fit","Watts",       "Leaf avg power"),
    ("agg_power_avg_w",           "metrics_distributed_fit","Watts",       "Xavier avg power"),
    ("simulated_latency_seconds", "metrics_distributed_fit","Seconds",     "Round wall-clock time"),
    ("data_transfer_size_bytes",  "metrics_distributed_fit","Bytes",       "Data transferred"),
    ("num_epochs_run",            "metrics_distributed_fit","Epochs",      "Local epochs per round"),
    ("training_time_s",           "metrics_distributed_fit","Seconds",     "Leaf training time"),
    ("throughput_samples_per_s",  "metrics_distributed_fit","Samples/s",   "Leaf training throughput"),
    ("avg_batch_time_ms",         "metrics_distributed_fit","ms",          "Avg batch time (leaf)"),
    ("hw_cpu_util_pct",           "metrics_distributed_fit","%",           "Leaf CPU utilisation"),
    ("hw_ram_util_pct",           "metrics_distributed_fit","%",           "Leaf RAM utilisation"),
    ("agg_hw_cpu_util_pct",       "metrics_distributed_fit","%",           "Xavier CPU utilisation"),
    ("agg_hw_power_total_soc_mw", "metrics_distributed_fit","mW",          "Xavier SoC total power"),
    ("agg_hw_temp_max_celsius",   "metrics_distributed_fit","°C",          "Xavier max temperature"),
    ("compression_ratio_applied", "metrics_distributed_fit","Ratio",       "Compression ratio (FLASH)"),
    ("model_size_bytes",          "metrics_distributed_fit","Bytes",       "Model size"),
]

COLORS = {
    "FLASH (HFL)": "#e63946",
    "FLARE (HFL)": "#2a9d8f",
    "FedAvg (HFL)": "#457b9d",
}


def plot_all(histories: Dict[str, Any], output_dir: Path, t_thr: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric_key, source_attr, ylabel, title in PLOT_SPECS:
        plt.figure(figsize=(10, 5))
        has_data = False
        for label, hist in histories.items():
            data = (hist.losses_distributed if source_attr == "losses_distributed"
                    else getattr(hist, source_attr, {}).get(metric_key, []))
            if not data:
                continue
            rounds, values = zip(*data)
            smooth = pd.Series(values).rolling(ROLLING_WINDOW, min_periods=1).mean().tolist()
            plt.plot(rounds, smooth, label=label,
                     color=COLORS.get(label), linewidth=1.8)
            has_data = True
        if not has_data:
            plt.close()
            continue
        if metric_key == "simulated_latency_seconds":
            plt.axhline(y=t_thr, color="black", linestyle="--", linewidth=1,
                        label=f"Threshold ({t_thr}s)")
        plt.title(title, fontsize=13)
        plt.xlabel("Round", fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        safe = (metric_key or "eval_loss").replace(" ", "_").replace("/", "_")
        path = output_dir / f"{safe}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  saved plot → {path}")


def plot_hw_dashboard(hw_csv_map: Dict[str, Path], output_dir: Path):
    hw_keys = [
        ("server_hw_cpu_util_pct",   "CPU util (%)"),
        ("server_hw_ram_util_pct",   "RAM util (%)"),
        ("server_energy_joules",     "Server energy (J)"),
        ("server_hw_temp_max_celsius","Max CPU temp (°C)"),
        ("server_hw_load_avg_1m",    "Load avg 1m"),
    ]
    fig, axes = plt.subplots(len(hw_keys), 1,
                             figsize=(12, 3 * len(hw_keys)), sharex=True)
    fig.suptitle("Chameleon Cloud — server hardware per round", fontsize=14)
    for ax, (key, label) in zip(axes, hw_keys):
        for exp, csv_path in hw_csv_map.items():
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if key in df.columns:
                smooth = df[key].rolling(ROLLING_WINDOW, min_periods=1).mean()
                ax.plot(df["round"], smooth, label=exp, color=COLORS.get(exp))
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Round")
    plt.tight_layout()
    path = output_dir / "chameleon_hw_dashboard.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved hw dashboard → {path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFL global server (Chameleon Cloud)")
    parser.add_argument("--rounds",      type=int, default=NUM_ROUNDS)
    parser.add_argument("--port",        type=int, default=8080)
    parser.add_argument("--output-dir",  type=str, default="./fl_results_hfl")
    parser.add_argument("--experiment",  type=str, default="all",
                        choices=["flash", "flare", "fedavg", "all"])
    parser.add_argument("--no-wait",     action="store_true",
                        help="Don't prompt between experiments (use when scripting)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Server] Chameleon Cloud — device: {hw_metrics.DEVICE}")
    print(f"[Server] Listening on 0.0.0.0:{args.port} for {args.rounds} rounds")
    print(f"[Server] hw snapshot:")
    for k, v in sorted(snapshot().items()):
        print(f"    {k}: {v:.4f}")
    print()

    init_params = ndarrays_to_parameters(get_parameters(SimpleNet()))
    common = dict(
        min_fit_clients=NUM_AGGREGATORS,
        min_evaluate_clients=NUM_AGGREGATORS,
        min_available_clients=NUM_AGGREGATORS,
        fit_metrics_aggregation_fn=_global_agg,
        evaluate_metrics_aggregation_fn=_global_agg,
        initial_parameters=init_params,
    )

    factories = {
        "flash":  lambda: FLASHGlobalStrategy(
                      bar_tau_r=TARGET_TAU,
                      t_thr=LATENCY_THRESHOLD,
                      compression_options=COMPRESSION_OPTIONS,
                      **common),
        "flare":  lambda: FLAREGlobalStrategy(**common),
        "fedavg": lambda: FedAvg(**common),
    }

    to_run = list(factories.keys()) if args.experiment == "all" else [args.experiment]
    histories: Dict[str, Any] = {}
    hw_csv_map: Dict[str, Path] = {}

    for i, name in enumerate(to_run):
        label = f"{name.upper()} (HFL)"
        print(f"\n{'='*62}")
        print(f"  Experiment: {label}")
        print(f"{'='*62}")
        print(
            f"  Start aggregator + clients:\n"
            f"    python aggregator.py --strategy {name} --rounds {args.rounds}"
            f" --server-address <THIS_IP>:{args.port}\n"
            f"    python clients.py --cid 0 --agg-address <XAVIER_IP>:8081"
            f" --strategy {name}\n"
            f"    python clients.py --cid 1 --agg-address <XAVIER_IP>:8081"
            f" --strategy {name}\n"
        )
        if not args.no_wait:
            input("  Press ENTER when aggregator + clients are running ...")
        elif i > 0:
            print("  Waiting 20s for aggregator + clients to restart ...")
            time.sleep(20)

        strategy = factories[name]()
        hw_logger = RoundHWLogger(strategy, output_dir, name)

        hist = fl.server.start_server(
            server_address=f"0.0.0.0:{args.port}",
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        )
        hw_logger.save()

        histories[label] = hist
        hw_csv_map[label] = output_dir / f"{name}_server_hw.csv"

        print(f"\n  Saving CSVs for {label} ...")
        save_csvs(hist, f"{name}_HFL", output_dir)

    if histories:
        print("\nGenerating plots ...")
        plot_all(histories, output_dir, LATENCY_THRESHOLD)
        plot_hw_dashboard(hw_csv_map, output_dir)

    print(f"\nAll done. Results in '{output_dir}/'")
