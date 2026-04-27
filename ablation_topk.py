# -*- coding: utf-8 -*-
"""
ablation_topk.py — Top-k Sparsification Ablation Analysis

Reads per-ratio result CSVs and produces comparison plots + a summary table.

Usage — run each compression ratio experiment first:

    # On server (repeat once per ratio, changing --fixed-r and --output-dir)
    python server.py --experiment flash --rounds 60 --no-wait \
        --fixed-r 1.0  --output-dir ./fl_results_hfl/r100
    python server.py --experiment flash --rounds 60 --no-wait \
        --fixed-r 0.75 --output-dir ./fl_results_hfl/r075
    python server.py --experiment flash --rounds 60 --no-wait \
        --fixed-r 0.5  --output-dir ./fl_results_hfl/r050
    python server.py --experiment flash --rounds 60 --no-wait \
        --fixed-r 0.25 --output-dir ./fl_results_hfl/r025

    # On aggregator (use matching --fixed-r)
    python aggregator.py --strategy flash --rounds 60 --fixed-r 0.5 ...

Then analyse:
    python ablation_topk.py --results-dirs \
        fl_results_hfl/r100 fl_results_hfl/r075 fl_results_hfl/r050 fl_results_hfl/r025 \
        --labels "r=1.0" "r=0.75" "r=0.5" "r=0.25" \
        --output-dir ./fl_results_hfl/ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RATIOS_DEFAULT = [1.0, 0.75, 0.5, 0.25]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MARKERS = ["o", "s", "^", "D"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_experiment(results_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """Load the three standard CSVs from an experiment directory."""
    def _try(name: str) -> Optional[pd.DataFrame]:
        p = results_dir / f"flash_HFL_{name}.csv"
        if not p.exists():
            # also check local/global baseline naming
            for stem in results_dir.glob(f"*_{name}.csv"):
                return pd.read_csv(stem)
            return None
        return pd.read_csv(p)

    return {
        "eval_loss":    _try("eval_loss"),
        "eval_metrics": _try("eval_metrics"),
        "fit_metrics":  _try("fit_metrics"),
    }


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def plot_accuracy_vs_rounds(experiments: List[Dict], labels: List[str], output_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for exp, label, color, marker in zip(experiments, labels, COLORS, MARKERS):
        df = exp.get("eval_metrics")
        if df is None or "accuracy" not in df.columns:
            continue
        rounds = df["round"].values
        acc = _smooth(df["accuracy"].values * 100)
        ax.plot(rounds, acc, label=label, color=color, marker=marker, markevery=10,
                linewidth=2, markersize=6)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Round — Top-k Ablation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_accuracy_vs_rounds.png", dpi=150)
    plt.close(fig)
    print(f"  saved ablation_accuracy_vs_rounds.png")


def plot_loss_vs_rounds(experiments: List[Dict], labels: List[str], output_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for exp, label, color, marker in zip(experiments, labels, COLORS, MARKERS):
        df = exp.get("eval_loss")
        if df is None or "eval_loss" not in df.columns:
            continue
        rounds = df["round"].values
        loss = _smooth(df["eval_loss"].values)
        ax.plot(rounds, loss, label=label, color=color, marker=marker, markevery=10,
                linewidth=2, markersize=6)
    ax.set_xlabel("Round")
    ax.set_ylabel("Eval Loss")
    ax.set_title("Eval Loss vs Round — Top-k Ablation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_loss_vs_rounds.png", dpi=150)
    plt.close(fig)
    print(f"  saved ablation_loss_vs_rounds.png")


def plot_payload_vs_accuracy(experiments: List[Dict], labels: List[str], output_dir: Path):
    """Scatter: mean communication payload (bytes) vs final accuracy — the core tradeoff."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for exp, label, color, marker in zip(experiments, labels, COLORS, MARKERS):
        fit = exp.get("fit_metrics")
        eval_m = exp.get("eval_metrics")
        if fit is None or eval_m is None:
            continue
        payload_col = next((c for c in fit.columns if "data_transfer_size_bytes" in c), None)
        if payload_col is None:
            continue
        mean_payload_kb = fit[payload_col].mean() / 1024.0
        final_acc = eval_m["accuracy"].iloc[-10:].mean() * 100  # avg last 10 rounds
        ax.scatter(mean_payload_kb, final_acc, color=color, marker=marker,
                   s=120, label=label, zorder=5)
        ax.annotate(label, (mean_payload_kb, final_acc),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Mean Payload per Round (KB)")
    ax.set_ylabel("Final Accuracy — avg last 10 rounds (%)")
    ax.set_title("Communication Payload vs Accuracy Tradeoff")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_payload_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  saved ablation_payload_vs_accuracy.png")


def plot_energy_vs_accuracy(experiments: List[Dict], labels: List[str], output_dir: Path):
    """Scatter: total energy (J) vs final accuracy."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for exp, label, color, marker in zip(experiments, labels, COLORS, MARKERS):
        fit = exp.get("fit_metrics")
        eval_m = exp.get("eval_metrics")
        if fit is None or eval_m is None:
            continue
        energy_col = next((c for c in fit.columns if "energy_joules" in c and "per_sample" not in c), None)
        if energy_col is None:
            continue
        total_energy_j = fit[energy_col].sum()
        final_acc = eval_m["accuracy"].iloc[-10:].mean() * 100
        ax.scatter(total_energy_j, final_acc, color=color, marker=marker,
                   s=120, label=label, zorder=5)
        ax.annotate(label, (total_energy_j, final_acc),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Total Training Energy (J)")
    ax.set_ylabel("Final Accuracy — avg last 10 rounds (%)")
    ax.set_title("Energy vs Accuracy Tradeoff")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_energy_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  saved ablation_energy_vs_accuracy.png")


# ── Summary table ──────────────────────────────────────────────────────────────

def build_summary(experiments: List[Dict], labels: List[str]) -> pd.DataFrame:
    rows = []
    for exp, label in zip(experiments, labels):
        row: Dict = {"label": label}
        eval_m = exp.get("eval_metrics")
        eval_l = exp.get("eval_loss")
        fit    = exp.get("fit_metrics")

        if eval_m is not None and "accuracy" in eval_m.columns:
            row["best_accuracy_%"]  = round(eval_m["accuracy"].max() * 100, 2)
            row["final_accuracy_%"] = round(eval_m["accuracy"].iloc[-1] * 100, 2)
            row["best_round"]       = int(eval_m.loc[eval_m["accuracy"].idxmax(), "round"])

        if eval_l is not None and "eval_loss" in eval_l.columns:
            row["best_loss"]  = round(eval_l["eval_loss"].min(), 4)
            row["final_loss"] = round(eval_l["eval_loss"].iloc[-1], 4)

        if fit is not None:
            payload_col = next((c for c in fit.columns if "data_transfer_size_bytes" in c), None)
            energy_col  = next((c for c in fit.columns if "energy_joules" in c
                                 and "per_sample" not in c), None)
            if payload_col:
                row["mean_payload_kb"] = round(fit[payload_col].mean() / 1024.0, 1)
                row["total_payload_mb"] = round(fit[payload_col].sum() / 1e6, 2)
            if energy_col:
                row["total_energy_j"] = round(fit[energy_col].sum(), 1)

        rows.append(row)
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(results_dirs: List[Path], labels: List[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [load_experiment(d) for d in results_dirs]
    loaded = [(d, e) for d, e in zip(results_dirs, experiments)
              if any(v is not None for v in e.values())]
    if not loaded:
        print("ERROR: No valid result CSVs found in any of the provided directories.")
        return

    print(f"[Ablation] Loaded {len(loaded)}/{len(results_dirs)} experiments")

    plot_accuracy_vs_rounds(experiments, labels, output_dir)
    plot_loss_vs_rounds(experiments, labels, output_dir)
    plot_payload_vs_accuracy(experiments, labels, output_dir)
    plot_energy_vs_accuracy(experiments, labels, output_dir)

    summary = build_summary(experiments, labels)
    summary_path = output_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n  saved {summary_path}")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-k sparsification ablation analysis")
    parser.add_argument("--results-dirs", nargs="+", required=True,
                        help="One directory per compression ratio experiment")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Legend labels (default: directory names)")
    parser.add_argument("--output-dir", type=str, default="./fl_results_hfl/ablation")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.results_dirs]
    lbls = args.labels if args.labels else [d.name for d in dirs]
    if len(lbls) != len(dirs):
        parser.error("--labels count must match --results-dirs count")

    run(dirs, lbls, Path(args.output_dir))
