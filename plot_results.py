"""
Research-quality figure generation for the FLASH HFL energy-efficiency paper.

Loads CSVs from fl_results_hfl/, generates publication-ready figures in
fl_results_hfl/figures/.  Designed for multi-strategy comparison (FLASH /
FLARE / FedAvg); gracefully falls back to single-strategy when others are
absent.

Usage:
    python plot_results.py [--results-dir fl_results_hfl] [--out-dir fl_results_hfl/figures]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# --- Style ------------------------------------------------------------------ #
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
})

# IEEE two-column figure width ≈ 3.5 in; full-width ≈ 7.16 in
COL1 = 3.5
COL2 = 7.16

STRATEGY_STYLE = {
    "flash": dict(color="#1f77b4", marker="o", label="FLASH"),
    "flare": dict(color="#ff7f0e", marker="s", label="FLARE"),
    "fedavg": dict(color="#2ca02c", marker="^", label="FedAvg"),
}


# --- Data loading ----------------------------------------------------------- #

def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.sort_values("round").reset_index(drop=True)
    return df


def load_strategy(results_dir: Path, strategy: str) -> dict[str, pd.DataFrame | None]:
    prefix = f"{strategy}_HFL"
    # server_hw is written directly by server.py without the _HFL_ infix
    server_hw_path = results_dir / f"{strategy}_server_hw.csv"
    if not server_hw_path.exists():
        server_hw_path = results_dir / f"{prefix}_server_hw.csv"
    return {
        "eval_loss": _load(results_dir / f"{prefix}_eval_loss.csv"),
        "eval_metrics": _load(results_dir / f"{prefix}_eval_metrics.csv"),
        "fit_metrics": _load(results_dir / f"{prefix}_fit_metrics.csv"),
        "server_hw": _load(server_hw_path),
    }


# --- Helpers ---------------------------------------------------------------- #

def _ax_round_xlabel(ax):
    ax.set_xlabel("Communication Round")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def _savefig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    print(f"  saved {path}")
    plt.close(fig)


def _smooth(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window, center=True, min_periods=1).mean()


# ---------------------------------------------------------------------------- #
# Figure 1 — Convergence (loss + accuracy)                                     #
# ---------------------------------------------------------------------------- #

def fig_convergence(data: dict, out_dir: Path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(COL2, 2.6))

    for strat, dfs in data.items():
        sty = STRATEGY_STYLE[strat]
        el = dfs["eval_loss"]
        em = dfs["eval_metrics"]

        if el is not None:
            ax_loss.plot(el["round"], el["eval_loss"],
                         color=sty["color"], marker=sty["marker"],
                         markevery=5, label=sty["label"])

        if em is not None and "accuracy" in em.columns:
            ax_acc.plot(em["round"], em["accuracy"] * 100,
                        color=sty["color"], marker=sty["marker"],
                        markevery=5, label=sty["label"])

    for ax, ylabel in [(ax_loss, "Eval Loss"), (ax_acc, "Test Accuracy (%)")]:
        _ax_round_xlabel(ax)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    ax_loss.set_title("(a) Convergence — Loss")
    ax_acc.set_title("(b) Convergence — Accuracy")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig1_convergence.pdf")


# ---------------------------------------------------------------------------- #
# Figure 2 — Per-round energy breakdown across tiers                           #
# ---------------------------------------------------------------------------- #

def fig_energy_breakdown(data: dict, out_dir: Path):
    for strat, dfs in data.items():
        fit = dfs["fit_metrics"]
        hw = dfs["server_hw"]
        if fit is None:
            continue

        rounds = fit["round"].values
        leaf_e = fit.get("leaf_energy_joules", pd.Series(np.nan, index=fit.index)).values
        agg_e = fit.get("agg_energy_joules", pd.Series(np.nan, index=fit.index)).values

        server_e = np.full(len(rounds), np.nan)
        if hw is not None and "server_energy_joules" in hw.columns:
            merged = pd.merge(fit[["round"]], hw[["round", "server_energy_joules"]], on="round", how="left")
            server_e = merged["server_energy_joules"].values

        fig, ax = plt.subplots(figsize=(COL2, 2.8))
        valid = ~(np.isnan(leaf_e) & np.isnan(agg_e) & np.isnan(server_e))
        rounds_v = rounds[valid]

        bottom = np.zeros(valid.sum())
        colors = ["#aec7e8", "#ffbb78", "#98df8a"]
        labels = ["Leaf (Pi5 / Nano)", "Aggregator (Xavier)", "Server (Chameleon)"]
        for arr, color, lbl in zip(
            [leaf_e[valid], agg_e[valid], server_e[valid]], colors, labels
        ):
            arr = np.nan_to_num(arr)
            ax.bar(rounds_v, arr, bottom=bottom, color=color, label=lbl,
                   width=0.8, edgecolor="none")
            bottom += arr

        _ax_round_xlabel(ax)
        ax.set_ylabel("Energy (J)")
        ax.set_title(f"Per-Round Energy by Tier — {STRATEGY_STYLE[strat]['label']}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        _savefig(fig, out_dir, f"fig2_energy_breakdown_{strat}.pdf")


# ---------------------------------------------------------------------------- #
# Figure 3 — Compression ratio over rounds (FLASH only)                        #
# ---------------------------------------------------------------------------- #

def fig_compression_ratio(data: dict, out_dir: Path):
    if "flash" not in data or data["flash"]["fit_metrics"] is None:
        return
    fit = data["flash"]["fit_metrics"]
    if "leaf_compression_ratio_applied" not in fit.columns:
        return

    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    r = fit["leaf_compression_ratio_applied"]
    ax.plot(fit["round"], r, color=STRATEGY_STYLE["flash"]["color"],
            marker="o", markevery=5)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="No compression")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Top-k Ratio r")
    ax.set_title("Adaptive Compression Ratio")
    _ax_round_xlabel(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig3_compression_ratio.pdf")


# ---------------------------------------------------------------------------- #
# Figure 4 — Communication overhead (data transfer bytes)                       #
# ---------------------------------------------------------------------------- #

def fig_comm_overhead(data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    plotted = False

    for strat, dfs in data.items():
        fit = dfs["fit_metrics"]
        if fit is None or "data_transfer_size_bytes" not in fit.columns:
            continue
        sty = STRATEGY_STYLE[strat]
        size_kb = fit["data_transfer_size_bytes"] / 1024
        ax.plot(fit["round"], size_kb,
                color=sty["color"], marker=sty["marker"],
                markevery=5, label=sty["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    _ax_round_xlabel(ax)
    ax.set_ylabel("Transfer Size (KB)")
    ax.set_title("Communication Overhead")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig4_comm_overhead.pdf")


# ---------------------------------------------------------------------------- #
# Figure 5 — Round duration                                                    #
# ---------------------------------------------------------------------------- #

def fig_round_duration(data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    plotted = False

    for strat, dfs in data.items():
        hw = dfs["server_hw"]
        if hw is None or "server_round_duration_s" not in hw.columns:
            continue
        sty = STRATEGY_STYLE[strat]
        ax.plot(hw["round"], hw["server_round_duration_s"],
                color=sty["color"], marker=sty["marker"],
                markevery=5, label=sty["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    _ax_round_xlabel(ax)
    ax.set_ylabel("Round Duration (s)")
    ax.set_title("Wall-Clock Round Time")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig5_round_duration.pdf")


# ---------------------------------------------------------------------------- #
# Figure 6 — Energy per sample                                                  #
# ---------------------------------------------------------------------------- #

def fig_energy_per_sample(data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    plotted = False

    for strat, dfs in data.items():
        fit = dfs["fit_metrics"]
        if fit is None or "leaf_energy_per_sample_j" not in fit.columns:
            continue
        sty = STRATEGY_STYLE[strat]
        eps = fit["leaf_energy_per_sample_j"] * 1000  # mJ
        ax.plot(fit["round"], _smooth(eps),
                color=sty["color"], marker=sty["marker"],
                markevery=5, label=sty["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    _ax_round_xlabel(ax)
    ax.set_ylabel("Energy / Sample (mJ)")
    ax.set_title("Leaf Energy Efficiency")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig6_energy_per_sample.pdf")


# ---------------------------------------------------------------------------- #
# Figure 7 — Average power per tier (bar chart summary)                         #
# ---------------------------------------------------------------------------- #

def fig_power_summary(data: dict, out_dir: Path):
    strategies, server_p, agg_p, leaf_p = [], [], [], []

    for strat, dfs in data.items():
        fit = dfs["fit_metrics"]
        hw = dfs["server_hw"]

        leaf_avg = (fit["leaf_power_avg_w"].mean()
                    if fit is not None and "leaf_power_avg_w" in fit.columns
                    else np.nan)
        agg_avg = (fit["agg_power_avg_w"].mean()
                   if fit is not None and "agg_power_avg_w" in fit.columns
                   else np.nan)
        srv_avg = (hw["server_power_avg_w"].mean()
                   if hw is not None and "server_power_avg_w" in hw.columns
                   else np.nan)

        strategies.append(STRATEGY_STYLE[strat]["label"])
        leaf_p.append(leaf_avg)
        agg_p.append(agg_avg)
        server_p.append(srv_avg)

    if not strategies:
        return

    x = np.arange(len(strategies))
    width = 0.25
    fig, ax = plt.subplots(figsize=(COL1, 2.8))
    ax.bar(x - width, leaf_p, width, label="Leaf", color="#aec7e8")
    ax.bar(x, agg_p, width, label="Aggregator", color="#ffbb78")
    ax.bar(x + width, server_p, width, label="Server", color="#98df8a")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel("Avg Power (W)")
    ax.set_title("Mean Power per Tier")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig7_power_summary.pdf")


# ---------------------------------------------------------------------------- #
# Figure 8 — Accuracy vs cumulative energy (efficiency frontier)                #
# ---------------------------------------------------------------------------- #

def fig_accuracy_vs_energy(data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    plotted = False

    for strat, dfs in data.items():
        fit = dfs["fit_metrics"]
        em = dfs["eval_metrics"]
        if fit is None or em is None:
            continue
        if "leaf_energy_joules" not in fit.columns and "energy_joules" not in fit.columns:
            continue
        if "accuracy" not in em.columns:
            continue

        # Sum all available tier energies for a true system-wide efficiency curve
        e_cols = [c for c in ["leaf_energy_joules", "agg_energy_joules"] if c in fit.columns]
        if not e_cols:
            e_cols = ["energy_joules"] if "energy_joules" in fit.columns else []
        if not e_cols:
            continue
        fit_e = fit[["round"] + e_cols].copy()
        fit_e["_total_e"] = fit_e[e_cols].sum(axis=1)
        merged = pd.merge(fit_e[["round", "_total_e"]], em[["round", "accuracy"]], on="round", how="inner")
        cum_e_kj = merged["_total_e"].cumsum() / 1000  # kJ
        sty = STRATEGY_STYLE[strat]
        ax.plot(cum_e_kj, merged["accuracy"] * 100,
                color=sty["color"], marker=sty["marker"],
                markevery=5, label=sty["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Cumulative Energy (kJ)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy vs. Energy Trade-off")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig8_accuracy_vs_energy.pdf")


# ---------------------------------------------------------------------------- #
# Figure 9 — Latency jitter over rounds                                         #
# ---------------------------------------------------------------------------- #

def fig_latency_jitter(data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    plotted = False

    for strat, dfs in data.items():
        fit = dfs["fit_metrics"]
        if fit is None or "leaf_latency_jitter_mean_s" not in fit.columns:
            continue
        sty = STRATEGY_STYLE[strat]
        ax.plot(fit["round"], fit["leaf_latency_jitter_mean_s"] * 1000,
                color=sty["color"], marker=sty["marker"],
                markevery=5, label=sty["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    _ax_round_xlabel(ax)
    ax.set_ylabel("Latency Jitter (ms)")
    ax.set_title("Communication Latency Jitter")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig9_latency_jitter.pdf")


# ---------------------------------------------------------------------------- #
# Figure 10 — Gradient norm decay (training stability)                          #
# ---------------------------------------------------------------------------- #

def fig_grad_norm(data: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    plotted = False

    for strat, dfs in data.items():
        fit = dfs["fit_metrics"]
        if fit is None or "leaf_grad_norm_mean" not in fit.columns:
            continue
        sty = STRATEGY_STYLE[strat]
        ax.plot(fit["round"], _smooth(fit["leaf_grad_norm_mean"]),
                color=sty["color"], marker=sty["marker"],
                markevery=5, label=sty["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    _ax_round_xlabel(ax)
    ax.set_ylabel("Mean Gradient Norm")
    ax.set_title("Training Stability (Grad Norm)")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, "fig10_grad_norm.pdf")


# ---------------------------------------------------------------------------- #
# Main                                                                          #
# ---------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="fl_results_hfl")
    parser.add_argument("--out-dir", default="fl_results_hfl/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    strategies = ["flash", "flare", "fedavg"]
    data = {}
    for s in strategies:
        dfs = load_strategy(results_dir, s)
        if any(v is not None for v in dfs.values()):
            data[s] = dfs
            print(f"  loaded: {s}")

    if not data:
        print(f"No CSV files found in {results_dir}. Run the experiment first.")
        sys.exit(1)

    print(f"\nGenerating figures -> {out_dir}/")
    fig_convergence(data, out_dir)
    fig_energy_breakdown(data, out_dir)
    fig_compression_ratio(data, out_dir)
    fig_comm_overhead(data, out_dir)
    fig_round_duration(data, out_dir)
    fig_energy_per_sample(data, out_dir)
    fig_power_summary(data, out_dir)
    fig_accuracy_vs_energy(data, out_dir)
    fig_latency_jitter(data, out_dir)
    fig_grad_norm(data, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
