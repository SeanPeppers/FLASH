"""
Research-quality figure generation for the FLASH HFL energy-efficiency paper.

Averages across multiple experimental runs per dataset, plots mean lines with
shaded ±1 std error bands, and saves publication-ready PNGs to charts/.

Usage:
    python plot_results.py                          # both datasets, charts/
    python plot_results.py --dataset mnist
    python plot_results.py --dataset ucihar
    python plot_results.py --out-dir my_charts
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
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.6",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 0,
})

FIG_W_SINGLE = 9.0   # single-panel figure width (in)
FIG_W_DOUBLE = 13.0  # double-panel figure width (in)
FIG_H        = 5.0   # figure height (in)

STRATEGY_STYLE = {
    "flash":         dict(color="#ff7f0e", label="FLASH"),
    "fixedcompress": dict(color="#2ca02c", label="FixedCompress"),
    "fedavg":        dict(color="#1f77b4", label="FedAvg"),
    "adamc":         dict(color="#d62728", label="adaMC"),
}

ROUND_SUFFIXES = ["first_round", "second_round", "third_round"]

_CI_LEAF_AGG = 0.434   # kg CO2/kWh  (FGCU, Florida FRCC grid)
_CI_SERVER   = 0.389   # kg CO2/kWh  (Chameleon TACC, Texas ERCOT grid)


# --- Multi-run data loading ------------------------------------------------- #

def _find_run_dirs(dataset_tag: str) -> list[Path]:
    dirs = []
    for suffix in ROUND_SUFFIXES:
        d = Path(f"fl_results_hfl_{dataset_tag}_{suffix}")
        if d.is_dir():
            dirs.append(d)
    return dirs


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "round" in df.columns:
        df = df.sort_values("round").reset_index(drop=True)
    return df


def _load_one_run(run_dir: Path, strategy: str) -> dict:
    prefix = f"{strategy}_HFL"
    server_hw = run_dir / f"{strategy}_server_hw.csv"
    if not server_hw.exists():
        server_hw = run_dir / f"{prefix}_server_hw.csv"
    return {
        "eval_loss":    _load_csv(run_dir / f"{prefix}_eval_loss.csv"),
        "eval_metrics": _load_csv(run_dir / f"{prefix}_eval_metrics.csv"),
        "fit_metrics":  _load_csv(run_dir / f"{prefix}_fit_metrics.csv"),
        "server_hw":    _load_csv(server_hw),
    }


def load_averaged(dataset_tag: str, strategy: str) -> dict:
    """
    Load CSVs from all run directories for this dataset/strategy and return
    a dict of {key: {'mean': DataFrame, 'std': DataFrame, 'n': int}}.
    Falls back to raw DataFrames when only one run exists.
    """
    run_dirs = _find_run_dirs(dataset_tag)
    if not run_dirs:
        return {}

    runs = [_load_one_run(d, strategy) for d in run_dirs]
    keys = ["eval_loss", "eval_metrics", "fit_metrics", "server_hw"]
    result = {}

    for key in keys:
        frames = [r[key] for r in runs if r[key] is not None]
        if not frames:
            result[key] = None
            continue
        if len(frames) == 1:
            result[key] = {"mean": frames[0], "std": None, "n": 1, "raw": frames}
            continue

        # Align on round index and average numeric columns
        numeric_cols = [c for c in frames[0].columns if c != "round"
                        and pd.api.types.is_numeric_dtype(frames[0][c])]
        ref_rounds = frames[0]["round"].values
        aligned = []
        for f in frames:
            f2 = f[f["round"].isin(ref_rounds)].set_index("round")
            aligned.append(f2[numeric_cols])

        combined = pd.concat(aligned, axis=0, keys=range(len(aligned)))
        mean_df = combined.groupby(level=1).mean().reset_index()
        std_df  = combined.groupby(level=1).std(ddof=1).reset_index()
        mean_df.rename(columns={"index": "round"}, inplace=True)
        std_df.rename(columns={"index": "round"}, inplace=True)
        result[key] = {"mean": mean_df, "std": std_df, "n": len(frames), "raw": frames}

    return result


# --- Plot helpers ------------------------------------------------------------ #

def _ax_round_xlabel(ax):
    ax.set_xlabel("Communication Round")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def _savefig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (Path(name).stem + ".png")
    fig.savefig(path)
    print(f"  saved {path}")
    plt.close(fig)


def _smooth(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window, center=True, min_periods=1).mean()


def _get_col(entry, col: str, smooth_w: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (rounds, mean_values, std_values) for a column from an averaged entry."""
    if entry is None:
        return None
    mean_df = entry["mean"]
    std_df  = entry["std"]
    if col not in mean_df.columns:
        return None
    rounds = mean_df["round"].values
    mean_v = mean_df[col].values.astype(float)
    std_v  = std_df[col].values.astype(float) if std_df is not None and col in std_df.columns else np.zeros_like(mean_v)
    if smooth_w > 1:
        mean_v = pd.Series(mean_v).rolling(smooth_w, center=True, min_periods=1).mean().values
        std_v  = pd.Series(std_v).rolling(smooth_w, center=True, min_periods=1).mean().values
    return rounds, mean_v, std_v


def _plot_with_band(ax, rounds, mean_v, std_v, color, label, alpha=0.15):
    ax.plot(rounds, mean_v, color=color, label=label)
    if np.any(std_v > 0):
        ax.fill_between(rounds, mean_v - std_v, mean_v + std_v,
                        color=color, alpha=alpha, linewidth=0)


def _system_energy_series(entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (rounds, mean_total_J, std_total_J) for leaf+agg combined energy."""
    if entry is None:
        return None
    mean_df = entry["mean"]
    std_df  = entry["std"]
    cols = [c for c in ["leaf_energy_joules", "agg_energy_joules"] if c in mean_df.columns]
    if not cols:
        return None
    rounds = mean_df["round"].values
    mean_v = mean_df[cols].sum(axis=1).values
    if std_df is not None:
        std_v = std_df[cols].sum(axis=1).values
    else:
        std_v = np.zeros_like(mean_v)
    return rounds, mean_v, std_v


def _gco2(energy_j: float, ci: float) -> float:
    return energy_j / 3_600_000 * ci * 1000


# --- Figures ---------------------------------------------------------------- #

def fig_convergence(data: dict, out_dir: Path, tag: str):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))

    for strat, entries in data.items():
        sty   = STRATEGY_STYLE[strat]
        color = sty["color"]
        label = sty["label"]

        res = _get_col(entries.get("eval_loss"), "eval_loss", smooth_w=5)
        if res:
            rounds, mean_v, std_v = res
            _plot_with_band(ax_loss, rounds, mean_v, std_v, color, label)

        res = _get_col(entries.get("eval_metrics"), "accuracy", smooth_w=5)
        if res:
            rounds, mean_v, std_v = res
            _plot_with_band(ax_acc, rounds, mean_v * 100, std_v * 100, color, label)

    ax_loss.set_yscale("log")
    ax_loss.set_title("Convergence — Eval Loss")
    ax_acc.set_title("Convergence — Test Accuracy")
    for ax, ylabel in [(ax_loss, "Eval Loss (log scale)"), (ax_acc, "Test Accuracy (%)")]:
        _ax_round_xlabel(ax)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    ax_acc.set_ylim(bottom=88.0 if tag == "UCI-HAR" else 98.0)
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig1_convergence_{tag}.png")


def fig_energy_breakdown(data: dict, out_dir: Path, tag: str):
    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        hw_entry  = entries.get("server_hw")
        if fit_entry is None:
            continue
        fit = fit_entry["mean"]

        rounds  = fit["round"].values
        leaf_e  = fit.get("leaf_energy_joules", pd.Series(np.nan, index=fit.index)).values
        agg_e   = fit.get("agg_energy_joules",  pd.Series(np.nan, index=fit.index)).values
        server_e = np.full(len(rounds), np.nan)
        if hw_entry is not None and "server_energy_joules" in hw_entry["mean"].columns:
            hw_mean = hw_entry["mean"]
            merged = pd.merge(fit[["round"]], hw_mean[["round", "server_energy_joules"]], on="round", how="left")
            server_e = merged["server_energy_joules"].values

        fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
        valid   = ~(np.isnan(leaf_e) & np.isnan(agg_e) & np.isnan(server_e))
        bottom  = np.zeros(valid.sum())
        colors  = ["#aec7e8", "#ffbb78", "#98df8a"]
        labels  = ["Leaf (Pi5 / Nano)", "Aggregator (Xavier)", "Server (Chameleon)"]
        for arr, color, lbl in zip([leaf_e[valid], agg_e[valid], server_e[valid]], colors, labels):
            arr = np.nan_to_num(arr)
            ax.bar(rounds[valid], arr, bottom=bottom, color=color, label=lbl, width=0.8, edgecolor="none")
            bottom += arr

        ax.set_title(f"Energy Breakdown — {STRATEGY_STYLE[strat]['label']}")
        _ax_round_xlabel(ax)
        ax.set_ylabel("Energy (J)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        _savefig(fig, out_dir, f"fig2_energy_breakdown_{strat}_{tag}.png")


def fig_compression_ratio(data: dict, out_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        if fit_entry is None:
            continue
        fit = fit_entry["mean"]
        if "leaf_compression_ratio_applied" not in fit.columns:
            continue
        sty = STRATEGY_STYLE[strat]
        r   = fit["leaf_compression_ratio_applied"]
        if r.nunique() == 1:
            ax.axhline(r.iloc[0], color=sty["color"], linestyle="--",
                       linewidth=1.6, label=f"{sty['label']} (fixed)")
        else:
            res = _get_col(entries["fit_metrics"], "leaf_compression_ratio_applied")
            if res:
                rounds, mean_v, std_v = res
                _plot_with_band(ax, rounds, mean_v, std_v, sty["color"], sty["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_ylim(0, 1.15)
    ax.set_title("Adaptive Compression Ratio")
    ax.set_ylabel("Compression Ratio $r$")
    _ax_round_xlabel(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig3_compression_ratio_{tag}.png")


def fig_comm_overhead(data: dict, out_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        if fit_entry is None:
            continue
        fit = fit_entry["mean"]
        col = "data_transfer_size_bytes"
        if col not in fit.columns:
            continue
        sty = STRATEGY_STYLE[strat]
        res = _get_col(entries["fit_metrics"], col)
        if res:
            rounds, mean_v, std_v = res
            _plot_with_band(ax, rounds, mean_v / 1024, std_v / 1024, sty["color"], sty["label"])
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title("Communication Overhead per Round")
    _ax_round_xlabel(ax)
    ax.set_ylabel("Transfer Size (KB)")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig4_comm_overhead_{tag}.png")


def fig_round_duration(data: dict, out_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        hw_entry = entries.get("server_hw")
        if hw_entry is None:
            continue
        res = _get_col(hw_entry, "server_round_duration_s")
        if res:
            rounds, mean_v, std_v = res
            sty = STRATEGY_STYLE[strat]
            _plot_with_band(ax, rounds, mean_v, std_v, sty["color"], sty["label"])
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title("Round Duration")
    _ax_round_xlabel(ax)
    ax.set_ylabel("Round Duration (s)")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig5_round_duration_{tag}.png")


def fig_energy_per_sample(data: dict, out_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        if fit_entry is None:
            continue
        res = _get_col(fit_entry, "leaf_energy_per_sample_j", smooth_w=3)
        if res:
            rounds, mean_v, std_v = res
            sty = STRATEGY_STYLE[strat]
            _plot_with_band(ax, rounds, mean_v * 1000, std_v * 1000, sty["color"], sty["label"])
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title("Energy per Sample")
    _ax_round_xlabel(ax)
    ax.set_ylabel("Energy / Sample (mJ)")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig6_energy_per_sample_{tag}.png")


def fig_power_summary(data: dict, out_dir: Path, tag: str):
    strategies, server_p, agg_p, leaf_p = [], [], [], []

    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        hw_entry  = entries.get("server_hw")
        fit = fit_entry["mean"] if fit_entry else None
        hw  = hw_entry["mean"]  if hw_entry  else None

        leaf_avg = fit["leaf_power_avg_w"].mean() if fit is not None and "leaf_power_avg_w" in fit.columns else np.nan
        agg_avg  = fit["agg_power_avg_w"].mean()  if fit is not None and "agg_power_avg_w"  in fit.columns else np.nan
        srv_avg  = hw["server_power_avg_w"].mean() if hw  is not None and "server_power_avg_w" in hw.columns else np.nan

        strategies.append(STRATEGY_STYLE[strat]["label"])
        leaf_p.append(leaf_avg)
        agg_p.append(agg_avg)
        server_p.append(srv_avg)

    if not strategies:
        return

    y      = np.arange(len(strategies))
    height = 0.22
    fig, ax = plt.subplots(figsize=(FIG_W_DOUBLE, FIG_H))
    ax.barh(y + height, leaf_p,   height, label="Leaf (Pi5/Nano)",    color="#aec7e8")
    ax.barh(y,          agg_p,    height, label="Aggregator (Xavier)", color="#ffbb78")
    ax.barh(y - height, server_p, height, label="Server (Chameleon)", color="#98df8a")
    ax.set_yticks(y)
    ax.set_yticklabels(strategies)
    ax.set_title("Mean Power per Tier")
    ax.set_xlabel("Mean Power (W)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig7_power_summary_{tag}.png")


def fig_accuracy_vs_energy(data: dict, out_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        em_entry  = entries.get("eval_metrics")
        if fit_entry is None or em_entry is None:
            continue

        fit = fit_entry["mean"]
        em  = em_entry["mean"]
        e_cols = [c for c in ["leaf_energy_joules", "agg_energy_joules"] if c in fit.columns]
        if not e_cols or "accuracy" not in em.columns:
            continue

        fit2 = fit[["round"] + e_cols].copy()
        fit2["_total_e"] = fit2[e_cols].sum(axis=1)
        merged = pd.merge(fit2[["round", "_total_e"]], em[["round", "accuracy"]], on="round")
        cum_kj = merged["_total_e"].cumsum() / 1000
        sty = STRATEGY_STYLE[strat]
        ax.plot(cum_kj, merged["accuracy"] * 100, color=sty["color"], label=sty["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title("Accuracy vs. Cumulative Energy")
    ax.set_xlabel("Cumulative Energy (kJ)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig8_accuracy_vs_energy_{tag}.png")


def fig_latency_jitter(data: dict, out_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        if fit_entry is None:
            continue
        res = _get_col(fit_entry, "leaf_latency_jitter_mean_s")
        if res:
            rounds, mean_v, std_v = res
            sty = STRATEGY_STYLE[strat]
            _plot_with_band(ax, rounds, mean_v * 1000, std_v * 1000, sty["color"], sty["label"])
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title("Latency Jitter")
    _ax_round_xlabel(ax)
    ax.set_ylabel("Latency Jitter (ms)")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig9_latency_jitter_{tag}.png")


def fig_grad_norm(data: dict, out_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        if fit_entry is None:
            continue
        res = _get_col(fit_entry, "leaf_grad_norm_mean", smooth_w=3)
        if res:
            rounds, mean_v, std_v = res
            sty = STRATEGY_STYLE[strat]
            _plot_with_band(ax, rounds, mean_v, std_v, sty["color"], sty["label"])
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title("Gradient Norm")
    _ax_round_xlabel(ax)
    ax.set_ylabel("Mean Gradient Norm")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig10_grad_norm_{tag}.png")


def fig_cumulative_energy(data: dict, out_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        if fit_entry is None:
            continue
        res = _system_energy_series(fit_entry)
        if res is None:
            continue
        rounds, mean_v, std_v = res
        cum_mean = np.cumsum(mean_v) / 1000
        cum_std  = np.sqrt(np.cumsum(std_v ** 2)) / 1000
        sty = STRATEGY_STYLE[strat]
        _plot_with_band(ax, rounds, cum_mean, cum_std, sty["color"], sty["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title("Cumulative Energy Consumption")
    _ax_round_xlabel(ax)
    ax.set_ylabel("Cumulative Energy (kJ)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig11_cumulative_energy_{tag}.png")


def fig_carbon_footprint(data: dict, out_dir: Path, tag: str):
    labels, co2_vals, co2_err = [], [], []

    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        if fit_entry is None:
            continue
        res = _system_energy_series(fit_entry)
        if res is None:
            continue
        _, mean_v, std_v = res
        labels.append(STRATEGY_STYLE[strat]["label"])
        co2_vals.append(_gco2(mean_v.sum(), _CI_LEAF_AGG))
        co2_err.append(_gco2(std_v.sum(), _CI_LEAF_AGG))

    if not labels:
        return

    x   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    bars = ax.bar(x, co2_vals, yerr=co2_err, capsize=5,
                  color=[STRATEGY_STYLE[s]["color"] for s in data], width=0.5)
    for bar, val in zip(bars, co2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(co2_err) * 1.1 + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Carbon Footprint (Leaf + Aggregator)")
    ax.set_ylabel("CO$_2$ Equivalent (g)")
    ax.set_ylim(0, max(co2_vals) * 1.3)
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig12_carbon_footprint_{tag}.png")


def fig_green_efficiency(data: dict, out_dir: Path, tag: str):
    labels, gei_vals, gei_err = [], [], []

    for strat, entries in data.items():
        em_entry  = entries.get("eval_metrics")
        fit_entry = entries.get("fit_metrics")
        if em_entry is None or fit_entry is None:
            continue
        if "accuracy" not in em_entry["mean"].columns:
            continue
        res = _system_energy_series(fit_entry)
        if res is None:
            continue

        # Compute GEI per run then report mean ± std
        raw_fits = fit_entry.get("raw", [fit_entry["mean"]])
        raw_ems  = em_entry.get("raw",  [em_entry["mean"]])
        geis = []
        for rf, re in zip(raw_fits, raw_ems):
            e_cols = [c for c in ["leaf_energy_joules", "agg_energy_joules"] if c in rf.columns]
            if not e_cols:
                continue
            total_kj = rf[e_cols].sum(axis=1).sum() / 1000
            best_acc = re["accuracy"].max() * 100
            geis.append(best_acc / total_kj)
        if not geis:
            continue
        labels.append(STRATEGY_STYLE[strat]["label"])
        gei_vals.append(np.mean(geis))
        gei_err.append(np.std(geis, ddof=1) if len(geis) > 1 else 0.0)

    if not labels:
        return

    x   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    bars = ax.bar(x, gei_vals, yerr=gei_err, capsize=5,
                  color=[STRATEGY_STYLE[s]["color"] for s in data], width=0.5)
    for bar, val in zip(bars, gei_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(gei_err or [0]) * 1.1 + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Green Efficiency Index")
    ax.set_ylabel("Accuracy (%) / Energy (kJ)")
    ax.set_ylim(0, max(gei_vals) * 1.25)
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig13_green_efficiency_{tag}.png")


def fig_thermal_profile(data: dict, out_dir: Path, tag: str):
    fig, (ax_leaf, ax_agg) = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        fit_entry = entries.get("fit_metrics")
        if fit_entry is None:
            continue
        sty = STRATEGY_STYLE[strat]

        res = _get_col(fit_entry, "leaf_hw_temp_max_celsius", smooth_w=5)
        if res:
            rounds, mean_v, std_v = res
            _plot_with_band(ax_leaf, rounds, mean_v, std_v, sty["color"], sty["label"])
            plotted = True

        res = _get_col(fit_entry, "agg_hw_temp_max_celsius", smooth_w=5)
        if res:
            rounds, mean_v, std_v = res
            _plot_with_band(ax_agg, rounds, mean_v, std_v, sty["color"], sty["label"])

    if not plotted:
        plt.close(fig)
        return

    ax_leaf.set_title("Thermal Profile — Leaf Tier")
    ax_agg.set_title("Thermal Profile — Aggregator Tier")
    for ax, ylabel in [(ax_leaf, "Leaf Max Temp (°C)"), (ax_agg, "Aggregator Max Temp (°C)")]:
        _ax_round_xlabel(ax)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    fig.tight_layout()
    _savefig(fig, out_dir, f"fig14_thermal_profile_{tag}.png")


def fig_convergence_loss_only(data: dict, out_dir: Path, tag: str):
    """Single-panel eval-loss only — used in the UCI-HAR three-panel figure."""
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        sty = STRATEGY_STYLE[strat]
        res = _get_col(entries.get("eval_loss"), "eval_loss", smooth_w=5)
        if res:
            rounds, mean_v, std_v = res
            _plot_with_band(ax, rounds, mean_v, std_v, sty["color"], sty["label"])
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_yscale("log")
    ax.set_title("Convergence — Evaluation Loss")
    _ax_round_xlabel(ax)
    ax.set_ylabel("Eval Loss (log scale)")
    ax.legend(loc="best")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig_convergence_loss_{tag}.png")


def fig_running_gei(data: dict, out_dir: Path, tag: str):
    """Running GEI = best-accuracy-so-far (%) / cumulative-energy (kJ) over rounds."""
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))
    plotted = False

    for strat, entries in data.items():
        em_entry  = entries.get("eval_metrics")
        fit_entry = entries.get("fit_metrics")
        if em_entry is None or fit_entry is None:
            continue
        mean_em  = em_entry["mean"]
        mean_fit = fit_entry["mean"]
        if "accuracy" not in mean_em.columns:
            continue
        e_cols = [c for c in ["leaf_energy_joules", "agg_energy_joules"] if c in mean_fit.columns]
        if not e_cols:
            continue

        merged = pd.merge(
            mean_em[["round", "accuracy"]],
            mean_fit[["round"] + e_cols],
            on="round", how="inner"
        )
        cum_kj   = merged[e_cols].sum(axis=1).cumsum() / 1000
        best_acc = merged["accuracy"].expanding().max() * 100
        gei      = best_acc / (cum_kj + 1e-9)

        sty = STRATEGY_STYLE[strat]
        ax.plot(merged["round"].values, gei.values, color=sty["color"], label=sty["label"])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title("Running Green Efficiency Index")
    ax.set_xlim(left=5)   # skip init spike where cumulative energy is near-zero
    _ax_round_xlabel(ax)
    ax.set_ylabel("GEI  (Acc.\\ % / kJ)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig_running_gei_{tag}.png")


def fig_pareto_frontier(data: dict, out_dir: Path, tag: str):
    """Energy–loss Pareto frontier scatter across strategies."""
    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))

    points = []
    for strat, entries in data.items():
        fit_entry  = entries.get("fit_metrics")
        loss_entry = entries.get("eval_loss")
        if fit_entry is None or loss_entry is None:
            continue
        raw_fits   = fit_entry.get("raw",  [fit_entry["mean"]])
        raw_losses = loss_entry.get("raw", [loss_entry["mean"]])

        energies, losses = [], []
        for rf, rl in zip(raw_fits, raw_losses):
            e_cols = [c for c in ["leaf_energy_joules", "agg_energy_joules"] if c in rf.columns]
            if not e_cols or "eval_loss" not in rl.columns:
                continue
            energies.append(rf[e_cols].sum(axis=1).sum() / 1000)
            losses.append(float(rl["eval_loss"].iloc[-1]))

        if not energies:
            continue
        sty = STRATEGY_STYLE[strat]
        points.append(dict(
            energy     = np.mean(energies),
            energy_std = np.std(energies, ddof=1) if len(energies) > 1 else 0.0,
            loss       = np.mean(losses),
            loss_std   = np.std(losses, ddof=1)   if len(losses) > 1  else 0.0,
            label      = sty["label"],
            color      = sty["color"],
        ))

    if not points:
        plt.close(fig)
        return

    # Identify Pareto-optimal points (minimise both axes)
    pts = np.array([(p["energy"], p["loss"]) for p in points])
    is_pareto = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j and pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1] \
                    and (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1]):
                is_pareto[i] = False
                break

    # Shade dominated region (upper-right of Pareto front)
    pareto_e = pts[is_pareto, 0].min()
    pareto_l = pts[is_pareto, 1].min()
    e_max = pts[:, 0].max()
    l_max = pts[:, 1].max()
    e_margin = (e_max - pts[:, 0].min()) * 0.18
    l_margin = (l_max - pareto_l) * 0.25
    ax.fill_between(
        [pareto_e, e_max + e_margin],
        [pareto_l, pareto_l],
        [l_max + l_margin, l_max + l_margin],
        color="lightgray", alpha=0.45, zorder=1, label="Dominated region"
    )

    # Pareto-front L-shaped line
    ax.axvline(pareto_e, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
    ax.axhline(pareto_l, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

    # Plot each strategy
    for p in points:
        ax.errorbar(
            p["energy"], p["loss"],
            xerr=p["energy_std"], yerr=p["loss_std"],
            fmt="o", color=p["color"], markersize=13,
            capsize=5, capthick=1.8, elinewidth=1.8,
            label=p["label"], zorder=5
        )
        # Offset label so it doesn't overlap the marker
        offset = (6, 5) if p["energy"] < pareto_e + 5 else (-80, 5)
        ax.annotate(
            p["label"],
            xy=(p["energy"], p["loss"]),
            xytext=offset, textcoords="offset points",
            fontsize=11, color=p["color"], fontweight="bold"
        )

    ax.set_title(f"Energy–Loss Pareto Frontier ({tag})")
    ax.set_xlabel("Total Edge-Tier Energy (kJ)")
    ax.set_ylabel("Final Evaluation Loss (R60)")
    # Legend without the per-strategy items (labels shown via annotations)
    handles, labels = ax.get_legend_handles_labels()
    pareto_handle = [h for h, l in zip(handles, labels) if l == "Dominated region"]
    ax.legend(pareto_handle, ["Dominated region"], loc="upper left")
    fig.tight_layout()
    _savefig(fig, out_dir, f"fig_pareto_{tag}.png")


# --- Main ------------------------------------------------------------------- #

FIGURE_FUNCS = [
    fig_convergence, fig_energy_breakdown, fig_compression_ratio,
    fig_comm_overhead, fig_round_duration, fig_energy_per_sample,
    fig_power_summary, fig_accuracy_vs_energy, fig_latency_jitter,
    fig_grad_norm, fig_cumulative_energy, fig_carbon_footprint,
    fig_green_efficiency, fig_thermal_profile,
    fig_convergence_loss_only, fig_running_gei, fig_pareto_frontier,
]


def run_dataset(dataset_tag: str, out_dir: Path):
    run_dirs = _find_run_dirs(dataset_tag)
    if not run_dirs:
        print(f"  No run directories found for {dataset_tag}, skipping.")
        return

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_tag}  ({len(run_dirs)} runs)")
    print(f"{'='*60}")

    data = {}
    for strat in ["flash", "fixedcompress", "fedavg", "adamc"]:
        entries = load_averaged(dataset_tag, strat)
        if entries:
            data[strat] = entries
            print(f"  loaded: {strat}")

    if not data:
        print(f"  No data found for {dataset_tag}.")
        return

    tag_dir = out_dir / dataset_tag.lower().replace("-", "_")
    print(f"\n  Generating figures -> {tag_dir}/")
    for fn in FIGURE_FUNCS:
        try:
            fn(data, tag_dir, dataset_tag)
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {e}")

    print(f"  Done — {dataset_tag}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "ucihar", "all"], default="all")
    parser.add_argument("--out-dir", default="updated_charts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    targets = []
    if args.dataset in ("mnist", "all"):
        targets.append("MNIST")
    if args.dataset in ("ucihar", "all"):
        targets.append("UCI-HAR")

    for tag in targets:
        run_dataset(tag, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
