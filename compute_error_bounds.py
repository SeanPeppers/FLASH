"""
Compute mean ± std error bounds across multiple experimental runs.
Usage:
    python compute_error_bounds.py --dataset mnist
    python compute_error_bounds.py --dataset ucihar
    python compute_error_bounds.py --dataset all

Expects directories named:
    fl_results_hfl_MNIST_first_round/   fl_results_hfl_MNIST_second_round/   [fl_results_hfl_MNIST_third_round/]
    fl_results_hfl_UCI-HAR_first_round/ fl_results_hfl_UCI-HAR_second_round/ [fl_results_hfl_UCI-HAR_third_round/]
"""

import argparse
import os
import numpy as np
import pandas as pd

STRATEGIES = ["flash", "fedavg", "fixedcompress", "adamc"]
ROUND_SUFFIXES = ["first_round", "second_round", "third_round"]

ENERGY_COL   = "energy_joules"       # total (leaf+agg) per round in fit_metrics
LEAF_ENERGY  = "leaf_energy_joules"
AGG_ENERGY   = "agg_energy_joules"
TRANSFER_COL = "leaf_data_transfer_size_bytes"


def find_run_dirs(dataset_tag: str) -> list[str]:
    dirs = []
    for suffix in ROUND_SUFFIXES:
        d = f"fl_results_hfl_{dataset_tag}_{suffix}"
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def load_eval_loss(run_dir: str, strategy: str) -> pd.Series | None:
    path = os.path.join(run_dir, f"{strategy}_HFL_eval_loss.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="round")
    return df["eval_loss"]


def load_eval_metrics(run_dir: str, strategy: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, f"{strategy}_HFL_eval_metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col="round")


def load_fit_metrics(run_dir: str, strategy: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, f"{strategy}_HFL_fit_metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col="round")


def aggregate_runs(series_list: list[pd.Series]) -> pd.DataFrame:
    combined = pd.concat(series_list, axis=1)
    return pd.DataFrame({
        "mean": combined.mean(axis=1),
        "std":  combined.std(axis=1, ddof=1),
    })


def print_summary_table(dataset_tag: str, run_dirs: list[str]):
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_tag}  |  Runs found: {len(run_dirs)}")
    for i, d in enumerate(run_dirs, 1):
        print(f"    Run {i}: {d}")
    print(f"{'='*70}")

    key_rounds = [10, 30, 60]

    for strategy in STRATEGIES:
        loss_series, acc_series, leaf_e_series, agg_e_series, transfer_series = [], [], [], [], []

        for run_dir in run_dirs:
            loss = load_eval_loss(run_dir, strategy)
            metrics = load_eval_metrics(run_dir, strategy)
            fit = load_fit_metrics(run_dir, strategy)

            if loss is not None:
                loss_series.append(loss)
            if metrics is not None and "accuracy" in metrics.columns:
                acc_series.append(metrics["accuracy"])
            if fit is not None:
                if LEAF_ENERGY in fit.columns:
                    leaf_e_series.append(fit[LEAF_ENERGY])
                if AGG_ENERGY in fit.columns:
                    agg_e_series.append(fit[AGG_ENERGY])
                if TRANSFER_COL in fit.columns:
                    transfer_series.append(fit[TRANSFER_COL])

        if not loss_series:
            print(f"\n  [{strategy.upper()}] — no data found, skipping")
            continue

        print(f"\n  [{strategy.upper()}]  ({len(loss_series)} runs)")

        loss_agg = aggregate_runs(loss_series)
        acc_agg  = aggregate_runs(acc_series)  if acc_series  else None

        # Per-round totals for energy (sum leaf+agg per round across runs)
        if leaf_e_series and agg_e_series:
            total_e_per_run = [l + a for l, a in zip(leaf_e_series, agg_e_series)]
            energy_agg = aggregate_runs(total_e_per_run)
        else:
            energy_agg = None

        transfer_agg = aggregate_runs(transfer_series) if transfer_series else None

        # --- Convergence table ---
        print(f"  {'Round':>6}  {'Loss mean±std':>18}  {'Accuracy mean±std':>20}")
        print(f"  {'-'*6}  {'-'*18}  {'-'*20}")
        for r in key_rounds:
            if r not in loss_agg.index:
                continue
            lm, ls = loss_agg.loc[r, "mean"], loss_agg.loc[r, "std"]
            loss_str = f"{lm:.4f} ± {ls:.4f}" if len(loss_series) > 1 else f"{lm:.4f}"

            acc_str = "N/A"
            if acc_agg is not None and r in acc_agg.index:
                am, as_ = acc_agg.loc[r, "mean"], acc_agg.loc[r, "std"]
                acc_str = f"{am:.4f} ± {as_:.4f}" if len(acc_series) > 1 else f"{am:.4f}"

            print(f"  {r:>6}  {loss_str:>18}  {acc_str:>20}")

        # --- Cumulative energy summary (sum over all 60 rounds) ---
        if energy_agg is not None:
            cum_means = []
            cum_stds  = []
            for run_e in total_e_per_run:
                cum_means.append(run_e.sum())
            total_mean = np.mean(cum_means) / 1000  # J → kJ
            total_std  = np.std(cum_means, ddof=1) / 1000 if len(cum_means) > 1 else 0.0
            print(f"\n  Cumulative leaf+agg energy (60 rounds): "
                  f"{total_mean:.2f} ± {total_std:.2f} kJ")

        # --- Cumulative transfer ---
        if transfer_agg is not None:
            cum_transfer = []
            for t in transfer_series:
                cum_transfer.append(t.sum())
            tm = np.mean(cum_transfer) / 1e6
            ts = np.std(cum_transfer, ddof=1) / 1e6 if len(cum_transfer) > 1 else 0.0
            print(f"  Cumulative data transfer (60 rounds):    "
                  f"{tm:.2f} ± {ts:.2f} MB")

    # --- Cross-strategy comparison at R60 ---
    print(f"\n{'='*70}")
    print(f"  Cross-strategy summary at R60 ({dataset_tag})")
    print(f"  {'Strategy':>14}  {'Loss R60':>18}  {'Acc R60':>20}  {'CumEnergy kJ':>14}")
    print(f"  {'-'*14}  {'-'*18}  {'-'*20}  {'-'*14}")

    ref_energy = None
    ref_loss   = None

    for strategy in STRATEGIES:
        loss_series, acc_series, leaf_e_series, agg_e_series = [], [], [], []

        for run_dir in run_dirs:
            loss    = load_eval_loss(run_dir, strategy)
            metrics = load_eval_metrics(run_dir, strategy)
            fit     = load_fit_metrics(run_dir, strategy)
            if loss is not None:
                loss_series.append(loss)
            if metrics is not None and "accuracy" in metrics.columns:
                acc_series.append(metrics["accuracy"])
            if fit is not None:
                if LEAF_ENERGY in fit.columns:
                    leaf_e_series.append(fit[LEAF_ENERGY])
                if AGG_ENERGY in fit.columns:
                    agg_e_series.append(fit[AGG_ENERGY])

        if not loss_series:
            continue

        lm = np.mean([s.iloc[-1] for s in loss_series])
        ls = np.std([s.iloc[-1] for s in loss_series], ddof=1) if len(loss_series) > 1 else 0.0
        loss_str = f"{lm:.4f} ± {ls:.4f}" if len(loss_series) > 1 else f"{lm:.4f}"

        am, as_ = 0.0, 0.0
        if acc_series:
            am  = np.mean([s.iloc[-1] for s in acc_series])
            as_ = np.std([s.iloc[-1] for s in acc_series], ddof=1) if len(acc_series) > 1 else 0.0
        acc_str = f"{am:.4f} ± {as_:.4f}" if acc_series and len(acc_series) > 1 else f"{am:.4f}"

        energy_str = "N/A"
        if leaf_e_series and agg_e_series:
            cum = [((l + a).sum() / 1000) for l, a in zip(leaf_e_series, agg_e_series)]
            em = np.mean(cum)
            es = np.std(cum, ddof=1) if len(cum) > 1 else 0.0
            energy_str = f"{em:.2f} ± {es:.2f}" if len(cum) > 1 else f"{em:.2f}"
            if strategy == "fedavg":
                ref_energy = em
                ref_loss   = lm

        print(f"  {strategy.upper():>14}  {loss_str:>18}  {acc_str:>20}  {energy_str:>14}")

    # --- Savings vs FedAvg ---
    if ref_energy is not None:
        print(f"\n  Savings vs FedAvg (mean values):")
        for strategy in STRATEGIES:
            if strategy == "fedavg":
                continue
            loss_series, leaf_e_series, agg_e_series = [], [], []
            for run_dir in run_dirs:
                loss = load_eval_loss(run_dir, strategy)
                fit  = load_fit_metrics(run_dir, strategy)
                if loss is not None:
                    loss_series.append(loss)
                if fit is not None:
                    if LEAF_ENERGY in fit.columns:
                        leaf_e_series.append(fit[LEAF_ENERGY])
                    if AGG_ENERGY in fit.columns:
                        agg_e_series.append(fit[AGG_ENERGY])
            if not loss_series:
                continue
            lm = np.mean([s.iloc[-1] for s in loss_series])
            loss_delta = (ref_loss - lm) / ref_loss * 100
            energy_str = ""
            if leaf_e_series and agg_e_series:
                em = np.mean([((l + a).sum() / 1000) for l, a in zip(leaf_e_series, agg_e_series)])
                energy_delta = (ref_energy - em) / ref_energy * 100
                energy_str = f"  energy savings: {energy_delta:+.1f}%"
            print(f"    {strategy.upper():>14}: loss delta vs FedAvg: {loss_delta:+.1f}%{energy_str}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "ucihar", "all"], default="all")
    args = parser.parse_args()

    targets = []
    if args.dataset in ("mnist", "all"):
        targets.append("MNIST")
    if args.dataset in ("ucihar", "all"):
        targets.append("UCI-HAR")

    for tag in targets:
        run_dirs = find_run_dirs(tag)
        if not run_dirs:
            print(f"\nNo run directories found for {tag} — skipping.")
            continue
        print_summary_table(tag, run_dirs)


if __name__ == "__main__":
    main()
