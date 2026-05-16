"""
Component study figure generation for the FLASH TGCN paper ablation section.

Loads FL_FLASH_Component_Eval/component_eval_results.json and produces
5 publication-ready PNGs in component_study/.

Usage:
    python plot_component_study.py
    python plot_component_study.py --out-dir my_folder
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Style (matches plot_results.py) ---------------------------------------- #
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
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
    "lines.markersize": 6,
})

FIG_W_SINGLE = 9.0
FIG_W_DOUBLE = 13.0
FIG_H        = 5.0

R_COLORS = {
    0.25: "#d62728",
    0.5:  "#ff7f0e",
    0.75: "#2ca02c",
    1.0:  "#1f77b4",
}
R_LABELS = {
    0.25: "r=0.25 (75% compr.)",
    0.5:  "r=0.50 (50% compr.)",
    0.75: "r=0.75 (25% compr.)",
    1.0:  "r=1.00 (none)",
}

METHOD_COLORS = {
    "Size-Weighted":     "#ff7f0e",
    "Equal Weight":      "#1f77b4",
    "Loss-Inv-Weighted": "#d62728",
}

PIPELINE_COLORS = {
    "Training":    "#1f77b4",
    "Aggregation": "#ff7f0e",
    "Overhead":    "#2ca02c",
}


# ---------------------------------------------------------------------------- #
# C1 — FLAS Per-Client Compression Assignment
# Left:  scatter latency vs savings, coloured by ratio, marker = ok/missed
# Right: strip plot — all 50 clients as dots per ratio level (replaces bar chart)
# ---------------------------------------------------------------------------- #
def fig_c1_compression(c1, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))

    # ---- Left: scatter ----
    ax = axes[0]
    for r_val in [1.0, 0.75, 0.5, 0.25]:
        pts  = [x for x in c1 if x["chosen_r"] == r_val]
        lats = [x["est_lat"] for x in pts]
        saves= [x["savings"] for x in pts]
        oks  = [x["ok"]      for x in pts]
        color = R_COLORS[r_val]
        met_l = [l for l, o in zip(lats, oks) if o]
        met_s = [s for s, o in zip(saves, oks) if o]
        miss_l= [l for l, o in zip(lats, oks) if not o]
        miss_s= [s for s, o in zip(saves, oks) if not o]
        if met_l:
            ax.scatter(met_l,  met_s,  color=color, s=55, zorder=3,
                       label=R_LABELS[r_val])
        if miss_l:
            ax.scatter(miss_l, miss_s, color=color, s=75, marker="X",
                       zorder=3, edgecolors="k", linewidths=0.6)

    ax.axvline(0.5, color="0.35", linestyle="--", linewidth=1.4,
               label="Deadline (0.5 s)")
    ax.set_xlabel("Estimated Round Latency (s)")
    ax.set_ylabel("Bandwidth Savings (%)")
    ax.set_title("(a) Per-Client Compression Selection")
    # fold marker-type explanation into the legend as extra entries
    met_handle  = plt.Line2D([0], [0], marker="o", color="0.4", linestyle="none",
                             markersize=6, label="Filled = deadline met")
    miss_handle = plt.Line2D([0], [0], marker="X", color="0.4", linestyle="none",
                             markersize=6, markeredgecolor="k", markeredgewidth=0.5,
                             label="✕ = deadline missed")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [met_handle, miss_handle],
              labels  + ["Filled = deadline met", u"× = deadline missed"],
              fontsize=9, ncol=1, loc="lower right", handlelength=1.2)

    # ---- Right: strip plot (all 50 clients as dots per ratio) ----
    ax2 = axes[1]
    ax2.set_facecolor("white")
    rng = np.random.default_rng(42)
    r_vals_ordered = [1.0, 0.75, 0.5, 0.25]
    x_positions = {r: i for i, r in enumerate(r_vals_ordered)}

    for r_val in r_vals_ordered:
        pts   = [x for x in c1 if x["chosen_r"] == r_val]
        oks   = [x["ok"] for x in pts]
        count = len(pts)
        xi    = x_positions[r_val]
        jitter= rng.uniform(-0.18, 0.18, count)
        y_met  = [xi + j for j, o in zip(jitter, oks) if o]
        y_miss = [xi + j for j, o in zip(jitter, oks) if not o]
        ax2.scatter(y_met,  [1] * len(y_met),  color=R_COLORS[r_val],
                    s=55, zorder=3)
        ax2.scatter(y_miss, [1] * len(y_miss), color=R_COLORS[r_val],
                    s=75, marker="X", zorder=3, edgecolors="k", linewidths=0.6)
        # count label
        n_ok   = sum(oks)
        n_miss = count - n_ok
        label  = f"n={count}\n({n_ok} met, {n_miss} miss)"
        ax2.text(xi, 1.06, label, ha="center", va="bottom", fontsize=9,
                 color=R_COLORS[r_val])

    ax2.set_xlim(-0.5, len(r_vals_ordered) - 0.5)
    ax2.set_ylim(0.9, 1.2)
    ax2.set_xticks(range(len(r_vals_ordered)))
    ax2.set_xticklabels(
        ["r=1.00\n(none)", "r=0.75\n(25%)", "r=0.50\n(50%)", "r=0.25\n(75%)"],
        fontsize=10)
    ax2.set_yticks([])
    ax2.set_xlabel("Compression Ratio Assigned")
    ax2.set_title("(b) Client Distribution by Assigned Ratio (N = 50)")
    # horizontal grid off (y-axis is meaningless)
    ax2.yaxis.grid(False)
    ax2.xaxis.grid(True)

    fig.suptitle("FLAS: Latency-Aware Compression Selection",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(out_dir) / "fig_c1_compression.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------- #
# C2 — Adaptive Learning Rate (RALRS) vs Fixed LR
# Left:  loss curves (fixed annotations above each point)
# Right: diverging filled-area improvement curve (replaces bar chart)
# ---------------------------------------------------------------------------- #
def fig_c2_adaptive_lr(c2, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))

    taus   = [x["tau_i"] for x in c2]
    a_loss = [x["a_loss"] for x in c2]
    f_loss = [x["f_loss"] for x in c2]
    imp    = [x["imp"]    for x in c2]
    a_lr   = [x["a_lr"]  for x in c2]

    # ---- Left: loss lines ----
    ax = axes[0]
    ax.plot(taus, a_loss, color="#ff7f0e", marker="o", zorder=3,
            label="Adaptive LR (RALRS)")
    ax.plot(taus, f_loss, color="#1f77b4", linestyle="--", marker="s", zorder=3,
            label="Fixed LR = 0.01")
    # annotations: alternate above/below to avoid collision
    offsets_pts = [10, -16, 10, -16, 10]
    for tau, alr, aloss, yoff in zip(taus, a_lr, a_loss, offsets_pts):
        va = "bottom" if yoff > 0 else "top"
        ax.annotate(f"lr={alr:.4f}", xy=(tau, aloss),
                    xytext=(0, yoff), textcoords="offset points",
                    ha="center", va=va, fontsize=8, color="#cc5500",
                    arrowprops=dict(arrowstyle="-", color="#cc5500",
                                   lw=0.7, shrinkA=3, shrinkB=0))
    ax.set_xlabel(r"Target Epochs per Round $\tau$")
    ax.set_ylabel("Training Loss")
    ax.set_title(r"(a) Loss vs $\tau$: Adaptive vs Fixed LR")
    ax.legend(loc="lower right")
    ax.set_xticks(taus)
    ax.set_xticklabels([str(int(t)) for t in taus])
    # pad top so annotations don't clip
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.02, ymax + 0.04)

    # ---- Right: diverging filled-area gain curve ----
    ax2 = axes[1]
    taus_arr = np.array(taus)
    imp_arr  = np.array(imp)
    # fine interpolation for smooth fill
    tau_fine = np.linspace(taus_arr[0], taus_arr[-1], 300)
    imp_fine = np.interp(tau_fine, taus_arr, imp_arr)

    ax2.axhline(0, color="0.35", linewidth=1.2, zorder=2)
    ax2.fill_between(tau_fine, imp_fine, 0,
                     where=(imp_fine >= 0), alpha=0.25, color="#2ca02c",
                     label="Adaptive better")
    ax2.fill_between(tau_fine, imp_fine, 0,
                     where=(imp_fine <= 0), alpha=0.25, color="#d62728",
                     label="Fixed better")
    ax2.plot(tau_fine, imp_fine, color="#333333", linewidth=2.0, zorder=3)
    ax2.scatter(taus, imp, color=["#2ca02c" if v >= 0 else "#d62728" for v in imp],
                s=60, zorder=4)
    # annotate each point
    for tau, v in zip(taus, imp):
        yoff = 0.4 if v >= 0 else -0.5
        va   = "bottom" if v >= 0 else "top"
        ax2.text(tau, v + yoff, f"{v:+.2f}%", ha="center", va=va,
                 fontsize=9, fontweight="bold")
    ax2.set_xlabel(r"Target Epochs per Round $\tau$")
    ax2.set_ylabel("Loss Improvement over Fixed LR (%)")
    ax2.set_title(r"(b) RALRS Gain vs $\tau$")
    ax2.set_xticks(taus)
    ax2.set_xticklabels([str(int(t)) for t in taus])
    ax2.legend(loc="upper right")

    fig.suptitle("RALRS: Residual Adaptive Learning Rate Scheduling",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(out_dir) / "fig_c2_adaptive_lr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------- #
# C3 — Cost Model Calibration Convergence  (line charts — no change in type)
# ---------------------------------------------------------------------------- #
def fig_c3_calibration(c3, out_dir):
    rnds   = [x["rnd"]    for x in c3]
    kc_err = [x["kc_err"] for x in c3]
    km_err = [x["km_err"] for x in c3]
    kc_est = [x["kc_est"] for x in c3]
    km_est = [x["km_est"] for x in c3]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_DOUBLE, FIG_H))

    # ---- Left: error % (first 20 rounds) ----
    n  = 20
    ax = axes[0]
    ax.plot(rnds[:n], kc_err[:n], color="#ff7f0e", marker="o", markersize=5,
            label=r"$\hat{\kappa}_\mathrm{comp}$ error")
    ax.plot(rnds[:n], km_err[:n], color="#1f77b4", marker="s", markersize=5,
            label=r"$\hat{\kappa}_\mathrm{comm}$ error")
    ax.axvline(3, color="0.35", linestyle="--", linewidth=1.3,
               label="Round 3 (<1% error)")
    ax.axhspan(0, 1, alpha=0.08, color="#2ca02c", zorder=0)
    ax.text(3.3, 0.5, "<1% zone", fontsize=8, color="#2ca02c", va="center")
    ax.set_xlabel("Round")
    ax.set_ylabel("Estimation Error (%)")
    ax.set_title("(a) Cost Model Error (Rounds 1–20)")
    ax.legend(loc="upper right")
    ax.set_xlim(0.5, n + 0.5)

    # ---- Right: two y-axes so comp (0.1) and comm (0.5) are both readable ----
    ax2   = axes[1]
    color_c = "#ff7f0e"
    color_m = "#1f77b4"

    ln1, = ax2.plot(rnds, kc_est, color=color_c, linewidth=1.8,
                    label=r"$\hat{\kappa}_\mathrm{comp}$ (EMA)")
    ax2.axhline(0.1, color=color_c, linestyle="--", linewidth=1.2, alpha=0.7,
                label=r"True $\kappa_\mathrm{comp}=0.1$")
    ax2.set_xlabel("Round")
    ax2.set_ylabel(r"$\hat{\kappa}_\mathrm{comp}$", color=color_c)
    ax2.tick_params(axis="y", labelcolor=color_c)
    ax2.set_ylim(0.08, 0.14)
    ax2.set_title(r"(b) EMA Estimates vs True Values (100 Rounds)")

    ax3 = ax2.twinx()
    ax3.grid(False)
    ln2, = ax3.plot(rnds, km_est, color=color_m, linewidth=1.8,
                    label=r"$\hat{\kappa}_\mathrm{comm}$ (EMA)")
    ax3.axhline(0.5, color=color_m, linestyle="--", linewidth=1.2, alpha=0.7,
                label=r"True $\kappa_\mathrm{comm}=0.5$")
    ax3.set_ylabel(r"$\hat{\kappa}_\mathrm{comm}$", color=color_m)
    ax3.tick_params(axis="y", labelcolor=color_m)
    ax3.set_ylim(0.38, 0.56)

    lines = [ln1, ln2,
             plt.Line2D([0], [0], color=color_c, linestyle="--", linewidth=1.2, alpha=0.7),
             plt.Line2D([0], [0], color=color_m, linestyle="--", linewidth=1.2, alpha=0.7)]
    labels_leg = [r"$\hat{\kappa}_\mathrm{comp}$ (EMA)",
                  r"$\hat{\kappa}_\mathrm{comm}$ (EMA)",
                  r"True $\kappa_\mathrm{comp}=0.1$",
                  r"True $\kappa_\mathrm{comm}=0.5$"]
    ax2.legend(lines, labels_leg, fontsize=9, loc="center right")

    fig.suptitle("Cost Model Calibration Convergence",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(out_dir) / "fig_c3_calibration.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------- #
# C4 — Aggregation Method Comparison
# Completely redesigned: scatter of (divergence, loss) with 3 labeled points
# styled like the Pareto frontier figures used in the main results.
# ---------------------------------------------------------------------------- #
def fig_c4_aggregation(c4, out_dir):
    methods = [x["method"] for x in c4]
    losses  = np.array([x["loss"] for x in c4])
    divs    = np.array([x["div"]  for x in c4])
    ms_arr  = np.array([x["ms"]   for x in c4])
    colors  = [METHOD_COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_H))

    # shade "worse-than-best" region (upper-right of the best point)
    best_i = int(np.argmin(losses))
    bx, by = divs[best_i], losses[best_i]
    ax.fill_between(
        [bx, divs.max() + 0.1], [by, by],
        [losses.max() + 0.0002, losses.max() + 0.0002],
        color="#d62728", alpha=0.07, zorder=0,
        label="Dominated region")

    # dashed crosshair at best point
    ax.axvline(bx, color="0.5", linestyle=":", linewidth=1.2)
    ax.axhline(by, color="0.5", linestyle=":", linewidth=1.2)

    for method, loss, div, color in zip(methods, losses, divs, colors):
        ax.scatter(div, loss, color=color, s=160, zorder=4,
                   edgecolors="k", linewidths=0.8)

    # annotations — offset each label to avoid overlap
    offsets = {
        "Size-Weighted":     (-0.005, -0.00025),
        "Equal Weight":      (-0.005,  0.00012),
        "Loss-Inv-Weighted": ( 0.003,  0.00008),
    }
    for method, loss, div, color in zip(methods, losses, divs, colors):
        dx, dy = offsets.get(method, (0.003, 0.00008))
        ax.annotate(
            method,
            xy=(div, loss),
            xytext=(div + dx, loss + dy),
            fontsize=10, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9,
                            shrinkA=5, shrinkB=0))

    ax.set_xlabel("Divergence from Centralised Model (%)")
    ax.set_ylabel("Aggregated Model Loss")
    ax.set_title("Fidelity-Weighted Aggregation: Method Comparison")

    # x-axis: zoom in tight around the data
    x_pad = (divs.max() - divs.min()) * 0.6
    ax.set_xlim(divs.min() - x_pad, divs.max() + x_pad * 1.5)
    y_pad = (losses.max() - losses.min()) * 0.8
    ax.set_ylim(losses.min() - y_pad, losses.max() + y_pad * 1.2)

    # summary timing text box
    timing_str = "\n".join(
        [f"{m}: {t:.1f} ms" for m, t in zip(methods, ms_arr)])
    ax.text(0.98, 0.97, "Aggregation time\n" + timing_str,
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.6", alpha=0.9))

    dominated_patch = mpatches.Patch(color="#d62728", alpha=0.2,
                                     label="Dominated region")
    ax.legend(handles=[dominated_patch], loc="upper left")

    fig.tight_layout()
    out = Path(out_dir) / "fig_c4_aggregation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------- #
# C5 — End-to-End Pipeline Analysis
# Single-panel radar chart comparing FLASH vs FedAvg on 4 performance dimensions.
# ---------------------------------------------------------------------------- #
def fig_c5_pipeline(c5, out_dir):
    flash  = c5[0]   # mode="flash"
    fedavg = c5[1]   # mode="fedavg"

    categories = ["Deadline\nMet", "Bandwidth\nSaved", "Loss\nQuality", "Latency\nCompliance"]
    N = len(categories)

    # Normalise all dimensions to [0, 1] with 1 = best
    # Deadline met: raw fraction (0.92 vs 0.46)
    flash_deadline  = flash["deadline_pct"]
    fedavg_deadline = fedavg["deadline_pct"]

    # Bandwidth saved: 1 - avg_r  (lower ratio = more compression = more savings)
    flash_bw   = 1 - flash["avg_r"]   # 0.32
    fedavg_bw  = 1 - fedavg["avg_r"]  # 0.0

    # Loss quality: invert loss; normalise to [0, 1] across the two strategies
    inv_f = 1 / flash["loss"]
    inv_a = 1 / fedavg["loss"]
    flash_lq   = inv_f / (inv_f + inv_a)   # ~0.785
    fedavg_lq  = inv_a / (inv_f + inv_a)   # ~0.215

    # Latency compliance: fraction of rounds where avg_lat < threshold (0.5 s)
    # FLASH avg_lat=0.45 → compliant; FedAvg avg_lat=0.6421 → not
    # Express as normalised score: threshold / avg_lat, capped at 1
    thresh = 0.5
    flash_lat  = min(1.0, thresh / flash["avg_lat"])
    fedavg_lat = min(1.0, thresh / fedavg["avg_lat"])

    def clamp(v): return max(0.0, min(1.0, v))

    flash_vals  = [clamp(flash_deadline),  clamp(flash_bw),
                   clamp(flash_lq),        clamp(flash_lat)]
    fedavg_vals = [clamp(fedavg_deadline), clamp(fedavg_bw),
                   clamp(fedavg_lq),       clamp(fedavg_lat)]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    flash_vals  += [flash_vals[0]]
    fedavg_vals += [fedavg_vals[0]]
    angles      += [angles[0]]

    fig, ax = plt.subplots(figsize=(FIG_W_SINGLE, FIG_W_SINGLE * 0.82),
                           subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, flash_vals,  color="#ff7f0e", linewidth=2.4, label="FLASH")
    ax.fill(angles, flash_vals,  color="#ff7f0e", alpha=0.20)
    ax.plot(angles, fedavg_vals, color="#1f77b4", linewidth=2.4,
            linestyle="--", label="FedAvg")
    ax.fill(angles, fedavg_vals, color="#1f77b4", alpha=0.13)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=8.5)
    ax.yaxis.grid(True, linestyle="--", color="0.78", linewidth=0.7)
    ax.xaxis.grid(True, linestyle="--", color="0.78", linewidth=0.7)

    # annotate actual values next to each vertex
    angle_arr = np.array(angles[:-1])
    for vals, color in [(flash_vals, "#cc5500"), (fedavg_vals, "#0055aa")]:
        for ang, val in zip(angle_arr, vals[:-1]):
            x = (val + 0.12) * np.cos(ang - np.pi / 2)
            y = (val + 0.12) * np.sin(ang - np.pi / 2)
            # convert polar to Cartesian for annotation
            ax.annotate(f"{val:.2f}",
                        xy=(ang, val), xytext=(ang, val + 0.13),
                        fontsize=8, color=color, ha="center", va="center")

    ax.set_title("Performance Profile: FLASH vs FedAvg\n(normalised, higher = better)",
                 fontsize=12, pad=20)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=11, framealpha=0.9)

    fig.tight_layout()
    out = Path(out_dir) / "fig_c5_pipeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Generate FLASH component study figures.")
    parser.add_argument("--data",
                        default="FL_FLASH_Component_Eval/component_eval_results.json")
    parser.add_argument("--out-dir", default="component_study")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.data) as f:
        data = json.load(f)

    print(f"Generating component study figures -> {args.out_dir}/")
    fig_c1_compression(data["c1_compression"], args.out_dir)
    fig_c2_adaptive_lr(data["c2_adaptive_lr"], args.out_dir)
    fig_c3_calibration(data["c3_calibration"], args.out_dir)
    fig_c4_aggregation(data["c4_aggregation"], args.out_dir)
    fig_c5_pipeline(data["c5_pipeline"],       args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
