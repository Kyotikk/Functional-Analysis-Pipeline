"""
run_plots.py  –  Generate presentation-quality figures from the functional-analysis pipeline.

Plots produced
--------------
1. icf_dual_axis.png       – ICF scatter per domain (effort vs capacity, coloured by R4 stage)
2. correlation_summary.png – Spearman ρ bar chart per domain × modality with CI whiskers
3. effort_by_r4_stage.png  – Box plots of effort distribution per R4 stage and domain
4. effort_variability_by_r4_and_domain.png – Effort spread within R4 classes
5. clinical_case_comparison.png – Two individuals with similar capacity but different effort profiles
6. modality_correlation_heatmap.png – Heatmap of modality-level Spearman ρ across domains
5. effort_variability_by_r4_and_domain.png – Effort spread within R4 classes, including domain-wise grouped boxplots

Usage
-----
    python run_plots.py \
        --combined-csv  output/correlation/run2/combined_analysis.csv \
        --correlation-csv output/correlation/run2/r4_correlation.csv \
        --output-dir    output/plots/run1

Defaults match the paths above so the script works with no arguments if run from
the functional-analysis-pipeline directory.
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── colour palettes ───────────────────────────────────────────────────────────
R4_COLOURS = {
    5: "#8cff00",   # green  – healthy
    4: "#f1c40f",   # yellow – mild impairment
    3: "#e97132",   # orange – moderate
    2: "#bb1301",   # red    – severe
    1: "#510000",   # purple – very severe
}

MODALITY_COLOURS = {
    "hr_hrv":    "#e74c3c",
    "eda":       "#3498db",
    "imu_wrist": "#2ecc71",
    "imu_chest": "#9b59b6",
    "overall":   "#34495e",
}

DOMAIN_ORDER = ["Basic Movements", "Walking", "Oral Care", "Grooming"]


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 – ICF dual-axis scatter
# ─────────────────────────────────────────────────────────────────────────────
def plot_icf_dual_axis(combined: pd.DataFrame, output_dir: Path) -> None:
    """
    Per-domain scatter: x = physiological effort (-100..100), y = capacity stage (1–5).
    Points coloured by clinical R4 stage. Quadrant lines + labels added.
    Clinical case pair highlighted with bold edge styling.
    """
    # Find clinical case pair (same logic as plot_clinical_case_comparison)
    sub_all = combined.dropna(subset=["effort", "r4_stage", "capacity_stage", "domain"]).copy()
    subj_list = sub_all["subject_id"].unique()
    case_pair = None
    best_score = 0

    for i, s1 in enumerate(subj_list):
        s1_data = sub_all[sub_all["subject_id"] == s1]
        s1_cap = s1_data["capacity_stage"].mean()
        s1_eff = s1_data["effort"].mean()

        for s2 in subj_list[i+1:]:
            s2_data = sub_all[sub_all["subject_id"] == s2]
            s2_cap = s2_data["capacity_stage"].mean()
            s2_eff = s2_data["effort"].mean()

            cap_sim = 1.0 - abs(s1_cap - s2_cap) / 5.0
            eff_diff = abs(s1_eff - s2_eff)
            score = cap_sim * (1 + eff_diff / 200.0)

            if score > best_score and eff_diff > 30:
                best_score = score
                case_pair = {s1, s2}

    domains = [d for d in DOMAIN_ORDER if d in combined["domain"].unique()]
    n = len(domains)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 7 * nrows),
                             squeeze=False)
    fig.suptitle("ICF Dual-Axis: Physiological Effort vs Functional Capacity",
                 fontsize=16, fontweight="bold", y=1.03)

    # quadrant labels (effort midpoint = 0, capacity midpoint = 3)
    quadrant_props = dict(fontsize=8, ha="center", va="center",
                          color="grey", style="italic", alpha=0.6)
    quadrant_def = [
        # (x, y, label)
        (-55, 4.5, "Healthy\n(low effort, high capacity)"),
        (55,  4.5, "Compensating\n(high effort, high capacity)"),
        (-55, 1.5, "Coasting\n(low effort, low capacity)"),
        (55,  1.5, "Impaired\n(high effort, low capacity)"),
    ]

    for idx, domain in enumerate(domains):
        ax = axes[idx // ncols][idx % ncols]
        sub = combined[combined["domain"] == domain].dropna(
            subset=["effort", "capacity_stage"])

        if sub.empty:
            ax.set_visible(False)
            continue

        # quadrant lines
        ax.axvline(0, color="lightgrey", lw=1, ls="--", zorder=0)
        ax.axhline(3,   color="lightgrey", lw=1, ls="--", zorder=0)

        # quadrant text
        for qx, qy, ql in quadrant_def:
            ax.text(qx, qy, ql, **quadrant_props, transform=ax.transData)

        # scatter – add jitter on capacity_stage (integer) for readability
        rng = np.random.default_rng(42)
        jitter_y = rng.uniform(-0.15, 0.15, len(sub))
        jitter_x = rng.uniform(-3, 3, len(sub))


        # Always include all R4 stages present in this domain in the legend, even if only a highlighted subject has that stage
        r4_stages_present = sorted(sub["r4_stage"].dropna().unique())
        legend_handles = []
        for r4 in r4_stages_present:
            mask = sub["r4_stage"] == r4
            # Separate clinical case subjects from regular subjects
            if case_pair:
                case_mask = mask & sub["subject_id"].isin(case_pair)
                regular_mask = mask & ~sub["subject_id"].isin(case_pair)
            else:
                case_mask = pd.Series([False] * len(mask))
                regular_mask = mask

            # Plot regular subjects (no legend label here)
            if regular_mask.any():
                ax.scatter(
                    sub.loc[regular_mask, "effort"] + jitter_x[regular_mask.values],
                    sub.loc[regular_mask, "capacity_stage"] + jitter_y[regular_mask.values],
                    c=R4_COLOURS.get(int(r4), "grey"),
                    s=70, edgecolors="white", linewidths=0.5, zorder=3,
                )

            # Plot clinical case subjects with bold edge styling
            if case_mask.any():
                ax.scatter(
                    sub.loc[case_mask, "effort"] + jitter_x[case_mask.values],
                    sub.loc[case_mask, "capacity_stage"] + jitter_y[case_mask.values],
                    c=R4_COLOURS.get(int(r4), "grey"),
                    s=70, edgecolors="black", linewidths=2.0, zorder=4,
                )

            # Add to legend (one handle per R4 stage present)
            legend_handles.append(
                mpatches.Patch(color=R4_COLOURS.get(int(r4), "grey"), label=f"R4 stage {int(r4)}")
            )

        # Add legend for this subplot with all present R4 stages
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.7)

        ax.set_xlim(-105, 105)
        ax.set_ylim(0.5, 5.5)
        ax.set_xlabel("Physiological Effort Score (-100 to +100, 0 = HC baseline)")
        ax.set_ylabel("Sensor-Derived Capacity Stage")
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_title(domain, fontweight="bold")

        # reliable vs unreliable styling via alpha patches
        if "effort_reliable" in sub.columns:
            unreliable = sub[sub["effort_reliable"] == False]
            if not unreliable.empty:
                ax.scatter(
                    unreliable["effort"],
                    unreliable["capacity_stage"],
                    s=70, facecolors="none", edgecolors="grey",
                    linewidths=1.5, zorder=4, label="Unreliable effort",
                )

        ax.legend(fontsize=8, loc="lower right", framealpha=0.7)

    # hide unused panels
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # shared R4 legend
    handles = [mpatches.Patch(color=c, label=f"R4 stage {k}")
               for k, c in sorted(R4_COLOURS.items())]
    fig.legend(handles=handles, title="Clinical R4 stage",
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03),
               framealpha=0.7)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.10, hspace=0.32, wspace=0.22)
    out = output_dir / "icf_dual_axis.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 – HC vs Nursing Home domain effort comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_hc_vs_nh_domain_effort(hc_csv: Path, nh_csv: Path, output_dir: Path) -> None:
    """
    Side-by-side domain effort comparison between healthy controls (HC)
    and nursing home (NH) cohort.
    """
    hc_df = pd.read_csv(hc_csv)
    nh_df = pd.read_csv(nh_csv)

    domain_cols = [
        (d, f"{d}_effort")
        for d in DOMAIN_ORDER
        if f"{d}_effort" in hc_df.columns and f"{d}_effort" in nh_df.columns
    ]
    if not domain_cols:
        print("  [skip] No shared '<domain>_effort' columns found in HC/NH files.")
        return

    fig, axes = plt.subplots(1, len(domain_cols), figsize=(5.2 * len(domain_cols), 6), squeeze=False)
    fig.suptitle("Domain Effort Comparison: HC vs Nursing Home",
                 fontsize=15, fontweight="bold", y=0.99)

    rng = np.random.default_rng(42)
    hc_color = "#1f77b4"
    nh_color = "#e97132"

    for idx, (domain, col) in enumerate(domain_cols):
        ax = axes[0][idx]
        hc_vals = pd.to_numeric(hc_df[col], errors="coerce").dropna().values
        nh_vals = pd.to_numeric(nh_df[col], errors="coerce").dropna().values
        if len(hc_vals) == 0 or len(nh_vals) == 0:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            [hc_vals, nh_vals],
            positions=[0, 1],
            widths=0.55,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color="grey"),
            capprops=dict(color="grey"),
            flierprops=dict(marker="o", markerfacecolor="grey", markersize=4, alpha=0.35),
        )
        bp["boxes"][0].set_facecolor(hc_color)
        bp["boxes"][1].set_facecolor(nh_color)
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_alpha(0.75)

        jitter_hc = rng.uniform(-0.09, 0.09, len(hc_vals))
        jitter_nh = rng.uniform(-0.09, 0.09, len(nh_vals))
        ax.scatter(np.zeros(len(hc_vals)) + jitter_hc, hc_vals, s=14, color=hc_color,
                   alpha=0.45, edgecolors="none", zorder=3)
        ax.scatter(np.ones(len(nh_vals)) + jitter_nh, nh_vals, s=14, color=nh_color,
                   alpha=0.45, edgecolors="none", zorder=3)

        hc_mean = float(np.mean(hc_vals))
        nh_mean = float(np.mean(nh_vals))
        delta = nh_mean - hc_mean
        ax.scatter([0, 1], [hc_mean, nh_mean], marker="D", s=45, color="black", zorder=4)
        ax.text(0.5, 0.97, f"Δ mean = {delta:+.1f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["HC", "Nursing Home"])
        ax.set_ylim(-105, 105)
        ax.set_title(domain, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Effort Score (-100 to +100)")

    legend_handles = [
        mpatches.Patch(facecolor=hc_color, edgecolor="none", label="Healthy Control"),
        mpatches.Patch(facecolor=nh_color, edgecolor="none", label="Nursing Home"),
        Line2D([0], [0], marker="D", color="black", lw=0, markersize=6, label="Mean"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.8)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.87, bottom=0.16, wspace=0.25)
    out = output_dir / "hc_vs_nursing_home_domain_effort.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    diff = np.subtract.outer(x, y)
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(x) * len(y)))


def plot_hc_vs_nh_domain_effort_stats(hc_csv: Path, nh_csv: Path, output_dir: Path) -> None:
    """HC vs NH domain effort with p-values and Cliff's delta annotations."""
    hc_df = pd.read_csv(hc_csv)
    nh_df = pd.read_csv(nh_csv)

    domain_cols = [
        (d, f"{d}_effort")
        for d in DOMAIN_ORDER
        if f"{d}_effort" in hc_df.columns and f"{d}_effort" in nh_df.columns
    ]
    if not domain_cols:
        print("  [skip] No shared '<domain>_effort' columns found in HC/NH files.")
        return

    fig, axes = plt.subplots(1, len(domain_cols), figsize=(5.6 * len(domain_cols), 6.4), squeeze=False)
    fig.suptitle("Healthy Control vs Nursing Home Effort by Domain (with Statistics)",
                 fontsize=15, fontweight="bold", y=0.99)

    rng = np.random.default_rng(42)
    hc_color = "#1f77b4"
    nh_color = "#e97132"

    for idx, (domain, col) in enumerate(domain_cols):
        ax = axes[0][idx]
        hc_vals = pd.to_numeric(hc_df[col], errors="coerce").dropna().values
        nh_vals = pd.to_numeric(nh_df[col], errors="coerce").dropna().values
        if len(hc_vals) == 0 or len(nh_vals) == 0:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            [hc_vals, nh_vals],
            positions=[0, 1],
            widths=0.52,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color="grey"),
            capprops=dict(color="grey"),
            flierprops=dict(marker="o", markerfacecolor="grey", markersize=4, alpha=0.3),
        )
        bp["boxes"][0].set_facecolor(hc_color)
        bp["boxes"][1].set_facecolor(nh_color)
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_alpha(0.75)

        ax.scatter(np.zeros(len(hc_vals)) + rng.uniform(-0.09, 0.09, len(hc_vals)), hc_vals,
                   s=14, color=hc_color, alpha=0.35, edgecolors="none", zorder=3)
        ax.scatter(np.ones(len(nh_vals)) + rng.uniform(-0.09, 0.09, len(nh_vals)), nh_vals,
                   s=14, color=nh_color, alpha=0.35, edgecolors="none", zorder=3)

        stat = mannwhitneyu(hc_vals, nh_vals, alternative="two-sided")
        p_val = float(stat.pvalue)
        star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        cliffs = _cliffs_delta(nh_vals, hc_vals)
        delta_mean = float(np.mean(nh_vals) - np.mean(hc_vals))

        ax.text(0.5, 0.1, f"Δmean={delta_mean:+.1f} | p={p_val:.3g} ({star})\nCliff's δ={cliffs:+.2f}",
                transform=ax.transAxes, ha="center", va="center", fontsize=9.0, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["HC", "NH"])
        ax.set_ylim(-105, 105)
        ax.set_title(domain, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Effort Score (-100 to +100)")

    legend_handles = [
        mpatches.Patch(facecolor=hc_color, edgecolor="none", label="Healthy Control"),
        mpatches.Patch(facecolor=nh_color, edgecolor="none", label="Nursing Home"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.025), framealpha=0.8)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.18, wspace=0.3)
    out = output_dir / "hc_vs_nursing_home_domain_effort_stats.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out}")


def plot_walking_stage_proxy_comparison(combined: pd.DataFrame, hc_csv: Path, output_dir: Path) -> None:
    """
    Walking effort comparison using stage-based proxy groups:
    - NH Walking stage 2 as wheelchair proxy
    - NH Walking stage 3 as walker proxy
    Compared against HC walking baseline.
    """
    hc_df = pd.read_csv(hc_csv)
    if "Walking_effort" not in hc_df.columns:
        print("  [skip] HC CSV missing Walking_effort column.")
        return

    walking = combined[combined["domain"] == "Walking"].dropna(subset=["effort", "capacity_stage"]).copy()
    if walking.empty:
        print("  [skip] No walking rows in combined data.")
        return

    hc_vals = pd.to_numeric(hc_df["Walking_effort"], errors="coerce").dropna().values
    nh_stage2 = walking[walking["capacity_stage"].astype(int) == 2]["effort"].values
    nh_stage3 = walking[walking["capacity_stage"].astype(int) == 3]["effort"].values

    if len(hc_vals) == 0 or (len(nh_stage2) == 0 and len(nh_stage3) == 0):
        print("  [skip] Missing data for walking stage proxy comparison.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), squeeze=False)
    ax_a, ax_b = axes[0]
    fig.suptitle("Walking Effort: HC Baseline vs NH Mobility-Proxy Groups",
                 fontsize=15, fontweight="bold", y=0.99)

    groups = [hc_vals, nh_stage2, nh_stage3]
    labels = ["HC", "NH stage2\n(wheelchair proxy)", "NH stage3\n(walker proxy)"]
    colors = ["#1f77b4", "#bb1301", "#e97132"]

    bp = ax_a.boxplot(
        groups,
        positions=[0, 1, 2],
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(color="grey"),
        capprops=dict(color="grey"),
        flierprops=dict(marker="o", markerfacecolor="grey", markersize=4, alpha=0.35),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(groups):
        if len(vals) == 0:
            continue
        ax_a.scatter(np.full(len(vals), i) + rng.uniform(-0.1, 0.1, len(vals)), vals,
                     s=14, color=colors[i], alpha=0.38, edgecolors="none", zorder=3)

    ax_a.set_xticks([0, 1, 2])
    ax_a.set_xticklabels(labels)
    ax_a.set_ylim(-105, 105)
    ax_a.set_ylabel("Effort Score (-100 to +100)")
    ax_a.set_title("A  Walking Effort Distributions", fontweight="bold", loc="center")
    ax_a.grid(axis="y", alpha=0.3)

    # Panel B: direct NH proxy contrast + summary stats
    valid_nh = []
    valid_lbl = []
    valid_col = []
    if len(nh_stage2) > 0:
        valid_nh.append(nh_stage2)
        valid_lbl.append("Stage2\n(wheelchair)")
        valid_col.append("#bb1301")
    if len(nh_stage3) > 0:
        valid_nh.append(nh_stage3)
        valid_lbl.append("Stage3\n(walker)")
        valid_col.append("#e97132")

    if valid_nh:
        bp2 = ax_b.boxplot(
            valid_nh,
            positions=np.arange(len(valid_nh)),
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color="grey"),
            capprops=dict(color="grey"),
            flierprops=dict(marker="o", markerfacecolor="grey", markersize=4, alpha=0.35),
        )
        for patch, c in zip(bp2["boxes"], valid_col):
            patch.set_facecolor(c)
            patch.set_alpha(0.78)

        ax_b.set_xticks(np.arange(len(valid_nh)))
        ax_b.set_xticklabels(valid_lbl)
        ax_b.set_ylim(-105, 105)
        ax_b.set_title("B  NH Mobility-Proxy Contrast", fontweight="bold", loc="center")
        ax_b.grid(axis="y", alpha=0.3)

        txt = [
            f"HC mean (walking): {np.mean(hc_vals):.1f}",
            f"NH stage2 mean: {np.mean(nh_stage2):.1f}" if len(nh_stage2) > 0 else "NH stage2 mean: n/a",
            f"NH stage3 mean: {np.mean(nh_stage3):.1f}" if len(nh_stage3) > 0 else "NH stage3 mean: n/a",
        ]
        if len(nh_stage2) > 0 and len(nh_stage3) > 0:
            mw = mannwhitneyu(nh_stage2, nh_stage3, alternative="two-sided")
            txt.append(f"stage2 vs stage3 p={mw.pvalue:.3g}")
        ax_b.text(0.03, 0.2, "\n".join(txt), transform=ax_b.transAxes,
                  ha="left", va="top", fontsize=9.5,
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="lightgrey"))

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14, wspace=0.26)
    out = output_dir / "walking_stage_proxy_effort_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 – Spearman ρ bar chart with CI whiskers
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_summary(corr: pd.DataFrame, output_dir: Path) -> None:
    """
    Horizontal bar chart of Spearman ρ per domain.
    Separate panels for each comparison (effort_vs_r4, capacity_vs_r4, …).
    Modality shown via colour; CI as whiskers; significant bars starred.
    """
    comparisons = corr["comparison"].unique()
    n = len(comparisons)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=False,
                             squeeze=False)
    fig.suptitle("Spearman ρ vs Clinical R4 Stage", fontsize=14,
                 fontweight="bold")

    MODALITY_LABELS = {
        "overall":   "Overall",
        "hr_hrv":    "HR / HRV",
        "eda":       "EDA",
        "imu_wrist": "IMU Wrist",
        "imu_chest": "IMU Chest (BioZ)",
    }

    for col_idx, comparison in enumerate(comparisons):
        ax = axes[0][col_idx]
        sub = corr[corr["comparison"] == comparison].copy()

        # build y-axis labels: "Domain · Modality"
        sub["label"] = sub.apply(
            lambda r: f"{r['domain']}\n({MODALITY_LABELS.get(r['modality'], r['modality'])})",
            axis=1,
        )
        sub = sub.sort_values(["domain", "modality"])
        sub = sub.reset_index(drop=True)

        y_pos = np.arange(len(sub))
        colours = [MODALITY_COLOURS.get(m, "grey") for m in sub["modality"]]

        bars = ax.barh(y_pos, sub["spearman_r"], color=colours,
                       alpha=0.8, height=0.65, zorder=2)

        # CI whiskers
        ax.errorbar(
            sub["spearman_r"], y_pos,
            xerr=[
                sub["spearman_r"] - sub["spearman_ci_lo"],
                sub["spearman_ci_hi"] - sub["spearman_r"],
            ],
            fmt="none", color="black", capsize=4, lw=1.2, zorder=3,
        )

        # significance stars
        for i, row in sub.iterrows():
            idx = sub.index.get_loc(i)
            p = row["spearman_p"]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if star:
                x = row["spearman_ci_hi"] + 0.03
                ax.text(x, idx, star, va="center", fontsize=10,
                        color="black", fontweight="bold")

        # reference line
        ax.axvline(0, color="black", lw=0.8, zorder=1)
        ax.set_xlim(-1.1, 1.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["label"], fontsize=9)
        ax.set_xlabel("Spearman ρ")
        ax.set_title(comparison.replace("_", " ").title(), fontweight="bold")
        ax.invert_yaxis()

        # shade grey background per domain
        domain_changes = [0] + list(
            (sub["domain"] != sub["domain"].shift()).cumsum().diff().fillna(1).eq(1).to_numpy().nonzero()[0]
        ) + [len(sub)]
        for di in range(len(domain_changes) - 1):
            start = domain_changes[di] - 0.5
            end   = domain_changes[di + 1] - 0.5
            if di % 2 == 1:
                ax.axhspan(start, end, color="whitesmoke", zorder=0)

    # modality legend
    mod_handles = [mpatches.Patch(color=c, label=MODALITY_LABELS.get(m, m))
                   for m, c in MODALITY_COLOURS.items()]
    fig.legend(handles=mod_handles, title="Modality", loc="lower center",
               ncol=len(mod_handles), bbox_to_anchor=(0.5, -0.05),
               framealpha=0.7)

    fig.tight_layout()
    out = output_dir / "correlation_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 – Effort distribution by R4 stage (box plots)
# ─────────────────────────────────────────────────────────────────────────────
def plot_effort_by_r4_stage(combined: pd.DataFrame, output_dir: Path) -> None:
    """
    Box plots: one panel per domain, x = R4 stage, y = effort score.
    Individual points overlaid.
    """
    domains = [d for d in DOMAIN_ORDER if d in combined["domain"].unique()]
    n = len(domains)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    fig.suptitle("Physiological Effort by Clinical R4 Stage",
                 fontsize=14, fontweight="bold")

    for idx, domain in enumerate(domains):
        ax = axes[0][idx]
        sub = combined[(combined["domain"] == domain)].dropna(
            subset=["effort", "r4_stage"])

        stages = sorted(sub["r4_stage"].unique())
        data_groups = [sub[sub["r4_stage"] == s]["effort"].values for s in stages]

        bp = ax.boxplot(
            data_groups,
            positions=range(len(stages)),
            widths=0.45,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(color="grey"),
            capprops=dict(color="grey"),
            flierprops=dict(marker="o", markerfacecolor="grey",
                            markersize=4, alpha=0.5),
        )

        for patch, s in zip(bp["boxes"], stages):
            patch.set_facecolor(R4_COLOURS.get(int(s), "grey"))
            patch.set_alpha(0.7)

        # overlay individual points with jitter
        rng = np.random.default_rng(42)
        for gi, (s, vals) in enumerate(zip(stages, data_groups)):
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(gi + jitter, vals,
                       c=R4_COLOURS.get(int(s), "grey"),
                       s=30, zorder=3, edgecolors="white", lw=0.5)

        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"R4-{int(s)}" for s in stages])
        ax.set_xlabel("Clinical R4 Stage")
        ax.set_ylabel("Physiological Effort (-100 to +100)" if idx == 0 else "")
        ax.set_ylim(-105, 105)
        ax.set_title(domain, fontweight="bold")
        ax.axhline(0, color="lightgrey", ls="--", lw=1, zorder=0)

    fig.tight_layout()
    out = output_dir / "effort_by_r4_stage.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3b – Effort variability within R4 class (with domain grouping)
# ─────────────────────────────────────────────────────────────────────────────
def plot_effort_variability_by_class(combined: pd.DataFrame, output_dir: Path) -> None:
    """
    Two-panel visualization focused on within-class spread.
      A) Effort vs R4 class (all domains pooled)
      B) Domain on x-axis, grouped boxplots colored by R4 class
    """
    sub = combined.dropna(subset=["effort", "r4_stage", "domain"]).copy()
    if sub.empty:
        print("  [skip] No effort/r4 data available for variability plot.")
        return

    sub["r4_stage"] = sub["r4_stage"].astype(int)
    domains = [d for d in DOMAIN_ORDER if d in sub["domain"].unique()]
    stages = sorted([s for s in sub["r4_stage"].unique() if s in R4_COLOURS])
    if not domains or not stages:
        print("  [skip] Missing domains or R4 stages for variability plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), squeeze=False)
    ax_a, ax_b = axes[0]
    fig.suptitle("Effort Score Variability Within Clinical R4 Classes",
                 fontsize=14, fontweight="bold")

    # Panel A: pooled by R4 class
    pooled_groups = [sub[sub["r4_stage"] == s]["effort"].values for s in stages]
    bp = ax_a.boxplot(
        pooled_groups,
        positions=np.arange(len(stages)),
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(color="grey"),
        capprops=dict(color="grey"),
        flierprops=dict(marker="o", markerfacecolor="grey", markersize=4, alpha=0.5),
    )
    for patch, s in zip(bp["boxes"], stages):
        patch.set_facecolor(R4_COLOURS.get(int(s), "grey"))
        patch.set_alpha(0.75)

    rng = np.random.default_rng(42)
    for i, (s, vals) in enumerate(zip(stages, pooled_groups)):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax_a.scatter(i + jitter, vals, s=18, color=R4_COLOURS[int(s)],
                     alpha=0.45, edgecolors="none", zorder=3)

    ax_a.set_xticks(np.arange(len(stages)))
    ax_a.set_xticklabels([f"R4-{s}" for s in stages])
    ax_a.set_xlabel("Clinical R4 Class")
    ax_a.set_ylabel("Computed Effort Score")
    ax_a.set_ylim(-105, 105)
    ax_a.set_title("A  Pooled Across Domains", fontweight="bold", loc="left")
    ax_a.grid(axis="y", color="lightgrey", alpha=0.6)

    # Panel B: grouped by domain, colored by R4 class
    x_base = np.arange(len(domains))
    n_stage = len(stages)
    group_width = 0.84
    box_w = group_width / max(n_stage, 1)

    legend_handles = []
    for j, s in enumerate(stages):
        pos = x_base - group_width / 2 + (j + 0.5) * box_w
        data_groups = [
            sub[(sub["domain"] == d) & (sub["r4_stage"] == s)]["effort"].values
            for d in domains
        ]

        bp_dom = ax_b.boxplot(
            data_groups,
            positions=pos,
            widths=box_w * 0.9,
            patch_artist=True,
            medianprops=dict(color="black", lw=1.5),
            whiskerprops=dict(color="grey", lw=1),
            capprops=dict(color="grey", lw=1),
            flierprops=dict(marker="o", markerfacecolor="grey", markersize=3, alpha=0.4),
        )
        for patch in bp_dom["boxes"]:
            patch.set_facecolor(R4_COLOURS[int(s)])
            patch.set_alpha(0.75)

        # sparse point overlay for visibility
        for d_idx, vals in enumerate(data_groups):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-box_w * 0.18, box_w * 0.18, len(vals))
            ax_b.scatter(np.full(len(vals), pos[d_idx]) + jitter, vals,
                         s=10, color=R4_COLOURS[int(s)], alpha=0.35,
                         edgecolors="none", zorder=3)

        legend_handles.append(
            mpatches.Patch(facecolor=R4_COLOURS[int(s)], edgecolor="none", label=f"R4-{s}")
        )

    ax_b.set_xticks(x_base)
    ax_b.set_xticklabels(domains)
    ax_b.set_xlabel("Domain")
    ax_b.set_ylabel("Computed Effort Score")
    ax_b.set_ylim(-105, 105)
    ax_b.set_title("B  Within Each Domain, Split by R4 Class", fontweight="bold", loc="left")
    ax_b.grid(axis="y", color="lightgrey", alpha=0.6)
    ax_b.legend(handles=legend_handles, title="Clinical R4", ncol=min(5, len(stages)),
                loc="upper center", bbox_to_anchor=(0.5, 1.02), framealpha=0.8)

    # source annotation
    fig.text(0.01, 0.01, "Data: combined_analysis.csv (effort, r4_stage, domain)",
             fontsize=8, color="dimgray")

    fig.tight_layout()
    out = output_dir / "effort_variability_by_r4_and_domain.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4b – Clinical case comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_clinical_case_comparison(combined: pd.DataFrame, output_dir: Path) -> None:
    """
    Visualize two individuals with similar sensor-derived capacity but
    different physiological effort profiles. Highlights clinical value of effort scoring.
    """
    sub = combined.dropna(subset=["effort", "r4_stage", "capacity_stage", "domain"]).copy()
    if sub.empty:
        print("  [skip] No data for case comparison.")
        return

    # Find pair: similar mean capacity, max effort difference
    subj_list = sub["subject_id"].unique()
    best_pair = None
    best_score = 0

    for i, s1 in enumerate(subj_list):
        s1_data = sub[sub["subject_id"] == s1]
        s1_cap = s1_data["capacity_stage"].mean()
        s1_eff = s1_data["effort"].mean()

        for s2 in subj_list[i+1:]:
            s2_data = sub[sub["subject_id"] == s2]
            s2_cap = s2_data["capacity_stage"].mean()
            s2_eff = s2_data["effort"].mean()

            # Score: same capacity, different effort
            cap_sim = 1.0 - abs(s1_cap - s2_cap) / 5.0  # normalize by max stage
            eff_diff = abs(s1_eff - s2_eff)
            score = cap_sim * (1 + eff_diff / 200.0)

            if score > best_score and eff_diff > 30:
                best_score = score
                best_pair = (s1, s2, s1_cap, s2_cap, s1_eff, s2_eff)

    if best_pair is None:
        print("  [skip] No suitable case pair found.")
        return

    subj_a, subj_b, cap_a, cap_b, eff_a, eff_b = best_pair

    # Prepare data for both subjects
    data_a = sub[sub["subject_id"] == subj_a].set_index("domain")
    data_b = sub[sub["subject_id"] == subj_b].set_index("domain")
    domains = [d for d in DOMAIN_ORDER if d in data_a.index or d in data_b.index]

    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(
        f"Clinical Case Comparison: Similar Capacity, Different Effort Profiles\n"
        f"{subj_a.upper()} (efficient) vs {subj_b.upper()} (compensatory)",
        fontsize=16, fontweight="bold", y=1.03,
    )

    gs = fig.add_gridspec(1, 3)

    # Panel A: Effort by domain
    ax_a = fig.add_subplot(gs[0, 0])
    efforts_a = [data_a.loc[d, "effort"] if d in data_a.index else np.nan for d in domains]
    efforts_b = [data_b.loc[d, "effort"] if d in data_b.index else np.nan for d in domains]
    x = np.arange(len(domains))
    w = 0.35
    ax_a.bar(x - w/2, efforts_a, w, label=subj_a.upper(), color="#510000", alpha=0.8)
    ax_a.bar(x + w/2, efforts_b, w, label=subj_b.upper(), color="#8cff00", alpha=0.8)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(domains, rotation=35, ha="right")
    ax_a.set_ylabel("Effort Score (-100 to +100)")
    finite_eff = np.array(efforts_a + efforts_b, dtype=float)
    finite_eff = finite_eff[np.isfinite(finite_eff)]
    min_eff = float(np.min(finite_eff)) if len(finite_eff) else -100.0
    max_eff = float(np.max(finite_eff)) if len(finite_eff) else 100.0
    ax_a.set_ylim(min(-105, min_eff * 1.15), max(105, max_eff * 1.15))
    ax_a.set_title("A  Effort by Domain", fontweight="bold", loc="left")
    ax_a.legend()
    ax_a.grid(axis="y", alpha=0.3)

    # Panel B: Capacity & R4 by domain
    ax_b = fig.add_subplot(gs[0, 1])
    caps_a = [data_a.loc[d, "capacity_stage"] if d in data_a.index else np.nan for d in domains]
    caps_b = [data_b.loc[d, "capacity_stage"] if d in data_b.index else np.nan for d in domains]

    ax_b.plot(x, caps_a, "o-", label=f"{subj_a} Capacity", linewidth=2, markersize=7, color="#510000")
    ax_b.plot(x, caps_b, "s-", label=f"{subj_b} Capacity", linewidth=2, markersize=7, color="#8cff00")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(domains, rotation=35, ha="right")
    ax_b.set_ylabel("Stage (1–5)")
    ax_b.set_ylim(0.5, 5.5)
    ax_b.set_title("B  Sensor Capacity & Clinical R4 Stage", fontweight="bold", loc="left")
    ax_b.legend(fontsize=8, loc="lower right")
    ax_b.grid(alpha=0.3)

    # Panel C: Effort distribution across all domains (pooled)
    ax_c = fig.add_subplot(gs[0, 2])
    all_efforts_a = data_a["effort"].dropna().values
    all_efforts_b = data_b["effort"].dropna().values

    ax_c.scatter(
        np.random.normal(1, 0.04, len(all_efforts_a)),
        all_efforts_a,
        s=100, alpha=0.6, color="#510000", label=subj_a.upper(), edgecolors="white", lw=1,
    )
    ax_c.scatter(
        np.random.normal(2, 0.04, len(all_efforts_b)),
        all_efforts_b,
        s=100, alpha=0.6, color="#8cff00", label=subj_b.upper(), edgecolors="white", lw=1,
    )

    # Add means
    ax_c.hlines(eff_a, 0.8, 1.2, colors="#510000", linewidth=3, linestyle="--", label=f"{subj_a} mean")
    ax_c.hlines(eff_b, 1.8, 2.2, colors="#8cff00", linewidth=3, linestyle="--", label=f"{subj_b} mean")

    ax_c.set_xticks([1, 2])
    ax_c.set_xticklabels([subj_a.upper(), subj_b.upper()])
    ax_c.set_ylabel("Effort Score")
    pooled = np.concatenate([all_efforts_a, all_efforts_b]).astype(float)
    pooled = pooled[np.isfinite(pooled)]
    min_pooled = float(np.min(pooled)) if len(pooled) else -100.0
    max_pooled = float(np.max(pooled)) if len(pooled) else 100.0
    ax_c.set_ylim(min(-105, min_pooled * 1.15), max(105, max_pooled * 1.15))
    ax_c.set_title("C  Effort Distribution Across Domains (pooled)", fontweight="bold", loc="left")
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.grid(axis="y", alpha=0.3)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.13, wspace=0.28)
    out = output_dir / "clinical_case_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out} ({subj_a} vs {subj_b})")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 – Modality correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_modality_heatmap(corr: pd.DataFrame, output_dir: Path) -> None:
    """
    Heatmap of Spearman ρ: rows = modality, columns = domain.
    Only effort_vs_r4 rows (non-overall modalities).
    """
    sub = corr[
        (corr["comparison"] == "effort_vs_r4") &
        (corr["modality"] != "overall")
    ].copy()

    if sub.empty:
        print("  [skip] No modality data for heatmap.")
        return

    MODALITY_LABELS = {
        "hr_hrv":    "HR / HRV",
        "eda":       "EDA",
        "imu_wrist": "IMU Wrist",
        "imu_chest": "IMU Chest (BioZ)",
    }
    domain_cols = [d for d in DOMAIN_ORDER if d in sub["domain"].unique()]

    # build both pivot tables with the same index order
    pivot = sub.pivot_table(
        index="modality", columns="domain", values="spearman_r"
    ).reindex(columns=domain_cols)

    p_pivot = sub.pivot_table(
        index="modality", columns="domain", values="spearman_p"
    ).reindex(index=pivot.index, columns=domain_cols)  # same row order as pivot

    # rename index AFTER both pivots are aligned
    row_labels = [MODALITY_LABELS.get(m, m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.suptitle("Modality Spearman ρ - Effort vs R4 Stage",
                 fontsize=13, fontweight="bold")

    vmax = 1.0
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    ax.set_xticks(range(len(domain_cols)))
    ax.set_xticklabels(domain_cols, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    # annotate cells with ρ value + significance
    for row_i in range(len(pivot.index)):
        for col_j in range(len(pivot.columns)):
            val = pivot.values[row_i, col_j]
            if np.isnan(val):
                continue
            p    = p_pivot.values[row_i, col_j]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            txt  = f"{val:+.2f}{star}"
            colour = "white" if abs(val) > 0.6 else "black"
            ax.text(col_j, row_i, txt, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=colour)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=10)
    ax.set_xlabel("")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    out = output_dir / "modality_correlation_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 – Class imbalance in R4 labels
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_imbalance(r4_csv: Path, output_dir: Path) -> None:
    """
    Three-panel figure:
      A) Overall histogram across all domains: count + % per R4 stage
      B) Stacked bar per domain showing R4 stage proportions
      C) Subject × domain heatmap coloured by R4 stage
    """
    df_raw = pd.read_csv(r4_csv)
    domain_cols = [c for c in df_raw.columns if c != "Participant"]

    # long format
    long = (
        df_raw.melt(id_vars="Participant", value_vars=domain_cols,
                    var_name="domain", value_name="r4_stage")
        .dropna(subset=["r4_stage"])
    )
    long["r4_stage"] = long["r4_stage"].astype(int)

    all_stages = [1, 2, 3, 4, 5]
    stage_labels = [f"Stage {s}" for s in all_stages]
    stage_colours = [R4_COLOURS[s] for s in all_stages]
    total = len(long)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Clinical R4 Label Distribution – Nursing Home Cohort (n=24 recordings)",
                 fontsize=15, fontweight="bold", y=1.01)

    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax_hist   = fig.add_subplot(gs[0, 0])
    ax_stack  = fig.add_subplot(gs[0, 1])
    ax_heat   = fig.add_subplot(gs[1, :])

    # ── Panel A: overall histogram ────────────────────────────────────────
    counts = long["r4_stage"].value_counts().reindex(all_stages, fill_value=0)
    bars = ax_hist.bar(
        [str(s) for s in all_stages], counts.values,
        color=stage_colours, edgecolor="white", linewidth=1.2, zorder=2,
    )
    ax_hist.set_xlabel("R4 Stage")
    ax_hist.set_ylabel("Count (subject × domain observations)")
    ax_hist.set_title("A  Overall Stage Distribution", fontweight="bold", loc="left")
    ax_hist.grid(axis="y", color="lightgrey", zorder=0)

    # annotate bars with count and %
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / total
        ax_hist.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # highlight stages 4+5 with a brace-style annotation
    high_pct = 100 * counts[[4, 5]].sum() / total
    ax_hist.annotate(
        f"{high_pct:.0f}% are\nStage 4 or 5",
        xy=(1.5, counts[[4, 5]].max() * 0.5),
        xytext=(2.3, counts[[4, 5]].max() * 0.85),
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
        fontsize=10, color="#e74c3c", fontweight="bold",
    )

    # ── Panel B: stacked bar per domain ───────────────────────────────────
    domain_order_full = [
        "Basic Movements", "Walking", "Oral Care", "Grooming",
        "Orientation", "Communication", "Mental Activity", "Swallowing",
        "Eating", "Excretion", "Bathing", "Dressing & Undressing",
        "Leisure", "Interaction",
    ]
    domains_present = [d for d in domain_order_full if d in domain_cols]

    # proportion per domain per stage
    pivot_pct = (
        long.groupby(["domain", "r4_stage"]).size()
        .unstack(fill_value=0)
        .reindex(columns=all_stages, fill_value=0)
        .reindex(domains_present)
    )
    pivot_pct_norm = pivot_pct.div(pivot_pct.sum(axis=1), axis=0) * 100

    bottom = np.zeros(len(domains_present))
    x_pos  = np.arange(len(domains_present))
    for stage, colour in zip(all_stages, stage_colours):
        vals = pivot_pct_norm[stage].values if stage in pivot_pct_norm.columns else np.zeros(len(domains_present))
        ax_stack.bar(x_pos, vals, bottom=bottom, color=colour,
                     edgecolor="white", linewidth=0.5, label=f"Stage {stage}")
        bottom += vals

    # add % labels for stage 4+5 combined
    high_pcts = pivot_pct_norm[[4, 5]].sum(axis=1).values if 4 in pivot_pct_norm.columns else np.zeros(len(domains_present))
    for xi, hp in zip(x_pos, high_pcts):
        if hp > 5:
            ax_stack.text(xi, 101, f"{hp:.0f}%", ha="center", va="bottom",
                          fontsize=7.5, color="#e74c3c", fontweight="bold")

    ax_stack.set_xticks(x_pos)
    ax_stack.set_xticklabels(
        [d.replace(" & ", "\n& ") for d in domains_present],
        rotation=40, ha="right", fontsize=8.5,
    )
    ax_stack.set_ylabel("Proportion (%)")
    ax_stack.set_ylim(0, 115)
    ax_stack.set_title("B  Stage Proportions per Domain", fontweight="bold", loc="left")
    ax_stack.legend(title="R4 Stage", loc="upper left", fontsize=8,
                    bbox_to_anchor=(1.01, 1), framealpha=0.8)

    # ── Panel C: subject × domain heatmap ────────────────────────────────
    pivot_heat = df_raw.set_index("Participant")[domains_present]
    # sort subjects by mean R4 stage descending (most impaired at top)
    pivot_heat = pivot_heat.loc[pivot_heat.mean(axis=1).sort_values(ascending=False).index]

    # build custom colour map from R4_COLOURS
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([R4_COLOURS[s] for s in all_stages])
    norm = BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ncolors=5)

    im = ax_heat.imshow(
        pivot_heat.values, cmap=cmap, norm=norm,
        aspect="auto", interpolation="none",
    )

    ax_heat.set_xticks(range(len(domains_present)))
    ax_heat.set_xticklabels(
        [d.replace(" & ", "\n& ") for d in domains_present],
        rotation=35, ha="right", fontsize=8.5,
    )
    ax_heat.set_yticks(range(len(pivot_heat)))
    ax_heat.set_yticklabels(pivot_heat.index, fontsize=8)
    ax_heat.set_title("C  R4 Stage per Subject × Domain  (sorted by mean stage)",
                      fontweight="bold", loc="left")
    ax_heat.tick_params(length=0)
    ax_heat.spines[:].set_visible(False)

    # annotate each cell with the stage value
    for ri in range(pivot_heat.shape[0]):
        for ci in range(pivot_heat.shape[1]):
            val = pivot_heat.values[ri, ci]
            if not np.isnan(val):
                txt_col = "white" if int(val) in (4, 5) else "black"
                ax_heat.text(ci, ri, str(int(val)), ha="center", va="center",
                             fontsize=7.5, color=txt_col)

    # discrete colorbar
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                        shrink=0.6, pad=0.01, ticks=all_stages)
    cbar.set_label("R4 Stage", fontsize=10)
    cbar.ax.set_yticklabels([f"Stage {s}" for s in all_stages])

    fig.text(0.01, 0.01, f"Source: {r4_csv}", fontsize=8, color="dimgray")

    # ── save ─────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.05, right=0.92, top=0.93, bottom=0.08,
                        hspace=0.42, wspace=0.32)
    out = output_dir / "class_imbalance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate scatter plots for the functional-analysis pipeline.")
    p.add_argument(
        "--combined-csv",
        default="output/correlation/run2/combined_analysis.csv",
        help="Path to combined_analysis.csv (default: %(default)s)",
    )
    p.add_argument(
        "--correlation-csv",
        default="output/correlation/run2/r4_correlation.csv",
        help="Path to r4_correlation.csv (default: %(default)s)",
    )
    p.add_argument(
        "--r4-csv",
        default="docs/R4-scores/R4_scores_nursing_home.csv",
        help="Path to R4_scores_nursing_home.csv (default: %(default)s)",
    )
    p.add_argument(
        "--hc-effort-csv",
        default="output/effort/hc_baseline_run1/effort_scores.csv",
        help="Path to HC effort scores CSV (default: %(default)s)",
    )
    p.add_argument(
        "--nh-effort-csv",
        default="output/effort/real_run2/effort_scores.csv",
        help="Path to nursing-home effort scores CSV (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        default="output/plots/run1",
        help="Directory to write PNG files (default: %(default)s)",
    )
    p.add_argument(
        "--plots",
        nargs="+",
        choices=["icf", "correlation", "boxplot", "variability", "clinical", "heatmap", "imbalance", "hc-vs-nh", "hc-vs-nh-stats", "walking-stratified", "all"],
        default=["all"],
        help="Which plots to generate (default: all)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    pipeline_root = Path(__file__).parent
    combined_path = pipeline_root / args.combined_csv
    corr_path     = pipeline_root / args.correlation_csv
    output_dir    = pipeline_root / args.output_dir
    r4_path       = pipeline_root / args.r4_csv
    hc_effort_path = pipeline_root / args.hc_effort_csv
    nh_effort_path = pipeline_root / args.nh_effort_csv

    want_all = "all" in args.plots
    want     = set(args.plots)

    # ── validate inputs ──────────────────────────────────────────────────────
    # imbalance only needs r4_csv; other plots need combined + corr
    need_combined = want_all or bool(want & {"icf", "correlation", "boxplot", "variability", "clinical", "heatmap", "walking-stratified"})
    need_r4       = want_all or "imbalance" in want
    need_hc_nh    = bool(want & {"hc-vs-nh", "hc-vs-nh-stats", "walking-stratified"})

    paths_to_check = []
    if need_combined:
        paths_to_check += [combined_path, corr_path]
    if need_r4:
        paths_to_check.append(r4_path)
    if need_hc_nh:
        paths_to_check += [hc_effort_path, nh_effort_path]

    missing = [p for p in paths_to_check if not p.exists()]
    if missing:
        sys.exit("ERROR: file(s) not found:\n" + "\n".join(str(p) for p in missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ── load data ────────────────────────────────────────────────────────────
    combined = pd.read_csv(combined_path) if need_combined else None
    corr     = pd.read_csv(corr_path)     if need_combined else None

    # ── generate plots ───────────────────────────────────────────────────────
    if want_all or "icf" in want:
        print("Generating ICF dual-axis scatter…")
        plot_icf_dual_axis(combined, output_dir)

    if want_all or "correlation" in want:
        print("Generating correlation summary bar chart…")
        plot_correlation_summary(corr, output_dir)

    if want_all or "boxplot" in want:
        print("Generating effort-by-R4-stage box plots…")
        plot_effort_by_r4_stage(combined, output_dir)

    if want_all or "variability" in want:
        print("Generating effort variability by R4 class…")
        plot_effort_variability_by_class(combined, output_dir)

    if want_all or "clinical" in want:
        print("Generating clinical case comparison…")
        plot_clinical_case_comparison(combined, output_dir)

    if want_all or "heatmap" in want:
        print("Generating modality correlation heatmap…")
        plot_modality_heatmap(corr, output_dir)

    if want_all or "imbalance" in want:
        print("Generating R4 class imbalance figures…")
        plot_class_imbalance(r4_path, output_dir)

    if "hc-vs-nh" in want:
        print("Generating HC vs nursing-home domain comparison…")
        plot_hc_vs_nh_domain_effort(hc_effort_path, nh_effort_path, output_dir)

    if "hc-vs-nh-stats" in want:
        print("Generating HC vs nursing-home domain comparison with statistics…")
        plot_hc_vs_nh_domain_effort_stats(hc_effort_path, nh_effort_path, output_dir)

    if "walking-stratified" in want:
        print("Generating walking mobility-proxy stratified comparison…")
        plot_walking_stage_proxy_comparison(combined, hc_effort_path, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
