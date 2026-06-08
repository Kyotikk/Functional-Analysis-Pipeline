#!/usr/bin/env python
"""
Plot simulation effort score distributions per condition using HC anchors.
Uses sim_run2_clean (subjects 2-5, HC-anchored) to show per-condition spread
across domains and modalities.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

SCORES_PATH = "functional-analysis-pipeline/output/effort/sim_run2_clean/effort_scores.csv"
OUT_DIR = Path("functional-analysis-pipeline/output/plots/score_histograms")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_ORDER = ["healthy", "elderly", "severe"]
COLORS = {"healthy": "#2ca02c", "elderly": "#ff7f0e", "severe": "#d62728"}

df = pd.read_csv(SCORES_PATH)
df["condition"] = df["subject_id"].str.extract(r"sim_(healthy|elderly|severe)_")[0]

domains = ["Basic Movements", "Walking", "Oral Care", "Grooming"]
modalities = ["hr_hrv", "imu_wrist", "imu_bioz", "eda"]
modality_labels = {
    "hr_hrv":   "HR / HRV",
    "imu_wrist": "IMU Wrist",
    "imu_bioz":  "IMU Bioz",
    "eda":       "EDA",
}

# ── Helper ────────────────────────────────────────────────────────────────────
def jitter_x(x, n, spread=0.15):
    return x + np.linspace(-spread, spread, n)


def strip_ax(ax, col_prefix, label, ylim=(-105, 105)):
    """Plot per-condition strip with individual points + mean line."""
    for i, cond in enumerate(CONDITION_ORDER):
        vals = df[df["condition"] == cond][col_prefix].dropna().values
        if len(vals) == 0:
            continue
        xs = jitter_x(i, len(vals))
        ax.scatter(xs, vals, color=COLORS[cond], s=60, zorder=3, alpha=0.85, edgecolors="black", linewidths=0.5)
        ax.hlines(np.mean(vals), i - 0.2, i + 0.2, colors=COLORS[cond], linewidths=2.5, zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c.capitalize() for c in CONDITION_ORDER], fontsize=9)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="HC baseline")
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylabel("Centered effort score (HC anchors)")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Overall domain effort per condition
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
fig.suptitle("Simulation Effort by Domain (HC Anchors, subjects 2–5)", fontsize=13, fontweight="bold")

for ax, domain in zip(axes, domains):
    col = f"{domain}_effort"
    strip_ax(ax, col, domain)
    if ax != axes[0]:
        ax.set_ylabel("")

legend_patches = [mpatches.Patch(color=COLORS[c], label=c.capitalize()) for c in CONDITION_ORDER]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
fig.savefig(OUT_DIR / "sim_hc_anchors_domains.png", dpi=150, bbox_inches="tight")
print("Saved: sim_hc_anchors_domains.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: HR/HRV modality per domain (core physiology)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
fig.suptitle("Simulation Effort – HR/HRV Modality Only (HC Anchors)", fontsize=13, fontweight="bold")

for ax, domain in zip(axes, domains):
    col = f"{domain}_effort_hr_hrv"
    if col not in df.columns:
        ax.set_visible(False)
        continue
    strip_ax(ax, col, domain)
    if ax != axes[0]:
        ax.set_ylabel("")

fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
fig.savefig(OUT_DIR / "sim_hc_anchors_hrhrv.png", dpi=150, bbox_inches="tight")
print("Saved: sim_hc_anchors_hrhrv.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: All modalities × all domains heatmap of per-condition means
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(modalities), len(domains), figsize=(14, 10), sharey=True, sharex=True)
fig.suptitle("Per-Modality Effort Distributions (HC Anchors)\nRows = modality, Columns = domain", fontsize=12, fontweight="bold")

for row_i, mod in enumerate(modalities):
    for col_i, domain in enumerate(domains):
        ax = axes[row_i][col_i]
        col_name = f"{domain}_effort_{mod}"
        if col_name not in df.columns:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_visible(True)
            if row_i == 0:
                ax.set_title(domain, fontsize=9, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(modality_labels[mod], fontsize=9)
            continue

        for i, cond in enumerate(CONDITION_ORDER):
            vals = df[df["condition"] == cond][col_name].dropna().values
            if len(vals) == 0:
                continue
            xs = jitter_x(i, len(vals), spread=0.12)
            ax.scatter(xs, vals, color=COLORS[cond], s=40, zorder=3, alpha=0.8, edgecolors="black", linewidths=0.4)
            ax.hlines(np.mean(vals), i - 0.18, i + 0.18, colors=COLORS[cond], linewidths=2, zorder=4)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_ylim(-105, 105)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["H", "E", "S"], fontsize=8)
        if row_i == 0:
            ax.set_title(domain, fontsize=9, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel(modality_labels[mod], fontsize=9)

fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout()
fig.savefig(OUT_DIR / "sim_hc_anchors_all_modalities.png", dpi=150, bbox_inches="tight")
print("Saved: sim_hc_anchors_all_modalities.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Text summary: ordering check per modality × domain
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ORDERING CHECK: Is healthy ≤ elderly ≤ severe? (per modality × domain)")
print("="*70)

header = f"{'Modality':<14}" + "".join(f"{d[:10]:<12}" for d in domains)
print(header)
print("-" * len(header))

for mod in ["overall"] + modalities:
    row_str = f"{modality_labels.get(mod, 'Overall'):<14}"
    for domain in domains:
        col_name = f"{domain}_effort" if mod == "overall" else f"{domain}_effort_{mod}"
        if col_name not in df.columns:
            row_str += f"{'N/A':<12}"
            continue
        means = {}
        for cond in CONDITION_ORDER:
            vals = df[df["condition"] == cond][col_name].dropna().values
            means[cond] = float(np.mean(vals)) if len(vals) > 0 else np.nan
        h, e, s = means["healthy"], means["elderly"], means["severe"]
        ordered = (h <= e <= s) if not any(np.isnan(v) for v in [h, e, s]) else None
        marker = "✓" if ordered else ("✗" if ordered is False else "?")
        row_str += f"{h:.0f}→{e:.0f}→{s:.0f}  {marker}".ljust(12)
    print(row_str)

print("\n(H→E→S = healthy mean → elderly mean → severe mean)\n")
