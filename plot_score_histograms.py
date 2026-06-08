#!/usr/bin/env python
"""Generate histogram plots for capacity and effort score distributions."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


_SIM_ID_RE = re.compile(r"^sim_(healthy|elderly|severe)_\d+$", re.IGNORECASE)


def _existing_effort_cols(df: pd.DataFrame) -> List[str]:
    return [
        c
        for c in df.columns
        if c.endswith("_effort")
        and not c.endswith("_effort_reliable")
        and "_effort_" not in c
    ]


def _save_hist(
    values: pd.Series,
    out_path: Path,
    title: str,
    bins: int,
    xlim: tuple[float, float] | None = None,
) -> None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        raise ValueError(f"No numeric values available for plot: {title}")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(clean, bins=bins, edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_capacity_overall_centered(
    cap_df: pd.DataFrame,
    out_path: Path,
    cap_cols: List[str],
) -> None:
    values = cap_df[cap_cols].stack(future_stack=True)
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("No numeric capacity values available.")

    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(clean, bins=bins, edgecolor="black", alpha=0.85, rwidth=0.9)
    ax.set_title("Autosorted Capacity Scores (All Domains)")
    ax.set_xlabel("Capacity stage")
    ax.set_ylabel("Count")
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_capacity_per_domain(
    cap_df: pd.DataFrame,
    out_path: Path,
    cap_cols: List[str],
) -> None:
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, col in enumerate(cap_cols):
        vals = pd.to_numeric(cap_df[col], errors="coerce").dropna()
        ax = axes_flat[i]
        ax.hist(vals, bins=bins, edgecolor="black", alpha=0.85, rwidth=0.9)
        ax.set_title(col)
        ax.set_xlim(0.5, 5.5)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(axis="y", alpha=0.25)

    axes[1, 0].set_xlabel("Capacity stage")
    axes[1, 1].set_xlabel("Capacity stage")
    axes[0, 0].set_ylabel("Count")
    axes[1, 0].set_ylabel("Count")
    fig.suptitle("Autosorted Capacity Scores by Domain", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_effort_overlay_hist(
    hc_values: pd.Series,
    real_values: pd.Series,
    out_path: Path,
    bins: int,
) -> None:
    hc_clean = pd.to_numeric(hc_values, errors="coerce").dropna()
    real_clean = pd.to_numeric(real_values, errors="coerce").dropna()
    if hc_clean.empty or real_clean.empty:
        raise ValueError("Need non-empty HC and real effort values for overlay histogram.")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(hc_clean, bins=bins, range=(-100, 100), alpha=0.5, label="HC baseline", edgecolor="black")
    ax.hist(real_clean, bins=bins, range=(-100, 100), alpha=0.5, label="Nursing home", edgecolor="black")
    ax.set_title("Effort Score Distribution: HC vs Nursing Home")
    ax.set_xlabel("Effort score")
    ax.set_ylabel("Count")
    ax.set_xlim(-100, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_sim_condition_overlay_hist(
    sim_df: pd.DataFrame,
    out_path: Path,
    bins: int,
) -> None:
    effort_cols = _existing_effort_cols(sim_df)
    if not effort_cols:
        raise ValueError("No effort columns found in simulation CSV.")

    long_df = sim_df[["subject_id", *effort_cols]].copy()
    long_df["condition"] = long_df["subject_id"].astype(str).str.extract(_SIM_ID_RE, expand=False)
    long_df = long_df.dropna(subset=["condition"])
    if long_df.empty:
        raise ValueError("No simulation subject IDs matched sim_healthy/sim_elderly/sim_severe.")

    values = []
    for condition in ["healthy", "elderly", "severe"]:
        condition_vals = long_df.loc[long_df["condition"] == condition, effort_cols].stack(future_stack=True)
        clean = pd.to_numeric(condition_vals, errors="coerce").dropna()
        if not clean.empty:
            values.append((condition, clean))

    if not values:
        raise ValueError("No numeric simulation effort values available for grouped histogram.")

    colours = {
        "healthy": "#2ecc71",
        "elderly": "#f39c12",
        "severe": "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for condition, clean in values:
        ax.hist(
            clean,
            bins=bins,
            range=(-100, 100),
            alpha=0.45,
            label=f"sim_{condition}",
            edgecolor="black",
            color=colours.get(condition),
        )

    ax.set_title("Effort Score Distribution: Simulation Conditions")
    ax.set_xlabel("Effort score")
    ax.set_ylabel("Count")
    ax.set_xlim(-100, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Create histograms for capacity and effort scores.")
    parser.add_argument(
        "--capacity-csv",
        type=Path,
        default=here / "output" / "capacity" / "batch_20260407_213748_real_run1" / "capacity_scores.csv",
        help="Path to autosorted capacity scores CSV.",
    )
    parser.add_argument(
        "--effort-hc-csv",
        type=Path,
        default=here / "output" / "effort" / "hc_baseline_run1" / "effort_scores.csv",
        help="Path to HC baseline effort scores CSV.",
    )
    parser.add_argument(
        "--effort-real-csv",
        type=Path,
        default=here / "output" / "effort" / "real_run2" / "effort_scores.csv",
        help="Path to nursing-home effort scores CSV.",
    )
    parser.add_argument(
        "--effort-sim-csv",
        type=Path,
        default=None,
        help="Optional path to simulation effort scores CSV for grouped condition histograms.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "output" / "plots" / "score_histograms",
        help="Directory for output PNG files.",
    )
    parser.add_argument(
        "--bins-capacity",
        type=int,
        default=8,
        help="Bin count for capacity histogram.",
    )
    parser.add_argument(
        "--bins-effort",
        type=int,
        default=20,
        help="Bin count for effort histograms.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap_df = pd.read_csv(args.capacity_csv)
    cap_cols = ["Basic Movements", "Walking", "Oral Care", "Grooming"]
    _save_capacity_overall_centered(
        cap_df=cap_df,
        out_path=args.output_dir / "capacity_autosorted_hist.png",
        cap_cols=cap_cols,
    )
    _save_capacity_per_domain(
        cap_df=cap_df,
        out_path=args.output_dir / "capacity_autosorted_hist_by_domain.png",
        cap_cols=cap_cols,
    )

    hc_df = pd.read_csv(args.effort_hc_csv)
    hc_effort_cols = _existing_effort_cols(hc_df)
    hc_values = hc_df[hc_effort_cols].stack(future_stack=True)
    _save_hist(
        values=hc_values,
        out_path=args.output_dir / "effort_hc_baseline_hist.png",
        title="Effort Scores: Healthy Control Baseline (All Domains)",
        bins=args.bins_effort,
        xlim=(-100, 100),
    )

    real_df = pd.read_csv(args.effort_real_csv)
    real_effort_cols = _existing_effort_cols(real_df)
    real_values = real_df[real_effort_cols].stack(future_stack=True)
    _save_hist(
        values=real_values,
        out_path=args.output_dir / "effort_nursing_home_hist.png",
        title="Effort Scores: Nursing Home Patients (All Domains)",
        bins=args.bins_effort,
        xlim=(-100, 100),
    )
    _save_effort_overlay_hist(
        hc_values=hc_values,
        real_values=real_values,
        out_path=args.output_dir / "effort_hc_vs_nursing_home_overlay_hist.png",
        bins=args.bins_effort,
    )

    if args.effort_sim_csv is not None:
        sim_df = pd.read_csv(args.effort_sim_csv)
        _save_sim_condition_overlay_hist(
            sim_df=sim_df,
            out_path=args.output_dir / "effort_sim_conditions_overlay_hist.png",
            bins=args.bins_effort,
        )

    print("Saved:")
    print(args.output_dir / "capacity_autosorted_hist.png")
    print(args.output_dir / "capacity_autosorted_hist_by_domain.png")
    print(args.output_dir / "effort_hc_baseline_hist.png")
    print(args.output_dir / "effort_nursing_home_hist.png")
    print(args.output_dir / "effort_hc_vs_nursing_home_overlay_hist.png")
    if args.effort_sim_csv is not None:
        print(args.output_dir / "effort_sim_conditions_overlay_hist.png")


if __name__ == "__main__":
    main()
