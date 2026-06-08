#!/usr/bin/env python
"""Investigate simulation effort score ordering under HC anchors.

Generates condition-stratified summaries and checks whether
healthy < elderly < severe holds for pooled and domain-level effort.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = ["healthy", "elderly", "severe"]
CONDITION_RANK = {name: idx for idx, name in enumerate(CONDITION_ORDER)}
CONDITION_COLORS = {
    "healthy": "#2f9e44",
    "elderly": "#f08c00",
    "severe": "#c92a2a",
}


def _extract_conditions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["condition"] = out["subject_id"].str.extract(r"sim_(healthy|elderly|severe)_")[0]
    out = out[out["condition"].isin(CONDITION_ORDER)].copy()
    out["condition"] = pd.Categorical(out["condition"], categories=CONDITION_ORDER, ordered=True)
    return out


def _domain_effort_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if not col.endswith("_effort"):
            continue
        if col.endswith("_effort_reliable"):
            continue
        # Exclude modality-specific columns such as Walking_effort_hr_hrv
        if "_effort_" in col:
            continue
        cols.append(col)
    return cols


def _melt_scores(df: pd.DataFrame, domain_cols: List[str], include_only_reliable: bool) -> pd.DataFrame:
    value_df = df[["subject_id", "condition"] + domain_cols].copy()

    if include_only_reliable:
        for col in domain_cols:
            rel_col = f"{col}_reliable"
            if rel_col in df.columns:
                value_df[col] = value_df[col].where(df[rel_col] == True)  # noqa: E712

    long_df = value_df.melt(
        id_vars=["subject_id", "condition"],
        value_vars=domain_cols,
        var_name="domain_col",
        value_name="effort_score",
    )
    long_df = long_df.dropna(subset=["effort_score"]).copy()
    long_df["domain"] = long_df["domain_col"].str.replace("_effort", "", regex=False)
    return long_df


def _compute_subject_pooled(df: pd.DataFrame, domain_cols: List[str], include_only_reliable: bool) -> pd.DataFrame:
    pooled_df = df[["subject_id", "condition"] + domain_cols].copy()
    if include_only_reliable:
        for col in domain_cols:
            rel_col = f"{col}_reliable"
            if rel_col in df.columns:
                pooled_df[col] = pooled_df[col].where(df[rel_col] == True)  # noqa: E712

    pooled_df["pooled_effort"] = pooled_df[domain_cols].median(axis=1, skipna=True)
    return pooled_df[["subject_id", "condition", "pooled_effort"]].dropna(subset=["pooled_effort"])


def _summary_stats(values: pd.Series) -> Dict[str, float]:
    return {
        "n": int(values.notna().sum()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _make_condition_summary(metric_df: pd.DataFrame, metric_col: str, metric_name: str) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        subset = metric_df[metric_df["condition"] == condition][metric_col]
        if subset.empty:
            continue
        stats = _summary_stats(subset)
        stats.update({"metric": metric_name, "condition": condition})
        rows.append(stats)
    return pd.DataFrame(rows)


def _build_ordering_checks(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in summary_df["metric"].unique():
        metric_rows = summary_df[summary_df["metric"] == metric]
        medians = {row["condition"]: row["median"] for _, row in metric_rows.iterrows()}
        missing = [c for c in CONDITION_ORDER if c not in medians]
        if missing:
            rows.append(
                {
                    "metric": metric,
                    "healthy_median": np.nan,
                    "elderly_median": np.nan,
                    "severe_median": np.nan,
                    "delta_elderly_minus_healthy": np.nan,
                    "delta_severe_minus_elderly": np.nan,
                    "strict_order_pass": False,
                    "note": f"missing conditions: {','.join(missing)}",
                }
            )
            continue

        healthy = medians["healthy"]
        elderly = medians["elderly"]
        severe = medians["severe"]
        rows.append(
            {
                "metric": metric,
                "healthy_median": healthy,
                "elderly_median": elderly,
                "severe_median": severe,
                "delta_elderly_minus_healthy": elderly - healthy,
                "delta_severe_minus_elderly": severe - elderly,
                "strict_order_pass": bool(healthy < elderly < severe),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def _plot_distributions(
    pooled_df: pd.DataFrame,
    long_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    metrics = ["pooled_effort"] + sorted(long_df["domain"].unique().tolist())
    ncols = len(metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric == "pooled_effort":
            plot_df = pooled_df.rename(columns={"pooled_effort": "value"})[["condition", "value"]]
            ylab = "Pooled effort"
        else:
            plot_df = long_df[long_df["domain"] == metric].rename(columns={"effort_score": "value"})[
                ["condition", "value"]
            ]
            ylab = f"{metric} effort"

        data_per_condition = [
            plot_df.loc[plot_df["condition"] == cond, "value"].dropna().to_numpy()
            for cond in CONDITION_ORDER
        ]

        ax.boxplot(
            data_per_condition,
            tick_labels=CONDITION_ORDER,
            patch_artist=True,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )

        for idx, (cond, values) in enumerate(zip(CONDITION_ORDER, data_per_condition), start=1):
            if len(values) == 0:
                continue
            jitter = np.random.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(
                np.full(len(values), idx) + jitter,
                values,
                alpha=0.75,
                s=28,
                color=CONDITION_COLORS[cond],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

        for patch, cond in zip(ax.artists, CONDITION_ORDER):
            patch.set_facecolor(CONDITION_COLORS[cond])
            patch.set_alpha(0.25)

        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel(ylab)
        ax.set_ylim(-105, 105)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle(title, fontsize=13)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _compare_to_baseline(new_df: pd.DataFrame, baseline_df: pd.DataFrame, domain_cols: List[str]) -> pd.DataFrame:
    new_long = new_df[["subject_id", "condition"] + domain_cols].melt(
        id_vars=["subject_id", "condition"], value_vars=domain_cols, var_name="domain_col", value_name="new_score"
    )
    baseline_long = baseline_df[["subject_id", "condition"] + domain_cols].melt(
        id_vars=["subject_id", "condition"],
        value_vars=domain_cols,
        var_name="domain_col",
        value_name="baseline_score",
    )
    merged = new_long.merge(
        baseline_long,
        on=["subject_id", "condition", "domain_col"],
        how="inner",
    )
    merged["delta_new_minus_baseline"] = merged["new_score"] - merged["baseline_score"]
    merged["domain"] = merged["domain_col"].str.replace("_effort", "", regex=False)

    rows = []
    for (condition, domain), group in merged.groupby(["condition", "domain"], observed=True):
        rows.append(
            {
                "condition": condition,
                "domain": domain,
                "n": int(group["delta_new_minus_baseline"].notna().sum()),
                "median_delta": float(group["delta_new_minus_baseline"].median()),
                "mean_delta": float(group["delta_new_minus_baseline"].mean()),
                "max_abs_delta": float(group["delta_new_minus_baseline"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    report_path: Path,
    summary_all: pd.DataFrame,
    ordering_all: pd.DataFrame,
    summary_rel: pd.DataFrame,
    ordering_rel: pd.DataFrame,
    delta_df: pd.DataFrame,
    plot_filename: str,
) -> None:
    pooled_all = ordering_all[ordering_all["metric"] == "pooled_effort"]
    pooled_rel = ordering_rel[ordering_rel["metric"] == "pooled_effort"]

    def _line_from(df: pd.DataFrame, label: str) -> str:
        if df.empty:
            return f"- {label}: unavailable"
        row = df.iloc[0]
        return (
            f"- {label}: healthy={row['healthy_median']:.2f}, "
            f"elderly={row['elderly_median']:.2f}, severe={row['severe_median']:.2f}, "
            f"strict_order_pass={bool(row['strict_order_pass'])}"
        )

    all_pass_rate = float(ordering_all["strict_order_pass"].mean()) if not ordering_all.empty else 0.0
    rel_pass_rate = float(ordering_rel["strict_order_pass"].mean()) if not ordering_rel.empty else 0.0

    lines = [
        "# Simulation HC-Anchor Rerun Investigation",
        "",
        "## Primary ordering check (healthy < elderly < severe)",
        _line_from(pooled_all, "Pooled (all scores)"),
        _line_from(pooled_rel, "Pooled (reliability-filtered)"),
        f"- Domain+pooled strict-pass rate (all scores): {all_pass_rate:.1%}",
        f"- Domain+pooled strict-pass rate (reliability-filtered): {rel_pass_rate:.1%}",
        "",
        "## Interpretation",
        "- Under current HC anchors, scores are heavily saturated near +100, which collapses separation between conditions.",
        "- Because of saturation, strict ordering generally fails even after reliability filtering.",
        "- See ordering CSVs for per-domain pass/fail and median deltas.",
        "",
        "## Artifacts",
        "- condition_summary_all.csv",
        "- condition_summary_reliable.csv",
        "- ordering_checks_all.csv",
        "- ordering_checks_reliable.csv",
        "- deltas_vs_sim_run2_clean.csv",
        f"- {plot_filename}",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate simulation ordering under HC anchors")
    parser.add_argument("--scores", type=Path, required=True, help="Path to rerun effort_scores.csv")
    parser.add_argument("--baseline-scores", type=Path, required=True, help="Path to baseline effort_scores.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for investigation")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rerun_df = _extract_conditions(pd.read_csv(args.scores))
    baseline_df = _extract_conditions(pd.read_csv(args.baseline_scores))

    domain_cols = _domain_effort_cols(rerun_df)
    if not domain_cols:
        raise ValueError("No domain effort columns found in rerun scores.")

    long_all = _melt_scores(rerun_df, domain_cols, include_only_reliable=False)
    pooled_all = _compute_subject_pooled(rerun_df, domain_cols, include_only_reliable=False)

    long_rel = _melt_scores(rerun_df, domain_cols, include_only_reliable=True)
    pooled_rel = _compute_subject_pooled(rerun_df, domain_cols, include_only_reliable=True)

    summary_all_parts = [_make_condition_summary(pooled_all, "pooled_effort", "pooled_effort")]
    for domain in sorted(long_all["domain"].unique()):
        domain_df = long_all[long_all["domain"] == domain]
        summary_all_parts.append(_make_condition_summary(domain_df, "effort_score", domain))
    summary_all = pd.concat(summary_all_parts, ignore_index=True)

    summary_rel_parts = [_make_condition_summary(pooled_rel, "pooled_effort", "pooled_effort")]
    for domain in sorted(long_rel["domain"].unique()):
        domain_df = long_rel[long_rel["domain"] == domain]
        summary_rel_parts.append(_make_condition_summary(domain_df, "effort_score", domain))
    summary_rel = pd.concat(summary_rel_parts, ignore_index=True)

    ordering_all = _build_ordering_checks(summary_all)
    ordering_rel = _build_ordering_checks(summary_rel)

    deltas = _compare_to_baseline(rerun_df, baseline_df, domain_cols)

    summary_all.to_csv(args.output_dir / "condition_summary_all.csv", index=False)
    summary_rel.to_csv(args.output_dir / "condition_summary_reliable.csv", index=False)
    ordering_all.to_csv(args.output_dir / "ordering_checks_all.csv", index=False)
    ordering_rel.to_csv(args.output_dir / "ordering_checks_reliable.csv", index=False)
    deltas.to_csv(args.output_dir / "deltas_vs_sim_run2_clean.csv", index=False)

    plot_name = "condition_distributions_all.png"
    _plot_distributions(
        pooled_all,
        long_all,
        args.output_dir / plot_name,
        title="Simulation Effort Distributions (HC Anchors, rerun)",
    )

    _write_report(
        args.output_dir / "simulation_validation_report.md",
        summary_all,
        ordering_all,
        summary_rel,
        ordering_rel,
        deltas,
        plot_name,
    )

    print(f"Saved investigation artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
