"""Write all Phase-4 correlation outputs to disk and print summary tables.

Outputs
-------
r4_correlation.csv
    One row per (domain × comparison × modality).  Contains Spearman ρ,
    Kendall τ, bootstrap 95 % CIs, p-values, and capacity-concordance
    extras (weighted κ, % exact, % within-1).

modality_r4_correlation.csv
    Filtered view of r4_correlation.csv — only per-modality effort rows.

combined_analysis.csv
    Per-subject × domain table with sensor capacity stage, effort,
    ICF-inspired capacity / performance / composite, and R4 stage.

stage_effort_r4_summary.csv
    Mean / median effort per (R4 stage × domain) — extends the existing
    sensor-internal stage_effort_summary with R4 as the stratifier.

gap_analysis.csv
    Subjects where sensor capacity stage and R4 stage diverge by more
    than `gap_threshold` stages, with their effort scores included.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .validator import CorrelationResult, ValidationResult, results_to_dataframe

logger = logging.getLogger(__name__)

_DOMAINS = ["Basic Movements", "Walking", "Oral Care", "Grooming"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _print_correlation_table(corr_df: pd.DataFrame) -> None:
    """Print a readable summary of the primary correlation results."""
    primary = corr_df[
        (corr_df["comparison"].isin(
            ["effort_vs_r4", "capacity_vs_r4", "composite_vs_r4"]
        )) &
        (corr_df["modality"] == "overall")
    ].copy()

    if primary.empty:
        logger.warning("No primary correlation results to display.")
        return

    print("\n" + "=" * 72)
    print("  Phase 4 — Sensor Scores vs Clinical R4 Stages")
    print("=" * 72)
    print(f"  {'Domain':<20}  {'Comparison':<20}  {'n':>3}  "
          f"{'ρ':>7}  {'95% CI':>16}  {'p':>7}  {'sig':>3}")
    print("  " + "-" * 68)

    for _, row in primary.iterrows():
        ci = f"[{row['spearman_ci_lo']:+.3f}, {row['spearman_ci_hi']:+.3f}]"
        sig = _sig_star(row["spearman_p"])
        print(
            f"  {row['domain']:<20}  {row['comparison']:<20}  "
            f"{int(row['n_subjects']):>3}  "
            f"{row['spearman_r']:+.3f}  {ci:>16}  "
            f"{row['spearman_p']:>7.3f}  {sig:>3}"
        )

    # Capacity concordance extras
    cap_rows = primary[primary["comparison"] == "capacity_vs_r4"]
    if not cap_rows.empty and cap_rows["weighted_kappa"].notna().any():
        print()
        print("  Capacity-stage concordance (sensor vs R4):")
        print(f"  {'Domain':<20}  {'κ (weighted)':>12}  {'% exact':>8}  {'% within-1':>10}")
        print("  " + "-" * 56)
        for _, row in cap_rows.iterrows():
            kap  = f"{row['weighted_kappa']:+.3f}" if pd.notna(row["weighted_kappa"]) else " N/A"
            pex  = f"{row['pct_exact']:.1f}%"      if pd.notna(row["pct_exact"])      else " N/A"
            pw1  = f"{row['pct_within_1']:.1f}%"   if pd.notna(row["pct_within_1"])   else " N/A"
            print(f"  {row['domain']:<20}  {kap:>12}  {pex:>8}  {pw1:>10}")

    print("=" * 72)
    print("  * p<0.05   ** p<0.01   *** p<0.001")
    print()


def _print_modality_table(corr_df: pd.DataFrame) -> None:
    """Print per-modality effort vs R4 Spearman ρ summary."""
    mod_rows = corr_df[
        (corr_df["comparison"] == "effort_vs_r4") &
        (corr_df["modality"] != "overall")
    ].copy()

    if mod_rows.empty:
        return

    print("  Per-modality effort vs R4 (Spearman ρ):")
    print(f"  {'Domain':<20}  {'Modality':<12}  {'n':>3}  {'ρ':>7}  {'p':>7}  {'sig':>3}")
    print("  " + "-" * 58)
    for _, row in mod_rows.sort_values(["domain", "modality"]).iterrows():
        sig = _sig_star(row["spearman_p"])
        print(
            f"  {row['domain']:<20}  {row['modality']:<12}  "
            f"{int(row['n_subjects']):>3}  "
            f"{row['spearman_r']:+.3f}  "
            f"{row['spearman_p']:>7.3f}  {sig:>3}"
        )
    print()


# ---------------------------------------------------------------------------
# Stage × effort × R4 summary
# ---------------------------------------------------------------------------

def _build_stage_effort_r4_summary(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median effort per (R4 stage × domain)."""
    rows = []
    for domain in _DOMAINS:
        sub = combined_df[combined_df["domain"] == domain].copy()
        sub = sub[sub["effort"].notna() & sub["effort_reliable"] == True]
        if sub.empty:
            continue
        for r4_stage, grp in sub.groupby("r4_stage"):
            rows.append({
                "r4_label":    domain,
                "r4_stage":    r4_stage,
                "n_subjects":  len(grp),
                "mean_effort": grp["effort"].mean(),
                "median_effort": grp["effort"].median(),
                "std_effort":  grp["effort"].std(),
                "min_effort":  grp["effort"].min(),
                "max_effort":  grp["effort"].max(),
            })
    return pd.DataFrame(rows).sort_values(["r4_label", "r4_stage"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def _build_gap_analysis(
    combined_df: pd.DataFrame,
    gap_threshold: int = 1,
) -> pd.DataFrame:
    """Subjects where sensor capacity stage and R4 stage diverge > threshold."""
    df = combined_df.copy()
    df["stage_r4_gap"] = (df["capacity_stage"] - df["r4_stage"]).abs()
    gap_df = df[df["stage_r4_gap"] > gap_threshold].copy()
    return gap_df.sort_values(["domain", "stage_r4_gap"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_outputs(
    validation_result: ValidationResult,
    merged_df: pd.DataFrame,
    output_dir: Path,
    gap_threshold: int = 1,
) -> None:
    """Write all output CSVs and print summary tables to stdout.

    Parameters
    ----------
    validation_result:
        Output of :func:`correlation.validator.validate`.
    merged_df:
        Merged sensor + R4 DataFrame from :func:`correlation.merger.load_and_merge`.
    output_dir:
        Directory in which to write output files.
    gap_threshold:
        Stages of divergence to flag in gap analysis (default 1 = only
        subjects differing by 2+ stages are flagged).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # r4_correlation.csv — full correlation table
    # ------------------------------------------------------------------
    corr_df = results_to_dataframe(validation_result.correlations)
    corr_path = output_dir / "r4_correlation.csv"
    corr_df.to_csv(corr_path, index=False)
    logger.info(f"Wrote {corr_path}")

    # ------------------------------------------------------------------
    # modality_r4_correlation.csv — modality sub-table
    # ------------------------------------------------------------------
    mod_df = corr_df[
        (corr_df["comparison"] == "effort_vs_r4") &
        (corr_df["modality"] != "overall")
    ].copy()
    mod_path = output_dir / "modality_r4_correlation.csv"
    mod_df.to_csv(mod_path, index=False)
    logger.info(f"Wrote {mod_path}")

    # ------------------------------------------------------------------
    # combined_analysis.csv — per-subject × domain ICF-mapped table
    # ------------------------------------------------------------------
    if validation_result.combined_df is not None:
        combined_path = output_dir / "combined_analysis.csv"
        validation_result.combined_df.to_csv(combined_path, index=False)
        logger.info(f"Wrote {combined_path}")

        # stage_effort_r4_summary.csv
        summary_df = _build_stage_effort_r4_summary(validation_result.combined_df)
        summary_path = output_dir / "stage_effort_r4_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Wrote {summary_path}")

        # gap_analysis.csv
        gap_df = _build_gap_analysis(validation_result.combined_df, gap_threshold)
        gap_path = output_dir / "gap_analysis.csv"
        gap_df.to_csv(gap_path, index=False)
        n_gap = len(gap_df["subject_id"].unique()) if not gap_df.empty else 0
        logger.info(
            f"Wrote {gap_path} "
            f"({n_gap} subjects with |sensor_stage - R4| > {gap_threshold})"
        )

    # ------------------------------------------------------------------
    # Print summary to stdout
    # ------------------------------------------------------------------
    _print_correlation_table(corr_df)
    _print_modality_table(corr_df)
