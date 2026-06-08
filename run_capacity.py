#!/usr/bin/env python3
"""
Functional Analysis Pipeline — Capacity Qualifier

Assigns R4 capacity stages (1–5) per domain to all subjects in a
HR-metric-extractor batch output directory.

Usage examples
--------------
# Score a single batch:
python run_capacity.py \
    --batch-dir HR-metric-extractor/output_batch/batch_20260407_213748_real_run1 \
    --rules functional-analysis-pipeline/config/capacity_rules.yaml \
    --output-dir functional-analysis-pipeline/output/capacity

# Score multiple batches together (e.g. HC + nursing home):
python run_capacity.py \
    --batch-dir batch_HC batch_patients \
    --rules functional-analysis-pipeline/config/capacity_rules.yaml \
    --output-dir functional-analysis-pipeline/output/capacity

# Compare against ground-truth R4 labels:
python run_capacity.py \
    --batch-dir ... --rules ... --output-dir ... \
    --icf-labels HR-metric-extractor/R4-scores/R4_scores_nursing_home.csv \
    --icf-id-col Participant
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

from capacity import run_batch_from_config
from capacity.rules import load_rules

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# R4 label columns expected in the ICF ground-truth CSV (order matters for display)
_DEFAULT_R4_COLUMNS = ["Basic Movements", "Walking", "Oral Care", "Grooming"]


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assign R4 capacity stages from HR-metric-extractor batch outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--batch-dir",
        nargs="+",
        required=True,
        metavar="DIR",
        help="One or more HR-metric-extractor batch output directories.",
    )
    p.add_argument(
        "--rules",
        default=Path(__file__).parent / "config" / "capacity_rules.yaml",
        type=Path,
        metavar="YAML",
        help="Path to capacity_rules.yaml (default: config/capacity_rules.yaml).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for output CSVs. If omitted, results are printed only.",
    )
    p.add_argument(
        "--subject-glob",
        default="sub_*",
        metavar="GLOB",
        help="Glob pattern for subject sub-directories (default: sub_*).",
    )
    p.add_argument(
        "--icf-labels",
        type=Path,
        default=None,
        metavar="CSV",
        help="Optional ground-truth R4 CSV for accuracy comparison.",
    )
    p.add_argument(
        "--icf-id-col",
        default="Participant",
        metavar="COL",
        help="Subject ID column name in the ICF labels CSV (default: Participant).",
    )
    p.add_argument(
        "--icf-r4-cols",
        nargs="+",
        default=_DEFAULT_R4_COLUMNS,
        metavar="COL",
        help="R4 domain columns in the ICF labels CSV.",
    )
    return p.parse_args(argv)


def _merge_and_compare(
    summary: pd.DataFrame,
    icf_path: Path,
    id_col: str,
    r4_cols: List[str],
) -> pd.DataFrame:
    """Merge predicted stages with ground-truth labels and report accuracy."""
    gt = pd.read_csv(icf_path)
    gt.columns = [c.strip() for c in gt.columns]

    if id_col not in gt.columns:
        logger.error(
            "ID column '%s' not found in %s. Available: %s",
            id_col, icf_path, list(gt.columns),
        )
        return summary

    gt = gt.rename(columns={id_col: "subject_id"})
    gt["subject_id"] = gt["subject_id"].astype(str).str.strip()
    summary["subject_id"] = summary["subject_id"].astype(str).str.strip()

    merged = summary.merge(
        gt[["subject_id"] + [c for c in r4_cols if c in gt.columns]],
        on="subject_id",
        how="left",
        suffixes=("_pred", "_gt"),
    )

    print("\n── Stage comparison (predicted vs. ground truth) ──────────────────")
    for col in r4_cols:
        pred_col = col if col in merged.columns else f"{col}_pred"
        gt_col = f"{col}_gt"
        if pred_col not in merged.columns or gt_col not in merged.columns:
            continue
        valid = merged[[pred_col, gt_col]].dropna()
        if valid.empty:
            continue
        exact = (valid[pred_col] == valid[gt_col]).mean()
        within1 = (abs(valid[pred_col] - valid[gt_col]) <= 1).mean()
        mae = abs(valid[pred_col] - valid[gt_col]).mean()
        print(f"  {col:20s}  exact={exact:.0%}  within±1={within1:.0%}  MAE={mae:.2f}  (n={len(valid)})")

    return merged


def main(argv=None) -> int:
    args = _parse_args(argv)

    if not args.rules.exists():
        logger.error("Rules file not found: %s", args.rules)
        return 1

    domain_rules = load_rules(args.rules)
    logger.info("Loaded rules for domains: %s", list(domain_rules.keys()))

    all_summaries = []
    for batch_path_str in args.batch_dir:
        batch_path = Path(batch_path_str)
        if not batch_path.exists():
            logger.error("Batch directory not found: %s", batch_path)
            continue

        logger.info("Processing batch: %s", batch_path)
        summary_df, _ = run_batch_from_config(
            batch_dir=batch_path,
            rules_path=args.rules,
            output_dir=args.output_dir / batch_path.name if args.output_dir else None,
            subject_glob=args.subject_glob,
        )
        if not summary_df.empty:
            summary_df["batch"] = batch_path.name
            all_summaries.append(summary_df)

    if not all_summaries:
        logger.warning("No subjects were scored.")
        return 0

    combined = pd.concat(all_summaries, ignore_index=True)

    # Save combined summary if multiple batches
    if args.output_dir and len(args.batch_dir) > 1:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "capacity_scores_combined.csv"
        combined.to_csv(out_path, index=False)
        logger.info("Saved combined scores → %s", out_path)

    # Print summary table
    display_cols = ["subject_id", "batch"] + [
        c for c in combined.columns
        if c not in ("subject_id", "batch") and not c.endswith("_capped")
    ]
    display_cols = [c for c in display_cols if c in combined.columns]
    print("\n── Capacity scores ────────────────────────────────────────────────")
    print(combined[display_cols].to_string(index=False))

    # Report capped-flag subjects
    capped_cols = [c for c in combined.columns if c.endswith("_capped")]
    for cc in capped_cols:
        capped = combined[combined[cc] == True]["subject_id"].tolist()
        if capped:
            domain = cc.replace("_capped", "")
            print(f"\n  NOTE ({domain}): {len(capped)} subject(s) may have higher capacity "
                  f"than assigned (not_assessable stages skipped): {capped}")

    # Optional ground-truth comparison
    if args.icf_labels:
        if not args.icf_labels.exists():
            logger.error("ICF labels file not found: %s", args.icf_labels)
        else:
            _merge_and_compare(combined, args.icf_labels, args.icf_id_col, args.icf_r4_cols)

    return 0


if __name__ == "__main__":
    sys.exit(main())
