#!/usr/bin/env python
"""CLI entry point for Phase 4 — R4 clinical validation (ICF-inspired).

Usage examples
--------------
# Minimal (uses defaults for config paths)
python run_correlation.py \\
    --capacity-csv output/capacity/batch_20260407_213748_real_run1/capacity_scores.csv \\
    --effort-csv   output/effort/real_run2/effort_scores.csv \\
    --r4-csv       ../HR-metric-extractor/R4-scores/R4_scores_nursing_home.csv \\
    --output-dir   output/correlation/run2

# Reliable effort scores only
python run_correlation.py \\
    --capacity-csv output/capacity/batch_20260407_213748_real_run1/capacity_scores.csv \\
    --effort-csv   output/effort/real_run2/effort_scores.csv \\
    --r4-csv       ../HR-metric-extractor/R4-scores/R4_scores_nursing_home.csv \\
    --output-dir   output/correlation/run2_reliable \\
    --reliable-only

# Custom bootstrap count
python run_correlation.py ... --n-bootstrap 2000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from correlation.merger    import load_and_merge
from correlation.validator import validate
from correlation.reporter  import write_outputs


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 4: correlate sensor capacity/effort scores with clinical R4 stages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--capacity-csv", required=True,
        help="Path to capacity_scores.csv (output of run_capacity.py).",
    )
    p.add_argument(
        "--effort-csv", required=True,
        help="Path to effort_scores.csv (output of run_effort.py).",
    )
    p.add_argument(
        "--r4-csv", required=True,
        help="Path to clinical R4 scores CSV (R4_scores_nursing_home.csv).",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory in which to write output CSVs.",
    )
    p.add_argument(
        "--reliable-only", action="store_true", default=False,
        help=(
            "Exclude subjects with all-unreliable effort scores from the merged "
            "table.  Individual per-domain reliability is still preserved in the "
            "output CSVs regardless of this flag."
        ),
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of bootstrap resamples for 95%% CI estimation.",
    )
    p.add_argument(
        "--gap-threshold", type=int, default=1,
        help=(
            "Minimum stage gap (|sensor_stage - R4|) to flag a subject in the "
            "gap analysis output (default: 1 means only gaps >= 2 are flagged)."
        ),
    )
    p.add_argument("--verbose", action="store_true", default=False)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    capacity_csv = Path(args.capacity_csv)
    effort_csv   = Path(args.effort_csv)
    r4_csv       = Path(args.r4_csv)
    output_dir   = Path(args.output_dir)

    # Validate inputs
    for p, name in [
        (capacity_csv, "--capacity-csv"),
        (effort_csv,   "--effort-csv"),
        (r4_csv,       "--r4-csv"),
    ]:
        if not p.exists():
            logger.error(f"{name}: file not found: {p}")
            return 1

    # ------------------------------------------------------------------
    # Step 1: Merge
    # ------------------------------------------------------------------
    logger.info("Step 1/3 — Merging capacity, effort, and R4 scores …")
    merged_df = load_and_merge(
        capacity_csv=capacity_csv,
        effort_csv=effort_csv,
        r4_csv=r4_csv,
        reliable_only=args.reliable_only,
    )
    logger.info(f"  Merged: {len(merged_df)} subjects")

    if merged_df.empty:
        logger.error("No subjects after merge — check input files and ID formats.")
        return 1

    # ------------------------------------------------------------------
    # Step 2: Validate
    # ------------------------------------------------------------------
    logger.info(f"Step 2/3 — Running correlations (n_bootstrap={args.n_bootstrap}) …")
    validation_result = validate(merged_df, n_bootstrap=args.n_bootstrap)

    n_results = len(validation_result.correlations)
    logger.info(f"  Computed {n_results} correlation results")

    # ------------------------------------------------------------------
    # Step 3: Write outputs
    # ------------------------------------------------------------------
    logger.info(f"Step 3/3 — Writing outputs to {output_dir} …")
    write_outputs(
        validation_result=validation_result,
        merged_df=merged_df,
        output_dir=output_dir,
        gap_threshold=args.gap_threshold,
    )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
