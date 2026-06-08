#!/usr/bin/env python
"""CLI for computing feature importance (Spearman correlation with R4 stage).

Usage
-----
python run_feature_importance.py \\
    --patient-batch-dir HR-metric-extractor/output_batch/batch_20260407_213748_real_run1 \\
    --hc-batch-dir      HR-metric-extractor/output_batch/batch_20260403_232300_HC_window_run \\
    --capacity-scores   output/capacity/batch_20260407_213748_real_run1/capacity_scores.csv \\
    --output-dir        output/feature_importance/run1 \\
    --top               10
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from effort.reference import load_effort_config
from effort.feature_importance import (
    compute_feature_importance,
    print_top_features,
    plot_feature_importance,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def main() -> None:
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Rank features by Spearman correlation with R4 capacity stage."
    )
    parser.add_argument("--patient-batch-dir", required=True, type=Path)
    parser.add_argument("--hc-batch-dir", required=True, type=Path)
    parser.add_argument(
        "--capacity-scores", required=True, type=Path,
        help="Path to capacity_scores.csv produced by run_capacity.py",
    )
    parser.add_argument(
        "--config", type=Path,
        default=script_dir / "config" / "effort_config.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=script_dir / "output" / "feature_importance",
    )
    parser.add_argument(
        "--subject-glob", default="sub_*",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Print top-N features per domain × activity (default: 10).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate PNG visualisations from the feature-importance table.",
    )
    parser.add_argument(
        "--plot-top", type=int, default=10,
        help="Top-N features per activity shown in the top-features plot (default: 10).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    for label, p in [
        ("--patient-batch-dir", args.patient_batch_dir),
        ("--hc-batch-dir", args.hc_batch_dir),
        ("--capacity-scores", args.capacity_scores),
        ("--config", args.config),
    ]:
        if not p.exists():
            logger.error("%s does not exist: %s", label, p)
            sys.exit(1)

    config = load_effort_config(args.config)

    df = compute_feature_importance(
        patient_batch_dir=args.patient_batch_dir,
        hc_batch_dir=args.hc_batch_dir,
        config=config,
        capacity_scores_csv=args.capacity_scores,
        output_dir=args.output_dir,
        subject_glob=args.subject_glob,
    )

    if not df.empty:
        print_top_features(df, top_n=args.top)
        print(f"\nFull table saved to: {args.output_dir / 'feature_importance.csv'}")
        if args.plot:
            plot_paths = plot_feature_importance(df, output_dir=args.output_dir, top_n=args.plot_top)
            if plot_paths:
                print("\nGenerated plots:")
                for p in plot_paths:
                    print(f"- {p}")


if __name__ == "__main__":
    main()
