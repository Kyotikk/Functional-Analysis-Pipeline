#!/usr/bin/env python
"""CLI entry point for the physiological effort scorer (Phase 3).

Usage examples
--------------
# Score patients against HC reference, using default config
python run_effort.py \\
    --patient-batch-dir HR-metric-extractor/output_batch/real_run2 \\
    --hc-batch-dir     HR-metric-extractor/output_batch/hc_run_2 \\
    --output-dir       output/effort/run1

# Specify a custom config
python run_effort.py \\
    --patient-batch-dir ... \\
    --hc-batch-dir     ... \\
    --config           config/effort_config.yaml \\
    --output-dir       output/effort/run1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from effort.reference import build_reference, load_effort_config
from effort.scorer import score_subject, SubjectEffortResult
from effort.batch_scorer import run_batch


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _print_summary_table(summary_df: pd.DataFrame) -> None:
    effort_cols = [c for c in summary_df.columns if c.endswith("_effort") and not c.endswith("_reliable")]
    reliable_cols = [c for c in summary_df.columns if c.endswith("_effort_reliable")]

    print("\n=== Effort Scores (-100 to +100, 0 = HC baseline, higher = more effort) ===\n")
    display = summary_df[["subject_id"] + effort_cols].copy()
    # Flag unreliable scores with *
    for ec in effort_cols:
        rc = ec + "_reliable"
        if rc in summary_df.columns:
            display[ec] = display.apply(
                lambda row: f"{row[ec]:.1f}*" if row.get(rc) is False else (
                    f"{row[ec]:.1f}" if row[ec] is not None else "N/A"
                ),
                axis=1,
            )
    print(display.to_string(index=False))
    print("\n* = at least one activity has low reliability (window count or feature coverage)")


def main() -> None:
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Score physiological effort against HC reference population."
    )
    parser.add_argument(
        "--patient-batch-dir", required=True, type=Path,
        help="Directory containing patient subject subdirectories.",
    )
    parser.add_argument(
        "--hc-batch-dir", required=True, type=Path,
        help="Directory containing healthy-control subject subdirectories.",
    )
    parser.add_argument(
        "--config", type=Path,
        default=script_dir / "config" / "effort_config.yaml",
        help="Path to effort_config.yaml (default: config/effort_config.yaml).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write effort_scores.csv and effort_details.csv. "
             "Defaults to output/effort/<timestamp>/.",
    )
    parser.add_argument(
        "--subject-glob", default="sub_*",
        help="Glob pattern applied to both patient and HC directories unless overridden.",
    )
    parser.add_argument(
        "--patient-subject-glob", default=None,
        help="Optional glob pattern for patient subject directories.",
    )
    parser.add_argument(
        "--hc-subject-glob", default=None,
        help="Optional glob pattern for healthy-control subject directories.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print results without writing CSV files.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate paths
    for label, p in [("--patient-batch-dir", args.patient_batch_dir),
                     ("--hc-batch-dir", args.hc_batch_dir),
                     ("--config", args.config)]:
        if not p.exists():
            logger.error("%s does not exist: %s", label, p)
            sys.exit(1)

    config = load_effort_config(args.config)
    patient_subject_glob = args.patient_subject_glob or args.subject_glob
    hc_subject_glob = args.hc_subject_glob or args.subject_glob

    if args.no_save:
        # Score and print only
        references = build_reference(args.hc_batch_dir, config, subject_glob=hc_subject_glob)
        subject_dirs = sorted(
            d for d in args.patient_batch_dir.iterdir()
            if d.is_dir() and d.match(patient_subject_glob)
        )
        rows = []
        for sdir in subject_dirs:
            result: SubjectEffortResult = score_subject(sdir, config, references)
            rows.append(result.to_summary_row())
        _print_summary_table(pd.DataFrame(rows))
        return

    # Full run with CSV output
    from datetime import datetime
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = script_dir / "output" / "effort" / f"batch_{stamp}"

    run_batch(
        patient_batch_dir=args.patient_batch_dir,
        hc_batch_dir=args.hc_batch_dir,
        config=config,
        output_dir=args.output_dir,
        patient_subject_glob=patient_subject_glob,
        hc_subject_glob=hc_subject_glob,
    )

    # Print summary from saved file
    summary_path = args.output_dir / "effort_scores.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        _print_summary_table(summary_df)
        print(f"\nFull results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
