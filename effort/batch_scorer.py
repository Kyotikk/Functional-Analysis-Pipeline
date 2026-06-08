"""Batch effort scoring: iterate a directory of subjects and produce CSVs."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .reference import ActivityReference, EffortConfig, build_reference, load_effort_config
from .scorer import SubjectEffortResult, score_subject

logger = logging.getLogger(__name__)


def run_batch(
    patient_batch_dir: Path,
    hc_batch_dir: Path,
    config: EffortConfig,
    output_dir: Path,
    patient_subject_glob: str = "sub_*",
    hc_subject_glob: str = "sub_*",
) -> None:
    """Build HC reference, score all patients, write output CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build reference profiles
    references: Dict[str, ActivityReference] = build_reference(
        hc_batch_dir, config, subject_glob=hc_subject_glob
    )

    subject_dirs = sorted(
        d for d in patient_batch_dir.iterdir()
        if d.is_dir() and d.match(patient_subject_glob)
    )
    logger.info("Scoring %d subjects from %s", len(subject_dirs), patient_batch_dir)

    summary_rows = []
    detail_rows = []

    for sdir in subject_dirs:
        result: SubjectEffortResult = score_subject(sdir, config, references)
        summary_rows.append(result.to_summary_row())
        detail_rows.extend(result.to_activity_rows())
        logger.debug("Scored %s", result.subject_id)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_path = output_dir / "effort_scores.csv"
    detail_path = output_dir / "effort_details.csv"

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    logger.info("Wrote %s (%d subjects)", summary_path, len(summary_df))
    logger.info("Wrote %s (%d rows)", detail_path, len(detail_df))


def run_batch_from_config(
    patient_batch_dir: Path,
    hc_batch_dir: Path,
    config_path: Path,
    output_dir: Optional[Path] = None,
    patient_subject_glob: str = "sub_*",
    hc_subject_glob: str = "sub_*",
) -> None:
    config = load_effort_config(config_path)

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            patient_batch_dir.parent.parent  # up from batch dir
            / "functional-analysis-pipeline"
            / "output"
            / "effort"
            / f"batch_{stamp}"
        )

    run_batch(
        patient_batch_dir,
        hc_batch_dir,
        config,
        output_dir,
        patient_subject_glob=patient_subject_glob,
        hc_subject_glob=hc_subject_glob,
    )
