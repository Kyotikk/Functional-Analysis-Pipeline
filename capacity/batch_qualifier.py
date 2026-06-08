"""Batch capacity scoring across a HR-metric-extractor output directory."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .qualifier import SubjectCapacityResult, score_subject
from .rules import DomainRules, load_rules

logger = logging.getLogger(__name__)


def run_batch(
    batch_dir: Path,
    domain_rules: Dict[str, DomainRules],
    output_dir: Optional[Path] = None,
    subject_glob: str = "sub_*",
) -> Tuple[pd.DataFrame, List[SubjectCapacityResult]]:
    """Score all subjects in a batch directory.

    Parameters
    ----------
    batch_dir:
        Root directory produced by HR-metric-extractor batch processing.
        Expects sub-directories matching ``subject_glob``.
    domain_rules:
        Loaded rules dict from ``load_rules()``.
    output_dir:
        If provided, writes ``capacity_scores.csv`` (summary) and
        ``capacity_evidence.csv`` (per-stage trace) here.
    subject_glob:
        Glob pattern for subject sub-directories.

    Returns
    -------
    summary_df, results
        ``summary_df`` is a DataFrame with one row per subject containing
        assigned stages and ``*_capped`` flags.
        ``results`` is the list of raw ``SubjectCapacityResult`` objects.
    """
    subject_dirs = sorted(
        d for d in batch_dir.iterdir() if d.is_dir() and d.match(subject_glob)
    )

    if not subject_dirs:
        logger.warning(
            "No subject directories matching '%s' found in %s", subject_glob, batch_dir
        )
        return pd.DataFrame(), []

    results: List[SubjectCapacityResult] = []
    for subject_dir in subject_dirs:
        try:
            result = score_subject(subject_dir, domain_rules)
            results.append(result)
            logger.info("Scored %-15s  stages: %s", result.subject_id, result.r4_vector)
        except Exception as exc:
            logger.error("Failed to score %s: %s", subject_dir.name, exc)

    if not results:
        return pd.DataFrame(), []

    summary_df = pd.DataFrame([r.to_summary_row() for r in results])

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / "capacity_scores.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved summary  → %s", summary_path)

        evidence_rows = [row for r in results for row in r.to_evidence_rows()]
        evidence_df = pd.DataFrame(evidence_rows)
        evidence_path = output_dir / "capacity_evidence.csv"
        evidence_df.to_csv(evidence_path, index=False)
        logger.info("Saved evidence → %s", evidence_path)

    return summary_df, results


def run_batch_from_config(
    batch_dir: Path,
    rules_path: Path,
    output_dir: Optional[Path] = None,
    subject_glob: str = "sub_*",
) -> Tuple[pd.DataFrame, List[SubjectCapacityResult]]:
    """Convenience wrapper that loads rules from a YAML path before running."""
    domain_rules = load_rules(rules_path)
    return run_batch(
        batch_dir=batch_dir,
        domain_rules=domain_rules,
        output_dir=output_dir,
        subject_glob=subject_glob,
    )
