"""Merge sensor-derived capacity/effort scores with clinical R4 scores.

Subject-ID normalisation
------------------------
The R4 CSV uses ``subj_A1`` style while the sensor pipeline uses ``sub_A1``.
All IDs are normalised to lowercase ``sub_*`` before joining:
  - ``subj_`` prefix → ``sub_``
  - ``Sub_`` (mixed case) → ``sub_``
  - underscore-separated digits/letters are kept as-is after the prefix strip

The join is an inner join: only subjects present in *all three* inputs are kept.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Domains for which we have sensor pipeline outputs
_DOMAINS = ["Basic Movements", "Walking", "Oral Care", "Grooming"]


def _normalise_id(subject_id: str) -> str:
    """Normalise subject IDs to lowercase ``sub_*`` form."""
    s = str(subject_id).strip().lower()
    # Replace leading subj_ or sub_ variants (including capitalised)
    s = re.sub(r"^subj_", "sub_", s)
    return s


def load_and_merge(
    capacity_csv: Path,
    effort_csv: Path,
    r4_csv: Path,
    reliable_only: bool = False,
) -> pd.DataFrame:
    """Load, normalise, and inner-join the three score tables.

    Parameters
    ----------
    capacity_csv:
        Output of ``run_capacity.py`` — columns: ``subject_id``,
        ``{domain}``, ``{domain}_capped`` for each domain.
    effort_csv:
        Output of ``run_effort.py`` — columns: ``subject_id``,
        ``{domain}_effort``, ``{domain}_effort_reliable``,
        ``{domain}_effort_{modality}`` …
    r4_csv:
        Clinical R4 score sheet — column ``Participant`` + one column
        per R4 domain (14 domains total).
    reliable_only:
        If True, rows where *all* domain effort scores are unreliable
        are excluded from the merged output.  Individual per-row
        reliability is preserved in ``{domain}_effort_reliable`` columns
        for downstream filtering.

    Returns
    -------
    DataFrame with columns:

    - ``subject_id`` (canonical ``sub_*`` form)
    - ``{domain}_stage``         — sensor capacity stage (int 1–5)
    - ``{domain}_effort``        — sensor effort score (float -100–100, or NaN)
    - ``{domain}_effort_reliable`` — bool
    - ``{domain}_effort_{mod}``  — per-modality effort scores
    - ``{domain}_r4``            — clinical R4 stage (float 1–5)

    for all domains in ``_DOMAINS``.
    """
    # ------------------------------------------------------------------
    # Load raw tables
    # ------------------------------------------------------------------
    cap_df = pd.read_csv(capacity_csv)
    eff_df = pd.read_csv(effort_csv)
    r4_df  = pd.read_csv(r4_csv)

    # ------------------------------------------------------------------
    # Normalise subject IDs
    # ------------------------------------------------------------------
    cap_df["subject_id"] = cap_df["subject_id"].apply(_normalise_id)
    eff_df["subject_id"] = eff_df["subject_id"].apply(_normalise_id)
    r4_df["subject_id"]  = r4_df["Participant"].apply(_normalise_id)
    r4_df = r4_df.drop(columns=["Participant"])

    # ------------------------------------------------------------------
    # Rename capacity columns: "{domain}" → "{domain}_stage"
    # Keep only the 4 MVP domains; drop the _capped flags
    # ------------------------------------------------------------------
    cap_rename = {d: f"{d}_stage" for d in _DOMAINS}
    cap_keep   = ["subject_id"] + [f"{d}_stage" for d in _DOMAINS]
    cap_df = cap_df.rename(columns=cap_rename)[cap_keep]

    # ------------------------------------------------------------------
    # Rename R4 columns: "{domain}" → "{domain}_r4"
    # Keep only MVP domains
    # ------------------------------------------------------------------
    r4_rename = {d: f"{d}_r4" for d in _DOMAINS}
    r4_keep   = ["subject_id"] + [f"{d}_r4" for d in _DOMAINS]
    r4_df = r4_df.rename(columns=r4_rename)[r4_keep]

    # ------------------------------------------------------------------
    # Inner join all three
    # ------------------------------------------------------------------
    merged = (
        cap_df
        .merge(eff_df,  on="subject_id", how="inner")
        .merge(r4_df,   on="subject_id", how="inner")
    )

    n = len(merged)
    logger.info(f"Merged dataset: {n} subjects after inner join")

    if n < 5:
        logger.warning(
            f"Only {n} subjects matched across all three inputs — "
            "check ID normalisation."
        )

    # Log dropped subjects per source
    all_cap = set(cap_df["subject_id"])
    all_eff = set(eff_df["subject_id"])
    all_r4  = set(r4_df["subject_id"])
    matched = set(merged["subject_id"])

    dropped = (all_cap | all_eff | all_r4) - matched
    if dropped:
        logger.info(f"Subjects not matched (dropped): {sorted(dropped)}")

    # ------------------------------------------------------------------
    # Optional: keep only rows where at least one domain effort is reliable
    # ------------------------------------------------------------------
    if reliable_only:
        reliable_cols = [f"{d}_effort_reliable" for d in _DOMAINS
                         if f"{d}_effort_reliable" in merged.columns]
        if reliable_cols:
            mask = merged[reliable_cols].any(axis=1)
            n_before = len(merged)
            merged = merged[mask].copy()
            logger.info(
                f"reliable_only=True: kept {len(merged)}/{n_before} subjects "
                "with at least one reliable domain effort score"
            )

    merged = merged.reset_index(drop=True)
    return merged
