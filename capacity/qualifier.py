"""Core capacity qualifier engine.

Loads HR-metric-extractor activity interval files for a single subject and
evaluates R4 stage criteria against them, producing a fully explainable result.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .rules import DomainRules, StageCheck, load_rules

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Map rule 'source' keys → CSV filenames inside a subject output directory.
# ---------------------------------------------------------------------------
_SOURCE_FILE_MAP: Dict[str, str] = {
    "activity_standing": "activity_standing.csv",
    "activity_transfer": "activity_transfer.csv",
    "activity_bed_transfer": "activity_bed_transfer.csv",
    "activity_dental_care": "activity_dental_care.csv",
    "activity_put_toothpaste": "activity_put_toothpaste.csv",
    "activity_rinse_mouth": "activity_rinse_mouth.csv",
    "activity_hair_care": "activity_hair_care.csv",
    "activity_washing_face": "activity_washing_face.csv",
    "activity_washing_hands": "activity_washing_hands.csv",
    "propulsion": "propulsion_activities.csv",
    "resting": "resting_activities.csv",
}

_REQUIRED_COLUMNS = {"activity", "t_start", "t_end", "duration_sec"}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CheckEvidence:
    """Evidence produced by evaluating one StageCheck."""

    source: str
    n_matched_intervals: int
    matched_activities: List[str]
    total_duration_sec: float
    satisfied: bool


@dataclass
class StageEvidence:
    """Evidence produced by evaluating one stage within a domain."""

    stage: int
    description: str
    satisfied: bool
    not_assessable: bool
    check_evidence: List[CheckEvidence] = field(default_factory=list)
    note: str = ""


@dataclass
class DomainResult:
    """Capacity scoring result for a single domain."""

    domain_name: str
    r4_label: str
    assigned_stage: int
    """The R4 stage assigned (1–5). Stage 1 is the default when no evidence is found."""

    capped_by_not_assessable: bool
    """True when one or more higher stages were skipped due to not_assessable=True,
    meaning the true stage could be higher than assigned."""

    stage_evidence: Dict[int, StageEvidence]
    """Full evidence trace keyed by stage number (for audit/explainability)."""


@dataclass
class SubjectCapacityResult:
    """Complete capacity scoring result for one subject."""

    subject_id: str
    domain_results: Dict[str, DomainResult]

    @property
    def r4_vector(self) -> Dict[str, int]:
        """Compact dict: r4_label → assigned_stage."""
        return {v.r4_label: v.assigned_stage for v in self.domain_results.values()}

    def to_summary_row(self) -> dict:
        """Return a flat dict suitable for a summary CSV row."""
        row: dict = {"subject_id": self.subject_id}
        for result in self.domain_results.values():
            row[result.r4_label] = result.assigned_stage
            row[f"{result.r4_label}_capped"] = result.capped_by_not_assessable
        return row

    def to_evidence_rows(self) -> List[dict]:
        """Return one row per domain×stage for a detailed evidence CSV."""
        rows = []
        for domain_key, result in self.domain_results.items():
            for stage_num, ev in result.stage_evidence.items():
                row: dict = {
                    "subject_id": self.subject_id,
                    "domain": result.r4_label,
                    "stage": stage_num,
                    "description": ev.description,
                    "not_assessable": ev.not_assessable,
                    "satisfied": ev.satisfied,
                    "assigned": result.assigned_stage == stage_num,
                }
                for i, ce in enumerate(ev.check_evidence):
                    row[f"check_{i}_source"] = ce.source
                    row[f"check_{i}_n_intervals"] = ce.n_matched_intervals
                    row[f"check_{i}_total_dur_sec"] = round(ce.total_duration_sec, 1)
                    row[f"check_{i}_satisfied"] = ce.satisfied
                    row[f"check_{i}_activities"] = "|".join(ce.matched_activities)
                rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Activity file loading
# ---------------------------------------------------------------------------


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["activity", "t_start", "t_end", "duration_sec"])


def load_activity_files(subject_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all recognised activity CSV files from a subject output directory.

    Missing files or files without the required columns are silently replaced
    with empty DataFrames so that downstream checks treat them as 'no evidence'.
    """
    result: Dict[str, pd.DataFrame] = {}

    for source_key, filename in _SOURCE_FILE_MAP.items():
        filepath = subject_dir / filename
        if not filepath.exists():
            result[source_key] = _empty_df()
            continue

        try:
            df = pd.read_csv(filepath)
        except Exception as exc:
            logger.warning("Could not read %s for %s: %s", filename, subject_dir.name, exc)
            result[source_key] = _empty_df()
            continue

        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing or df.empty:
            result[source_key] = _empty_df()
            continue

        df = df.copy()
        df["activity"] = df["activity"].astype(str).str.strip().str.lower()
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0.0)
        df["t_start"] = pd.to_numeric(df["t_start"], errors="coerce")
        df["t_end"] = pd.to_numeric(df["t_end"], errors="coerce")
        # Drop rows with negative or zero duration (invalid intervals)
        df = df[df["duration_sec"] > 0].reset_index(drop=True)
        result[source_key] = df

    return result


# ---------------------------------------------------------------------------
# Stage evaluation
# ---------------------------------------------------------------------------


def _apply_check(check: StageCheck, activities: Dict[str, pd.DataFrame]) -> CheckEvidence:
    """Evaluate one StageCheck against the loaded activity DataFrames."""
    df = activities.get(check.source, _empty_df())

    if df.empty:
        return CheckEvidence(
            source=check.source,
            n_matched_intervals=0,
            matched_activities=[],
            total_duration_sec=0.0,
            satisfied=False,
        )

    mask = pd.Series(True, index=df.index)

    # Include filter (OR across keywords)
    if check.keywords:
        kw_lower = [k.lower() for k in check.keywords]
        mask &= df["activity"].apply(lambda a: any(kw in a for kw in kw_lower))

    # Exclude filter
    if check.exclude_keywords:
        excl_lower = [k.lower() for k in check.exclude_keywords]
        mask &= ~df["activity"].apply(lambda a: any(ex in a for ex in excl_lower))

    # Duration filter
    if check.min_duration_sec > 0:
        mask &= df["duration_sec"] >= check.min_duration_sec

    matched = df[mask]
    n = len(matched)

    return CheckEvidence(
        source=check.source,
        n_matched_intervals=n,
        matched_activities=sorted(matched["activity"].unique().tolist()) if n > 0 else [],
        total_duration_sec=float(matched["duration_sec"].sum()) if n > 0 else 0.0,
        satisfied=n >= check.min_occurrences,
    )


def _evaluate_stage(
    stage_num: int,
    stage_rule,
    activities: Dict[str, pd.DataFrame],
) -> StageEvidence:
    """Evaluate all checks for a stage and return combined evidence."""
    if stage_rule.not_assessable:
        return StageEvidence(
            stage=stage_num,
            description=stage_rule.description,
            satisfied=False,
            not_assessable=True,
            check_evidence=[],
            note=stage_rule.note,
        )

    check_evidences = [_apply_check(c, activities) for c in stage_rule.checks]
    # Stage is satisfied only if ALL checks pass (AND logic).
    # A stage with no checks (empty list) is not satisfied.
    all_satisfied = bool(check_evidences) and all(ce.satisfied for ce in check_evidences)

    return StageEvidence(
        stage=stage_num,
        description=stage_rule.description,
        satisfied=all_satisfied,
        not_assessable=False,
        check_evidence=check_evidences,
        note=stage_rule.note,
    )


# ---------------------------------------------------------------------------
# Domain and subject scoring
# ---------------------------------------------------------------------------


def assign_domain_stage(
    domain_rules: DomainRules,
    activities: Dict[str, pd.DataFrame],
) -> DomainResult:
    """Assign the highest satisfied R4 stage for one domain.

    Algorithm:
      1. Evaluate all defined stages (2–5) against the evidence.
      2. Scan from Stage 5 downward; skip not_assessable stages.
      3. Return the first stage whose checks all pass.
      4. Default to Stage 1 if no stage is satisfied.
    """
    stage_numbers = sorted(domain_rules.stages.keys(), reverse=True)  # 5 → 4 → 3 → 2 ...

    # Evaluate every stage up-front for the full evidence trace.
    stage_evidence: Dict[int, StageEvidence] = {
        s: _evaluate_stage(s, domain_rules.stages[s], activities)
        for s in stage_numbers
    }

    assigned_stage = 1  # default
    capped_by_not_assessable = False

    for stage_num in stage_numbers:
        ev = stage_evidence[stage_num]
        if ev.not_assessable:
            continue
        if ev.satisfied:
            assigned_stage = stage_num
            # Flag if any higher stages were skipped due to not_assessable
            higher = [s for s in stage_numbers if s > stage_num]
            capped_by_not_assessable = any(stage_evidence[s].not_assessable for s in higher)
            break

    return DomainResult(
        domain_name=domain_rules.name,
        r4_label=domain_rules.r4_label,
        assigned_stage=assigned_stage,
        capped_by_not_assessable=capped_by_not_assessable,
        stage_evidence=stage_evidence,
    )


def score_subject(
    subject_dir: Path,
    domain_rules: Dict[str, DomainRules],
    subject_id: Optional[str] = None,
) -> SubjectCapacityResult:
    """Score all domains for a single subject directory."""
    sid = subject_id or subject_dir.name
    activities = load_activity_files(subject_dir)
    domain_results = {
        key: assign_domain_stage(rules, activities)
        for key, rules in domain_rules.items()
    }
    return SubjectCapacityResult(subject_id=sid, domain_results=domain_results)
