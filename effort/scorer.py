"""Per-subject effort scoring against the HC reference profiles."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .reference import (
    ActivityReference,
    EffortConfig,
    _augment_with_statistics,
    _per_window_deviation,
    _reduce,
    _select_features,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ActivityEffortResult:
    """Effort score for one subject × one activity."""

    activity_name: str
    raw_score: Optional[float]
    """Median per-window deviation (unnormalised). None if insufficient windows."""

    effort_score: Optional[float]
    """Centered effort score in [-100, 100]. None if insufficient windows."""

    n_windows: int
    n_features_used: int
    reliable: bool
    """True when n_windows >= min_windows and feature coverage is adequate."""

    reliability_note: str = ""
    modality_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    """Per-modality centered effort scores (same -100..100 scale as effort_score)."""


@dataclass
class DomainEffortResult:
    """Aggregated effort score for one subject × one domain."""

    domain_name: str
    r4_label: str
    effort_score: Optional[float]
    """Weighted average of activity effort scores. None if all activities missing."""

    activity_results: Dict[str, ActivityEffortResult] = field(default_factory=dict)
    reliable: bool = False
    reliability_note: str = ""
    modality_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    """Weighted-average per-modality scores across domain activities."""


@dataclass
class SubjectEffortResult:
    """Complete effort scoring result for one subject."""

    subject_id: str
    domain_results: Dict[str, DomainEffortResult]

    def to_summary_row(self) -> dict:
        row: dict = {"subject_id": self.subject_id}
        for dr in self.domain_results.values():
            row[f"{dr.r4_label}_effort"] = round(dr.effort_score, 2) if dr.effort_score is not None else None
            row[f"{dr.r4_label}_effort_reliable"] = dr.reliable
            for mod, score in dr.modality_scores.items():
                row[f"{dr.r4_label}_effort_{mod}"] = round(score, 2) if score is not None else None
        return row

    def to_activity_rows(self) -> List[dict]:
        rows = []
        for dr in self.domain_results.values():
            for ar in dr.activity_results.values():
                modality_cols = {
                    f"effort_{mod}": round(s, 2) if s is not None else None
                    for mod, s in ar.modality_scores.items()
                }
                rows.append({
                    "subject_id": self.subject_id,
                    "domain": dr.r4_label,
                    "activity": ar.activity_name,
                    "raw_score": ar.raw_score,
                    "effort_score": round(ar.effort_score, 2) if ar.effort_score is not None else None,
                    "n_windows": ar.n_windows,
                    "n_features": ar.n_features_used,
                    "reliable": ar.reliable,
                    "note": ar.reliability_note,
                    **modality_cols,
                })
        return rows


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------


def _normalise(
    raw: float,
    anchor_minus_100: float,
    anchor_0: float,
    anchor_100: float,
) -> float:
    """Map raw deviation to centered [-100, 100] scale.

    Piecewise mapping keeps anchor_0 as the neutral baseline:
      - raw >= anchor_0: maps toward +100 using (anchor_0, anchor_100)
      - raw <  anchor_0: maps toward -100 using (anchor_minus_100, anchor_0)
    """
    if raw >= anchor_0:
        if anchor_100 <= anchor_0:
            return 0.0
        scaled = (raw - anchor_0) / (anchor_100 - anchor_0) * 100.0
    else:
        if anchor_0 <= anchor_minus_100:
            return 0.0
        scaled = -(anchor_0 - raw) / (anchor_0 - anchor_minus_100) * 100.0
    return float(np.clip(scaled, -100.0, 100.0))


def score_activity(
    subject_dir: Path,
    act_cfg,                     # ActivityConfig
    reference: ActivityReference,
    scoring_cfg,                 # ScoringConfig
) -> ActivityEffortResult:
    """Score one activity for one subject."""
    fp = subject_dir / act_cfg.file
    if not fp.exists():
        return ActivityEffortResult(
            activity_name=act_cfg.name,
            raw_score=None, effort_score=None,
            n_windows=0, n_features_used=0, reliable=False,
            reliability_note="file not found",
        )

    df = pd.read_csv(fp)
    if df.empty:
        return ActivityEffortResult(
            activity_name=act_cfg.name,
            raw_score=None, effort_score=None,
            n_windows=0, n_features_used=0, reliable=False,
            reliability_note="file empty",
        )

    if scoring_cfg.augment_with_statistics:
        raw_feature_cols = _select_features(df, scoring_cfg.exclude_features)
        df = _augment_with_statistics(
            df,
            raw_feature_cols,
            scoring_cfg.statistical_transforms,
        )

    # Align to reference features (only use features present in both)
    present = [c for c in reference.feature_names if c in df.columns]
    if not present:
        return ActivityEffortResult(
            activity_name=act_cfg.name,
            raw_score=None, effort_score=None,
            n_windows=0, n_features_used=0, reliable=False,
            reliability_note="no matching features",
        )

    feat_df = df[present].apply(pd.to_numeric, errors="coerce")

    # Impute NaNs with the HC median for those features
    ref_idx = [reference.feature_names.index(c) for c in present]
    hc_med_sub = reference.hc_median[ref_idx]
    hc_mad_sub = reference.hc_mad[ref_idx]
    feat_df = feat_df.fillna(pd.Series(hc_med_sub, index=present))

    n_windows = len(feat_df)
    n_features = len(present)

    if n_windows < scoring_cfg.min_windows:
        return ActivityEffortResult(
            activity_name=act_cfg.name,
            raw_score=None, effort_score=None,
            n_windows=n_windows, n_features_used=n_features, reliable=False,
            reliability_note=f"only {n_windows} windows (min {scoring_cfg.min_windows})",
        )

    per_window = _per_window_deviation(
        feat_df, present,
        hc_med_sub, hc_mad_sub,
        scoring_cfg.epsilon, scoring_cfg.feature_reducer,
        scoring_cfg.inverse_feature_patterns,
    )

    raw = _reduce(per_window, scoring_cfg.window_reducer)
    effort = _normalise(
        raw,
        reference.norm_anchor_minus_100,
        reference.norm_anchor_0,
        reference.norm_anchor_100,
    )

    feature_coverage = n_features / len(reference.feature_names)
    reliable = feature_coverage >= 0.5
    note = "" if reliable else f"low feature coverage {feature_coverage:.0%}"

    # Per-modality sub-scores
    modality_scores: Dict[str, Optional[float]] = {}
    for mod_name, prefixes in scoring_cfg.modality_groups.items():
        mod_feats = [f for f in present if any(f.startswith(p) for p in prefixes)]
        if not mod_feats:
            modality_scores[mod_name] = None
            continue
        mod_idx = [reference.feature_names.index(f) for f in mod_feats]
        mod_pw = _per_window_deviation(
            feat_df, mod_feats,
            reference.hc_median[mod_idx], reference.hc_mad[mod_idx],
            scoring_cfg.epsilon, scoring_cfg.feature_reducer,
            scoring_cfg.inverse_feature_patterns,
        )
        mod_raw = _reduce(mod_pw, scoring_cfg.window_reducer)
        modality_scores[mod_name] = _normalise(
            mod_raw,
            reference.norm_anchor_minus_100,
            reference.norm_anchor_0,
            reference.norm_anchor_100,
        )

    return ActivityEffortResult(
        activity_name=act_cfg.name,
        raw_score=raw,
        effort_score=effort,
        n_windows=n_windows,
        n_features_used=n_features,
        reliable=reliable,
        reliability_note=note,
        modality_scores=modality_scores,
    )


def score_domain(
    subject_dir: Path,
    domain_cfg,                              # DomainEffortConfig
    references: Dict[str, ActivityReference],
    scoring_cfg,
) -> DomainEffortResult:
    """Score all activities in a domain and aggregate to domain effort score."""
    act_results: Dict[str, ActivityEffortResult] = {}

    for act_cfg in domain_cfg.activities:
        ref = references.get(act_cfg.name)
        if ref is None:
            logger.debug("No reference for activity '%s', skipping", act_cfg.name)
            act_results[act_cfg.name] = ActivityEffortResult(
                activity_name=act_cfg.name,
                raw_score=None, effort_score=None,
                n_windows=0, n_features_used=0, reliable=False,
                reliability_note="no HC reference built",
            )
            continue
        act_results[act_cfg.name] = score_activity(subject_dir, act_cfg, ref, scoring_cfg)

    # Weighted average over activities that have a valid, reliable score
    weighted_sum = 0.0
    weight_total = 0.0
    any_reliable = False
    for act_cfg in domain_cfg.activities:
        ar = act_results[act_cfg.name]
        if ar.effort_score is not None:
            weighted_sum += ar.effort_score * act_cfg.weight
            weight_total += act_cfg.weight
            if ar.reliable:
                any_reliable = True

    if weight_total == 0.0:
        return DomainEffortResult(
            domain_name=domain_cfg.name,
            r4_label=domain_cfg.r4_label,
            effort_score=None,
            activity_results=act_results,
            reliable=False,
            reliability_note="no activities scored",
        )

    domain_score = weighted_sum / weight_total
    reliable_acts = sum(1 for ar in act_results.values() if ar.reliable)
    total_acts = len(domain_cfg.activities)
    note = f"{reliable_acts}/{total_acts} activities reliable" if not any_reliable else ""

    # Aggregate per-modality scores with same weights
    all_modalities: set = set()
    for ar in act_results.values():
        all_modalities.update(ar.modality_scores.keys())

    domain_modality_scores: Dict[str, Optional[float]] = {}
    for mod in all_modalities:
        mod_wsum = 0.0
        mod_wtot = 0.0
        for act_cfg in domain_cfg.activities:
            ar = act_results[act_cfg.name]
            ms = ar.modality_scores.get(mod)
            if ms is not None:
                mod_wsum += ms * act_cfg.weight
                mod_wtot += act_cfg.weight
        domain_modality_scores[mod] = mod_wsum / mod_wtot if mod_wtot > 0 else None

    return DomainEffortResult(
        domain_name=domain_cfg.name,
        r4_label=domain_cfg.r4_label,
        effort_score=domain_score,
        activity_results=act_results,
        reliable=any_reliable,
        reliability_note=note,
        modality_scores=domain_modality_scores,
    )


def score_subject(
    subject_dir: Path,
    config: EffortConfig,
    references: Dict[str, ActivityReference],
    subject_id: Optional[str] = None,
) -> SubjectEffortResult:
    """Score all domains for one subject."""
    sid = subject_id or subject_dir.name
    domain_results = {
        key: score_domain(subject_dir, domain_cfg, references, config.scoring)
        for key, domain_cfg in config.domains.items()
    }
    return SubjectEffortResult(subject_id=sid, domain_results=domain_results)
