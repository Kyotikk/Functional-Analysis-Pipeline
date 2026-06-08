"""Load effort configuration and HC reference profiles.

A reference profile holds, per activity, the per-feature robust centre
(median) and scale (MAD) computed from all healthy-control windows.
It is the sole source of truth for deviation scoring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

_META_COLS = {
    "activity_idx", "activity_name", "t_start", "t_end", "duration_sec",
}


@dataclass
class ActivityConfig:
    name: str
    file: str
    weight: float = 1.0


@dataclass
class DomainEffortConfig:
    name: str
    r4_label: str
    activities: List[ActivityConfig]


@dataclass
class ScoringConfig:
    feature_reducer: str = "median"      # "median" | "mean" | "trimmed_mean" | "iqr_mean" | "huber"
    window_reducer: str = "median"       # "median" | "mean" | "trimmed_mean" | "iqr_mean" | "huber"
    min_windows: int = 3
    normalization_p100: float = 95.0
    epsilon: float = 1e-6
    exclude_features: List[str] = field(default_factory=list)
    modality_groups: Dict[str, List[str]] = field(default_factory=dict)
    """Maps modality name → list of feature-name prefixes (startswith matching)."""
    inverse_feature_patterns: List[str] = field(default_factory=list)
    """Feature patterns where lower values indicate higher effort (e.g., rmssd, sdnn).
    Features matching these patterns use signed deviation: (hc_median - window) instead of (window - hc_median)."""
    augment_with_statistics: bool = False
    """If true, append activity-level feature statistics as additional columns for scoring."""
    statistical_transforms: List[str] = field(default_factory=lambda: ["min", "max", "mean", "std"])
    """Statistics to append per feature when augmentation is enabled."""


@dataclass
class EffortConfig:
    scoring: ScoringConfig
    domains: Dict[str, DomainEffortConfig]

    @property
    def all_activities(self) -> Dict[str, ActivityConfig]:
        """Flat dict: activity_name → ActivityConfig across all domains."""
        out: Dict[str, ActivityConfig] = {}
        for domain in self.domains.values():
            for act in domain.activities:
                out[act.name] = act
        return out


def load_effort_config(path: Path) -> EffortConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    sc_raw = raw.get("scoring", {})
    raw_modalities = sc_raw.get("modality_groups", {})
    modality_groups = {
        name: list(prefixes)
        for name, prefixes in raw_modalities.items()
    }
    scoring = ScoringConfig(
        feature_reducer=sc_raw.get("feature_reducer", "median"),
        window_reducer=sc_raw.get("window_reducer", "median"),
        min_windows=int(sc_raw.get("min_windows", 3)),
        normalization_p100=float(sc_raw.get("normalization_p100", 95.0)),
        epsilon=float(sc_raw.get("epsilon", 1e-6)),
        exclude_features=list(sc_raw.get("exclude_features", [])),
        modality_groups=modality_groups,
        inverse_feature_patterns=list(sc_raw.get("inverse_feature_patterns", [])),
        augment_with_statistics=bool(sc_raw.get("augment_with_statistics", False)),
        statistical_transforms=list(sc_raw.get("statistical_transforms", ["min", "max", "mean", "std"])),
    )

    domains: Dict[str, DomainEffortConfig] = {}
    for domain_key, domain_data in raw.get("domains", {}).items():
        activities = [
            ActivityConfig(
                name=act_name,
                file=act_data["file"],
                weight=float(act_data.get("weight", 1.0)),
            )
            for act_name, act_data in domain_data.get("activities", {}).items()
        ]
        domains[domain_key] = DomainEffortConfig(
            name=domain_key,
            r4_label=domain_data.get("r4_label", domain_key),
            activities=activities,
        )

    return EffortConfig(scoring=scoring, domains=domains)


# ---------------------------------------------------------------------------
# Reference profile
# ---------------------------------------------------------------------------


@dataclass
class ActivityReference:
    """Per-activity reference computed from HC windows."""

    activity_name: str
    feature_names: List[str]
    hc_median: np.ndarray        # shape (n_features,)
    hc_mad: np.ndarray           # shape (n_features,)
    n_hc_windows: int
    n_hc_subjects: int

    # HC subject-level raw deviation scores (used for centered normalisation)
    hc_subject_scores: np.ndarray   # shape (n_hc_subjects,)
    norm_anchor_minus_100: float    # p(100-XX) of HC scores       → maps to -100
    norm_anchor_0: float            # median of HC subject scores   → maps to 0
    norm_anchor_100: float          # pXX of HC subject scores      → maps to +100


def _select_features(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Return numeric columns minus metadata and explicit exclusions."""
    excl = _META_COLS | set(exclude)
    return [
        c for c in df.columns
        if c not in excl and pd.api.types.is_numeric_dtype(df[c])
    ]


def _load_activity_windows(
    subject_dirs: List[Path],
    filename: str,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Concatenate all windows for one activity across multiple subject dirs."""
    frames = []
    for sdir in subject_dirs:
        fp = sdir / filename
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if df.empty:
            continue
        if feature_cols is not None:
            present = [c for c in feature_cols if c in df.columns]
            df = df[present]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _reduce(arr: np.ndarray, method: str) -> float:
    vals = np.asarray(arr, dtype=float)

    # Handle common reducers fast with numpy built-ins.
    if method == "mean":
        return float(np.nanmean(vals))
    if method == "median":
        return float(np.nanmedian(vals))

    x = vals[np.isfinite(vals)]
    if len(x) == 0:
        return float("nan")

    if method == "trimmed_mean":
        # Trim 20% from each tail for robust central tendency.
        if len(x) < 5:
            return float(np.mean(x))
        k = int(np.floor(0.2 * len(x)))
        x_sorted = np.sort(x)
        if 2 * k >= len(x_sorted):
            return float(np.mean(x_sorted))
        return float(np.mean(x_sorted[k:len(x_sorted) - k]))

    if method == "iqr_mean":
        q1, q3 = np.percentile(x, [25, 75])
        in_iqr = x[(x >= q1) & (x <= q3)]
        if len(in_iqr) == 0:
            return float(np.median(x))
        return float(np.mean(in_iqr))

    if method == "huber":
        # Iteratively reweighted location estimate with Huber weights.
        mu = float(np.median(x))
        mad = float(np.median(np.abs(x - mu)))
        sigma = max(1.4826 * mad, 1e-9)
        c = 1.5
        for _ in range(50):
            r = x - mu
            a = np.abs(r)
            w = np.ones_like(a)
            mask = a > c * sigma
            w[mask] = (c * sigma) / a[mask]
            denom = np.sum(w)
            if denom <= 0:
                break
            new_mu = float(np.sum(w * x) / denom)
            if abs(new_mu - mu) < 1e-7:
                mu = new_mu
                break
            mu = new_mu
        return mu

    raise ValueError(
        f"Unsupported reducer '{method}'. "
        "Expected one of: mean, median, trimmed_mean, iqr_mean, huber"
    )


def _augment_with_statistics(
    windows: pd.DataFrame,
    feature_cols: List[str],
    transforms: List[str],
) -> pd.DataFrame:
    """Append per-feature activity-level statistics as additional columns.

    Added columns are constant over rows for a given activity file and named
    as: <feature>__<transform>, where transform is one of min|max|mean|std.
    """
    if not feature_cols or not transforms:
        return windows

    valid = [t for t in transforms if t in {"min", "max", "mean", "std"}]
    if not valid:
        return windows

    base = windows[feature_cols].apply(pd.to_numeric, errors="coerce")

    stats = {
        "min": base.min(axis=0, skipna=True),
        "max": base.max(axis=0, skipna=True),
        "mean": base.mean(axis=0, skipna=True),
        "std": base.std(axis=0, skipna=True, ddof=0),
    }

    derived_cols = {}
    for tr in valid:
        series = stats[tr].fillna(0.0)
        for feat in feature_cols:
            derived_cols[f"{feat}__{tr}"] = float(series.loc[feat])

    if not derived_cols:
        return windows.copy()

    derived_df = pd.DataFrame(derived_cols, index=windows.index)
    return pd.concat([windows.copy(), derived_df], axis=1)


def _per_window_deviation(
    windows: pd.DataFrame,
    feature_cols: List[str],
    hc_median: np.ndarray,
    hc_mad: np.ndarray,
    epsilon: float,
    feature_reducer: str,
    inverse_feature_patterns: List[str] = None,
) -> np.ndarray:
    """Return a 1-D array of per-window deviation scalars.
    
    For most features, deviation = (value - hc_median) / (hc_mad + epsilon).
    For inverse features (e.g., rmssd, sdnn), deviation = (hc_median - value) / (hc_mad + epsilon),
    because lower values indicate higher effort.
    
    All deviations are absolute-valued to keep positive for aggregation.
    """
    if inverse_feature_patterns is None:
        inverse_feature_patterns = []
    
    mat = windows[feature_cols].to_numpy(dtype=float, na_value=np.nan)
    # Shape: (n_windows, n_features)
    
    # Determine which features are inverse
    is_inverse = np.array([
        any(feat.startswith(p) or p in feat for p in inverse_feature_patterns)
        for feat in feature_cols
    ])
    
    # Compute signed deviations for inverse features
    # For normal features: deviation = |window - hc| / mad (absolute deviation)
    # For inverse features where HIGHER values = LESS effort (e.g., HRV):
    #   - If window > hc (high HRV/low effort): flip sign → negative deviation
    #   - If window < hc (low HRV/high effort):  flip sign → positive deviation
    #   This flipping inverts the feature's contribution relative to its magnitude
    dev = np.zeros_like(mat)
    
    # Normal features: (value - hc) / mad, take absolute
    dev[:, ~is_inverse] = np.abs((mat[:, ~is_inverse] - hc_median[~is_inverse]) / (hc_mad[~is_inverse] + epsilon))
    
    # Inverse features: -(value - hc) / mad = (hc - value) / mad, then absolute
    # This flips the sign: high HRV → negative after flip → low effort
    #                      low HRV → positive after flip → high effort
    dev[:, is_inverse] = np.abs((hc_median[is_inverse] - mat[:, is_inverse]) / (hc_mad[is_inverse] + epsilon))
    
    # Fast path for common reducers keeps augmented runs tractable.
    if feature_reducer == "mean":
        return np.nanmean(dev, axis=1)
    if feature_reducer == "median":
        return np.nanmedian(dev, axis=1)

    # Robust reducers are computed row-wise.
    return np.array([_reduce(row, feature_reducer) for row in dev], dtype=float)


def build_reference(
    hc_batch_dir: Path,
    config: EffortConfig,
    subject_glob: str = "sub_*",
) -> Dict[str, ActivityReference]:
    """Build per-activity reference profiles from an HC batch directory.

    Returns a dict: activity_name → ActivityReference.
    """
    hc_dirs = sorted(d for d in hc_batch_dir.iterdir() if d.is_dir() and d.match(subject_glob))
    if not hc_dirs:
        raise ValueError(f"No subject directories found in {hc_batch_dir}")

    logger.info("Building HC reference from %d subjects in %s", len(hc_dirs), hc_batch_dir)

    sc = config.scoring
    references: Dict[str, ActivityReference] = {}

    for act_cfg in config.all_activities.values():
        # Load all HC windows for this activity
        all_windows = _load_activity_windows(hc_dirs, act_cfg.file)
        if all_windows.empty:
            logger.warning("No HC windows found for activity '%s' (%s)", act_cfg.name, act_cfg.file)
            continue

        if sc.augment_with_statistics:
            raw_feature_cols = _select_features(all_windows, sc.exclude_features)
            all_windows = _augment_with_statistics(
                all_windows,
                raw_feature_cols,
                sc.statistical_transforms,
            )

        feature_cols = _select_features(all_windows, sc.exclude_features)
        if not feature_cols:
            logger.warning("No usable features for activity '%s'", act_cfg.name)
            continue

        feat_df = all_windows[feature_cols].apply(pd.to_numeric, errors="coerce")

        # Drop features with >50% missing across all HC windows
        valid_ratio = feat_df.notna().mean()
        feature_cols = [c for c in feature_cols if valid_ratio[c] >= 0.5]
        feat_df = feat_df[feature_cols]

        # Impute remaining NaNs with column median before computing stats
        feat_df = feat_df.fillna(feat_df.median())

        hc_median = feat_df.median().to_numpy()
        hc_mad = (feat_df - feat_df.median()).abs().median().to_numpy()

        # Compute per-HC-subject deviation scores (for normalisation anchors)
        hc_subject_scores: List[float] = []
        for sdir in hc_dirs:
            fp = sdir / act_cfg.file
            if not fp.exists():
                continue
            sub_df = pd.read_csv(fp)
            if sub_df.empty:
                continue
            if sc.augment_with_statistics:
                sub_raw_feature_cols = _select_features(sub_df, sc.exclude_features)
                sub_df = _augment_with_statistics(
                    sub_df,
                    sub_raw_feature_cols,
                    sc.statistical_transforms,
                )
            present = [c for c in feature_cols if c in sub_df.columns]
            if not present:
                continue
            sub_feat = sub_df[present].apply(pd.to_numeric, errors="coerce").fillna(
                pd.Series(hc_median, index=feature_cols)[present]
            )
            per_window = _per_window_deviation(
                sub_feat, present,
                hc_median[[feature_cols.index(c) for c in present]],
                hc_mad[[feature_cols.index(c) for c in present]],
                sc.epsilon, sc.feature_reducer,
                sc.inverse_feature_patterns,
            )
            if len(per_window) >= sc.min_windows:
                hc_subject_scores.append(_reduce(per_window, sc.window_reducer))

        hc_arr = np.array(hc_subject_scores) if hc_subject_scores else np.array([0.0])
        norm_0 = float(np.median(hc_arr))
        norm_100 = float(np.percentile(hc_arr, sc.normalization_p100))
        norm_m100 = float(np.percentile(hc_arr, 100.0 - sc.normalization_p100))
        if norm_100 <= norm_0:
            norm_100 = norm_0 + 1.0   # degenerate guard
        if norm_m100 >= norm_0:
            norm_m100 = norm_0 - 1.0  # degenerate guard

        references[act_cfg.name] = ActivityReference(
            activity_name=act_cfg.name,
            feature_names=feature_cols,
            hc_median=hc_median,
            hc_mad=hc_mad,
            n_hc_windows=len(all_windows),
            n_hc_subjects=len(hc_dirs),
            hc_subject_scores=hc_arr,
            norm_anchor_minus_100=norm_m100,
            norm_anchor_0=norm_0,
            norm_anchor_100=norm_100,
        )
        logger.info(
            "  %-20s  %4d windows  %3d features  "
            "HC score: p%g=%.3f  median=%.3f  p%g=%.3f",
            act_cfg.name, len(all_windows), len(feature_cols),
            100.0 - sc.normalization_p100, norm_m100, norm_0,
            sc.normalization_p100, norm_100,
        )

    return references
