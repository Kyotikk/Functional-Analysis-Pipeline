"""Feature importance: Spearman correlation of per-feature HC-deviation against R4 stage.

For each (domain, activity, feature), compute:
  1. Per-subject median absolute deviation from HC median, normalised by HC MAD
     — one scalar per subject per feature
  2. Spearman rank correlation of that scalar with the subject's R4 stage
     (from capacity_scores.csv)

Output: feature_importance.csv
  domain, r4_label, activity, modality, feature, spearman_r, p_value, abs_r, n_subjects
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .reference import ActivityReference, EffortConfig, build_reference, _select_features

MODALITY_COLOURS = {
    "eda":       "#8cff00",
    "hr_hrv":    "#f1c40f",
    "imu_bioz":  "#e97132",
    "imu_chest": "#bb1301",
    "imu_wrist": "#510000",    
}

logger = logging.getLogger(__name__)

# Map domain key → column name in capacity_scores.csv
# The capacity pipeline writes "{r4_label}" as the column (no suffix).
def _stage_col(r4_label: str) -> str:
    return r4_label


def _per_subject_feature_deviations(
    subject_dirs: List[Path],
    act_cfg,                   # ActivityConfig
    reference: ActivityReference,
    epsilon: float,
    inverse_feature_patterns: List[str],
) -> pd.DataFrame:
    """Return a DataFrame (subject_id × feature) of per-subject median deviations.

    Each cell = median over all windows of |window[f] - hc_median[f]| / (hc_mad[f] + eps).
    """
    rows = []
    for sdir in subject_dirs:
        fp = sdir / act_cfg.file
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if df.empty:
            continue
        present = [c for c in reference.feature_names if c in df.columns]
        if not present:
            continue
        feat_df = df[present].apply(pd.to_numeric, errors="coerce")

        ref_idx = [reference.feature_names.index(c) for c in present]
        hc_med = reference.hc_median[ref_idx]
        hc_mad = reference.hc_mad[ref_idx]

        # shape: (n_windows, n_present_features)
        mat = feat_df.to_numpy(dtype=float, na_value=np.nan)
        # Impute NaN windows with HC median for each feature
        for fi, col in enumerate(present):
            nan_mask = np.isnan(mat[:, fi])
            mat[nan_mask, fi] = hc_med[fi]

        dev = np.zeros_like(mat)
        # Determine which features are inverse
        is_inverse = np.array([
            any(feat.startswith(p) or p in feat for p in inverse_feature_patterns)
            for feat in present
        ])
        # Compute signed deviations
        dev[:, ~is_inverse] = (mat[:, ~is_inverse] - hc_med[~is_inverse]) / (hc_mad[~is_inverse] + epsilon)
        dev[:, is_inverse] = (hc_med[is_inverse] - mat[:, is_inverse]) / (hc_mad[is_inverse] + epsilon)
        dev = np.abs(dev)   # (n_windows, n_features)
        median_dev = np.nanmedian(dev, axis=0)             # (n_features,)

        row = {"subject_id": sdir.name}
        row.update(dict(zip(present, median_dev.tolist())))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("subject_id")


def _assign_modality(feature: str, modality_groups: Dict[str, List[str]]) -> str:
    for mod_name, prefixes in modality_groups.items():
        if any(feature.startswith(p) for p in prefixes):
            return mod_name
    return "other"


def compute_feature_importance(
    patient_batch_dir: Path,
    hc_batch_dir: Path,
    config: EffortConfig,
    capacity_scores_csv: Path,
    output_dir: Path,
    subject_glob: str = "sub_*",
) -> pd.DataFrame:
    """Build feature importance table and write to output_dir/feature_importance.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    references = build_reference(hc_batch_dir, config, subject_glob=subject_glob)
    capacity_df = pd.read_csv(capacity_scores_csv)

    subject_dirs = sorted(
        d for d in patient_batch_dir.iterdir()
        if d.is_dir() and d.match(subject_glob)
    )
    logger.info("Computing feature importance for %d subjects", len(subject_dirs))

    sc = config.scoring
    records: List[dict] = []

    for domain_key, domain_cfg in config.domains.items():
        stage_col = _stage_col(domain_cfg.r4_label)
        if stage_col not in capacity_df.columns:
            logger.warning("Column '%s' not found in capacity scores, skipping domain '%s'",
                           stage_col, domain_key)
            continue

        stage_series = capacity_df.set_index("subject_id")[stage_col].dropna()

        for act_cfg in domain_cfg.activities:
            ref = references.get(act_cfg.name)
            if ref is None:
                continue

            feat_matrix = _per_subject_feature_deviations(
                subject_dirs, act_cfg, ref, sc.epsilon, sc.inverse_feature_patterns
            )
            if feat_matrix.empty:
                continue

            # Align subjects: keep only those with both effort deviation and R4 stage
            common_subjects = feat_matrix.index.intersection(stage_series.index)
            if len(common_subjects) < 5:
                logger.warning(
                    "Only %d subjects overlap for activity '%s' — skipping (need ≥5)",
                    len(common_subjects), act_cfg.name,
                )
                continue

            feat_sub = feat_matrix.loc[common_subjects]
            stage_sub = stage_series.loc[common_subjects].astype(float)

            for feat in feat_sub.columns:
                x = feat_sub[feat].to_numpy(dtype=float)
                y = stage_sub.to_numpy(dtype=float)
                # Drop pairs where either is NaN
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 5:
                    continue
                if np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
                    continue
                r, p = stats.spearmanr(x[mask], y[mask])
                if np.isnan(r):
                    # Happens when x or y is constant for the compared subjects.
                    continue
                records.append({
                    "domain": domain_key,
                    "r4_label": domain_cfg.r4_label,
                    "activity": act_cfg.name,
                    "modality": _assign_modality(feat, sc.modality_groups),
                    "feature": feat,
                    "spearman_r": round(float(r), 4),
                    "p_value": round(float(p), 6),
                    "abs_r": round(abs(float(r)), 4),
                    "n_subjects": int(mask.sum()),
                })

    if not records:
        logger.warning("No feature importance records computed.")
        return pd.DataFrame()

    result_df = pd.DataFrame(records).sort_values(
        ["domain", "activity", "abs_r"], ascending=[True, True, False]
    )

    out_path = output_dir / "feature_importance.csv"
    result_df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d feature-activity pairs)", out_path, len(result_df))

    return result_df


def print_top_features(df: pd.DataFrame, top_n: int = 10) -> None:
    """Print top-N features per domain × activity by |Spearman r|."""
    for (domain, activity), grp in df.groupby(["domain", "activity"]):
        top = grp.nlargest(top_n, "abs_r")
        print(f"\n--- {domain} / {activity} ---")
        print(top[["modality", "feature", "spearman_r", "p_value", "n_subjects"]]
              .to_string(index=False))


def plot_feature_importance(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> List[Path]:
    """Create PNG plots for feature importance and return written file paths.

    Generated plots:
      1) top_features_by_activity.png
      2) modality_mean_abs_r_by_activity.png
      3) modality_mean_abs_r_by_domain.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        logger.warning("Feature-importance DataFrame is empty; no plots generated.")
        return []

    written: List[Path] = []

    # 1) Top-N features per (domain, activity) as horizontal bar subplots.
    groups = list(df.groupby(["domain", "activity"], sort=True))
    n_panels = len(groups)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, max(4 * n_rows, 6)))
    axes_arr = np.array(axes).reshape(-1)

    for idx, ((domain, activity), grp) in enumerate(groups):
        ax = axes_arr[idx]
        top = grp.nlargest(top_n, "abs_r").sort_values("abs_r", ascending=True)
        y = np.arange(len(top))
        ax.barh(y, top["abs_r"].values)
        ax.set_yticks(y)
        ax.set_yticklabels(top["feature"].values, fontsize=8)
        ax.set_xlabel("|Spearman r|")
        ax.set_title(f"{domain} / {activity}")
        ax.grid(axis="x", alpha=0.3)

    for idx in range(n_panels, len(axes_arr)):
        axes_arr[idx].axis("off")

    fig.tight_layout()
    p1 = output_dir / "top_features_by_activity.png"
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(p1)

    # 2) Modality summary per (domain, activity)
    mod_act = (
        df.groupby(["domain", "activity", "modality"], as_index=False)["abs_r"]
        .mean()
        .rename(columns={"abs_r": "mean_abs_r"})
    )
    labels = mod_act.apply(lambda r: f"{r['domain']}\n{r['activity']}", axis=1)
    pivot = (
        mod_act.assign(label=labels)
        .pivot(index="label", columns="modality", values="mean_abs_r")
        .fillna(0.0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(16, max(6, 0.4 * len(pivot))))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[col].values, width=width, label=col, color=MODALITY_COLOURS.get(col, None))
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Mean |Spearman r|")
    ax.set_title("Average Modality Importance by Domain / Activity")
    ax.legend(title="Modality")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = output_dir / "modality_mean_abs_r_by_activity.png"
    fig.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(p2)

    # 3) Modality summary per domain
    mod_dom = (
        df.groupby(["domain", "modality"], as_index=False)["abs_r"]
        .mean()
        .rename(columns={"abs_r": "mean_abs_r"})
    )
    pivot_dom = (
        mod_dom.pivot(index="domain", columns="modality", values="mean_abs_r")
        .fillna(0.0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot_dom.index))
    width = 0.8 / max(1, len(pivot_dom.columns))
    for i, col in enumerate(pivot_dom.columns):
        ax.bar(x + i * width, pivot_dom[col].values, width=width, label=col, color=MODALITY_COLOURS.get(col, None))
    ax.set_xticks(x + width * (len(pivot_dom.columns) - 1) / 2)
    ax.set_xticklabels(pivot_dom.index, rotation=20, ha="right")
    ax.set_ylabel("Mean |Spearman r|")
    ax.set_title("Average Modality Importance by Domain")
    ax.legend(title="Modality")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p3 = output_dir / "modality_mean_abs_r_by_domain.png"
    fig.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(p3)

    logger.info("Wrote %d feature-importance plot(s) to %s", len(written), output_dir)
    return written
