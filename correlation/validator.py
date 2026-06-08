"""Statistical validation of sensor scores against clinical R4 stages.

Analyses
--------
1. Effort vs R4          — Spearman ρ + Kendall τ + 95 % bootstrap CI
                           (expected: negative — higher effort → lower R4)
2. Capacity stage vs R4  — Spearman ρ + Kendall τ + bootstrap CI +
                           weighted Cohen's κ + % exact / within-1
                           (expected: positive — higher stage → higher R4)
3. Composite vs R4       — ICF-inspired composite = mean(icf_capacity,
                           icf_performance); Spearman ρ + bootstrap CI
4. Per-modality vs R4    — Spearman ρ for each modality sub-score vs R4

All bootstrap CIs use percentile method on 1000 resamples (configurable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

_DOMAINS = ["Basic Movements", "Walking", "Oral Care", "Grooming"]
_MODALITIES = ["hr_hrv", "eda", "imu_wrist", "imu_bioz", "imu_chest"]

# ICF-inspired scale mappings
# capacity stage 1-5  →  icf_capacity 4-0  (5-stage = 0 problem)
# effort -100..100    →  icf_performance 0-4  (negative effort treated as 0 problem)
_ICF_EFFORT_DIVISOR = 25.0   # effort +100 → ICF 4


def _icf_capacity(stage: float) -> float:
    return 5.0 - stage   # [0, 4]


def _icf_performance(effort: float) -> float:
    return min(max(effort, 0.0) / _ICF_EFFORT_DIVISOR, 4.0)   # [0, 4]


def _composite(icf_cap: float, icf_perf: float) -> float:
    return (icf_cap + icf_perf) / 2.0   # [0, 4]


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------

def _bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap CI for Spearman ρ."""
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    rs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if len(set(xi)) < 2 or len(set(yi)) < 2:
            continue
        r, _ = stats.spearmanr(xi, yi)
        rs.append(r)
    if len(rs) < 10:
        return (np.nan, np.nan)
    alpha = (100 - ci) / 2
    return (float(np.percentile(rs, alpha)), float(np.percentile(rs, 100 - alpha)))


def _bootstrap_kendall(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap CI for Kendall τ."""
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    taus = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if len(set(xi)) < 2 or len(set(yi)) < 2:
            continue
        tau, _ = stats.kendalltau(xi, yi)
        taus.append(tau)
    if len(taus) < 10:
        return (np.nan, np.nan)
    alpha = (100 - ci) / 2
    return (float(np.percentile(taus, alpha)), float(np.percentile(taus, 100 - alpha)))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CorrelationResult:
    domain: str
    comparison: str        # e.g. "effort_vs_r4", "capacity_vs_r4", "composite_vs_r4"
    modality: Optional[str]  # None for overall; modality name for per-modality rows
    n_subjects: int
    spearman_r: float
    spearman_p: float
    spearman_ci_lo: float
    spearman_ci_hi: float
    kendall_tau: float
    kendall_p: float
    kendall_ci_lo: float
    kendall_ci_hi: float
    # Capacity concordance extras (only populated for capacity_vs_r4)
    weighted_kappa: Optional[float] = None
    pct_exact: Optional[float] = None
    pct_within_1: Optional[float] = None


@dataclass
class ValidationResult:
    correlations: List[CorrelationResult] = field(default_factory=list)
    combined_df: Optional[pd.DataFrame] = None  # per-subject ICF-mapped table


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_corr(
    x: np.ndarray,
    y: np.ndarray,
    domain: str,
    comparison: str,
    modality: Optional[str],
    n_bootstrap: int,
) -> Optional[CorrelationResult]:
    """Compute Spearman + Kendall + bootstrap CIs for paired arrays."""
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 3:
        logger.warning(
            f"  {domain} | {comparison}: only {n} valid pairs — skipping"
        )
        return None

    r, p_r   = stats.spearmanr(xv, yv)
    tau, p_t = stats.kendalltau(xv, yv)
    ci_r     = _bootstrap_spearman(xv, yv, n_bootstrap=n_bootstrap)
    ci_t     = _bootstrap_kendall(xv, yv, n_bootstrap=n_bootstrap)

    return CorrelationResult(
        domain=domain,
        comparison=comparison,
        modality=modality,
        n_subjects=n,
        spearman_r=float(r),
        spearman_p=float(p_r),
        spearman_ci_lo=ci_r[0],
        spearman_ci_hi=ci_r[1],
        kendall_tau=float(tau),
        kendall_p=float(p_t),
        kendall_ci_lo=ci_t[0],
        kendall_ci_hi=ci_t[1],
    )


def _capacity_concordance(
    sensor_stages: np.ndarray,
    r4_stages: np.ndarray,
) -> Tuple[Optional[float], float, float]:
    """Return (weighted_kappa, pct_exact, pct_within_1)."""
    mask = np.isfinite(sensor_stages) & np.isfinite(r4_stages)
    sv, rv = sensor_stages[mask].astype(int), r4_stages[mask].astype(int)
    if len(sv) < 3:
        return (None, np.nan, np.nan)

    # Clip to valid R4 range 1-5
    sv = np.clip(sv, 1, 5)
    rv = np.clip(rv, 1, 5)

    # Weighted kappa (linear weights)
    try:
        kappa = float(cohen_kappa_score(sv, rv, weights="linear"))
    except Exception as exc:
        logger.warning(f"  Weighted kappa failed: {exc}")
        kappa = None

    pct_exact    = float(np.mean(sv == rv)) * 100
    pct_within_1 = float(np.mean(np.abs(sv - rv) <= 1)) * 100
    return kappa, pct_exact, pct_within_1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(
    merged_df: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> ValidationResult:
    """Run all correlation analyses against R4 stages.

    Parameters
    ----------
    merged_df:
        Output of :func:`correlation.merger.load_and_merge`.
    n_bootstrap:
        Number of bootstrap resamples for CI estimation.

    Returns
    -------
    :class:`ValidationResult` containing all :class:`CorrelationResult`
    objects and the per-subject combined DataFrame.
    """
    result = ValidationResult()
    combined_rows = []

    for domain in _DOMAINS:
        stage_col   = f"{domain}_stage"
        effort_col  = f"{domain}_effort"
        reliable_col = f"{domain}_effort_reliable"
        r4_col      = f"{domain}_r4"

        if r4_col not in merged_df.columns:
            logger.warning(f"R4 column '{r4_col}' missing — skipping {domain}")
            continue

        r4 = merged_df[r4_col].to_numpy(dtype=float)

        # ------------------------------------------------------------------
        # 1. Effort vs R4  (reliable rows only for this comparison)
        # ------------------------------------------------------------------
        if effort_col in merged_df.columns:
            effort = merged_df[effort_col].to_numpy(dtype=float)
            if reliable_col in merged_df.columns:
                reliable_mask = merged_df[reliable_col].fillna(False).to_numpy(dtype=bool)
                x_eff = np.where(reliable_mask, effort, np.nan)
            else:
                x_eff = effort

            cr = _compute_corr(x_eff, r4, domain, "effort_vs_r4", None, n_bootstrap)
            if cr:
                result.correlations.append(cr)
                logger.info(
                    f"  {domain} | effort_vs_r4  : ρ={cr.spearman_r:+.3f} "
                    f"(p={cr.spearman_p:.3f}, n={cr.n_subjects}) "
                    f"95%CI [{cr.spearman_ci_lo:+.3f}, {cr.spearman_ci_hi:+.3f}]"
                )

        # ------------------------------------------------------------------
        # 2. Capacity stage vs R4
        # ------------------------------------------------------------------
        if stage_col in merged_df.columns:
            stages = merged_df[stage_col].to_numpy(dtype=float)
            cr = _compute_corr(stages, r4, domain, "capacity_vs_r4", None, n_bootstrap)
            if cr:
                kappa, pct_exact, pct_within_1 = _capacity_concordance(stages, r4)
                cr.weighted_kappa = kappa
                cr.pct_exact      = pct_exact
                cr.pct_within_1   = pct_within_1
                result.correlations.append(cr)
                logger.info(
                    f"  {domain} | capacity_vs_r4: ρ={cr.spearman_r:+.3f} "
                    f"(p={cr.spearman_p:.3f}, n={cr.n_subjects}) "
                    f"κ={kappa:.3f}" if kappa is not None else
                    f"  {domain} | capacity_vs_r4: ρ={cr.spearman_r:+.3f} "
                    f"(p={cr.spearman_p:.3f}, n={cr.n_subjects})"
                )

        # ------------------------------------------------------------------
        # 3. Composite vs R4
        # ------------------------------------------------------------------
        if stage_col in merged_df.columns and effort_col in merged_df.columns:
            stages = merged_df[stage_col].to_numpy(dtype=float)
            effort = merged_df[effort_col].to_numpy(dtype=float)
            if reliable_col in merged_df.columns:
                reliable_mask = merged_df[reliable_col].fillna(False).to_numpy(dtype=bool)
                eff_for_comp = np.where(reliable_mask, effort, np.nan)
            else:
                eff_for_comp = effort

            icf_cap  = 5.0 - stages                                        # [0, 4]
            icf_perf = np.clip(np.maximum(eff_for_comp, 0.0) / _ICF_EFFORT_DIVISOR, 0, 4)  # [0, 4]
            composite = (icf_cap + icf_perf) / 2.0

            cr = _compute_corr(composite, r4, domain, "composite_vs_r4", None, n_bootstrap)
            if cr:
                result.correlations.append(cr)
                logger.info(
                    f"  {domain} | composite_vs_r4: ρ={cr.spearman_r:+.3f} "
                    f"(p={cr.spearman_p:.3f}, n={cr.n_subjects})"
                )

        # ------------------------------------------------------------------
        # 4. Per-modality effort vs R4
        # ------------------------------------------------------------------
        for mod in _MODALITIES:
            mod_col = f"{domain}_effort_{mod}"
            if mod_col not in merged_df.columns:
                continue
            mod_vals = merged_df[mod_col].to_numpy(dtype=float)
            cr = _compute_corr(mod_vals, r4, domain, "effort_vs_r4", mod, n_bootstrap)
            if cr:
                result.correlations.append(cr)

        # ------------------------------------------------------------------
        # Build per-subject ICF-mapped row
        # ------------------------------------------------------------------
        for _, row in merged_df.iterrows():
            stage_val  = row.get(stage_col, np.nan)
            effort_val = row.get(effort_col, np.nan)
            r4_val     = row.get(r4_col, np.nan)
            rel_val    = row.get(reliable_col, False)

            icf_cap_v  = _icf_capacity(float(stage_val))  if pd.notna(stage_val)  else np.nan
            icf_perf_v = _icf_performance(float(effort_val)) if pd.notna(effort_val) and rel_val else np.nan
            comp_v     = _composite(icf_cap_v, icf_perf_v) if pd.notna(icf_cap_v) and pd.notna(icf_perf_v) else np.nan

            combined_rows.append({
                "subject_id":      row["subject_id"],
                "domain":          domain,
                "capacity_stage":  stage_val,
                "effort":          effort_val,
                "effort_reliable": rel_val,
                "r4_stage":        r4_val,
                "icf_capacity":    icf_cap_v,
                "icf_performance": icf_perf_v,
                "composite":       comp_v,
            })

    result.combined_df = pd.DataFrame(combined_rows)
    return result


def results_to_dataframe(correlations: List[CorrelationResult]) -> pd.DataFrame:
    """Convert a list of CorrelationResult to a flat DataFrame."""
    rows = []
    for cr in correlations:
        rows.append({
            "domain":          cr.domain,
            "comparison":      cr.comparison,
            "modality":        cr.modality if cr.modality else "overall",
            "n_subjects":      cr.n_subjects,
            "spearman_r":      cr.spearman_r,
            "spearman_p":      cr.spearman_p,
            "spearman_ci_lo":  cr.spearman_ci_lo,
            "spearman_ci_hi":  cr.spearman_ci_hi,
            "kendall_tau":     cr.kendall_tau,
            "kendall_p":       cr.kendall_p,
            "kendall_ci_lo":   cr.kendall_ci_lo,
            "kendall_ci_hi":   cr.kendall_ci_hi,
            "weighted_kappa":  cr.weighted_kappa,
            "pct_exact":       cr.pct_exact,
            "pct_within_1":    cr.pct_within_1,
        })
    return pd.DataFrame(rows)
