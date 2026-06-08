from __future__ import annotations

import argparse
import itertools
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from effort.reference import ActivityReference, build_reference, load_effort_config
from effort.scorer import score_subject


LOG = logging.getLogger("sim_ordering_sweep")

CONDITION_ORDER = ["healthy", "elderly", "severe"]
REDUCERS = ["median", "mean", "trimmed_mean", "iqr_mean", "huber"]


@dataclass
class SweepResult:
    feature_reducer: str
    window_reducer: str
    normalization_p100: float
    augment_with_statistics: bool
    n_subjects: int
    pooled_healthy: float
    pooled_elderly: float
    pooled_severe: float
    pooled_pass: bool
    pooled_margin_sum: float
    n_metrics_checked: int
    n_metrics_pass: int
    pass_rate: float
    margin_sum: float
    saturation_rate: float
    objective_score: float


def _parse_float_list(raw: str) -> List[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _parse_str_list(raw: str) -> List[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one string value.")
    return values


def _extract_condition(subject_id: str) -> str | None:
    parts = subject_id.split("_")
    if len(parts) >= 3 and parts[0] == "sim" and parts[1] in CONDITION_ORDER:
        return parts[1]
    return None


def _score_subjects(
    patient_dirs: Iterable[Path],
    references: Dict[str, ActivityReference],
    config,
) -> pd.DataFrame:
    rows = []
    for sdir in patient_dirs:
        result = score_subject(sdir, config, references, subject_id=sdir.name)
        row = result.to_summary_row()
        condition = _extract_condition(sdir.name)
        if condition is None:
            continue
        row["condition"] = condition
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    return df


def _domain_effort_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if not col.endswith("_effort"):
            continue
        if col.endswith("_effort_reliable") or "_effort_" in col:
            continue
        cols.append(col)
    return cols


def _set_norm_anchors(references: Dict[str, ActivityReference], p100: float) -> Dict[str, ActivityReference]:
    tuned: Dict[str, ActivityReference] = {}
    for act, ref in references.items():
        hc = np.asarray(ref.hc_subject_scores, dtype=float)
        if hc.size == 0:
            hc = np.array([0.0], dtype=float)
        anchor_0 = float(np.median(hc))
        anchor_p = float(np.percentile(hc, p100))
        anchor_m = float(np.percentile(hc, 100.0 - p100))
        if anchor_p <= anchor_0:
            anchor_p = anchor_0 + 1.0
        if anchor_m >= anchor_0:
            anchor_m = anchor_0 - 1.0

        tuned[act] = ActivityReference(
            activity_name=ref.activity_name,
            feature_names=ref.feature_names,
            hc_median=ref.hc_median,
            hc_mad=ref.hc_mad,
            n_hc_windows=ref.n_hc_windows,
            n_hc_subjects=ref.n_hc_subjects,
            hc_subject_scores=ref.hc_subject_scores,
            norm_anchor_minus_100=anchor_m,
            norm_anchor_0=anchor_0,
            norm_anchor_100=anchor_p,
        )
    return tuned


def _evaluate_ordering(summary_df: pd.DataFrame, effort_cols: List[str]) -> dict:
    if summary_df.empty:
        return {
            "pooled_healthy": math.nan,
            "pooled_elderly": math.nan,
            "pooled_severe": math.nan,
            "pooled_pass": False,
            "pooled_margin_sum": -1e6,
            "n_metrics_checked": 0,
            "n_metrics_pass": 0,
            "pass_rate": 0.0,
            "margin_sum": -1e6,
            "saturation_rate": 1.0,
            "objective_score": -1e9,
        }

    pooled = summary_df[["subject_id", "condition"] + effort_cols].copy()
    pooled["pooled_effort"] = pooled[effort_cols].median(axis=1, skipna=True)

    sat_vals = pooled["pooled_effort"].dropna().to_numpy(dtype=float)
    saturation_rate = float(np.mean(np.abs(sat_vals) >= 99.9)) if len(sat_vals) else 1.0

    metrics = ["pooled_effort"] + effort_cols
    n_pass = 0
    margin_sum = 0.0
    checked = 0

    pooled_medians = {}
    for metric in metrics:
        grouped = summary_df.groupby("condition", observed=True)[metric].median() if metric != "pooled_effort" else pooled.groupby("condition", observed=True)[metric].median()
        if any(c not in grouped.index for c in CONDITION_ORDER):
            continue

        healthy = float(grouped.loc["healthy"])
        elderly = float(grouped.loc["elderly"])
        severe = float(grouped.loc["severe"])
        d1 = elderly - healthy
        d2 = severe - elderly
        strict = bool(healthy < elderly < severe)

        checked += 1
        if strict:
            n_pass += 1
        margin_sum += d1 + d2

        if metric == "pooled_effort":
            pooled_medians = {
                "pooled_healthy": healthy,
                "pooled_elderly": elderly,
                "pooled_severe": severe,
                "pooled_pass": strict,
                "pooled_margin_sum": d1 + d2,
            }

    pass_rate = (n_pass / checked) if checked else 0.0

    # Prioritize restoring strict ordering, then separation, then less saturation.
    objective = (
        10_000.0 * n_pass
        + 1_000.0 * (1.0 if pooled_medians.get("pooled_pass", False) else 0.0)
        + 10.0 * margin_sum
        - 100.0 * saturation_rate
    )

    return {
        **pooled_medians,
        "n_metrics_checked": checked,
        "n_metrics_pass": n_pass,
        "pass_rate": pass_rate,
        "margin_sum": margin_sum,
        "saturation_rate": saturation_rate,
        "objective_score": objective,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep effort settings for simulation condition ordering")
    parser.add_argument(
        "--patient-batch-dir",
        type=Path,
        default=ROOT.parent / "HR-metric-extractor" / "output_batch" / "sim_run2",
    )
    parser.add_argument(
        "--hc-batch-dir",
        type=Path,
        default=ROOT.parent / "HR-metric-extractor" / "output_batch" / "hc_run_2",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "effort_config.yaml")
    parser.add_argument("--patient-subject-glob", default="sim_*_[2-5]")
    parser.add_argument("--hc-subject-glob", default="sub_*")
    parser.add_argument(
        "--normalization-grid",
        default="85,90,95,97.5,99",
        help="Comma-separated p100 values to test.",
    )
    parser.add_argument(
        "--feature-reducers",
        default=",".join(REDUCERS),
        help="Comma-separated feature reducers to test.",
    )
    parser.add_argument(
        "--window-reducers",
        default=",".join(REDUCERS),
        help="Comma-separated window reducers to test.",
    )
    parser.add_argument(
        "--augment-options",
        default="false",
        help="Comma-separated values from {false,true}; default false for speed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "validation" / "sim_ordering_sensitivity_sweep",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    if not args.patient_batch_dir.exists():
        raise FileNotFoundError(f"Missing patient batch dir: {args.patient_batch_dir}")
    if not args.hc_batch_dir.exists():
        raise FileNotFoundError(f"Missing HC batch dir: {args.hc_batch_dir}")

    norm_values = _parse_float_list(args.normalization_grid)
    feature_reducers = _parse_str_list(args.feature_reducers)
    window_reducers = _parse_str_list(args.window_reducers)
    for name in feature_reducers + window_reducers:
        if name not in REDUCERS:
            raise ValueError(f"Unsupported reducer '{name}'. Expected one of: {', '.join(REDUCERS)}")
    augment_tokens = [x.strip().lower() for x in args.augment_options.split(",") if x.strip()]
    augment_values: List[bool] = []
    for tok in augment_tokens:
        if tok not in {"true", "false"}:
            raise ValueError(f"Invalid augment option '{tok}'. Use true/false.")
        augment_values.append(tok == "true")
    if not augment_values:
        augment_values = [False]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(
        d for d in args.patient_batch_dir.iterdir() if d.is_dir() and d.match(args.patient_subject_glob)
    )
    if not patient_dirs:
        raise ValueError("No simulation subjects found with given patient glob.")

    base_config = load_effort_config(args.config)

    results: List[SweepResult] = []
    combo_iter = list(itertools.product(augment_values, feature_reducers, window_reducers))
    total_combos = len(combo_iter) * len(norm_values)
    done = 0

    LOG.info(
        "Sweep start: %d reducer/augment combos x %d normalization values = %d settings",
        len(combo_iter),
        len(norm_values),
        total_combos,
    )

    for augment, feat_red, win_red in combo_iter:
        cfg = load_effort_config(args.config)
        cfg.scoring.feature_reducer = feat_red
        cfg.scoring.window_reducer = win_red
        cfg.scoring.augment_with_statistics = augment

        LOG.info(
            "Building references for augment=%s feature=%s window=%s",
            augment,
            feat_red,
            win_red,
        )
        references = build_reference(args.hc_batch_dir, cfg, subject_glob=args.hc_subject_glob)

        for p100 in norm_values:
            tuned_refs = _set_norm_anchors(references, p100)
            cfg.scoring.normalization_p100 = p100

            summary_df = _score_subjects(patient_dirs, tuned_refs, cfg)
            effort_cols = _domain_effort_cols(summary_df)
            metrics = _evaluate_ordering(summary_df, effort_cols)

            row = SweepResult(
                feature_reducer=feat_red,
                window_reducer=win_red,
                normalization_p100=float(p100),
                augment_with_statistics=augment,
                n_subjects=len(summary_df),
                pooled_healthy=float(metrics["pooled_healthy"]),
                pooled_elderly=float(metrics["pooled_elderly"]),
                pooled_severe=float(metrics["pooled_severe"]),
                pooled_pass=bool(metrics["pooled_pass"]),
                pooled_margin_sum=float(metrics["pooled_margin_sum"]),
                n_metrics_checked=int(metrics["n_metrics_checked"]),
                n_metrics_pass=int(metrics["n_metrics_pass"]),
                pass_rate=float(metrics["pass_rate"]),
                margin_sum=float(metrics["margin_sum"]),
                saturation_rate=float(metrics["saturation_rate"]),
                objective_score=float(metrics["objective_score"]),
            )
            results.append(row)
            done += 1

            if done % 10 == 0 or done == total_combos:
                LOG.info("Progress: %d/%d settings evaluated", done, total_combos)

    df = pd.DataFrame([r.__dict__ for r in results])
    df = df.sort_values(
        ["n_metrics_pass", "pooled_pass", "margin_sum", "objective_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    summary_csv = args.output_dir / "sweep_summary.csv"
    df.to_csv(summary_csv, index=False)

    top_csv = args.output_dir / "sweep_top20.csv"
    df.head(20).to_csv(top_csv, index=False)

    report_lines = [
        "# Simulation Ordering Sensitivity Sweep",
        "",
        f"Settings evaluated: {len(df)}",
        f"Patient batch: {args.patient_batch_dir}",
        f"HC batch: {args.hc_batch_dir}",
        f"Patient glob: {args.patient_subject_glob}",
        f"HC glob: {args.hc_subject_glob}",
        f"Normalization grid: {','.join(str(x) for x in norm_values)}",
        f"Augment options: {','.join(str(x).lower() for x in augment_values)}",
        "",
    ]
    if not df.empty:
        best = df.iloc[0]
        report_lines += [
            "## Best setting",
            f"- feature_reducer: {best['feature_reducer']}",
            f"- window_reducer: {best['window_reducer']}",
            f"- normalization_p100: {best['normalization_p100']}",
            f"- augment_with_statistics: {best['augment_with_statistics']}",
            f"- n_metrics_pass: {best['n_metrics_pass']}/{best['n_metrics_checked']}",
            f"- pooled ordering pass: {best['pooled_pass']}",
            f"- pooled medians: healthy={best['pooled_healthy']:.2f}, elderly={best['pooled_elderly']:.2f}, severe={best['pooled_severe']:.2f}",
            f"- saturation_rate: {best['saturation_rate']:.3f}",
            "",
            "## Files",
            "- sweep_summary.csv",
            "- sweep_top20.csv",
        ]

    (args.output_dir / "README.md").write_text("\n".join(report_lines), encoding="utf-8")
    LOG.info("Wrote %s", summary_csv)
    LOG.info("Wrote %s", top_csv)
    LOG.info("Wrote %s", args.output_dir / "README.md")


if __name__ == "__main__":
    main()
