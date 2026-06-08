from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
import traceback

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from correlation.merger import load_and_merge
from effort.batch_scorer import run_batch
from effort.reference import load_effort_config


DOMAINS = ["Basic Movements", "Walking", "Oral Care", "Grooming"]
REDUCERS = ["median", "mean", "trimmed_mean", "iqr_mean", "huber"]


@dataclass
class ComboResult:
    tag: str
    feature_reducer: str
    window_reducer: str
    neg_corr_score: float
    n_sig_negative: int


def evaluate_combo(capacity_csv: Path, effort_csv: Path, r4_csv: Path) -> tuple[float, int]:
    merged = load_and_merge(capacity_csv, effort_csv, r4_csv, reliable_only=False)
    neg_score = 0.0
    n_sig_neg = 0

    for domain in DOMAINS:
        x = merged[f"{domain}_effort"].to_numpy(dtype=float)
        y = merged[f"{domain}_r4"].to_numpy(dtype=float)
        mask = pd.notna(x) & pd.notna(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 3:
            continue
        rho, pval = stats.spearmanr(x, y)
        if pd.notna(rho):
            neg_score += -float(rho)
            if rho < 0 and pval < 0.05:
                n_sig_neg += 1

    return neg_score, n_sig_neg


def rewrite_default_reducers(config_path: Path, feature_reducer: str, window_reducer: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    text_new = re.sub(
        r'(^\s*feature_reducer:\s*)"[^"]+"',
        rf'\1"{feature_reducer}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text_new = re.sub(
        r'(^\s*window_reducer:\s*)"[^"]+"',
        rf'\1"{window_reducer}"',
        text_new,
        count=1,
        flags=re.MULTILINE,
    )
    config_path.write_text(text_new, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "config" / "effort_config.yaml"

    patient_dir = root.parent / "HR-metric-extractor" / "output_batch" / "batch_20260604_005145_real_run3"
    hc_dir = root.parent / "HR-metric-extractor" / "output_batch" / "hc_run_2"
    r4_csv = root.parent / "HR-metric-extractor" / "R4-scores" / "R4_scores_nursing_home.csv"
    capacity_csv = root / "output" / "capacity" / "batch_20260604_005145_real_run3" / "capacity_scores.csv"

    out_root = root / "output" / "effort" / "reducer_sweep_robust_aug_real_run3"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_now = load_effort_config(config_path)
    default_feature = cfg_now.scoring.feature_reducer
    default_window = cfg_now.scoring.window_reducer
    default_tag = f"f-{default_feature}_w-{default_window}_aug"

    results: list[ComboResult] = []
    failures: list[tuple[str, str]] = []

    for feat_red in REDUCERS:
        for win_red in REDUCERS:
            tag = f"f-{feat_red}_w-{win_red}_aug"
            out_dir = out_root / tag
            out_csv = out_dir / "effort_scores.csv"
            out_dir.mkdir(parents=True, exist_ok=True)

            if not out_csv.exists():
                print(f"RUN  {tag}", flush=True)
                try:
                    cfg = load_effort_config(config_path)
                    cfg.scoring.feature_reducer = feat_red
                    cfg.scoring.window_reducer = win_red
                    cfg.scoring.augment_with_statistics = True
                    run_batch(
                        patient_batch_dir=patient_dir,
                        hc_batch_dir=hc_dir,
                        config=cfg,
                        output_dir=out_dir,
                        patient_subject_glob="sub_*",
                        hc_subject_glob="sub_*",
                    )
                    print(f"DONE {tag}", flush=True)
                except Exception as exc:
                    failures.append((tag, str(exc)))
                    print(f"FAIL {tag}: {exc}", flush=True)
                    traceback.print_exc()
                    continue
            else:
                print(f"SKIP {tag}", flush=True)

            try:
                neg, n_sig = evaluate_combo(capacity_csv, out_csv, r4_csv)
                results.append(
                    ComboResult(
                        tag=tag,
                        feature_reducer=feat_red,
                        window_reducer=win_red,
                        neg_corr_score=neg,
                        n_sig_negative=n_sig,
                    )
                )
            except Exception as exc:
                failures.append((tag, f"evaluation: {exc}"))
                print(f"FAIL {tag} evaluation: {exc}", flush=True)
                traceback.print_exc()

    summary_df = pd.DataFrame(
        {
            "config": [r.tag for r in results],
            "feature_reducer": [r.feature_reducer for r in results],
            "window_reducer": [r.window_reducer for r in results],
            "neg_corr_score": [r.neg_corr_score for r in results],
            "n_sig_negative": [r.n_sig_negative for r in results],
        }
    )

    if summary_df.empty:
        raise RuntimeError("No successful sweep results were produced.")

    summary_df = summary_df.sort_values(["neg_corr_score", "n_sig_negative"], ascending=[False, False])
    summary_csv = out_root / "reducer_sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    if failures:
        failures_path = out_root / "failures.log"
        failures_path.write_text(
            "\n".join([f"{tag}\t{msg}" for tag, msg in failures]),
            encoding="utf-8",
        )

    best_row = summary_df.iloc[0]
    default_rows = summary_df.loc[summary_df["config"] == default_tag]
    if default_rows.empty:
        print(f"WARN default config {default_tag} not found in successful results; no auto-update", flush=True)
        print(f"Saved summary: {summary_csv}", flush=True)
        return

    default_row = default_rows.iloc[0]
    best_score = float(best_row["neg_corr_score"])
    default_score = float(default_row["neg_corr_score"])

    print("\n=== AUG robust sweep top 10 ===", flush=True)
    print(summary_df.head(10).to_string(index=False), flush=True)
    print(
        f"\nCurrent default: {default_tag} score={default_score:.6f}; "
        f"Best: {best_row['config']} score={best_score:.6f}",
        flush=True,
    )

    if best_score > default_score and best_row["config"] != default_tag:
        rewrite_default_reducers(
            config_path=config_path,
            feature_reducer=str(best_row["feature_reducer"]),
            window_reducer=str(best_row["window_reducer"]),
        )
        print(
            f"AUTO-UPDATED defaults to feature={best_row['feature_reducer']} "
            f"window={best_row['window_reducer']}",
            flush=True,
        )
    else:
        print("Default retained (no strictly better neg_corr_score found).", flush=True)

    print(f"Saved summary: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()