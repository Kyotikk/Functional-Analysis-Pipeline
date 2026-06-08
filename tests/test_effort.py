"""Unit tests for the effort scorer.

All tests use synthetic in-memory data — no real files required.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from effort.reference import (
    ActivityConfig,
    ActivityReference,
    DomainEffortConfig,
    EffortConfig,
    ScoringConfig,
    _augment_with_statistics,
    _per_window_deviation,
    _reduce,
    _select_features,
)
from effort.scorer import (
    DomainEffortResult,
    SubjectEffortResult,
    _normalise,
    score_activity,
    score_domain,
    score_subject,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scoring_cfg(**kwargs) -> ScoringConfig:
    defaults = dict(
        feature_reducer="median",
        window_reducer="median",
        min_windows=2,
        normalization_p100=95.0,
        epsilon=1e-6,
        exclude_features=[],
    )
    defaults.update(kwargs)
    return ScoringConfig(**defaults)


def _make_reference(
    activity_name: str = "test_act",
    features: list[str] = None,
    hc_median: list[float] = None,
    hc_mad: list[float] = None,
    hc_subject_scores: list[float] = None,
    norm_anchor_minus_100: float = -1.0,
    norm_anchor_0: float = 0.5,
    norm_anchor_100: float = 2.0,
) -> ActivityReference:
    if features is None:
        features = ["f1", "f2", "f3"]
    if hc_median is None:
        hc_median = [1.0] * len(features)
    if hc_mad is None:
        hc_mad = [1.0] * len(features)
    if hc_subject_scores is None:
        hc_subject_scores = [0.3, 0.5, 0.6]
    return ActivityReference(
        activity_name=activity_name,
        feature_names=features,
        hc_median=np.array(hc_median, dtype=float),
        hc_mad=np.array(hc_mad, dtype=float),
        n_hc_windows=30,
        n_hc_subjects=len(hc_subject_scores),
        hc_subject_scores=np.array(hc_subject_scores, dtype=float),
        norm_anchor_minus_100=norm_anchor_minus_100,
        norm_anchor_0=norm_anchor_0,
        norm_anchor_100=norm_anchor_100,
    )


def _make_windows_csv(n_rows: int, values: dict, path: Path) -> None:
    df = pd.DataFrame({k: [v] * n_rows for k, v in values.items()})
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Test _select_features
# ---------------------------------------------------------------------------


class TestSelectFeatures:
    def test_excludes_meta_cols(self):
        df = pd.DataFrame({"activity_name": ["x"], "t_start": [0], "f1": [1.0], "f2": [2.0]})
        result = _select_features(df, exclude=[])
        assert "activity_name" not in result
        assert "t_start" not in result
        assert "f1" in result
        assert "f2" in result

    def test_excludes_non_numeric(self):
        df = pd.DataFrame({"label": ["a", "b"], "f1": [1.0, 2.0]})
        result = _select_features(df, exclude=[])
        assert "label" not in result
        assert "f1" in result

    def test_explicit_exclusion(self):
        df = pd.DataFrame({"f1": [1.0], "f2": [2.0], "n_beats": [10]})
        result = _select_features(df, exclude=["n_beats"])
        assert "n_beats" not in result
        assert "f1" in result


# ---------------------------------------------------------------------------
# Test _per_window_deviation
# ---------------------------------------------------------------------------


class TestPerWindowDeviation:
    def test_zero_deviation_at_median(self):
        df = pd.DataFrame({"f1": [1.0, 1.0], "f2": [2.0, 2.0]})
        hc_median = np.array([1.0, 2.0])
        hc_mad = np.array([1.0, 1.0])
        result = _per_window_deviation(df, ["f1", "f2"], hc_median, hc_mad, 1e-6, "median")
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-9)

    def test_deviation_scales_with_distance(self):
        df = pd.DataFrame({"f1": [3.0, 5.0]})   # distances 2.0 and 4.0
        hc_median = np.array([1.0])
        hc_mad = np.array([1.0])
        result = _per_window_deviation(df, ["f1"], hc_median, hc_mad, 0.0, "median")
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(4.0)

    def test_feature_reducer_mean_vs_median(self):
        df = pd.DataFrame({"f1": [3.0], "f2": [1.0]})   # devs: 2.0, 0.0
        hc_median = np.array([1.0, 1.0])
        hc_mad = np.array([1.0, 1.0])
        median_res = _per_window_deviation(df, ["f1", "f2"], hc_median, hc_mad, 0.0, "median")
        mean_res   = _per_window_deviation(df, ["f1", "f2"], hc_median, hc_mad, 0.0, "mean")
        assert median_res[0] == pytest.approx(1.0)  # median(2.0, 0.0) = 1.0
        assert mean_res[0]   == pytest.approx(1.0)  # mean(2.0, 0.0) = 1.0

    def test_large_deviation_means_higher_score(self):
        hc_median = np.array([0.0])
        hc_mad = np.array([1.0])
        df_low  = pd.DataFrame({"f1": [1.0]})
        df_high = pd.DataFrame({"f1": [10.0]})
        low  = _per_window_deviation(df_low,  ["f1"], hc_median, hc_mad, 0.0, "median")[0]
        high = _per_window_deviation(df_high, ["f1"], hc_median, hc_mad, 0.0, "median")[0]
        assert high > low


class TestFeatureStatisticsAugmentation:
    def test_adds_min_max_mean_std_columns(self):
        df = pd.DataFrame({"f1": [1.0, 3.0], "f2": [2.0, 6.0]})
        out = _augment_with_statistics(df, ["f1", "f2"], ["min", "max", "mean", "std"])

        expected_cols = {
            "f1__min", "f1__max", "f1__mean", "f1__std",
            "f2__min", "f2__max", "f2__mean", "f2__std",
        }
        assert expected_cols.issubset(set(out.columns))
        assert out["f1__min"].iloc[0] == pytest.approx(1.0)
        assert out["f1__max"].iloc[0] == pytest.approx(3.0)
        assert out["f1__mean"].iloc[0] == pytest.approx(2.0)
        assert out["f1__std"].iloc[0] == pytest.approx(1.0)

    def test_invalid_transforms_are_ignored(self):
        df = pd.DataFrame({"f1": [1.0, 3.0]})
        out = _augment_with_statistics(df, ["f1"], ["median", "bogus", "min"])
        assert "f1__min" in out.columns
        assert "f1__median" not in out.columns
        assert "f1__bogus" not in out.columns


class TestReduceRobustMethods:
    def test_trimmed_mean_reduces_outlier_impact(self):
        arr = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
        mean_val = _reduce(arr, "mean")
        trimmed_val = _reduce(arr, "trimmed_mean")
        assert trimmed_val < mean_val
        assert trimmed_val == pytest.approx(0.0)

    def test_iqr_mean_reduces_outlier_impact(self):
        arr = np.array([0.0, 0.0, 0.0, 1.0, 100.0])
        mean_val = _reduce(arr, "mean")
        iqr_val = _reduce(arr, "iqr_mean")
        assert iqr_val < mean_val
        assert iqr_val == pytest.approx(0.25, abs=1e-6)

    def test_huber_reduces_outlier_impact(self):
        arr = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
        mean_val = _reduce(arr, "mean")
        huber_val = _reduce(arr, "huber")
        assert huber_val < mean_val
        assert huber_val >= 0.0

    def test_invalid_reducer_raises(self):
        with pytest.raises(ValueError):
            _reduce(np.array([1.0, 2.0]), "not_a_reducer")


class TestPerWindowDeviationRobustReducers:
    def test_trimmed_mean_feature_reducer(self):
        df = pd.DataFrame({"f1": [0.0], "f2": [0.0], "f3": [0.0], "f4": [0.0], "f5": [100.0]})
        hc_median = np.zeros(5)
        hc_mad = np.ones(5)
        out = _per_window_deviation(df, ["f1", "f2", "f3", "f4", "f5"], hc_median, hc_mad, 0.0, "trimmed_mean")
        assert out[0] == pytest.approx(0.0)

    def test_huber_feature_reducer(self):
        df = pd.DataFrame({"f1": [0.0], "f2": [0.0], "f3": [0.0], "f4": [0.0], "f5": [100.0]})
        hc_median = np.zeros(5)
        hc_mad = np.ones(5)
        out = _per_window_deviation(df, ["f1", "f2", "f3", "f4", "f5"], hc_median, hc_mad, 0.0, "huber")
        assert out[0] < 20.0


# ---------------------------------------------------------------------------
# Test _normalise
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_anchor_minus_100_maps_to_minus_100(self):
        assert _normalise(0.0, anchor_minus_100=0.0, anchor_0=0.5, anchor_100=2.0) == pytest.approx(-100.0)

    def test_anchor_0_maps_near_zero(self):
        assert _normalise(0.5, anchor_minus_100=0.0, anchor_0=0.5, anchor_100=2.0) == pytest.approx(0.0)

    def test_anchor_100_maps_to_100(self):
        assert _normalise(2.0, anchor_minus_100=0.0, anchor_0=0.5, anchor_100=2.0) == pytest.approx(100.0)

    def test_positive_midpoint_between_anchor_0_and_anchor_100(self):
        assert _normalise(1.25, anchor_minus_100=0.0, anchor_0=0.5, anchor_100=2.0) == pytest.approx(50.0)

    def test_negative_midpoint_between_anchor_minus_100_and_anchor_0(self):
        assert _normalise(0.25, anchor_minus_100=0.0, anchor_0=0.5, anchor_100=2.0) == pytest.approx(-50.0)

    def test_below_anchor_minus_100_clipped_to_minus_100(self):
        assert _normalise(-1.0, anchor_minus_100=0.0, anchor_0=0.5, anchor_100=2.0) == -100.0

    def test_values_above_anchor_100_clipped_to_plus_100(self):
        val = _normalise(3.5, anchor_minus_100=0.0, anchor_0=0.5, anchor_100=2.0)
        assert val == pytest.approx(100.0)  # clipped at +100


# ---------------------------------------------------------------------------
# Test score_activity
# ---------------------------------------------------------------------------


class TestScoreActivity:
    def test_missing_file_returns_none(self, tmp_path):
        act_cfg = ActivityConfig(name="test", file="missing.csv", weight=1.0)
        ref = _make_reference()
        sc = _make_scoring_cfg()
        result = score_activity(tmp_path, act_cfg, ref, sc)
        assert result.effort_score is None
        assert "not found" in result.reliability_note

    def test_empty_file_returns_none(self, tmp_path):
        csv = tmp_path / "act.csv"
        csv.write_text("f1,f2\n")   # header only
        act_cfg = ActivityConfig(name="test", file="act.csv", weight=1.0)
        ref = _make_reference(features=["f1", "f2"])
        sc = _make_scoring_cfg()
        result = score_activity(tmp_path, act_cfg, ref, sc)
        assert result.effort_score is None
        assert "empty" in result.reliability_note

    def test_insufficient_windows(self, tmp_path):
        csv = tmp_path / "act.csv"
        _make_windows_csv(1, {"f1": 2.0, "f2": 3.0}, csv)
        act_cfg = ActivityConfig(name="test", file="act.csv", weight=1.0)
        ref = _make_reference(features=["f1", "f2"])
        sc = _make_scoring_cfg(min_windows=3)
        result = score_activity(tmp_path, act_cfg, ref, sc)
        assert result.effort_score is None
        assert "1 windows" in result.reliability_note

    def test_hc_like_windows_score_near_zero(self, tmp_path):
        """Subject at HC median should score near 0 (after normalisation)."""
        csv = tmp_path / "act.csv"
        _make_windows_csv(10, {"f1": 1.0, "f2": 2.0}, csv)
        ref = _make_reference(
            features=["f1", "f2"],
            hc_median=[1.0, 2.0],
            hc_mad=[1.0, 1.0],
            norm_anchor_0=0.0,
            norm_anchor_100=2.0,
        )
        act_cfg = ActivityConfig(name="test", file="act.csv", weight=1.0)
        sc = _make_scoring_cfg()
        result = score_activity(tmp_path, act_cfg, ref, sc)
        assert result.effort_score is not None
        assert result.effort_score == pytest.approx(0.0, abs=5.0)

    def test_high_deviation_scores_higher_than_low(self, tmp_path):
        ref = _make_reference(
            features=["f1"],
            hc_median=[0.0],
            hc_mad=[1.0],
            norm_anchor_0=0.0,
            norm_anchor_100=5.0,
        )
        sc = _make_scoring_cfg()

        csv_lo = tmp_path / "lo.csv"
        csv_hi = tmp_path / "hi.csv"
        _make_windows_csv(5, {"f1": 1.0}, csv_lo)
        _make_windows_csv(5, {"f1": 10.0}, csv_hi)

        lo = score_activity(tmp_path, ActivityConfig("lo", "lo.csv", 1.0), ref, sc)
        hi = score_activity(tmp_path, ActivityConfig("hi", "hi.csv", 1.0), ref, sc)
        assert hi.effort_score > lo.effort_score

    def test_score_capped_at_200(self, tmp_path):
        ref = _make_reference(
            features=["f1"],
            hc_median=[0.0],
            hc_mad=[1.0],
            norm_anchor_minus_100=-1.0,
            norm_anchor_0=0.0,
            norm_anchor_100=1.0,
        )
        sc = _make_scoring_cfg()
        csv = tmp_path / "extreme.csv"
        _make_windows_csv(5, {"f1": 1000.0}, csv)
        result = score_activity(tmp_path, ActivityConfig("x", "extreme.csv", 1.0), ref, sc)
        assert result.effort_score <= 100.0

    def test_statistics_augmentation_affects_score_when_reference_matches(self, tmp_path):
        csv = tmp_path / "act.csv"
        pd.DataFrame({"f1": [1.0, 9.0]}).to_csv(csv, index=False)

        ref_no_stats = _make_reference(
            features=["f1"],
            hc_median=[0.0],
            hc_mad=[1.0],
            norm_anchor_0=0.0,
            norm_anchor_100=10.0,
        )
        sc_no_stats = _make_scoring_cfg(augment_with_statistics=False)
        base_res = score_activity(tmp_path, ActivityConfig("test", "act.csv", 1.0), ref_no_stats, sc_no_stats)

        aug_ref_features = ["f1", "f1__min", "f1__max", "f1__mean", "f1__std"]
        ref_with_stats = _make_reference(
            features=aug_ref_features,
            hc_median=[0.0, 0.0, 0.0, 0.0, 0.0],
            hc_mad=[1.0, 1.0, 1.0, 1.0, 1.0],
            norm_anchor_0=0.0,
            norm_anchor_100=10.0,
        )
        sc_with_stats = _make_scoring_cfg(
            augment_with_statistics=True,
            statistical_transforms=["min", "max", "mean", "std"],
        )
        aug_res = score_activity(tmp_path, ActivityConfig("test", "act.csv", 1.0), ref_with_stats, sc_with_stats)

        assert base_res.raw_score is not None
        assert aug_res.raw_score is not None
        assert aug_res.raw_score != pytest.approx(base_res.raw_score)


# ---------------------------------------------------------------------------
# Test score_domain
# ---------------------------------------------------------------------------


class TestScoreDomain:
    def _build_domain(self, activities) -> DomainEffortConfig:
        return DomainEffortConfig(
            name="test_domain",
            r4_label="Test Domain",
            activities=activities,
        )

    def test_all_missing_returns_none(self, tmp_path):
        domain = self._build_domain([
            ActivityConfig("act1", "a1.csv", 1.0),
            ActivityConfig("act2", "a2.csv", 1.0),
        ])
        refs = {}  # no references
        sc = _make_scoring_cfg()
        cfg = EffortConfig(scoring=sc, domains={"d": domain})
        result = score_domain(tmp_path, domain, refs, sc)
        assert result.effort_score is None

    def test_weighted_average(self, tmp_path):
        ref = _make_reference(
            features=["f1"],
            hc_median=[0.0],
            hc_mad=[1.0],
            norm_anchor_0=0.0,
            norm_anchor_100=10.0,
        )
        # act1 deviation = 2.0 → effort=20; act2 deviation = 8.0 → effort=80
        csv1 = tmp_path / "a1.csv"
        csv2 = tmp_path / "a2.csv"
        _make_windows_csv(5, {"f1": 2.0}, csv1)
        _make_windows_csv(5, {"f1": 8.0}, csv2)

        domain = self._build_domain([
            ActivityConfig("a1", "a1.csv", 1.0),
            ActivityConfig("a2", "a2.csv", 3.0),
        ])
        refs = {"a1": ref, "a2": ref}
        sc = _make_scoring_cfg()

        result = score_domain(tmp_path, domain, refs, sc)
        expected = (20.0 * 1.0 + 80.0 * 3.0) / (1.0 + 3.0)
        assert result.effort_score == pytest.approx(expected, abs=1.0)

    def test_partial_missing_uses_available(self, tmp_path):
        ref = _make_reference(
            features=["f1"],
            hc_median=[0.0],
            hc_mad=[1.0],
            norm_anchor_0=0.0,
            norm_anchor_100=5.0,
        )
        csv1 = tmp_path / "present.csv"
        _make_windows_csv(5, {"f1": 5.0}, csv1)

        domain = self._build_domain([
            ActivityConfig("present", "present.csv", 1.0),
            ActivityConfig("missing", "missing.csv", 1.0),
        ])
        refs = {"present": ref}  # no ref for "missing"
        sc = _make_scoring_cfg()

        result = score_domain(tmp_path, domain, refs, sc)
        assert result.effort_score is not None


# ---------------------------------------------------------------------------
# Test SubjectEffortResult
# ---------------------------------------------------------------------------


class TestSubjectEffortResult:
    def test_to_summary_row_includes_all_domains(self, tmp_path):
        ref = _make_reference(
            features=["f1"],
            hc_median=[0.0],
            hc_mad=[1.0],
            norm_anchor_0=0.0,
            norm_anchor_100=5.0,
        )
        csv = tmp_path / "dental.csv"
        _make_windows_csv(5, {"f1": 3.0}, csv)

        sc = _make_scoring_cfg()
        domain = DomainEffortConfig(
            name="oral_care",
            r4_label="Oral Care",
            activities=[ActivityConfig("dental_care", "dental.csv", 1.0)],
        )
        config = EffortConfig(scoring=sc, domains={"oral_care": domain})
        refs = {"dental_care": ref}

        result = score_subject(tmp_path, config, refs, subject_id="sub_test")
        row = result.to_summary_row()
        assert row["subject_id"] == "sub_test"
        assert "Oral Care_effort" in row

    def test_to_activity_rows_structure(self, tmp_path):
        ref = _make_reference(features=["f1"], hc_median=[0.0], hc_mad=[1.0],
                              norm_anchor_0=0.0, norm_anchor_100=5.0)
        csv = tmp_path / "a.csv"
        _make_windows_csv(5, {"f1": 2.0}, csv)

        sc = _make_scoring_cfg()
        domain = DomainEffortConfig(
            name="d",
            r4_label="D",
            activities=[ActivityConfig("act", "a.csv", 1.0)],
        )
        config = EffortConfig(scoring=sc, domains={"d": domain})
        refs = {"act": ref}

        result = score_subject(tmp_path, config, refs, subject_id="sub_x")
        rows = result.to_activity_rows()
        assert len(rows) == 1
        assert rows[0]["subject_id"] == "sub_x"
        assert rows[0]["activity"] == "act"
        assert "effort_score" in rows[0]
        assert "n_windows" in rows[0]
