"""Unit tests for the capacity qualifier.

All tests use synthetic in-memory DataFrames — no file I/O required.
Run with:  python -m pytest tests/test_capacity.py -v
"""
from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from capacity.rules import StageCheck, StageRule, DomainRules
from capacity.qualifier import (
    _apply_check,
    _evaluate_stage,
    assign_domain_stage,
    SubjectCapacityResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(**cols) -> pd.DataFrame:
    """Build a minimal activity DataFrame from keyword arguments."""
    return pd.DataFrame(cols)


def activities(**source_dfs) -> dict:
    """Build a fake activities dict for qualifier functions."""
    from capacity.qualifier import _empty_df, _SOURCE_FILE_MAP
    result = {k: _empty_df() for k in _SOURCE_FILE_MAP}
    result.update(source_dfs)
    return result


def simple_domain(stages: dict) -> DomainRules:
    return DomainRules(name="test", r4_label="Test", stages=stages)


# ---------------------------------------------------------------------------
# _apply_check
# ---------------------------------------------------------------------------

class TestApplyCheck:

    def test_no_source_file_not_satisfied(self):
        check = StageCheck(source="activity_standing", min_duration_sec=10.0, min_occurrences=1)
        result = _apply_check(check, {})
        assert not result.satisfied
        assert result.n_matched_intervals == 0

    def test_empty_dataframe_not_satisfied(self):
        check = StageCheck(source="activity_standing", min_occurrences=1)
        result = _apply_check(check, activities())
        assert not result.satisfied

    def test_keyword_include_match(self):
        df = make_df(
            activity=["standing", "sit to stand"],
            duration_sec=[200.0, 10.0],
            t_start=[0.0, 300.0],
            t_end=[200.0, 310.0],
        )
        check = StageCheck(
            source="activity_standing",
            keywords=["standing"],
            min_occurrences=1,
            min_duration_sec=0.0,
        )
        result = _apply_check(check, activities(activity_standing=df))
        assert result.satisfied
        assert result.n_matched_intervals == 1
        assert "standing" in result.matched_activities

    def test_keyword_exclude_filters_transitions(self):
        df = make_df(
            activity=["standing", "sit to stand", "stand to sit"],
            duration_sec=[200.0, 12.0, 10.0],
            t_start=[0.0, 300.0, 400.0],
            t_end=[200.0, 312.0, 410.0],
        )
        check = StageCheck(
            source="activity_standing",
            keywords=["stand", "standing"],
            exclude_keywords=[" to "],
            min_duration_sec=180.0,
            min_occurrences=1,
        )
        result = _apply_check(check, activities(activity_standing=df))
        assert result.satisfied
        assert result.n_matched_intervals == 1
        assert result.matched_activities == ["standing"]

    def test_min_duration_filters_short_intervals(self):
        df = make_df(
            activity=["standing", "standing"],
            duration_sec=[50.0, 200.0],
            t_start=[0.0, 100.0],
            t_end=[50.0, 300.0],
        )
        check = StageCheck(
            source="activity_standing",
            min_duration_sec=180.0,
            min_occurrences=1,
        )
        result = _apply_check(check, activities(activity_standing=df))
        assert result.satisfied
        assert result.n_matched_intervals == 1

    def test_min_occurrences_not_met(self):
        df = make_df(
            activity=["brush teeth"],
            duration_sec=[20.0],
            t_start=[0.0],
            t_end=[20.0],
        )
        check = StageCheck(source="activity_dental_care", min_occurrences=2)
        result = _apply_check(check, activities(activity_dental_care=df))
        assert not result.satisfied

    def test_total_duration_reported_correctly(self):
        df = make_df(
            activity=["wash hands", "wash hands"],
            duration_sec=[15.0, 20.0],
            t_start=[0.0, 100.0],
            t_end=[15.0, 120.0],
        )
        check = StageCheck(source="activity_washing_hands", min_occurrences=1)
        result = _apply_check(check, activities(activity_washing_hands=df))
        assert result.total_duration_sec == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# Oral Care stage discrimination
# ---------------------------------------------------------------------------

class TestOralCareStages:

    def _dental_care_df(self, labels, durations):
        n = len(labels)
        return make_df(
            activity=labels,
            duration_sec=durations,
            t_start=list(range(n)),
            t_end=[i + d for i, d in enumerate(durations)],
        )

    def _oral_care_domain(self):
        return simple_domain({
            4: StageRule(
                description="Set up + brush",
                checks=[StageCheck(source="activity_dental_care",
                                   keywords=["put toothpaste", "toothpaste"],
                                   min_occurrences=1, min_duration_sec=3.0)],
            ),
            3: StageRule(
                description="Brush only",
                checks=[StageCheck(source="activity_dental_care",
                                   keywords=["brush teeth", "clean teeth"],
                                   min_occurrences=1, min_duration_sec=10.0)],
            ),
            2: StageRule(
                description="Rinse only",
                checks=[StageCheck(source="activity_dental_care",
                                   keywords=["rinse mouth", "rinse"],
                                   min_occurrences=1, min_duration_sec=5.0)],
            ),
        })

    def test_stage4_when_toothpaste_and_brush_present(self):
        df = self._dental_care_df(
            ["put toothpaste", "brush teeth", "rinse mouth"],
            [5.0, 30.0, 15.0],
        )
        result = assign_domain_stage(self._oral_care_domain(), activities(activity_dental_care=df))
        assert result.assigned_stage == 4

    def test_stage3_when_only_brush_present(self):
        df = self._dental_care_df(["brush teeth", "rinse mouth"], [30.0, 15.0])
        result = assign_domain_stage(self._oral_care_domain(), activities(activity_dental_care=df))
        assert result.assigned_stage == 3

    def test_stage2_when_only_rinse_present(self):
        df = self._dental_care_df(["rinse mouth"], [15.0])
        result = assign_domain_stage(self._oral_care_domain(), activities(activity_dental_care=df))
        assert result.assigned_stage == 2

    def test_stage1_when_no_oral_care(self):
        result = assign_domain_stage(self._oral_care_domain(), activities())
        assert result.assigned_stage == 1

    def test_stage2_rinse_too_short_gives_stage1(self):
        df = self._dental_care_df(["rinse mouth"], [2.0])  # below 5 s threshold
        result = assign_domain_stage(self._oral_care_domain(), activities(activity_dental_care=df))
        assert result.assigned_stage == 1


# ---------------------------------------------------------------------------
# Basic Mobility stage discrimination
# ---------------------------------------------------------------------------

class TestBasicMobilityStages:

    def _standing_df(self, labels, durations):
        n = len(labels)
        return make_df(
            activity=labels, duration_sec=durations,
            t_start=list(range(n)), t_end=[i + d for i, d in enumerate(durations)],
        )

    def _basic_mobility_domain(self):
        return simple_domain({
            5: StageRule(
                description="Stand ≥15s (protocol-adjusted)",
                checks=[StageCheck(
                    source="activity_standing",
                    keywords=["stand", "standing"],
                    exclude_keywords=[" to "],
                    min_duration_sec=15.0, min_occurrences=1,
                )],
            ),
            4: StageRule(
                description="Transfer",
                checks=[StageCheck(
                    source="activity_transfer",
                    min_occurrences=1, min_duration_sec=5.0,
                )],
            ),
            3: StageRule(
                description="Lying to sit",
                checks=[StageCheck(
                    source="activity_bed_transfer",
                    keywords=["lying to sit", "sit to lying"],
                    min_occurrences=1, min_duration_sec=3.0,
                )],
            ),
            2: StageRule(
                description="Turn in bed",
                checks=[StageCheck(
                    source="activity_bed_transfer",
                    min_occurrences=1, min_duration_sec=3.0,
                )],
            ),
        })

    def test_stage5_with_long_standing(self):
        df = self._standing_df(["standing"], [20.0])  # protocol-adjusted threshold 15s
        result = assign_domain_stage(self._basic_mobility_domain(), activities(activity_standing=df))
        assert result.assigned_stage == 5

    def test_transitions_only_do_not_satisfy_stage5(self):
        """sit-to-stand and stand-to-sit are transitions, not maintained standing.
        They are below the 15s threshold AND filtered by the ' to ' exclude keyword."""
        df = self._standing_df(["sit to stand", "stand to sit"], [14.0, 10.0])
        transfer_df = make_df(
            activity=["sit to stand"], duration_sec=[14.0], t_start=[0.0], t_end=[14.0]
        )
        result = assign_domain_stage(
            self._basic_mobility_domain(),
            activities(activity_standing=df, activity_transfer=transfer_df),
        )
        assert result.assigned_stage == 4

    def test_stage4_with_transfers_only(self):
        transfer_df = make_df(
            activity=["sit to stand", "transfer from bed"],
            duration_sec=[14.0, 28.0],
            t_start=[0.0, 100.0],
            t_end=[14.0, 128.0],
        )
        result = assign_domain_stage(
            self._basic_mobility_domain(),
            activities(activity_transfer=transfer_df),
        )
        assert result.assigned_stage == 4

    def test_stage3_with_lying_to_sit(self):
        bed_df = make_df(
            activity=["lying to sit"], duration_sec=[10.0], t_start=[0.0], t_end=[10.0]
        )
        result = assign_domain_stage(
            self._basic_mobility_domain(),
            activities(activity_bed_transfer=bed_df),
        )
        assert result.assigned_stage == 3

    def test_stage2_turn_in_bed_only(self):
        bed_df = make_df(
            activity=["turn in bed (l)", "turn in bed (r)"],
            duration_sec=[8.0, 7.0],
            t_start=[0.0, 50.0],
            t_end=[8.0, 57.0],
        )
        result = assign_domain_stage(
            self._basic_mobility_domain(),
            activities(activity_bed_transfer=bed_df),
        )
        assert result.assigned_stage == 2

    def test_stage1_no_evidence(self):
        result = assign_domain_stage(self._basic_mobility_domain(), activities())
        assert result.assigned_stage == 1


# ---------------------------------------------------------------------------
# Walking stage discrimination
# ---------------------------------------------------------------------------

class TestWalkingStages:

    def _walking_domain(self):
        return simple_domain({
            5: StageRule(
                description="Public transport",
                not_assessable=True,
                checks=[],
            ),
            4: StageRule(
                description="Stairs",
                not_assessable=True,
                checks=[],
            ),
            3: StageRule(
                description="Walks without assistance",
                checks=[StageCheck(
                    source="propulsion",
                    keywords=["level walking", "walking", "walker", "ambulation"],
                    exclude_keywords=["self propulsion", "assisted propulsion"],
                    min_duration_sec=30.0, min_occurrences=1,
                )],
            ),
            2: StageRule(
                description="Wheelchair",
                checks=[StageCheck(
                    source="propulsion",
                    keywords=["self propulsion", "assisted propulsion"],
                    min_duration_sec=30.0, min_occurrences=1,
                )],
            ),
        })

    def test_stage3_level_walking(self):
        df = make_df(
            activity=["level walking"], duration_sec=[80.0], t_start=[0.0], t_end=[80.0]
        )
        result = assign_domain_stage(self._walking_domain(), activities(propulsion=df))
        assert result.assigned_stage == 3
        assert result.capped_by_not_assessable  # stages 4 & 5 skipped

    def test_stage2_only_wheelchair(self):
        df = make_df(
            activity=["self propulsion"], duration_sec=[60.0], t_start=[0.0], t_end=[60.0]
        )
        result = assign_domain_stage(self._walking_domain(), activities(propulsion=df))
        assert result.assigned_stage == 2

    def test_walker_classified_as_stage3_not_stage2(self):
        """A 'walker' is a walking aid — subjects using it CAN walk (Stage 3)."""
        df = make_df(
            activity=["walker"], duration_sec=[45.0], t_start=[0.0], t_end=[45.0]
        )
        result = assign_domain_stage(self._walking_domain(), activities(propulsion=df))
        assert result.assigned_stage == 3

    def test_stage1_no_propulsion(self):
        result = assign_domain_stage(self._walking_domain(), activities())
        assert result.assigned_stage == 1

    def test_capped_flag_true_for_walking(self):
        """Stages 4 and 5 are not_assessable, so any walking result is capped."""
        df = make_df(
            activity=["level walking"], duration_sec=[80.0], t_start=[0.0], t_end=[80.0]
        )
        result = assign_domain_stage(self._walking_domain(), activities(propulsion=df))
        assert result.capped_by_not_assessable is True


# ---------------------------------------------------------------------------
# Grooming stage discrimination
# ---------------------------------------------------------------------------

class TestGroomingStages:

    def _grooming_domain(self):
        return simple_domain({
            5: StageRule(description="Nail cutting", not_assessable=True, checks=[]),
            4: StageRule(
                description="Hair styling",
                checks=[StageCheck(source="activity_hair_care", min_occurrences=1, min_duration_sec=10.0)],
            ),
            3: StageRule(
                description="Wash face",
                checks=[StageCheck(source="activity_washing_face", min_occurrences=1, min_duration_sec=10.0)],
            ),
            2: StageRule(
                description="Wash hands",
                checks=[StageCheck(source="activity_washing_hands", min_occurrences=1, min_duration_sec=10.0)],
            ),
        })

    def test_stage4_hair_care(self):
        df = make_df(activity=["style beard/hair"], duration_sec=[30.0], t_start=[0.0], t_end=[30.0])
        result = assign_domain_stage(self._grooming_domain(), activities(activity_hair_care=df))
        assert result.assigned_stage == 4

    def test_stage3_only_face_wash(self):
        df = make_df(activity=["wash face"], duration_sec=[20.0], t_start=[0.0], t_end=[20.0])
        result = assign_domain_stage(self._grooming_domain(), activities(activity_washing_face=df))
        assert result.assigned_stage == 3

    def test_stage2_only_hand_wash(self):
        df = make_df(activity=["wash hands"], duration_sec=[15.0], t_start=[0.0], t_end=[15.0])
        result = assign_domain_stage(self._grooming_domain(), activities(activity_washing_hands=df))
        assert result.assigned_stage == 2

    def test_stage1_no_grooming(self):
        result = assign_domain_stage(self._grooming_domain(), activities())
        assert result.assigned_stage == 1


# ---------------------------------------------------------------------------
# Evidence output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def test_summary_row_has_all_domains(self):
        from capacity.rules import load_rules
        rules_path = Path(__file__).parent.parent / "config" / "capacity_rules.yaml"
        domain_rules = load_rules(rules_path)
        fake_activities = {k: pd.DataFrame(columns=["activity", "t_start", "t_end", "duration_sec"])
                          for k in [
                              "activity_standing", "activity_transfer", "activity_bed_transfer",
                              "activity_dental_care", "activity_hair_care",
                              "activity_washing_face", "activity_washing_hands",
                              "propulsion", "resting",
                          ]}
        from capacity.qualifier import score_subject
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_dir = Path(tmpdir) / "sub_test"
            subject_dir.mkdir()
            # Write empty CSVs so load_activity_files finds files
            for name in ["activity_standing.csv", "activity_transfer.csv",
                         "activity_bed_transfer.csv", "activity_dental_care.csv",
                         "activity_hair_care.csv", "activity_washing_face.csv",
                         "activity_washing_hands.csv", "propulsion_activities.csv",
                         "resting_activities.csv"]:
                (subject_dir / name).write_text(
                    "activity,t_start,t_end,duration_sec\n"
                )
            result = score_subject(subject_dir, domain_rules)

        row = result.to_summary_row()
        assert "subject_id" in row
        assert "Basic Movements" in row
        assert "Walking" in row
        assert "Oral Care" in row
        assert "Grooming" in row

    def test_all_stage1_when_all_files_empty(self):
        from capacity.rules import load_rules
        rules_path = Path(__file__).parent.parent / "config" / "capacity_rules.yaml"
        domain_rules = load_rules(rules_path)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_dir = Path(tmpdir) / "sub_empty"
            subject_dir.mkdir()
            for name in ["activity_standing.csv", "activity_transfer.csv",
                         "activity_bed_transfer.csv", "activity_dental_care.csv",
                         "activity_hair_care.csv", "activity_washing_face.csv",
                         "activity_washing_hands.csv", "propulsion_activities.csv",
                         "resting_activities.csv"]:
                (subject_dir / name).write_text("activity,t_start,t_end,duration_sec\n")
            from capacity.qualifier import score_subject
            result = score_subject(subject_dir, domain_rules)

        for domain_result in result.domain_results.values():
            assert domain_result.assigned_stage == 1, (
                f"{domain_result.r4_label} expected Stage 1, got {domain_result.assigned_stage}"
            )

    def test_evidence_rows_count(self):
        """to_evidence_rows should produce one row per domain × stage."""
        from capacity.rules import load_rules
        rules_path = Path(__file__).parent.parent / "config" / "capacity_rules.yaml"
        domain_rules = load_rules(rules_path)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_dir = Path(tmpdir) / "sub_evrows"
            subject_dir.mkdir()
            for name in ["activity_standing.csv", "activity_transfer.csv",
                         "activity_bed_transfer.csv", "activity_dental_care.csv",
                         "activity_hair_care.csv", "activity_washing_face.csv",
                         "activity_washing_hands.csv", "propulsion_activities.csv",
                         "resting_activities.csv"]:
                (subject_dir / name).write_text("activity,t_start,t_end,duration_sec\n")
            from capacity.qualifier import score_subject
            result = score_subject(subject_dir, domain_rules)

        rows = result.to_evidence_rows()
        # 4 domains × 4 stages each = 16 rows
        assert len(rows) == 16
