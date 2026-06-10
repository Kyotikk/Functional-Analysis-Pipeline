# Functional Analysis Pipeline

A Python pipeline for assessing **physical functional capacity** and **physiological effort** from wearable sensor data, aligned with the R4 clinical framework and ICF (International Classification of Functioning) model.

The pipeline consumes batch outputs from the upstream **HR-metric-extractor** tool and produces per-subject stage assignments, effort scores, and clinical correlation statistics across four functional domains: Basic Movements, Walking, Oral Care, and Grooming.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Entry Points](#entry-points)
  - [run_capacity.py — R4 Stage Assignment](#run_capacitypy--r4-stage-assignment)
  - [run_effort.py — Physiological Effort Scoring](#run_effortpy--physiological-effort-scoring)
  - [run_correlation.py — Clinical Validation](#run_correlationpy--clinical-validation)
  - [run_feature_importance.py — Feature Importance](#run_feature_importancepy--feature-importance)
  - [run_plots.py — Visualisations](#run_plotspy--visualisations)
- [Configuration](#configuration)
  - [Capacity Rules (capacity_rules.yaml)](#capacity-rules-capacity_rulesyaml)
  - [Effort Config (effort_config.yaml)](#effort-config-effort_configyaml)
- [Outputs](#outputs)
- [Modules](#modules)
- [Tests](#tests)
- [Key Concepts](#key-concepts)

---

## Overview

The pipeline implements a three-phase analysis on top of HR-metric-extractor batch outputs:

| Phase | Script | What it does |
|-------|--------|--------------|
| **Capacity** | `run_capacity.py` | Assigns R4 stages (1–5) per domain from activity interval files |
| **Effort** | `run_effort.py` | Scores physiological effort (−100 to +100) relative to a healthy-control (HC) population |
| **Correlation** | `run_correlation.py` | Correlates sensor scores with clinical R4 labels; computes Spearman ρ, Kendall τ, Cohen's κ, and bootstrap CIs |

A fourth script, `run_feature_importance.py`, ranks sensor features by Spearman correlation with R4 stage, and `run_plots.py` generates presentation-quality figures from the correlation outputs.

---

## Pipeline Architecture

```
HR-metric-extractor
  ├── output_batch/patient_run/        ─┐
  └── output_batch/hc_run/             ─┤─ run_capacity.py ──→ capacity_scores.csv
                                        └─ run_effort.py ────→ effort_scores.csv
                                                                        │
                                                  R4 clinical CSV ──────┤
                                                                         ↓
                                              run_correlation.py ──→ correlation CSVs
                                                                         │
                                                                         ↓
                                                   run_plots.py ────→ figures (PNG)
```

Each subject's data lives in a subdirectory matching the glob `sub_*` inside the batch output directory. The pipeline reads activity interval CSVs and HR-metric CSVs from those subdirectories.

---

## Project Structure

```
functional-analysis-pipeline/
├── run_capacity.py             # CLI: R4 stage assignment
├── run_effort.py               # CLI: physiological effort scoring
├── run_correlation.py          # CLI: clinical correlation analysis
├── run_feature_importance.py   # CLI: feature importance ranking
├── run_plots.py                # CLI: figure generation
│
├── capacity/                   # Capacity qualifier package
│   ├── __init__.py
│   ├── qualifier.py            # Core stage-assignment engine
│   ├── batch_qualifier.py      # Batch runner for multiple subjects
│   └── rules.py                # YAML rule loader and data structures
│
├── effort/                     # Effort scorer package
│   ├── __init__.py
│   ├── reference.py            # HC reference profile builder
│   ├── scorer.py               # Per-subject effort scoring
│   ├── batch_scorer.py         # Batch runner
│   └── feature_importance.py   # Spearman-based feature ranking
│
├── correlation/                # Clinical validation package
│   ├── __init__.py
│   ├── merger.py               # Merges capacity, effort, and R4 CSVs
│   ├── validator.py            # Statistical correlation analysis
│   └── reporter.py             # Output CSV and report writer
│
├── config/
│   ├── capacity_rules.yaml     # R4 stage rules per domain
│   ├── effort_config.yaml      # Effort scoring parameters and domain/activity map
│   └── effort_config_no_imu.yaml
│
├── docs/
│   └── R4-scores/              # Reference R4 ground-truth label CSVs
│
├── tests/
│   ├── test_capacity.py
│   └── test_effort.py
│
├── output/                     # Generated outputs (gitignored)
└── scripts/                    # Research/sensitivity analysis scripts (gitignored)
```

---

## Installation

Python 3.10+ is recommended (the project was developed and tested on CPython 3.10).

**Install dependencies:**

```bash
pip install pandas numpy scipy scikit-learn matplotlib pyyaml
```

No `setup.py` or `pyproject.toml` is included — the package is intended to be run directly from the project root.

**Verify the install by running the test suite:**

```bash
python -m pytest tests/ -v
```

---

## Quick Start

A typical end-to-end run (assuming HR-metric-extractor has already produced batch output):

```bash
# 1. Assign R4 capacity stages
python run_capacity.py \
    --batch-dir path/to/patient_batch \
    --output-dir output/capacity/run1

# 2. Score physiological effort against HC reference
python run_effort.py \
    --patient-batch-dir path/to/patient_batch \
    --hc-batch-dir      path/to/hc_batch \
    --output-dir        output/effort/run1

# 3. Correlate sensor scores with clinical R4 labels
python run_correlation.py \
    --capacity-csv output/capacity/run1/capacity_scores.csv \
    --effort-csv   output/effort/run1/effort_scores.csv \
    --r4-csv       docs/R4-scores/R4_scores_nursing_home.csv \
    --output-dir   output/correlation/run1

# 4. Generate figures
python run_plots.py \
    --combined-csv    output/correlation/run1/combined_analysis.csv \
    --correlation-csv output/correlation/run1/r4_correlation.csv \
    --output-dir      output/plots/run1
```

---

## Entry Points

### `run_capacity.py` — R4 Stage Assignment

Reads activity interval CSVs from one or more HR-metric-extractor batch directories and assigns an R4 stage (1–5) per domain for each subject. Stage 5 indicates full capacity; Stage 1 is the default when no evidence is found.

```bash
python run_capacity.py \
    --batch-dir <DIR> [<DIR> ...] \
    --rules     config/capacity_rules.yaml \
    --output-dir output/capacity
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-dir` | *(required)* | One or more batch directories; multiple directories are combined |
| `--rules` | `config/capacity_rules.yaml` | Path to the YAML rules file |
| `--output-dir` | *(print only)* | Directory to write `capacity_scores.csv`; omit to print only |
| `--subject-glob` | `sub_*` | Glob pattern for subject subdirectories |
| `--icf-labels` | — | Optional ground-truth R4 CSV for accuracy comparison |
| `--icf-id-col` | `Participant` | ID column name in the ground-truth CSV |

**Output:** `capacity_scores.csv` — one row per subject with a stage column per domain, plus `_capped` flag columns indicating subjects where higher stages were skipped due to `not_assessable` rules.

---

### `run_effort.py` — Physiological Effort Scoring

Builds a per-activity reference profile from the HC cohort, then scores each patient subject's physiological deviation from that reference on a centered −100 to +100 scale (0 = HC median; +100 = HC 95th percentile).

```bash
python run_effort.py \
    --patient-batch-dir <DIR> \
    --hc-batch-dir      <DIR> \
    --output-dir        output/effort/run1
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--patient-batch-dir` | *(required)* | Directory of patient subject folders |
| `--hc-batch-dir` | *(required)* | Directory of healthy-control subject folders |
| `--config` | `config/effort_config.yaml` | Effort scoring configuration |
| `--output-dir` | `output/effort/<timestamp>/` | Directory for output CSVs |
| `--subject-glob` | `sub_*` | Glob applied to both patient and HC directories |
| `--patient-subject-glob` | — | Override glob for patients only |
| `--hc-subject-glob` | — | Override glob for HC only |
| `--no-save` | `False` | Print scores to console without writing files |
| `-v / --verbose` | `False` | Enable debug logging |

**Output:** Two CSVs in `--output-dir`:
- `effort_scores.csv` — one row per subject, effort score and reliability flag per domain
- `effort_details.csv` — per-activity breakdown with modality-level sub-scores

Scores flagged with `*` in the console summary have at least one activity with low reliability (insufficient windows or feature coverage).

---

### `run_correlation.py` — Clinical Validation

Merges capacity and effort scores with clinical R4 labels, then computes a battery of statistics: Spearman ρ, Kendall τ, weighted Cohen's κ, exact/within-1 agreement, and ICF-inspired composite scores. All correlation CIs are estimated by bootstrap (default 1000 resamples).

```bash
python run_correlation.py \
    --capacity-csv output/capacity/run1/capacity_scores.csv \
    --effort-csv   output/effort/run1/effort_scores.csv \
    --r4-csv       docs/R4-scores/R4_scores_nursing_home.csv \
    --output-dir   output/correlation/run1
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--capacity-csv` | *(required)* | Output of `run_capacity.py` |
| `--effort-csv` | *(required)* | Output of `run_effort.py` |
| `--r4-csv` | *(required)* | Clinical R4 scores CSV |
| `--output-dir` | *(required)* | Directory for output CSVs |
| `--reliable-only` | `False` | Exclude subjects with all-unreliable effort scores |
| `--n-bootstrap` | `1000` | Number of bootstrap resamples for 95% CI |
| `--gap-threshold` | `1` | Minimum stage gap to flag in gap analysis (default flags gaps ≥ 2) |
| `--verbose` | `False` | Enable debug logging |

**Expected correlation directions:**
- Effort vs R4: **negative** (higher physiological effort → lower functional capacity)
- Capacity stage vs R4: **positive** (higher assigned stage → higher clinical R4)

**Output CSVs in `--output-dir`:** `r4_correlation.csv`, `combined_analysis.csv`, plus a gap analysis file.

---

### `run_feature_importance.py` — Feature Importance

Ranks all sensor features by Spearman correlation with the assigned R4 capacity stage. Useful for identifying which physiological signals drive capacity discrimination.

```bash
python run_feature_importance.py \
    --patient-batch-dir path/to/patient_batch \
    --hc-batch-dir      path/to/hc_batch \
    --capacity-scores   output/capacity/run1/capacity_scores.csv \
    --output-dir        output/feature_importance/run1 \
    --top 10 \
    --plot
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--patient-batch-dir` | *(required)* | Patient batch directory |
| `--hc-batch-dir` | *(required)* | HC batch directory |
| `--capacity-scores` | *(required)* | `capacity_scores.csv` from `run_capacity.py` |
| `--top` | `10` | Print top-N features per domain × activity |
| `--plot` | `False` | Generate PNG visualisations |
| `--plot-top` | `10` | Features shown per activity in the plot |

**Output:** `feature_importance.csv` and optional PNG plots in `--output-dir`.

---

### `run_plots.py` — Visualisations

Generates a set of presentation-quality figures from correlation analysis outputs.

```bash
python run_plots.py \
    --combined-csv    output/correlation/run1/combined_analysis.csv \
    --correlation-csv output/correlation/run1/r4_correlation.csv \
    --output-dir      output/plots/run1
```

**Figures produced:**

| File | Description |
|------|-------------|
| `icf_dual_axis.png` | ICF scatter per domain (effort vs capacity, coloured by R4 stage) |
| `correlation_summary.png` | Spearman ρ bar chart per domain × modality with CI whiskers |
| `effort_by_r4_stage.png` | Box plots of effort distribution per R4 stage and domain |
| `effort_variability_by_r4_and_domain.png` | Effort spread within R4 classes |
| `clinical_case_comparison.png` | Two individuals with similar capacity but different effort profiles |
| `modality_correlation_heatmap.png` | Heatmap of modality-level Spearman ρ across domains |

---

## Configuration

### Capacity Rules (`capacity_rules.yaml`)

Defines the R4 stage criteria for each domain. Stages run from 1 (worst) to 5 (best). The qualifier scans descending from Stage 5; the first satisfied stage is assigned. Stage 1 is the default when no evidence is found.

**Stage criteria structure:**

```yaml
domains:
  basic_mobility:
    r4_label: "Basic Movements"
    stages:
      5:
        description: "Maintains standing position (≥3 min without assistance)"
        checks:
          - source: activity_standing
            keywords: []           # [] = accept all rows
            min_occurrences: 1
            min_duration_sec: 180
      4:
        description: "..."
        checks: [...]
      # ... stages 3, 2; stage 1 is the default (no checks needed)
```

**Check fields:**

| Field | Description |
|-------|-------------|
| `source` | Activity file key (mapped to a CSV in the subject directory) |
| `keywords` | Substring include-filter on the `activity` column (OR logic; empty = accept all) |
| `exclude_keywords` | Substring exclude-filter applied after include filter |
| `min_occurrences` | Minimum rows that must survive all filters |
| `min_duration_sec` | Per-row minimum duration threshold |

Setting `not_assessable: true` on a stage marks it as unevaluable from protocol data. The qualifier skips it and sets `capped_by_not_assessable: true` in the result, signalling the true capacity may be higher.

---

### Effort Config (`effort_config.yaml`)

Controls the effort scoring algorithm and maps domains to their constituent activity files.

**Scoring section:**

```yaml
scoring:
  feature_reducer: "median"       # How to collapse per-feature deviations → one score per window
  window_reducer: "median"        # How to collapse per-window scores → one score per activity
  min_windows: 3                  # Subjects below this get NaN and a reliability warning
  normalization_p100: 95          # HC p95 maps to +100; HC p5 maps to −100
  epsilon: 1.0e-6                 # Added to MAD to avoid division by zero
  augment_with_statistics: false  # Append min/max/mean/std as derived features
  exclude_features: [...]         # Features always excluded from scoring
  modality_groups: {...}          # Maps feature prefixes to modality labels
  inverse_feature_patterns: [...] # Features where lower value = higher effort (e.g. rmssd, sdnn)
```

Available reducer options: `median`, `mean`, `trimmed_mean`, `iqr_mean`, `huber`.

**Domains section:**

```yaml
domains:
  basic_mobility:
    r4_label: "Basic Movements"
    activities:
      standing:
        file: activity_standing_hr_metrics.csv
        weight: 1.0
      transfer:
        file: activity_transfer_hr_metrics.csv
        weight: 1.0
```

Each domain maps to one or more activity files. Weights are used when averaging activity scores to a domain-level score.

An `effort_config_no_imu.yaml` variant is included for runs where IMU features are unavailable.

---

## Outputs

All outputs are written under `output/` (gitignored). The typical directory layout after a full run:

```
output/
├── capacity/
│   └── <batch_name>/
│       └── capacity_scores.csv
├── effort/
│   └── <batch_name>/
│       ├── effort_scores.csv       # Per-subject domain scores and reliability flags
│       └── effort_details.csv      # Per-activity breakdown with modality sub-scores
├── correlation/
│   └── <run_name>/
│       ├── combined_analysis.csv   # Merged capacity + effort + R4 per subject × domain
│       ├── r4_correlation.csv      # Correlation statistics per domain
│       └── gap_analysis.csv        # Subjects with large stage discrepancies
└── plots/
    └── <run_name>/
        └── *.png
```

---

## Modules

### `capacity`

| Module | Responsibility |
|--------|---------------|
| `rules.py` | Loads `capacity_rules.yaml` into `DomainRules` / `StageRule` / `StageCheck` dataclasses |
| `qualifier.py` | Core engine: loads activity CSVs for one subject, evaluates stage checks, assigns stages with full audit trail via `CheckEvidence` / `StageEvidence` dataclasses |
| `batch_qualifier.py` | Iterates over all subject directories in a batch and calls `qualifier.score_subject` |

### `effort`

| Module | Responsibility |
|--------|---------------|
| `reference.py` | Builds per-activity HC reference profiles (median + MAD per feature); loads `effort_config.yaml` |
| `scorer.py` | Scores one subject against the HC reference; normalises to −100…+100 scale; produces `SubjectEffortResult` with per-modality breakdown |
| `batch_scorer.py` | Batch runner; writes `effort_scores.csv` and `effort_details.csv` |
| `feature_importance.py` | Computes Spearman ρ between each feature and R4 capacity stage; optional PNG plots |

### `correlation`

| Module | Responsibility |
|--------|---------------|
| `merger.py` | Joins capacity, effort, and clinical R4 CSVs on `subject_id`; optionally filters to reliable-only subjects |
| `validator.py` | Computes Spearman ρ, Kendall τ, weighted Cohen's κ, exact/within-1 agreement, and bootstrap 95% CIs; ICF composite score analysis; per-modality correlations |
| `reporter.py` | Writes `combined_analysis.csv`, `r4_correlation.csv`, and gap analysis outputs |

---

## Tests

Unit tests are in `tests/` and use synthetic in-memory data — no real files or HR-metric-extractor outputs are required to run them.

```bash
# Run all tests
python -m pytest tests/ -v

# Run only capacity tests
python -m pytest tests/test_capacity.py -v

# Run only effort tests
python -m pytest tests/test_effort.py -v
```

---

## Key Concepts

**R4 domains** — the four functional domains assessed by this pipeline, aligned with the clinical R4 framework: Basic Movements, Walking, Oral Care, Grooming.

**R4 stage (1–5)** — capacity level within a domain. Stage 5 indicates the highest functional capacity (e.g., maintains standing ≥3 min independently); Stage 1 is the lowest (default when no evidence is found). Assignment is cumulative: satisfying Stage 4 implies Stage 3 capabilities are also present.

**Effort score (−100 to +100)** — a subject's physiological deviation from the HC population during an activity. A score of 0 equals the HC median; +100 equals the HC 95th percentile (configurable via `normalization_p100`). Higher scores indicate greater physiological strain. Scores outside ±100 are clipped.

**Reliability flag** — an effort score is marked unreliable when the subject has fewer than `min_windows` valid signal windows for an activity, or when feature coverage is insufficient. Unreliable scores are included in outputs but flagged so downstream analyses can filter them.

**Modality sub-scores** — the effort scorer splits features into modality groups (HR/HRV, EDA, IMU wrist, IMU bioz, IMU chest) and reports a separate effort score per modality, enabling per-sensor-type analysis.

**ICF composite** — an ICF-inspired composite score derived from capacity stage and effort: `capacity → icf_capacity (0–4)` and `effort → icf_performance (0–4)`, averaged to a single `composite (0–4)`. Used in `run_correlation.py` to mirror the ICF body-function / activity framework.

**Healthy Controls (HC)** — the reference population against which effort is normalised. HC subjects are expected to come from the same measurement protocol but without functional impairment.

**capped_by_not_assessable** — a flag in `capacity_scores.csv` indicating that one or more stages were skipped during assignment because they were marked `not_assessable` in the rules. The true capacity of that subject may be higher than the assigned stage.
