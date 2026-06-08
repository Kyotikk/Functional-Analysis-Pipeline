"""Data structures and YAML loader for R4 capacity rules."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class StageCheck:
    """A single evidence check against one activity source file."""

    source: str
    """File-name prefix key (resolved to a CSV path by the qualifier)."""

    keywords: List[str] = field(default_factory=list)
    """Include filter: at least one keyword must appear as a substring of the
    activity label (case-insensitive). Empty list = accept all rows."""

    exclude_keywords: List[str] = field(default_factory=list)
    """Exclude filter applied after the include filter: rows whose activity
    label contains any of these substrings are rejected."""

    min_occurrences: int = 1
    """Minimum number of rows that must survive all filters."""

    min_duration_sec: float = 0.0
    """Per-row minimum duration. Rows below this threshold are excluded."""


@dataclass
class StageRule:
    """Criteria for a single R4 stage within a domain."""

    description: str
    checks: List[StageCheck]
    not_assessable: bool = False
    """True when this stage cannot be evaluated from controlled-protocol data."""
    note: str = ""


@dataclass
class DomainRules:
    """All stage rules for one R4 domain."""

    name: str
    """Internal key (e.g. 'basic_mobility')."""

    r4_label: str
    """Human-readable label matching the R4 scores CSV column name."""

    stages: Dict[int, StageRule]
    """Mapping of stage number → StageRule. Stages 1–5, where 5 = best."""


def load_rules(path: Path) -> Dict[str, DomainRules]:
    """Parse a capacity rules YAML file.

    Returns a dict mapping domain key → DomainRules.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if "domains" not in raw:
        raise ValueError(f"Rules file {path} must have a top-level 'domains' key.")

    domains: Dict[str, DomainRules] = {}
    for domain_key, domain_data in raw["domains"].items():
        r4_label = domain_data.get("r4_label", domain_key)
        stages: Dict[int, StageRule] = {}

        for stage_num_raw, stage_data in domain_data.get("stages", {}).items():
            stage_num = int(stage_num_raw)
            checks: List[StageCheck] = []
            for chk in stage_data.get("checks", []):
                checks.append(
                    StageCheck(
                        source=chk["source"],
                        keywords=[str(k) for k in chk.get("keywords", [])],
                        exclude_keywords=[str(k) for k in chk.get("exclude_keywords", [])],
                        min_occurrences=int(chk.get("min_occurrences", 1)),
                        min_duration_sec=float(chk.get("min_duration_sec", 0.0)),
                    )
                )
            stages[stage_num] = StageRule(
                description=stage_data.get("description", ""),
                checks=checks,
                not_assessable=bool(stage_data.get("not_assessable", False)),
                note=str(stage_data.get("note", "")),
            )

        domains[domain_key] = DomainRules(
            name=domain_key,
            r4_label=r4_label,
            stages=stages,
        )

    return domains
