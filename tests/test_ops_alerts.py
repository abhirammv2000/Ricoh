"""The alert rules only mean something if every metric, outcome and error code they name exists.

promtool checks the syntax of ops/alerts.yml and does not know what the service exports, so a renamed metric
would leave an alert that is valid and never fires. This ties the two together.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from api import errors
from api.metrics import OUTCOMES, Metrics

ALERTS = Path(__file__).resolve().parent.parent / "ops" / "alerts.yml"


@pytest.fixture(scope="module")
def rules():
    data = yaml.safe_load(ALERTS.read_text(encoding="utf-8"))
    return [rule for group in data["groups"] for rule in group["rules"]]


@pytest.fixture(scope="module")
def exported_series():
    # a metric with labels has no sample line until it is first used, but it always has a TYPE line, so the
    # declarations are what to read. A histogram is exposed as three series
    text = Metrics().render()[0].decode()
    names = set()
    for name, kind in re.findall(r"^# TYPE ([a-z_]+) ([a-z]+)$", text, re.M):
        names.add(name)
        if kind == "histogram":
            names |= {f"{name}_bucket", f"{name}_sum", f"{name}_count"}
    return names


def test_every_rule_has_a_name_an_expression_a_severity_and_a_runbook_link(rules):
    assert len(rules) >= 10
    for rule in rules:
        assert rule["alert"].startswith("Citera") and rule["expr"].strip()
        assert rule["labels"]["severity"] in ("page", "ticket")
        assert rule["annotations"]["summary"] and rule["annotations"]["runbook"].startswith("docs/OPERATIONS.md#")


def test_every_metric_an_alert_names_is_exported_by_the_service(rules, exported_series):
    problems = []
    for rule in rules:
        for name in set(re.findall(r"\bcitera_[a-z_]+\b", rule["expr"])):
            if name not in exported_series:
                problems.append(f"{rule['alert']} uses {name}")
    assert not problems, problems


def test_every_outcome_and_error_code_an_alert_names_exists(rules):
    codes = {value.code for value in vars(errors).values() if isinstance(value, errors.PublicError)}
    for rule in rules:
        for outcome in re.findall(r'outcome="([a-z]+)"', rule["expr"]):
            assert outcome in OUTCOMES, (rule["alert"], outcome)
        for alternation in re.findall(r'outcome=~"([a-z|]+)"', rule["expr"]):
            assert set(alternation.split("|")) <= set(OUTCOMES), (rule["alert"], alternation)
        for code in re.findall(r'code="([a-z_]+)"', rule["expr"]):
            assert code in codes, (rule["alert"], code)


def test_every_runbook_anchor_points_at_a_heading_in_the_runbook(rules):
    runbook = (ALERTS.parent.parent / "docs" / "OPERATIONS.md").read_text(encoding="utf-8")
    headings = {
        re.sub(r"[^a-z0-9]", "", line.lstrip("# ").strip().lower())
        for line in runbook.splitlines()
        if line.startswith("#")
    }
    for rule in rules:
        anchor = rule["annotations"]["runbook"].split("#", 1)[1]
        assert anchor in headings, f"{rule['alert']} links to #{anchor}, which is not a heading in docs/OPERATIONS.md"
