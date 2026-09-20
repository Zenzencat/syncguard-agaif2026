"""Review finding: a flagged tower could show severity 0.349 in the Prioritize table, well
below the 0.52 alert threshold, because hysteresis keeps a tower's alert_state 'alerting'
across a 3-in/5-out debounce even after its latest reading's severity has dipped -- so the
column was really showing "latest reading", not "what triggered the flag". Decision: relabel
the column rather than add peak-severity tracking (no schema/data-model change needed, and
the value genuinely is the latest reading -- api/exposure.py's rank_priority() sources it
straight from latest_severity_per_tower()).
"""
from __future__ import annotations


def test_prioritize_severity_column_is_labeled_latest_reading(client):
    html = client.get("/dashboard").text
    assert "Latest reading (SIMULATED)" in html
    assert ">SIMULATED severity</div></div>" not in html, (
        "the old label implied this value is what triggered the flag, not just the latest reading"
    )
