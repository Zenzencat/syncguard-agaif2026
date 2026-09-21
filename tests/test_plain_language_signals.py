"""Incident-detail panel: plain-language SHAP signals (display layer only).

The panel fetches the same /events/{id}/explain data as before and translates it client-side:
a friendly name + one-line description per feature (a single lookup table), and a strength/
direction phrase derived from the SHAP value's sign and its relative size among the signals
shown. Raw feature names and exact SHAP values stay one click away under "Technical detail".

Static checks against the served dashboard (same convention as tests/test_dashboard_map_click.py:
one HTML file, inline JS). The pure helpers are additionally executed with node when it is
installed (skipped otherwise), so the strength rule is tested as behaviour, not just as text.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from api.schemas import FeatureVector

# Jargon a non-specialist would have to look up. Must not appear in a friendly NAME.
JARGON = ("AGC", "SNR", "C/N0", "CN0", "DOP", "L1", "Doppler", "pseudorange", "SHAP", "PVT", "RINEX", "u-blox")


def _html(client) -> str:
    return client.get("/dashboard").text


def _script(html: str) -> str:
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def _labels(html: str) -> dict:
    m = re.search(r"/\*FEATURE_LABELS_BEGIN\*/(.*?)/\*FEATURE_LABELS_END\*/", html, re.S)
    assert m, "the feature lookup table markers must be present"
    return json.loads(m.group(1))


def _pure_block(html: str) -> str:
    m = re.search(r"// ---- plain-language signals BEGIN.*?// ---- plain-language signals END ----", html, re.S)
    assert m
    return m.group(0)


def _function_body(js: str, name: str) -> str:
    m = re.search(r"function\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{", js)
    assert m, f"expected function {name}() to exist"
    depth, i = 1, m.end()
    while depth and i < len(js):
        depth += {"{": 1, "}": -1}.get(js[i], 0)
        i += 1
    return js[m.end():i - 1]


# ---------------------------------------------------------------- the lookup table

def test_lookup_covers_exactly_the_23_model_features(client):
    features = set(FeatureVector.model_fields)
    assert len(features) == 23
    assert set(_labels(_html(client))) == features, "every feature needs a label, and no stale extras"


def test_every_label_is_short_plain_and_unique(client):
    labels = _labels(_html(client))
    names = [v["name"] for v in labels.values()]
    assert len(set(names)) == len(names), "two features share a display name"
    for feature, v in labels.items():
        assert v["name"].strip() and v["desc"].strip(), feature
        assert len(v["name"].split()) <= 6, f"{feature}: name should be a few words"
        assert len(v["desc"]) <= 170, f"{feature}: description should be one short line"
        for term in JARGON:
            assert term.lower() not in v["name"].lower(), f"{feature}: name uses jargon {term!r}"
        assert "shap" not in v["desc"].lower(), feature


def test_labels_keep_the_corrections_found_against_the_code(client):
    """Regressions for things a first-draft label got wrong when checked against
    extract_features.py / INGESTION_CONTRACT.md / measured data."""
    labels = _labels(_html(client))
    # every *_std is a spread ACROSS SATELLITES at one epoch, not variation over time
    # (pr_doppler_residual_std uses the wording the owner chose for the whole residual pair; see below)
    for f in ("snr_l1_std", "doppler_l1_std"):
        assert "one satellite to another" in labels[f]["desc"], f
    # the RF-monitor block is antenna path 01 only -- never claim both antennas
    for f in ("jam_ind_mean", "agc_cnt_mean", "noise_per_ms_mean"):
        assert "both" not in labels[f]["desc"].lower() and "antennas" not in labels[f]["desc"].lower(), f
    # 'AGC spikes under jamming' is NOT supported by the real data (median unchanged for jamming
    # recordings, lower for spoofing/meaconing) -- the description must not assert a direction
    agc = labels["agc_cnt_mean"]["desc"].lower()
    assert "spike" not in agc and "jamming" not in agc
    # tracked-on-L1 vs used-in-the-fix are two different counts
    assert "tracking" in labels["n_sats_l1"]["desc"] and "actually used" in labels["numSV"]["desc"]
    # accuracy fields are the receiver's own ESTIMATE of error, and higher = less certain
    for f in ("hAcc", "vAcc", "sAcc", "headAcc"):
        assert "own estimate" in labels[f]["desc"] and "uncertainty" in labels[f]["name"].lower(), f
    # the range-vs-Doppler residual is per-satellite self-consistency (distance change vs frequency
    # shift), not satellites agreeing with each other. Owner-chosen wording, applied to both the
    # (average) and (spread) rows.
    for f in ("pr_doppler_residual_mean", "pr_doppler_residual_std"):
        assert "each satellite's measured distance change matches its measured frequency shift" in labels[f]["desc"], f
        assert labels[f]["name"].startswith("Motion-consistency check"), f
    assert labels["pr_doppler_residual_mean"]["name"].endswith("(average)")
    assert labels["pr_doppler_residual_std"]["name"].endswith("(spread)")
    assert "first minute" in labels["pos_dev_m"]["desc"]


# ---------------------------------------------------------------- strength / direction rule (node)

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _run_node(html: str, expr: str):
    js = _pure_block(html) + f"\nconsole.log(JSON.stringify({expr}));"
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@needs_node
def test_strength_rule_is_relative_and_at_most_one_row_is_strong(client):
    html = _html(client)
    phrases = lambda vals: [s["phrase"] for s in _run_node(  # noqa: E731
        html, "signalStrengths(" + json.dumps([{"shap_value": v} for v in vals]) + ")")]
    # one signal outweighs the other two combined -> only it is "strongly"
    assert phrases([0.30, 0.05, -0.04]) == ["Strongly suggests an attack", "Suggests an attack",
                                            "Points away from an attack"]
    # three near-equal signals: none is strongly (rank alone must not create emphasis)
    assert phrases([0.10, 0.09, 0.11]) == ["Suggests an attack"] * 3
    # sign decides direction; a dominant negative is "strongly points away"
    assert phrases([-0.30, 0.05, 0.04])[0] == "Strongly points away from an attack"
    # exactly half of the shown magnitude does not OUTWEIGH the rest
    assert phrases([0.2, 0.1, 0.1]) == ["Suggests an attack"] * 3
    # zero contribution is neither direction
    assert phrases([0, 0, 0]) == ["Did not push the verdict either way"] * 3
    # the model is binary: wording must never name a specific attack type
    for vals in ([0.3, 0.05, -0.04], [-0.3, 0.05, 0.04]):
        for p in phrases(vals):
            assert not re.search(r"spoof|jam|meacon", p, re.I)


@needs_node
def test_unknown_feature_shows_its_raw_name_instead_of_disappearing(client):
    html = _html(client)
    assert _run_node(html, "friendlyFeature('some_future_feature')")["name"] == "some_future_feature"
    assert _run_node(html, "friendlyFeature('constructor')")["known"] is False   # not an object-prototype hit
    assert _run_node(html, "friendlyFeature('hAcc')")["known"] is True


# ---------------------------------------------------------------- render path

def test_incident_panel_is_plain_language_by_default_with_technical_detail_one_click_away(client):
    js = _script(_html(client))
    detail = _function_body(js, "renderIncidentDetail")
    assert "plainSignalsHtml(" in detail
    assert "shapBarsHtml(" not in detail, "the incident panel must not show raw bars/decimals by default"
    body = _function_body(js, "plainSignalsHtml")
    assert '<details class="tech-detail"' in body and "<summary>Technical detail</summary>" in body
    # nothing hidden permanently: raw feature name, value, exact SHAP value and direction are all there
    for needle in ("f.feature", "f.feature_value", "sv.toFixed(4)", "f.direction"):
        assert needle in body, needle


def test_display_layer_only_same_explain_endpoint_and_raw_explorer_untouched(client):
    js = _script(_html(client))
    assert "/events/${inc.top_event_id}/explain" in _function_body(js, "renderIncidentDetail")
    # the advanced raw-log explorer still uses the technical bars
    assert "shapBarsHtml(topFeatures)" in _function_body(js, "renderExplanation")


def test_copied_incident_summary_uses_the_same_plain_vocabulary(client):
    js = _script(_html(client))
    body = _function_body(js, "incidentSummaryText")
    assert "friendlyFeature(f.feature).name" in body and "signalStrengths(" in body
