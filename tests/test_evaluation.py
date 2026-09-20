"""Dashboard evidence must come from the shipped model, not stale HTML constants."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from api.evaluation import TEST_SELECTION, plain_feature_name
from api.main import EVALUATION_DATASET_PATH


def _held_out_frame():
    df = pd.read_parquet(EVALUATION_DATASET_PATH)
    run_ids = set()
    for attack_type, scenario_id, rover_state in TEST_SELECTION:
        matches = df.loc[
            (df["attack_type"] == attack_type)
            & (df["scenario_id"] == scenario_id)
            & (df["rover_state"] == rover_state), "run_id"
        ].unique()
        assert len(matches) == 1
        run_ids.add(matches[0])
    return df[df["run_id"].isin(run_ids)].copy()


def test_served_evaluation_equals_independent_shipped_model_computation(client):
    served = client.get("/evaluation").json()
    model = client.app.state.model_service
    held_out = _held_out_frame()
    y_true = held_out["attack"].astype(int).to_numpy()
    probability = model.pipeline.predict_proba(held_out[model.feature_cols])[:, 1]
    predicted = (probability >= model.decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    assert np.isclose(served["decision_threshold"], 0.52)
    assert served["confusion_matrix"] == {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    } == {"tn": 2093, "fp": 1044, "fn": 690, "tp": 10250}
    m = served["metrics"]
    assert np.isclose(m["accuracy"], (tn + tp) / len(y_true))
    assert np.isclose(m["attack_recall"], tp / (tp + fn))
    assert np.isclose(m["clean_recall"], tn / (tn + fp))
    assert np.isclose(m["false_alarm_rate"], fp / (fp + tn))
    assert np.isclose(m["attack_precision"], tp / (tp + fp))
    assert np.isclose(m["roc_auc"], roc_auc_score(y_true, probability))
    assert np.isclose(m["pr_auc"], average_precision_score(y_true, probability))

    independent_recall = held_out.assign(predicted=predicted).query("attack == 1").groupby(
        "attack_type"
    )["predicted"].mean().to_dict()
    assert {r["attack_type"]: r["recall"] for r in served["per_attack_recall"]} == independent_recall
    assert len(served["feature_importances"]) == 10
    assert all(row["plain_name"] != row["feature"] for row in served["feature_importances"])
    assert served["model_display"].startswith("Detector v2 - threshold 0.52 - ")


def test_dashboard_contains_no_hard_coded_detector_metrics(client):
    html = client.get("/dashboard").text
    assert '"confusion_matrix"' not in html
    assert '"attack_recall"' not in html
    assert '"per_attack_recall"' not in html
    assert "DATA.detector" not in html
    assert "improved_v2_threshold_tuning_only" not in html
    assert "false-alarm rate on clean data" in html.lower()


def test_global_header_is_operational_and_detector_summary_stays_on_evidence_tab(client):
    html = client.get("/dashboard").text
    header = html.split('<section class="kpi-strip"', 1)[1].split("</section>", 1)[0]
    assert "kpi-static" not in header
    assert "Real · detector, held-out" not in header
    for element_id in ("alert-pill", "active-alerts", "flagged-towers", "replay-count", "data-mode"):
        assert f'id="{element_id}"' in header
    assert 'id="detector-footer"' in html
    assert "m.attack_recall" in html
    assert "m.false_alarm_rate" in html


def test_plain_feature_names_cover_the_operator_groups():
    assert plain_feature_name("agc_cnt_mean").startswith("Receiver gain control")
    assert plain_feature_name("snr_l1_mean").startswith("Signal-to-noise ratio")
    assert plain_feature_name("noise_per_ms_mean").startswith("Radio noise floor")
    assert plain_feature_name("jam_ind_mean").startswith("Receiver jamming indicator")
    assert plain_feature_name("pr_doppler_residual_std") == "Doppler consistency"
    assert plain_feature_name("hAcc").startswith("Position solution quality")
    assert plain_feature_name("velN").startswith("Reported motion")
    assert plain_feature_name("numSV").startswith("Satellite tracking")
