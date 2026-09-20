"""Recompute dashboard evidence from the shipped model and fixed held-out recordings.

This module never fits or mutates a model.  It deliberately rebuilds the eight-recording
test selection from the processed feature table, then scores it at the threshold stored in
the deployed artifact so the dashboard cannot drift from what the API actually serves.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


TEST_SELECTION = [
    ("Jamming", "1.10.6", "dynamic"), ("Jamming", "1.6.4", "stationary"),
    ("Meaconing", "3.2.8", "dynamic"), ("Meaconing", "3.2.7", "stationary"),
    ("Spoofing", "2.3.2", "dynamic"), ("Spoofing", "2.1.1", "stationary"),
    ("Spoofing + Jamming", "2.6.4", "dynamic"),
    ("Spoofing + Jamming", "2.6.3", "stationary"),
]


def plain_feature_name(feature: str) -> str:
    """Return an operator-readable label while preserving no raw implementation jargon."""
    f = feature.lower()
    if f.startswith("agc_"):
        return "Receiver gain control (mean)" if f.endswith("_mean") else "Receiver gain control"
    if f.startswith("snr_"):
        statistic = "mean" if f.endswith("_mean") else "variation" if f.endswith("_std") else "minimum" if f.endswith("_min") else "value"
        return f"Signal-to-noise ratio ({statistic})"
    if f.startswith("noise_per_ms_"):
        return "Radio noise floor (mean)" if f.endswith("_mean") else "Radio noise floor"
    if f.startswith("jam_ind_"):
        return "Receiver jamming indicator (mean)" if f.endswith("_mean") else "Receiver jamming indicator"
    if f.startswith("doppler_") or f.startswith("pr_doppler_"):
        return "Doppler consistency"
    if f in {"pos_dev_m", "hacc", "vacc", "headacc", "sacc", "pdop"}:
        detail = {"pos_dev_m": "position deviation", "hacc": "horizontal accuracy",
                  "vacc": "vertical accuracy", "headacc": "heading accuracy",
                  "sacc": "speed accuracy", "pdop": "geometry quality"}[f]
        return f"Position solution quality ({detail})"
    if f.startswith("vel") or f == "gspeed":
        detail = {"veln": "north velocity", "vele": "east velocity",
                  "veld": "vertical velocity", "gspeed": "ground speed"}.get(f, "velocity")
        return f"Reported motion ({detail})"
    if f in {"n_sats_l1", "numsv", "fixtype"}:
        detail = {"n_sats_l1": "L1 satellites", "numsv": "satellites used",
                  "fixtype": "fix type"}[f]
        return f"Satellite tracking ({detail})"
    return feature.replace("_", " ").strip().title()


def model_display_name(model_service) -> str:
    return (f"Detector v2 - threshold {model_service.decision_threshold:.2f} - "
            f"{model_service.model_sha256[:12]}")


def compute_evaluation(model_service, dataset_path: Path) -> dict:
    df = pd.read_parquet(dataset_path)
    run_ids = set()
    for attack_type, scenario_id, rover_state in TEST_SELECTION:
        matches = df.loc[
            (df["attack_type"] == attack_type)
            & (df["scenario_id"] == scenario_id)
            & (df["rover_state"] == rover_state), "run_id"
        ].unique()
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one held-out run for {(attack_type, scenario_id, rover_state)}, "
                f"found {len(matches)}"
            )
        run_ids.add(matches[0])

    held_out = df[df["run_id"].isin(run_ids)].copy()
    y_true = held_out["attack"].astype(int).to_numpy()
    probabilities = model_service.pipeline.predict_proba(
        held_out[model_service.feature_cols]
    )[:, 1]
    predictions = (probabilities >= model_service.decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    held_out["prediction"] = predictions
    per_type = []
    for attack_type, group in held_out[held_out["attack"] == 1].groupby("attack_type"):
        per_type.append({
            "attack_type": attack_type,
            "n_rows": int(len(group)),
            "recall": float(group["prediction"].mean()),
        })

    rf = model_service.pipeline.named_steps["rf"]
    importance_order = np.argsort(rf.feature_importances_)[::-1][:10]
    feature_importances = [{
        "feature": model_service.feature_cols[i],
        "plain_name": plain_feature_name(model_service.feature_cols[i]),
        "importance": float(rf.feature_importances_[i]),
    } for i in importance_order]

    total = int(tn + fp + fn + tp)
    return {
        "model_display": model_display_name(model_service),
        "decision_threshold": model_service.decision_threshold,
        "held_out_recordings": len(run_ids),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "metrics": {
            "accuracy": float((tn + tp) / total),
            "attack_recall": float(tp / (tp + fn)),
            "clean_recall": float(tn / (tn + fp)),
            "false_alarm_rate": float(fp / (fp + tn)),
            "attack_precision": float(tp / (tp + fp)),
            "clean_precision": float(tn / (tn + fn)),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "clean_support": int(tn + fp),
            "attack_support": int(tp + fn),
            "total_support": total,
        },
        "per_attack_recall": per_type,
        "feature_importances": feature_importances,
    }
