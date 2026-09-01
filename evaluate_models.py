"""One-off verification script (not part of the pipeline): loads the two PERSISTED model
artifacts (models/model_baseline.joblib, models/model.joblib) and recomputes a full metrics
table directly against the real held-out TEST set -- same 8 recordings as both training
scripts use, rebuilt here independently from processed/syncguard_features.parquet rather than
trusting numbers already printed elsewhere. Confusion-matrix counts (TN/FP/FN/TP) are computed
directly via sklearn.metrics.confusion_matrix, and every rate (recall, FPR, FNR, accuracy) is
derived from those same counts, not from each other. Does not modify or retrain anything.

Run: python scripts_evaluate_models.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                              classification_report)

sys.stdout.reconfigure(encoding="utf-8")
SCRIPT_DIR = Path(__file__).resolve().parent

df = pd.read_parquet(SCRIPT_DIR / "processed" / "syncguard_features.parquet")
TEST_SELECTION = [
    ("Jamming", "1.10.6", "dynamic"), ("Jamming", "1.6.4", "stationary"),
    ("Meaconing", "3.2.8", "dynamic"), ("Meaconing", "3.2.7", "stationary"),
    ("Spoofing", "2.3.2", "dynamic"), ("Spoofing", "2.1.1", "stationary"),
    ("Spoofing + Jamming", "2.6.4", "dynamic"), ("Spoofing + Jamming", "2.6.3", "stationary"),
]
TEST_RUN_IDS = set()
for atype, sid, rstate in TEST_SELECTION:
    match = df.loc[(df["attack_type"] == atype) & (df["scenario_id"] == sid) &
                    (df["rover_state"] == rstate), "run_id"].unique()
    assert len(match) == 1
    TEST_RUN_IDS.add(match[0])
test_df = df[df["run_id"].isin(TEST_RUN_IDS)].copy()
y_test = test_df["attack"]
print(f"TEST set: {len(test_df)} rows / {test_df['run_id'].nunique()} recordings "
      f"(identical to both training scripts' held-out set)\n")


def evaluate(label, artifact_path):
    artifact = joblib.load(artifact_path)
    pipeline, feature_cols = artifact["pipeline"], artifact["feature_cols"]
    threshold = float(artifact.get("decision_threshold", 0.5))
    # Force single-threaded inference -- see api/model_service.py's comment. Without this,
    # RandomForestClassifier.predict_proba() with n_jobs>1 is not guaranteed bit-identical
    # across separate calls (confirmed: re-running this script's own predict_proba() call
    # gave a different confusion matrix than train_baseline_model.py's in-process
    # clf.predict() call did -- ~6 of 14077 rows, all within ~1e-15 of the 0.5 boundary).
    # n_jobs=1 is what api/main.py's /score endpoint uses in production, so these numbers
    # are also what the live API actually computes, not sklearn's internal .predict() path.
    pipeline.named_steps["rf"].n_jobs = 1
    X_test = test_df[feature_cols]

    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)  # rows=true [clean, attack], cols=pred [clean, attack]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    clean_recall = tn / (tn + fp)          # a.k.a. specificity, true negative rate
    fpr = fp / (fp + tn)                   # false positive rate -- clean rows wrongly flagged attack
    attack_recall = tp / (tp + fn)
    fnr = fn / (fn + tp)                   # false negative rate -- attack rows wrongly flagged clean
    clean_precision = tn / (tn + fn) if (tn + fn) else float("nan")
    attack_precision = tp / (tp + fp) if (tp + fp) else float("nan")
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    test_df["pred"] = y_pred
    per_type = test_df[test_df["attack"] == 1].groupby("attack_type").apply(
        lambda g: (g["pred"] == 1).mean(), include_groups=False)

    print("=" * 78)
    print(f"{label}  (threshold={threshold:.2f}, model_version={artifact.get('model_version')})")
    print("=" * 78)
    print(f"Confusion matrix [rows=true, cols=pred], order [clean, attack]:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Clean recall (TNR):  {clean_recall:.4f}")
    print(f"False Positive Rate: {fpr:.4f}  ({fp} of {fp+tn} true clean rows wrongly flagged attack)")
    print(f"Attack recall (TPR): {attack_recall:.4f}")
    print(f"False Negative Rate: {fnr:.4f}  ({fn} of {fn+tp} true attack rows wrongly flagged clean)")
    print(f"Clean precision:     {clean_precision:.4f}")
    print(f"Attack precision:    {attack_precision:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"PR-AUC:              {pr_auc:.4f}")
    print("Per-attack-type recall:")
    print(per_type.to_string())
    print()
    return dict(label=label, threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=accuracy, clean_recall=clean_recall, fpr=fpr,
                attack_recall=attack_recall, fnr=fnr, clean_precision=clean_precision,
                attack_precision=attack_precision, roc_auc=roc_auc, pr_auc=pr_auc,
                per_type=per_type)


baseline = evaluate("BASELINE", SCRIPT_DIR / "models" / "model_baseline.joblib")
improved = evaluate("IMPROVED", SCRIPT_DIR / "models" / "model.joblib")

print("=" * 78)
print("SIDE-BY-SIDE (baseline -> improved)")
print("=" * 78)
for key, fmt in [("accuracy", ".4f"), ("clean_recall", ".4f"), ("fpr", ".4f"),
                  ("attack_recall", ".4f"), ("fnr", ".4f"), ("roc_auc", ".4f"), ("pr_auc", ".4f")]:
    b, i = baseline[key], improved[key]
    print(f"{key:>16}: {b:{fmt}}  ->  {i:{fmt}}  (delta {i-b:+.4f})")
for atype in baseline["per_type"].index:
    b, i = baseline["per_type"][atype], improved["per_type"][atype]
    print(f"{atype:>24} recall: {b:.4f}  ->  {i:.4f}  (delta {i-b:+.4f})")
