"""
SyncGuard baseline classifier: RandomForest, attack (1) vs clean (0), per-epoch.

Methodology notes (see dataset_notes.md / baseline_model_report.md for the full writeup):
  - Train/test split is by RECORDING (run_id), not by row. Rows within a scenario are ~1Hz
    samples of a continuous, highly autocorrelated time series -- a random row split would put
    near-duplicate neighboring rows in both train and test and inflate scores.
  - Feature matrix excludes scenario metadata (attack_type, rover_state, bands, timestamps,
    run/scenario IDs) that a deployed detector would not have as ground truth -- only
    signal-derived measurements are used.
  - clock_drift_proxy_s is dropped (flat/uninformative, per review).
  - Class imbalance (77% attack / 23% clean) is handled via class_weight='balanced', not
    resampling, to keep all the data.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              average_precision_score, ConfusionMatrixDisplay)

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "processed"
df = pd.read_parquet(DATA_DIR / "syncguard_features.parquet")

# --- Held-out test recordings: one dynamic + one stationary run per attack_type, so every
# attack_type x rover_state combination present in the data is represented in both splits.
# Matched by (attack_type, scenario_id, rover_state) rather than the raw folder path, since
# one folder name contains a unicode ">=" glyph that's awkward to hardcode reliably. ---
TEST_SELECTION = [
    ("Jamming", "1.10.6", "dynamic"),
    ("Jamming", "1.6.4", "stationary"),
    ("Meaconing", "3.2.8", "dynamic"),
    ("Meaconing", "3.2.7", "stationary"),
    ("Spoofing", "2.3.2", "dynamic"),
    ("Spoofing", "2.1.1", "stationary"),
    ("Spoofing + Jamming", "2.6.4", "dynamic"),
    ("Spoofing + Jamming", "2.6.3", "stationary"),
]
TEST_RUN_IDS = set()
for atype, sid, rstate in TEST_SELECTION:
    match = df.loc[(df["attack_type"] == atype) & (df["scenario_id"] == sid) &
                    (df["rover_state"] == rstate), "run_id"].unique()
    assert len(match) == 1, f"Expected exactly 1 run for {(atype, sid, rstate)}, found {list(match)}"
    TEST_RUN_IDS.add(match[0])
print(f"Held-out run_ids ({len(TEST_RUN_IDS)}):")
for r in sorted(TEST_RUN_IDS):
    print(" ", r)

EXCLUDE_COLS = {
    "iTOW", "real_time", "lat", "lon", "height", "clock_drift_proxy_s",
    "attack", "scenario_id", "attack_type", "rover_state", "power_w_max", "bands",
    "n_attack_windows", "window_start_in_range", "run_id",
}
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print("Feature columns:", feature_cols)

is_test = df["run_id"].isin(TEST_RUN_IDS)
train_df, test_df = df[~is_test], df[is_test]
print(f"\nTrain: {len(train_df)} rows from {train_df['run_id'].nunique()} recordings")
print(f"Test:  {len(test_df)} rows from {test_df['run_id'].nunique()} recordings")
print(f"Test attack_type counts:\n{test_df['attack_type'].value_counts()}")

X_train, y_train = train_df[feature_cols], train_df["attack"]
X_test, y_test = test_df[feature_cols], test_df["attack"]

clf = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)),
])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

report_txt = classification_report(y_test, y_pred, target_names=["clean(0)", "attack(1)"], digits=3)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT (held-out recordings)")
print(report_txt)
print("Confusion matrix [rows=true, cols=pred], order [clean, attack]:")
print(cm)
print(f"\nROC-AUC: {roc_auc:.3f}")
print(f"PR-AUC (average precision): {pr_auc:.3f}")

# --- Per attack_type recall (only rows where the true label is attack=1) ---
test_df = test_df.copy()
test_df["pred"] = y_pred
per_type = test_df[test_df["attack"] == 1].groupby("attack_type").apply(
    lambda g: pd.Series({
        "n_attack_rows": len(g),
        "recall": (g["pred"] == 1).mean(),
    }), include_groups=False
)
print("\nPer-attack-type recall (of true attack rows, held-out set):")
print(per_type.to_string())

# --- Feature importances ---
importances = pd.Series(clf.named_steps["rf"].feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)
print("\nFeature importances:")
print(importances.to_string())

# --- Plots ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["clean", "attack"]).plot(
    ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion matrix (held-out recordings)")

importances.plot(kind="barh", ax=axes[1], color="tab:blue")
axes[1].invert_yaxis()
axes[1].set_title("RandomForest feature importances")
fig.tight_layout()
results_png = DATA_DIR / "baseline_model_results.png"
fig.savefig(results_png, dpi=130)
print(f"\nSaved plot to {results_png}")

# --- Write report ---
with open(SCRIPT_DIR / "baseline_model_report.md", "w", encoding="utf-8") as f:
    f.write("# SyncGuard baseline model -- RandomForest (attack vs. clean)\n\n")
    f.write("## Setup\n\n")
    f.write(f"- Features ({len(feature_cols)}): {', '.join(feature_cols)}\n")
    f.write("- `clock_drift_proxy_s` dropped (flat/uninformative).\n")
    f.write("- Scenario metadata (attack_type, rover_state, bands, timestamps, IDs) excluded from features -- not available to a deployed detector.\n")
    f.write("- Split: by recording (run_id), not by row, to avoid autocorrelation leakage. One dynamic + one stationary recording held out per attack_type.\n")
    f.write(f"- Train: {len(train_df)} rows / {train_df['run_id'].nunique()} recordings. Test: {len(test_df)} rows / {test_df['run_id'].nunique()} recordings.\n")
    f.write("- Class imbalance (77%/23% attack/clean) handled via `class_weight='balanced'`, not resampling.\n")
    f.write("- Model: RandomForestClassifier(n_estimators=300, class_weight='balanced'), median imputation for NaNs.\n\n")
    f.write("## Classification report (held-out recordings)\n\n```\n" + report_txt + "\n```\n\n")
    f.write(f"Confusion matrix [rows=true, cols=pred], order [clean, attack]:\n\n```\n{cm}\n```\n\n")
    f.write(f"ROC-AUC: {roc_auc:.3f}  \nPR-AUC (average precision): {pr_auc:.3f}\n\n")
    f.write("## Per-attack-type recall (held-out set, true attack rows only)\n\n")
    f.write(per_type.to_string() + "\n\n")
    f.write("## Feature importances\n\n```\n" + importances.to_string() + "\n```\n")

print("\nWrote baseline_model_report.md")
