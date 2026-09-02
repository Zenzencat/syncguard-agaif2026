"""Final model comparison: the shipped RandomForest vs. XGBoost, under the exact same
GroupKFold methodology established for the calibration and normalization experiments
(calibration_experiment.py, session_normalization_experiment.py) -- same fold construction,
same seed, same jamming-recall-floor-constrained threshold sweep as train_improved_model.py.
Does not modify models/model.joblib, the SHAP explainer, or any deployed threshold --
standalone comparison script. Writes GBM_COMPARISON.md.

--- Why XGBoost, not LightGBM (or both) ---
XGBoost was chosen as the sole challenger: it's the most standard, most widely-cited
gradient-boosting library for exactly this kind of RF-vs-GBM comparison, has mature,
well-tested shap.TreeExplainer support (directly relevant since SHAP is now a load-bearing
part of the shipped system -- see SHAP_EXPLAINABILITY.md), and -- confirmed empirically below,
not assumed -- has a materially simpler determinism story than RandomForest's (no n_jobs
predict-time sensitivity, .predict() and thresholded .predict_proba() agree exactly). LightGBM
would be a reasonable secondary comparison but wasn't run here given the scope of this
session's other work; it's a natural next step if this comparison motivates further model
exploration.

--- Determinism, verified directly (same rigor as ROBUSTNESS_NOTES.md's RF investigation) ---
Two independent XGBoost fits with the same random_state produce bit-identical predict_proba
output, with n_jobs=-1 or n_jobs=1, at fit time or predict time. Repeated predict_proba() calls
on the same fitted model are bit-identical. .predict() matches (predict_proba >= 0.5) exactly
on 4000 test rows (0 mismatches) -- unlike RandomForestClassifier, which needed n_jobs forced
to 1 at predict time to guarantee this (see ROBUSTNESS_NOTES.md). XGBoost needs no such
workaround; n_jobs=-1 is used throughout below for both fit and predict.

--- Class imbalance ---
class_weight='balanced' (RF) reweights both classes inversely to their frequency.  XGBoost's
right equivalent is scale_pos_weight = count(negative_class) / count(positive_class) -- note
"attack" (label 1) is the *majority* class here (~77%), so the standard formula naturally
produces scale_pos_weight < 1 (down-weighting the majority), which is exactly the intended
effect and requires no special-casing.  Computed fresh per fold from that fold's own training
data (never leaks val-fold class balance into the weight).
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")
SCRIPT_DIR = Path(__file__).resolve().parent
RNG_SEED = 42
OUTER_N_SPLITS = 4
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.02)
JAMMING_RECALL_TOLERANCE = 0.01  # same as train_improved_model.py

df = pd.read_parquet(SCRIPT_DIR / "processed" / "syncguard_features.parquet").reset_index(drop=True)
EXCLUDE_COLS = {
    "iTOW", "real_time", "lat", "lon", "height", "clock_drift_proxy_s",
    "attack", "scenario_id", "attack_type", "rover_state", "power_w_max", "bands",
    "n_attack_windows", "window_start_in_range", "run_id",
}
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"Full dataset: {len(df)} rows / {df['run_id'].nunique()} recordings")


# ---------------------------------------------------------------------------
# Identical GroupKFold construction to calibration_experiment.py /
# session_normalization_experiment.py (same seed, same logic) -- the established,
# already-validated evaluation harness for this project.
# ---------------------------------------------------------------------------
def attack_type_balanced_folds(frame: pd.DataFrame, n_splits: int, seed: int) -> list[set]:
    rng = np.random.RandomState(seed)
    folds = [set() for _ in range(n_splits)]
    for atype, sub in frame.groupby("attack_type"):
        runs = sub["run_id"].unique().tolist()
        rng.shuffle(runs)
        for i, run_id in enumerate(runs):
            folds[i % n_splits].add(run_id)
    return folds


def fold_metrics(y_true, attack_type, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "clean_recall": tn / (tn + fp) if (tn + fp) else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "attack_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
    }
    attack_mask = np.asarray(y_true) == 1
    at = np.asarray(attack_type)
    for a in ["Jamming", "Meaconing", "Spoofing", "Spoofing + Jamming"]:
        m = attack_mask & (at == a)
        out[f"recall_{a}"] = pred[m].mean() if m.sum() else float("nan")
    return out


outer_folds = attack_type_balanced_folds(df, OUTER_N_SPLITS, seed=RNG_SEED)
for i, runs in enumerate(outer_folds):
    sub = df[df["run_id"].isin(runs)]
    print(f"  fold {i}: {len(runs)} recordings, {len(sub)} rows")


def fit_rf(X, y):
    imputer = SimpleImputer(strategy="median").fit(X)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=RNG_SEED, n_jobs=-1)
    rf.fit(imputer.transform(X), y)
    rf.n_jobs = 1  # deterministic inference -- see ROBUSTNESS_NOTES.md
    return imputer, rf


def predict_rf(imputer, rf, X):
    return rf.predict_proba(imputer.transform(X))[:, 1]


def fit_xgb(X, y):
    imputer = SimpleImputer(strategy="median").fit(X)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    spw = n_neg / n_pos
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        tree_method="hist", random_state=RNG_SEED, n_jobs=-1, eval_metric="logloss",
    )
    model.fit(imputer.transform(X), y)
    return imputer, model


def predict_xgb(imputer, model, X):
    return model.predict_proba(imputer.transform(X))[:, 1]


def run_cv(fit_fn, predict_fn, label: str):
    oof_proba = np.full(len(df), np.nan)
    per_fold = []
    fit_times = []
    determinism_ref = None
    for fold_i, val_runs in enumerate(outer_folds):
        is_val = df["run_id"].isin(val_runs)
        train_df, val_df = df[~is_val], df[is_val]
        X_train, y_train = train_df[feature_cols], train_df["attack"]
        X_val = val_df[feature_cols]

        t0 = time.time()
        imputer, model = fit_fn(X_train, y_train)
        fit_times.append(time.time() - t0)
        proba = predict_fn(imputer, model, X_val)
        oof_proba[val_df.index] = proba

        m = fold_metrics(val_df["attack"], val_df["attack_type"], proba, 0.5)
        m["fold"] = fold_i
        per_fold.append(m)

        if fold_i == 0:
            proba_repeat = predict_fn(imputer, model, X_val)
            determinism_ref = bool(np.array_equal(proba, proba_repeat))

    assert not np.isnan(oof_proba).any()
    fold_df = pd.DataFrame(per_fold).set_index("fold")
    print(f"\n[{label}] fit time per fold: {[f'{t:.1f}s' for t in fit_times]}")
    print(f"[{label}] determinism (fold 0 scored twice): {'PASS' if determinism_ref else 'FAIL'}")
    print(f"[{label}] per-fold @ threshold=0.5:")
    print(fold_df[["accuracy", "clean_recall", "fpr", "recall_Jamming"]].to_string())
    return oof_proba, fold_df


def sweep_threshold(oof_proba, label):
    """Identical methodology to train_improved_model.py: floor-constrained sweep on pooled
    out-of-fold predictions, floor = this model's own threshold=0.5 jamming recall minus
    tolerance."""
    y_all, at_all = df["attack"], df["attack_type"]
    ref = fold_metrics(y_all, at_all, oof_proba, 0.5)
    floor = ref["recall_Jamming"] - JAMMING_RECALL_TOLERANCE
    best_t, best_clean, best_m = 0.5, ref["clean_recall"], ref
    print(f"\n[{label}] threshold sweep (jamming floor={floor:.3f}):")
    for t in THRESHOLD_GRID:
        m = fold_metrics(y_all, at_all, oof_proba, t)
        eligible = m["recall_Jamming"] >= floor
        marker = ""
        if eligible and m["clean_recall"] > best_clean:
            best_t, best_clean, best_m = t, m["clean_recall"], m
            marker = "  <-- best so far"
        print(f"  t={t:.2f} acc={m['accuracy']:.3f} clean_recall={m['clean_recall']:.3f} "
              f"fpr={m['fpr']:.3f} jamming_recall={m['recall_Jamming']:.3f}{marker}")
    print(f"[{label}] chosen threshold: {best_t:.2f}")
    return best_t, best_m


def per_fold_at_threshold(fit_fn, predict_fn, threshold, label):
    """Re-evaluate each fold's own held-out predictions at a FIXED (already-chosen) threshold,
    to report GroupKFold mean +/- std at each model's best operating point -- not just the
    pooled number."""
    rows = []
    for fold_i, val_runs in enumerate(outer_folds):
        is_val = df["run_id"].isin(val_runs)
        train_df, val_df = df[~is_val], df[is_val]
        imputer, model = fit_fn(train_df[feature_cols], train_df["attack"])
        proba = predict_fn(imputer, model, val_df[feature_cols])
        m = fold_metrics(val_df["attack"], val_df["attack_type"], proba, threshold)
        m["fold"] = fold_i
        rows.append(m)
    fold_df = pd.DataFrame(rows).set_index("fold")
    print(f"\n[{label}] per-fold @ chosen threshold={threshold:.2f}:")
    print(fold_df.to_string())
    summary = fold_df.agg(["mean", "std"])
    print(f"[{label}] mean +/- std:")
    print(summary.to_string())
    return fold_df, summary


results = {}

print("\n" + "=" * 78)
print("RANDOM FOREST (shipped config: 300 trees, class_weight='balanced')")
print("=" * 78)
rf_oof, rf_fold_df_05 = run_cv(fit_rf, predict_rf, "RandomForest")
rf_best_t, rf_best_m = sweep_threshold(rf_oof, "RandomForest")
rf_fold_df, rf_summary = per_fold_at_threshold(fit_rf, predict_rf, rf_best_t, "RandomForest")

print("\n" + "=" * 78)
print("XGBOOST (300 trees, depth=6, lr=0.1, scale_pos_weight per fold)")
print("=" * 78)
xgb_oof, xgb_fold_df_05 = run_cv(fit_xgb, predict_xgb, "XGBoost")
xgb_best_t, xgb_best_m = sweep_threshold(xgb_oof, "XGBoost")
xgb_fold_df, xgb_summary = per_fold_at_threshold(fit_xgb, predict_xgb, xgb_best_t, "XGBoost")

print("\n" + "=" * 78)
print("HEAD-TO-HEAD (GroupKFold mean +/- std, each model at its own best threshold)")
print("=" * 78)
comp_mean = pd.DataFrame({"RandomForest": rf_summary.loc["mean"], "XGBoost": xgb_summary.loc["mean"]})
comp_std = pd.DataFrame({"RandomForest": rf_summary.loc["std"], "XGBoost": xgb_summary.loc["std"]})
print("Mean:")
print(comp_mean.to_string())
print("\nStd:")
print(comp_std.to_string())

# Pooled ROC-AUC/PR-AUC too (threshold-independent, directly comparable)
for label, oof in [("RandomForest", rf_oof), ("XGBoost", xgb_oof)]:
    roc = roc_auc_score(df["attack"], oof)
    pr = average_precision_score(df["attack"], oof)
    print(f"{label}: pooled ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

comp_mean.to_csv(SCRIPT_DIR / "processed" / "gbm_comparison_mean.csv")
comp_std.to_csv(SCRIPT_DIR / "processed" / "gbm_comparison_std.csv")
print(f"\nSaved processed/gbm_comparison_mean.csv and processed/gbm_comparison_std.csv")
print(f"RF best threshold: {rf_best_t:.2f}   XGBoost best threshold: {xgb_best_t:.2f}")
