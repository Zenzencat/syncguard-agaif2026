"""Investigates whether GroupKFold cross-validation gives a more trustworthy per-attack-type
recall estimate than the single 8-recording TEST split, and whether probability calibration
(CalibratedClassifierCV, sigmoid/Platt and isotonic) on top of the RandomForest pipeline is a
genuine improvement -- building on the lesson from ROBUSTNESS_NOTES.md's earlier reweighting
attempt (looked good on one validation split, collapsed on the real held-out set).

Does NOT modify models/model.joblib, models/model_baseline.joblib, api/, or any deployed
threshold. Writes CALIBRATION_NOTES.md. Standalone analysis script.

--- Outer CV: the new "ground truth" evaluation harness ---
4-fold group-preserving K-fold (group=run_id, so no recording ever appears in both train and
validation within a fold -- same leakage discipline as extract_features.py's original
split-by-recording), across ALL 24 recordings this time, not just the 16 the shipped model
trained on. Plain sklearn GroupKFold doesn't consider labels at all; with as few as 4
recordings for the rarest attack type (Spoofing+Jamming), an unlucky assignment could leave a
fold with zero recordings of that type, making its per-fold recall meaningless for that class.
attack_type_balanced_folds() below fixes this by round-robining each attack type's recordings
across folds independently -- every fold gets a share of every type. n_splits=4 is the largest
split count where every fold is guaranteed at least one Spoofing+Jamming recording (4 total).

For each outer fold: refit the RandomForest with the *same* config as the shipped model
(n_estimators=300, class_weight='balanced', random_state=42, median imputation) on that fold's
training recordings, predict on its held-out recordings. Concatenating every fold's held-out
predictions gives one out-of-fold (OOF) probability per row across all 44639 rows, each scored
by a model that never saw that row's recording -- a genuinely out-of-sample estimate for the
full dataset, with per-fold breakdown giving variance, not just a point estimate.

--- Calibration: nested, no leakage ---
For each outer fold, calibration (CalibratedClassifierCV) is fit via a further 3-fold
group-preserving inner split *strictly within that fold's training recordings* -- the base RF
clones are each trained on 2 inner folds and calibrated on the 3rd, so calibration is always
evaluated on rows disjoint from what that specific base-model clone trained on, and the outer
fold's validation recordings are never touched by any of this. This is the standard
CalibratedClassifierCV(cv=<grouped splits>) pattern, not a hand-rolled train/calibrate split.

--- Determinism ---
Every RandomForestClassifier (outer, and every inner clone CalibratedClassifierCV fits) has
n_jobs forced to 1 before any predict_proba call -- see ROBUSTNESS_NOTES.md's writeup of why
n_jobs>1 predict_proba isn't guaranteed bit-reproducible across separate calls. Verified at the
end of this script by re-scoring the same fold twice and diffing.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (recall_score, confusion_matrix, roc_auc_score,
                              average_precision_score, brier_score_loss, log_loss)

sys.stdout.reconfigure(encoding="utf-8")
SCRIPT_DIR = Path(__file__).resolve().parent
RNG_SEED = 42
OUTER_N_SPLITS = 4
INNER_N_SPLITS = 3
SHIPPED_THRESHOLD = 0.52  # models/model.joblib's currently-deployed decision_threshold
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.02)
JAMMING_RECALL_TOLERANCE = 0.01  # same tolerance/rationale as train_improved_model.py

df = pd.read_parquet(SCRIPT_DIR / "processed" / "syncguard_features.parquet").reset_index(drop=True)
EXCLUDE_COLS = {
    "iTOW", "real_time", "lat", "lon", "height", "clock_drift_proxy_s",
    "attack", "scenario_id", "attack_type", "rover_state", "power_w_max", "bands",
    "n_attack_windows", "window_start_in_range", "run_id",
}
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"Full dataset: {len(df)} rows / {df['run_id'].nunique()} recordings")
print("Recordings per attack type:")
print(df.groupby("attack_type")["run_id"].nunique().to_string())
print()


def attack_type_balanced_folds(frame: pd.DataFrame, n_splits: int, seed: int) -> list[set]:
    rng = np.random.RandomState(seed)
    folds = [set() for _ in range(n_splits)]
    for atype, sub in frame.groupby("attack_type"):
        runs = sub["run_id"].unique().tolist()
        rng.shuffle(runs)
        for i, run_id in enumerate(runs):
            folds[i % n_splits].add(run_id)
    return folds


def fold_metrics(y_true: pd.Series, attack_type: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "clean_recall": tn / (tn + fp) if (tn + fp) else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "attack_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "fnr": fn / (fn + tp) if (fn + tp) else float("nan"),
    }
    attack_mask = y_true.values == 1
    for atype in ["Jamming", "Meaconing", "Spoofing", "Spoofing + Jamming"]:
        m = attack_mask & (attack_type.values == atype)
        out[f"recall_{atype}"] = pred[m].mean() if m.sum() else float("nan")
    return out


print("=" * 78)
print(f"OUTER CV: {OUTER_N_SPLITS}-fold, group-preserving (group=run_id), attack-type-balanced")
print("=" * 78)
outer_folds = attack_type_balanced_folds(df, OUTER_N_SPLITS, seed=RNG_SEED)
for i, runs in enumerate(outer_folds):
    sub = df[df["run_id"].isin(runs)]
    counts = sub.groupby("attack_type")["run_id"].nunique().to_dict()
    print(f"  fold {i}: {len(runs)} recordings, {len(sub)} rows -- {counts}")
print()

oof_uncal = np.full(len(df), np.nan)
oof_sigmoid = np.full(len(df), np.nan)
oof_isotonic = np.full(len(df), np.nan)
per_fold_uncal_metrics = []  # at SHIPPED_THRESHOLD, for the Step-1 sanity check
determinism_check = {}

for fold_i, val_runs in enumerate(outer_folds):
    print(f"--- outer fold {fold_i}/{OUTER_N_SPLITS - 1} ---")
    is_val = df["run_id"].isin(val_runs)
    train_df, val_df = df[~is_val].reset_index(drop=True), df[is_val]
    X_train, y_train = train_df[feature_cols], train_df["attack"]
    X_val, y_val = val_df[feature_cols], val_df["attack"]

    # --- uncalibrated base model, same config as the shipped model ---
    imputer = SimpleImputer(strategy="median").fit(X_train)
    Xi_train = imputer.transform(X_train)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=RNG_SEED, n_jobs=-1)
    rf.fit(Xi_train, y_train)
    rf.n_jobs = 1  # deterministic inference from here on
    Xi_val = imputer.transform(X_val)
    proba_uncal = rf.predict_proba(Xi_val)[:, 1]
    oof_uncal[val_df.index] = proba_uncal

    m = fold_metrics(y_val, val_df["attack_type"], proba_uncal, SHIPPED_THRESHOLD)
    per_fold_uncal_metrics.append(m)
    print(f"  uncalibrated @ {SHIPPED_THRESHOLD}: acc={m['accuracy']:.3f} clean_recall={m['clean_recall']:.3f} "
          f"fpr={m['fpr']:.3f} jamming_recall={m['recall_Jamming']:.3f}")

    if fold_i == 0:
        proba_uncal_repeat = rf.predict_proba(Xi_val)[:, 1]
        determinism_check["uncalibrated"] = bool(np.array_equal(proba_uncal, proba_uncal_repeat))

    # --- calibration: nested inner GroupKFold strictly within this fold's training recordings ---
    inner_folds = attack_type_balanced_folds(train_df, INNER_N_SPLITS, seed=RNG_SEED + fold_i)
    cv_splits = []
    for inner_val_runs in inner_folds:
        val_mask = train_df["run_id"].isin(inner_val_runs).values
        cv_splits.append((np.where(~val_mask)[0], np.where(val_mask)[0]))
    inner_counts = [train_df.iloc[te]["attack_type"].value_counts().to_dict() for _, te in cv_splits]
    print(f"  inner folds (for calibration, within this fold's train only): "
          f"{[len(te) for _, te in cv_splits]} rows -- attack-type coverage per inner fold: {inner_counts}")

    for method, target in [("sigmoid", "oof_sigmoid"), ("isotonic", "oof_isotonic")]:
        base_rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                          random_state=RNG_SEED, n_jobs=-1)
        calib = CalibratedClassifierCV(estimator=base_rf, method=method, cv=cv_splits, n_jobs=1)
        calib.fit(Xi_train, y_train)
        for cc in calib.calibrated_classifiers_:
            cc.estimator.n_jobs = 1  # deterministic inference for every inner clone
        proba_cal = calib.predict_proba(Xi_val)[:, 1]
        (oof_sigmoid if target == "oof_sigmoid" else oof_isotonic)[val_df.index] = proba_cal
        print(f"  {method} calibrated: brier={brier_score_loss(y_val, proba_cal):.4f} "
              f"(uncalibrated brier this fold: {brier_score_loss(y_val, proba_uncal):.4f})")

        if fold_i == 0:
            proba_cal_repeat = calib.predict_proba(Xi_val)[:, 1]
            determinism_check[method] = bool(np.array_equal(proba_cal, proba_cal_repeat))
    print()

assert not np.isnan(oof_uncal).any(), "every row should have exactly one OOF prediction"
assert not np.isnan(oof_sigmoid).any()
assert not np.isnan(oof_isotonic).any()

print("=" * 78)
print("DETERMINISM CHECK (same fold scored twice, arrays must be identical)")
print("=" * 78)
for k, v in determinism_check.items():
    print(f"  {k}: {'PASS -- bit-identical' if v else 'FAIL -- non-deterministic!'}")
print()

# ---------------------------------------------------------------------------
# STEP 1: uncalibrated CV baseline @ shipped threshold -- sanity check vs. evaluate_models.py
# ---------------------------------------------------------------------------
print("=" * 78)
print(f"STEP 1: uncalibrated model, CV out-of-fold, threshold={SHIPPED_THRESHOLD} "
      "(shipped threshold)")
print("=" * 78)
fold_df = pd.DataFrame(per_fold_uncal_metrics)
print("Per-fold:")
print(fold_df.to_string())
print("\nMean +/- std across folds:")
summary = fold_df.agg(["mean", "std"]).T
print(summary.to_string())

pooled = fold_metrics(df["attack"], df["attack_type"], oof_uncal, SHIPPED_THRESHOLD)
print(f"\nPooled (all {len(df)} rows, each out-of-fold):")
for k, v in pooled.items():
    print(f"  {k}: {v:.4f}")

evaluate_models_reference = dict(
    accuracy=0.8768, clean_recall=0.6672, fpr=0.3328, attack_recall=0.9369,
    recall_Jamming=0.8585, recall_Meaconing=0.9646, recall_Spoofing=0.9712,
    **{"recall_Spoofing + Jamming": 0.9846},
)
print("\nvs. evaluate_models.py's single 8-recording TEST split (for reference):")
for k, v in evaluate_models_reference.items():
    print(f"  {k}: CV={pooled.get(k, float('nan')):.4f}  single-split={v:.4f}  "
          f"delta={pooled.get(k, float('nan')) - v:+.4f}")

# ---------------------------------------------------------------------------
# STEP 2a: reliability -- Brier score, log-loss, calibration curves
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 2a: reliability (pooled out-of-fold predictions, all 44639 rows)")
print("=" * 78)
y_all = df["attack"]
reliability = {}
for label, proba in [("uncalibrated", oof_uncal), ("sigmoid", oof_sigmoid), ("isotonic", oof_isotonic)]:
    brier = brier_score_loss(y_all, proba)
    ll = log_loss(y_all, np.clip(proba, 1e-7, 1 - 1e-7))
    reliability[label] = {"brier": brier, "log_loss": ll}
    print(f"  {label:>13}: Brier={brier:.4f}  log-loss={ll:.4f}")

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
for label, proba, color in [("uncalibrated", oof_uncal, "tab:blue"),
                              ("sigmoid", oof_sigmoid, "tab:orange"),
                              ("isotonic", oof_isotonic, "tab:green")]:
    frac_pos, mean_pred = calibration_curve(y_all, proba, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label=label, color=color)
ax.set_xlabel("Mean predicted probability (per bin)")
ax.set_ylabel("Observed fraction of true attacks (per bin)")
ax.set_title("Reliability diagram -- out-of-fold predictions, all 44639 rows, 10 quantile bins")
ax.legend()
fig.tight_layout()
reliability_png = SCRIPT_DIR / "processed" / "calibration_reliability.png"
fig.savefig(reliability_png, dpi=130)
print(f"\nSaved {reliability_png}")

# ---------------------------------------------------------------------------
# STEP 2b: threshold sweep on each variant's OOF probabilities (own jamming-recall floor,
# same methodology as train_improved_model.py) -- "does a better threshold become available"
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("STEP 2b: threshold sweep per variant (jamming-recall-floor constrained)")
print("=" * 78)


def sweep_threshold(proba: np.ndarray) -> tuple[float, dict]:
    ref = fold_metrics(y_all, df["attack_type"], proba, 0.5)
    floor = ref["recall_Jamming"] - JAMMING_RECALL_TOLERANCE
    best_t, best_clean, best_m = 0.5, ref["clean_recall"], ref
    for t in THRESHOLD_GRID:
        m = fold_metrics(y_all, df["attack_type"], proba, t)
        if m["recall_Jamming"] >= floor and m["clean_recall"] > best_clean:
            best_t, best_clean, best_m = t, m["clean_recall"], m
    return best_t, best_m


results_table = {}
results_table["uncalibrated @ shipped 0.52"] = (SHIPPED_THRESHOLD, pooled)
for label, proba in [("uncalibrated", oof_uncal), ("sigmoid", oof_sigmoid), ("isotonic", oof_isotonic)]:
    t05 = fold_metrics(y_all, df["attack_type"], proba, 0.5)
    results_table[f"{label} @ 0.50 (untuned)"] = (0.5, t05)
    best_t, best_m = sweep_threshold(proba)
    results_table[f"{label} @ {best_t:.2f} (best, floor-constrained)"] = (best_t, best_m)
    print(f"\n{label}: best threshold = {best_t:.2f}")
    print(f"  clean_recall={best_m['clean_recall']:.4f}  fpr={best_m['fpr']:.4f}  "
          f"jamming_recall={best_m['recall_Jamming']:.4f}  accuracy={best_m['accuracy']:.4f}")

print("\n" + "=" * 78)
print("FULL COMPARISON TABLE")
print("=" * 78)
comp_df = pd.DataFrame({k: v[1] for k, v in results_table.items()}).T
comp_df.insert(0, "threshold", [v[0] for v in results_table.values()])
print(comp_df.to_string())
comp_df.to_csv(SCRIPT_DIR / "processed" / "calibration_comparison_table.csv")
print(f"\nSaved {SCRIPT_DIR / 'processed' / 'calibration_comparison_table.csv'}")
