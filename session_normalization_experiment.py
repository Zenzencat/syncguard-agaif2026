"""Tests whether session-relative normalization of the two features implicated in
FOLD_ANALYSIS.md (n_sats_l1, pr_doppler_residual_std) fixes or improves the clean-recall
weakness found there. Adds NEW features alongside the existing raw ones (never replaces them)
and validates with the same 4-fold, attack-type-balanced GroupKFold harness as
calibration_experiment.py (same seed, so fold 3 here is the identical set of recordings).

Does NOT modify models/model.joblib, models/model_baseline.joblib, api/, or any deployed
threshold. Writes NORMALIZATION_NOTES.md. Standalone analysis script.

--- Two normalization variants, deliberately kept separate ---

1. OFFLINE (whole-session median deviation): `value - median(value for all rows in this
   run_id)`. Uses only that recording's own feature values (no labels, no other recordings --
   doesn't violate the GroupKFold leakage discipline), but DOES use the full session including
   rows *after* the one being normalized -- not something a live streaming deployment could
   compute for an early row in an ongoing session. This is an "oracle ceiling": if this doesn't
   help, a deployable version won't either. If it does help, that's necessary but not
   sufficient -- the deployable version still has to be checked separately.

2. CAUSAL (trailing rolling-window median deviation): `value - median(value over the prior 60
   rows, i.e. rows[i-60:i], min 30 rows required or NaN)`. Fully computable in a live
   replay/streaming deployment (maintain a 60-reading rolling buffer per active
   tower/session) -- no lookahead. Deliberately a *rolling*, not *expanding-from-session-start*,
   window: an expanding window anchored at session start would stay biased toward whatever the
   session began with (checked directly -- recording 1.11.7's first 60 rows are entirely
   *inside* its attack window, with a high satellite count of 46; the clean segment that
   follows, with a real satellite count of 9, comes *after*, so a baseline anchored at session
   start would itself be contaminated by the attack and would be slow to adapt to the later
   regime change). A rolling window forgets old readings as it slides, so it can track a
   within-session regime change; an expanding one can't.

Cold start (fewer than 30 prior rows, or -- for the causal variant -- the very start of a
session) is left as NaN, same convention `pos_dev_m` already uses for dynamic scenarios in
extract_features.py, and handled by the existing SimpleImputer(strategy='median') the same way.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding="utf-8")
SCRIPT_DIR = Path(__file__).resolve().parent
RNG_SEED = 42
OUTER_N_SPLITS = 4
SHIPPED_THRESHOLD = 0.52
ROLLING_WINDOW = 60
MIN_HISTORY = 30
TARGET_FOLD = 3

df = pd.read_parquet(SCRIPT_DIR / "processed" / "syncguard_features.parquet").reset_index(drop=True)
df["real_time"] = pd.to_datetime(df["real_time"])
df = df.sort_values(["run_id", "real_time"]).reset_index(drop=True)

EXCLUDE_COLS = {
    "iTOW", "real_time", "lat", "lon", "height", "clock_drift_proxy_s",
    "attack", "scenario_id", "attack_type", "rover_state", "power_w_max", "bands",
    "n_attack_windows", "window_start_in_range", "run_id",
}
base_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

# ---------------------------------------------------------------------------
# Build the new features
# ---------------------------------------------------------------------------
NORM_SOURCE_FEATURES = ["n_sats_l1", "pr_doppler_residual_std"]

for feat in NORM_SOURCE_FEATURES:
    df[f"{feat}_dev_offline"] = df.groupby("run_id")[feat].transform(lambda s: s - s.median())

    def causal_dev(s: pd.Series) -> pd.Series:
        baseline = s.shift(1).rolling(window=ROLLING_WINDOW, min_periods=MIN_HISTORY).median()
        return s - baseline

    df[f"{feat}_dev_causal"] = df.groupby("run_id")[feat].transform(causal_dev)

offline_feature_cols = base_feature_cols + [f"{f}_dev_offline" for f in NORM_SOURCE_FEATURES]
causal_feature_cols = base_feature_cols + [f"{f}_dev_causal" for f in NORM_SOURCE_FEATURES]

for f in NORM_SOURCE_FEATURES:
    off = df[f"{f}_dev_offline"]
    cau = df[f"{f}_dev_causal"]
    print(f"{f}: offline-dev coverage={off.notna().mean():.1%}  "
          f"causal-dev coverage={cau.notna().mean():.1%} (NaN = cold start, imputed same as pos_dev_m)")

# ---------------------------------------------------------------------------
# GroupKFold harness -- identical to calibration_experiment.py
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
assert TARGET_FOLD < len(outer_folds)


def run_cv(feature_cols: list[str], label: str):
    oof_proba = np.full(len(df), np.nan)
    per_fold = []
    determinism = None
    for fold_i, val_runs in enumerate(outer_folds):
        is_val = df["run_id"].isin(val_runs)
        train_df, val_df = df[~is_val], df[is_val]
        X_train, y_train = train_df[feature_cols], train_df["attack"]
        X_val = val_df[feature_cols]

        imputer = SimpleImputer(strategy="median").fit(X_train)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=RNG_SEED, n_jobs=-1)
        rf.fit(imputer.transform(X_train), y_train)
        rf.n_jobs = 1
        proba = rf.predict_proba(imputer.transform(X_val))[:, 1]
        oof_proba[val_df.index] = proba

        m = fold_metrics(val_df["attack"], val_df["attack_type"], proba, SHIPPED_THRESHOLD)
        m["fold"] = fold_i
        per_fold.append(m)

        if fold_i == TARGET_FOLD:
            proba_repeat = rf.predict_proba(imputer.transform(X_val))[:, 1]
            determinism = bool(np.array_equal(proba, proba_repeat))
            if len(feature_cols) > len(base_feature_cols):
                importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
                new_feats = [c for c in feature_cols if c not in base_feature_cols]
                print(f"\n[{label}] new-feature importances (fold {fold_i}'s model) and their rank "
                      f"out of {len(feature_cols)}:")
                for nf in new_feats:
                    rank = list(importances.index).index(nf) + 1
                    print(f"  {nf}: importance={importances[nf]:.5f}  rank={rank}/{len(feature_cols)}")

    assert not np.isnan(oof_proba).any()
    pooled = fold_metrics(df["attack"], df["attack_type"], oof_proba, SHIPPED_THRESHOLD)
    roc = roc_auc_score(df["attack"], oof_proba)
    pr = average_precision_score(df["attack"], oof_proba)
    pooled["roc_auc"], pooled["pr_auc"] = roc, pr
    fold_df = pd.DataFrame(per_fold).set_index("fold")
    print(f"\n=== {label}: per-fold clean_recall / fpr / jamming_recall ===")
    print(fold_df[["accuracy", "clean_recall", "fpr", "recall_Jamming"]].to_string())
    print(f"[{label}] fold {TARGET_FOLD} determinism check: "
          f"{'PASS' if determinism else 'FAIL -- non-deterministic!'}")
    return pooled, fold_df, oof_proba


results = {}
fold_tables = {}
for label, cols in [("baseline", base_feature_cols),
                     ("offline-normalized (oracle ceiling)", offline_feature_cols),
                     ("causal-normalized (deployable)", causal_feature_cols)]:
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    pooled, fold_df, oof = run_cv(cols, label)
    results[label] = pooled
    fold_tables[label] = fold_df

print("\n" + "=" * 78)
print("POOLED COMPARISON (all 44639 rows, out-of-fold, threshold=0.52)")
print("=" * 78)
comp = pd.DataFrame(results).T
print(comp.to_string())
comp.to_csv(SCRIPT_DIR / "processed" / "normalization_comparison_table.csv")

print("\n" + "=" * 78)
print(f"FOLD {TARGET_FOLD} SPECIFICALLY (the direct test)")
print("=" * 78)
fold3_comp = pd.DataFrame({label: t.loc[TARGET_FOLD] for label, t in fold_tables.items()}).T
print(fold3_comp.to_string())
fold3_comp.to_csv(SCRIPT_DIR / "processed" / "normalization_fold3_comparison.csv")
print(f"\nSaved comparison tables to processed/normalization_comparison_table.csv and "
      f"processed/normalization_fold3_comparison.csv")
