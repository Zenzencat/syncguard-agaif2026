"""Confirm the metrics from spoofing_features_experiment.py hold when the 5 new features that
did NOT rank (both Gini and SHAP put them 22nd-30th of 34: the carrier-phase / code-minus-
carrier / n_const group) are dropped, keeping only the 6 that did rank. If LEAN matches
BASE+NEW, the leaner set is what to integrate. Same harness; no SHAP (not needed here).
Read-only; modifies nothing.
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
PROC = SCRIPT_DIR / "processed"
RNG_SEED, N_SPLITS, SHIPPED_T, ROLL_WIN, ROLL_MIN = 42, 4, 0.52, 30, 15
ATTACK_TYPES = ["Jamming", "Meaconing", "Spoofing", "Spoofing + Jamming"]

base = pd.read_parquet(PROC / "syncguard_features.parquet")
base["real_time"] = base["real_time"].astype("datetime64[us]")
supp = pd.read_parquet(PROC / "spoofing_features_supplement.parquet")
supp["real_time"] = supp["real_time"].astype("datetime64[us]")
sp_cols = [c for c in supp.columns if c.startswith("sp_")]
b = base.sort_values(["run_id", "real_time"], kind="stable").reset_index(drop=True)
s = supp.sort_values(["run_id", "real_time"], kind="stable").reset_index(drop=True)
df = pd.concat([b, s[sp_cols].reset_index(drop=True)], axis=1)
df["nsat_l1_roll_std_30"] = df.groupby("run_id")["n_sats_l1"].transform(
    lambda x: x.shift(1).rolling(ROLL_WIN, min_periods=ROLL_MIN).std())

EXCLUDE = {"iTOW", "real_time", "lat", "lon", "height", "clock_drift_proxy_s", "attack",
           "scenario_id", "attack_type", "rover_state", "power_w_max", "bands",
           "n_attack_windows", "window_start_in_range", "run_id"}
base_cols = [c for c in base.columns if c not in EXCLUDE]

DROP = ["sp_cp_doppler_resid_std", "sp_cp_doppler_resid_maxabs",
        "sp_cmc_l1_step_std", "sp_cmc_l1_step_maxabs", "sp_n_const"]
KEEP = ["sp_frac_l2_tracked", "sp_cn0_l1_minus_l2_mean", "sp_frac_low_cn0",
        "sp_doppler_l1_mad", "sp_xconst_cn0_spread", "nsat_l1_roll_std_30"]

SETS = {"BASE": base_cols,
        "BASE+NEW(11)": base_cols + sp_cols + ["nsat_l1_roll_std_30"],
        "BASE+LEAN(6)": base_cols + KEEP}


def folds():
    rng = np.random.RandomState(RNG_SEED)
    out = [set() for _ in range(N_SPLITS)]
    for _, sub in df.groupby("attack_type"):
        runs = sub["run_id"].unique().tolist()
        rng.shuffle(runs)
        for i, r in enumerate(runs):
            out[i % N_SPLITS].add(r)
    return out


F = folds()


def fold_metrics(y, at, proba, t):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    o = {"accuracy": (tp + tn) / (tp + tn + fp + fn),
         "clean_recall": tn / (tn + fp), "fpr": fp / (fp + tn),
         "attack_recall": tp / (tp + fn)}
    am = np.asarray(y) == 1
    a = np.asarray(at)
    for k in ATTACK_TYPES:
        m = am & (a == k)
        o[f"recall_{k}"] = pred[m].mean() if m.sum() else np.nan
    return o


for label, cols in SETS.items():
    per_fold, oof = [], np.full(len(df), np.nan)
    for val_runs in F:
        isv = df["run_id"].isin(val_runs)
        tr, va = df[~isv], df[isv]
        imp = SimpleImputer(strategy="median").fit(tr[cols])
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=RNG_SEED, n_jobs=-1)
        rf.fit(imp.transform(tr[cols]), tr["attack"])
        rf.n_jobs = 1
        p = rf.predict_proba(imp.transform(va[cols]))[:, 1]
        oof[va.index] = p
        per_fold.append(fold_metrics(va["attack"], va["attack_type"], p, SHIPPED_T))
    fdf = pd.DataFrame(per_fold)
    roc = roc_auc_score(df["attack"], oof)
    pr = average_precision_score(df["attack"], oof)
    print(f"\n=== {label}  (@ t={SHIPPED_T}, GroupKFold mean +/- std) ===")
    for m in ["accuracy", "clean_recall", "fpr", "recall_Jamming", "recall_Meaconing",
              "recall_Spoofing", "recall_Spoofing + Jamming"]:
        print(f"  {m:<26} {fdf[m].mean():.3f} +/- {fdf[m].std():.3f}")
    print(f"  {'pooled ROC-AUC':<26} {roc:.4f}    pooled PR-AUC {pr:.4f}")
    print(f"  jamming per-fold: {fdf['recall_Jamming'].round(3).tolist()}")
