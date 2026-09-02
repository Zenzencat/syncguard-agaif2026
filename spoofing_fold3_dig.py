"""Diagnostic only (read-only): the spoofing-feature experiment improves jamming recall in
folds 0-2 but regresses it in fold 3. Fold 3's only jamming recording is 1.11.7 -- the
obstructed-sky dynamic drive already flagged in FOLD_ANALYSIS.md / NORMALIZATION_NOTES.md as
having an INVERTED satellite-count/attack relationship. This confirms whether the new features
concentrate their fold-3 jamming cost on 1.11.7 specifically, and by how much.

Does not modify anything. Mirrors spoofing_features_experiment.py's harness exactly.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

sys.stdout.reconfigure(encoding="utf-8")
SCRIPT_DIR = Path(__file__).resolve().parent
PROC = SCRIPT_DIR / "processed"
RNG_SEED, N_SPLITS, ROLL_WIN, ROLL_MIN = 42, 4, 30, 15

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
new_cols = sp_cols + ["nsat_l1_roll_std_30"]


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
for i, runs in enumerate(F):
    j = df[df["run_id"].isin(runs) & (df["attack_type"] == "Jamming")]["run_id"].unique()
    print(f"fold {i} jamming recordings: {sorted(j)}")

val_runs = F[3]
is_val = df["run_id"].isin(val_runs)
tr, va = df[~is_val], df[is_val]

for label, cols in [("BASE", base_cols), ("BASE+NEW", base_cols + new_cols)]:
    imp = SimpleImputer(strategy="median").fit(tr[cols])
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RNG_SEED, n_jobs=-1)
    rf.fit(imp.transform(tr[cols]), tr["attack"])
    rf.n_jobs = 1
    va = va.copy()
    va["proba"] = rf.predict_proba(imp.transform(va[cols]))[:, 1]
    print(f"\n=== {label} : fold-3 held-out recordings, recall by class ===")
    for t in (0.50, 0.52):
        va["pred"] = (va["proba"] >= t).astype(int)
        rows = []
        for rid, g in va.groupby("run_id"):
            atk = g[g["attack"] == 1]; cln = g[g["attack"] == 0]
            rows.append({"run_id": rid.split("\\")[-1], "type": g["attack_type"].iloc[0],
                         "atk_rows": len(atk), "atk_recall": (atk["pred"] == 1).mean() if len(atk) else np.nan,
                         "cln_rows": len(cln), "cln_recall": (cln["pred"] == 0).mean() if len(cln) else np.nan})
        print(f"  threshold {t}:")
        print(pd.DataFrame(rows).to_string(index=False))
