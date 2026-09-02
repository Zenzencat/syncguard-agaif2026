"""
Does adding spoofing-specific features (carrier-phase, L2, per-constellation structure --
see extract_spoofing_features.py and SPOOFING_FEATURES.md) improve the shipped detector
under the SAME GroupKFold harness every other experiment in this repo uses?

Same discipline as train_gbm_comparison.py / session_normalization_experiment.py /
calibration_experiment.py:
  - identical attack_type_balanced_folds(df, n_splits=4, seed=42) -- same fold membership
    as every prior experiment, group = recording (run_id), attack-type round-robined so
    every fold has every type.
  - shipped RandomForest config: 300 trees, class_weight='balanced', random_state=42,
    n_jobs=1 at predict time (the ROBUSTNESS_NOTES.md determinism fix).
  - BASE = the 23 shipped features. NEW = 10 supplementary sp_* columns + one causal
    trailing-window feature (nsat_l1_roll_std_30) built here from the existing n_sats_l1.
  - reported three ways: (a) per-fold at the shipped 0.52 threshold, mean +/- std;
    (b) each feature set at its own floor-constrained best threshold (jamming-recall floor
    = own t=0.5 value - 0.01, identical to train_improved_model.py), mean +/- std;
    (c) pooled out-of-fold ROC-AUC / PR-AUC.
  - SHAP: exact TreeExplainer over a full-data BASE+NEW fit -- do the NEW features actually
    rank, or does the model ignore them for the same u-blox RF fields? First-class result
    regardless of whether headline metrics move. Plus per-fold RF feature_importances_
    mean +/- std for the new features.
  - determinism: every fold scored twice, asserted bit-identical.

Does NOT modify models/*.joblib, extract_features.py, processed/syncguard_features.parquet,
api/, or any deployed threshold. Writes SPOOFING_FEATURES.md inputs into processed/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding="utf-8")
SCRIPT_DIR = Path(__file__).resolve().parent
PROC = SCRIPT_DIR / "processed"
RNG_SEED = 42
N_SPLITS = 4
SHIPPED_THRESHOLD = 0.52
THRESHOLD_GRID = np.arange(0.30, 0.71, 0.02)
JAMMING_RECALL_TOLERANCE = 0.01
ROLL_WIN, ROLL_MIN = 30, 15
ATTACK_TYPES = ["Jamming", "Meaconing", "Spoofing", "Spoofing + Jamming"]

# ---------------------------------------------------------------------------
# Load + join (left join: supplement is keyed on the identical (run_id, real_time) spine)
# ---------------------------------------------------------------------------
base = pd.read_parquet(PROC / "syncguard_features.parquet")
base["real_time"] = base["real_time"].astype("datetime64[us]")
supp = pd.read_parquet(PROC / "spoofing_features_supplement.parquet")
supp["real_time"] = supp["real_time"].astype("datetime64[us]")
sp_cols = [c for c in supp.columns if c.startswith("sp_")]

# The base parquet has repeated (run_id, real_time) rows in the two-scenario merged folders
# (e.g. "2.3.10,2.3.11"), so a keyed merge would cross-join the duplicates. The supplement is
# built from the IDENTICAL nav_pvt spine (same dropna(real_time) + stable sort by real_time,
# same sorted() scenario order) as extract_features.py, so the two align positionally within
# each recording. Verify that explicitly, then attach by position -- no key merge.
b = base.sort_values(["run_id", "real_time"], kind="stable").reset_index(drop=True)
s = supp.sort_values(["run_id", "real_time"], kind="stable").reset_index(drop=True)
assert len(b) == len(s), f"row count mismatch: base {len(b)} vs supp {len(s)}"
assert (b["run_id"].values == s["run_id"].values).all(), "run_id sequence mismatch"
assert (b["real_time"].values == s["real_time"].values).all(), "real_time sequence mismatch"
df = pd.concat([b, s[sp_cols].reset_index(drop=True)], axis=1)
print(f"aligned: {len(df)} rows (base {len(base)}, supp {len(supp)}) -- positional attach verified")
print("supplement coverage after attach:")
print(df[sp_cols].notna().mean().to_string())

# causal trailing-window satellite-count stability (built here, not in the extractor --
# strictly trailing, shift(1) before rolling, per recording; cold start -> NaN -> imputed)
df["nsat_l1_roll_std_30"] = (
    df.groupby("run_id")["n_sats_l1"]
      .transform(lambda s: s.shift(1).rolling(ROLL_WIN, min_periods=ROLL_MIN).std())
)
print(f"nsat_l1_roll_std_30 coverage: {df['nsat_l1_roll_std_30'].notna().mean():.1%}")

EXCLUDE_COLS = {
    "iTOW", "real_time", "lat", "lon", "height", "clock_drift_proxy_s",
    "attack", "scenario_id", "attack_type", "rover_state", "power_w_max", "bands",
    "n_attack_windows", "window_start_in_range", "run_id",
}
base_feature_cols = [c for c in base.columns if c not in EXCLUDE_COLS]
new_feature_cols = sp_cols + ["nsat_l1_roll_std_30"]
all_feature_cols = base_feature_cols + new_feature_cols
print(f"\nBASE features ({len(base_feature_cols)}): {base_feature_cols}")
print(f"NEW features ({len(new_feature_cols)}): {new_feature_cols}")


# ---------------------------------------------------------------------------
# Harness -- verbatim from train_gbm_comparison.py
# ---------------------------------------------------------------------------
def attack_type_balanced_folds(frame, n_splits, seed):
    rng = np.random.RandomState(seed)
    folds = [set() for _ in range(n_splits)]
    for atype, sub in frame.groupby("attack_type"):
        runs = sub["run_id"].unique().tolist()
        rng.shuffle(runs)
        for i, run_id in enumerate(runs):
            folds[i % n_splits].add(run_id)
    return folds


def fold_metrics(y_true, attack_type, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "clean_recall": tn / (tn + fp) if (tn + fp) else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "attack_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
    }
    attack_mask = np.asarray(y_true) == 1
    at = np.asarray(attack_type)
    for a in ATTACK_TYPES:
        m = attack_mask & (at == a)
        out[f"recall_{a}"] = pred[m].mean() if m.sum() else float("nan")
    return out


outer_folds = attack_type_balanced_folds(df, N_SPLITS, RNG_SEED)
for i, runs in enumerate(outer_folds):
    sub = df[df["run_id"].isin(runs)]
    print(f"  fold {i}: {len(runs)} recordings, {len(sub)} rows, "
          f"types={sorted(sub['attack_type'].unique())}")


def fit_rf(X, y):
    imp = SimpleImputer(strategy="median").fit(X)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RNG_SEED, n_jobs=-1)
    rf.fit(imp.transform(X), y)
    rf.n_jobs = 1  # deterministic inference -- ROBUSTNESS_NOTES.md
    return imp, rf


def run_cv(feature_cols, label):
    oof = np.full(len(df), np.nan)
    fold_imp = []
    for fold_i, val_runs in enumerate(outer_folds):
        is_val = df["run_id"].isin(val_runs)
        tr, va = df[~is_val], df[is_val]
        imp, rf = fit_rf(tr[feature_cols], tr["attack"])
        Xva = imp.transform(va[feature_cols])
        proba = rf.predict_proba(Xva)[:, 1]
        assert np.array_equal(proba, rf.predict_proba(Xva)[:, 1]), f"{label} fold {fold_i} non-deterministic"
        oof[va.index] = proba
        fold_imp.append(pd.Series(rf.feature_importances_, index=feature_cols))
    assert not np.isnan(oof).any()
    return oof, pd.DataFrame(fold_imp)


def per_fold_at(feature_cols, threshold):
    rows = []
    for fold_i, val_runs in enumerate(outer_folds):
        is_val = df["run_id"].isin(val_runs)
        tr, va = df[~is_val], df[is_val]
        imp, rf = fit_rf(tr[feature_cols], tr["attack"])
        proba = rf.predict_proba(imp.transform(va[feature_cols]))[:, 1]
        m = fold_metrics(va["attack"], va["attack_type"], proba, threshold)
        m["fold"] = fold_i
        rows.append(m)
    return pd.DataFrame(rows).set_index("fold")


def sweep(oof, label):
    y, at = df["attack"], df["attack_type"]
    ref = fold_metrics(y, at, oof, 0.5)
    floor = ref["recall_Jamming"] - JAMMING_RECALL_TOLERANCE
    best_t, best_clean = 0.5, ref["clean_recall"]
    print(f"\n[{label}] floor-constrained sweep (jamming floor={floor:.3f}, ref t=0.5 "
          f"clean={ref['clean_recall']:.3f} jam={ref['recall_Jamming']:.3f})")
    for t in THRESHOLD_GRID:
        m = fold_metrics(y, at, oof, t)
        ok = m["recall_Jamming"] >= floor
        mk = ""
        if ok and m["clean_recall"] > best_clean:
            best_t, best_clean, mk = t, m["clean_recall"], "  <-- best"
        print(f"  t={t:.2f} acc={m['accuracy']:.3f} clean={m['clean_recall']:.3f} "
              f"fpr={m['fpr']:.3f} jam={m['recall_Jamming']:.3f}{mk}")
    print(f"[{label}] chosen threshold: {best_t:.2f}")
    return best_t


results = {}
print("\n" + "=" * 78 + "\nBASE (23 shipped features)\n" + "=" * 78)
base_oof, base_fi = run_cv(base_feature_cols, "BASE")
print("\n" + "=" * 78 + "\nBASE + NEW spoofing-specific features\n" + "=" * 78)
new_oof, new_fi = run_cv(all_feature_cols, "BASE+NEW")

for label, cols, oof, fi in [("BASE", base_feature_cols, base_oof, base_fi),
                             ("BASE+NEW", all_feature_cols, new_oof, new_fi)]:
    shipped_tbl = per_fold_at(cols, SHIPPED_THRESHOLD)
    best_t = sweep(oof, label)
    best_tbl = per_fold_at(cols, best_t)
    roc = roc_auc_score(df["attack"], oof)
    pr = average_precision_score(df["attack"], oof)
    results[label] = dict(cols=cols, oof=oof, fi=fi, best_t=best_t,
                          shipped_tbl=shipped_tbl, best_tbl=best_tbl, roc=roc, pr=pr)

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
METRICS = ["accuracy", "clean_recall", "fpr", "attack_recall",
           "recall_Jamming", "recall_Meaconing", "recall_Spoofing", "recall_Spoofing + Jamming"]


def summ(tbl):
    return pd.DataFrame({"mean": tbl[METRICS].mean(), "std": tbl[METRICS].std()})


for op, key in [("SHIPPED threshold 0.52", "shipped_tbl"), ("own best threshold", "best_tbl")]:
    print("\n" + "=" * 78)
    print(f"GroupKFold mean +/- std  @ {op}")
    print("=" * 78)
    b, n = summ(results["BASE"][key]), summ(results["BASE+NEW"][key])
    comp = pd.DataFrame({
        "BASE mean": b["mean"], "BASE std": b["std"],
        "BASE+NEW mean": n["mean"], "BASE+NEW std": n["std"],
        "delta mean": n["mean"] - b["mean"],
    })
    if key == "best_tbl":
        print(f"(BASE t={results['BASE']['best_t']:.2f}, BASE+NEW t={results['BASE+NEW']['best_t']:.2f})")
    print(comp.to_string(float_format=lambda x: f"{x:+.4f}"))
    comp.to_csv(PROC / f"spoofing_comparison_{key}.csv")

print("\n" + "=" * 78)
print("Pooled out-of-fold (threshold-independent)")
print("=" * 78)
print(f"  BASE     ROC-AUC={results['BASE']['roc']:.4f}  PR-AUC={results['BASE']['pr']:.4f}")
print(f"  BASE+NEW ROC-AUC={results['BASE+NEW']['roc']:.4f}  PR-AUC={results['BASE+NEW']['pr']:.4f}")

print("\n" + "=" * 78)
print("Per-fold JAMMING recall (the metric this attempt must not regress)")
print("=" * 78)
for op, key in [("t=0.52", "shipped_tbl"), ("own best t", "best_tbl")]:
    bj = results["BASE"][key]["recall_Jamming"]
    nj = results["BASE+NEW"][key]["recall_Jamming"]
    print(f"  @ {op:>10}:  BASE {bj.mean():.3f}+/-{bj.std():.3f} per-fold={bj.round(3).tolist()}")
    print(f"  {'':>13}  NEW  {nj.mean():.3f}+/-{nj.std():.3f} per-fold={nj.round(3).tolist()}")

# ---------------------------------------------------------------------------
# RF Gini importance of the NEW features -- per-fold mean +/- std (fast; same measure
# NORMALIZATION_NOTES.md used). Printed before SHAP so the key ranking result exists even
# if the (slow, exact) SHAP pass below is interrupted.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("RF Gini importance -- BASE+NEW model, per-fold mean +/- std, rank among 34")
print("=" * 78)
fi = results["BASE+NEW"]["fi"]                     # 4 x 34
fi_mean = fi.mean().sort_values(ascending=False)
gini_rank = {f: i + 1 for i, f in enumerate(fi_mean.index)}
print(f"{'feature':<30}{'gini mean':>11}{'gini std':>10}{'rank/34':>9}")
for f in new_feature_cols:
    print(f"{f:<30}{fi[f].mean():>11.5f}{fi[f].std():>10.5f}{gini_rank[f]:>7}/34")
print(f"\nfull ranking (all 34):")
for i, (f, v) in enumerate(fi_mean.items()):
    tag = "  <-- NEW" if f in new_feature_cols else ""
    print(f"  {i+1:>2}. {f:<28} {v:.5f}{tag}")
fi_mean.to_frame("gini_mean").assign(
    is_new=[f in new_feature_cols for f in fi_mean.index]).to_csv(PROC / "spoofing_gini_ranking.csv")

# ---------------------------------------------------------------------------
# SHAP: do the NEW features actually rank? (full-data fit -- descriptive, not a perf metric)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SHAP feature ranking -- full-data BASE+NEW fit")
print("=" * 78)
imp_all = SimpleImputer(strategy="median").fit(df[all_feature_cols])
rf_all = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RNG_SEED, n_jobs=-1)
rf_all.fit(imp_all.transform(df[all_feature_cols]), df["attack"])
rf_all.n_jobs = 1

rng = np.random.RandomState(RNG_SEED)
sample_idx = rng.choice(len(df), size=800, replace=False)
Xs = imp_all.transform(df.iloc[sample_idx][all_feature_cols])
explainer = shap.TreeExplainer(rf_all)
sv = explainer.shap_values(Xs, check_additivity=False)
sv_attack = sv[:, :, 1] if getattr(sv, "ndim", 2) == 3 else sv
mean_abs = pd.Series(np.abs(sv_attack).mean(axis=0), index=all_feature_cols).sort_values(ascending=False)
# determinism
sv2 = explainer.shap_values(Xs, check_additivity=False)
sv2_attack = sv2[:, :, 1] if getattr(sv2, "ndim", 2) == 3 else sv2
print(f"SHAP determinism (scored twice, bit-identical): {np.array_equal(sv_attack, sv2_attack)}")

rank = {f: i + 1 for i, f in enumerate(mean_abs.index)}
gini_all = pd.Series(rf_all.feature_importances_, index=all_feature_cols).sort_values(ascending=False)
gini_rank = {f: i + 1 for i, f in enumerate(gini_all.index)}
fold_fi_new = results["BASE+NEW"]["fi"][new_feature_cols]

print(f"\n{'feature':<30}{'meanabsSHAP':>13}{'SHAP rank':>11}{'gini rank':>11}"
      f"{'foldGini mean+/-std':>22}")
for f in new_feature_cols:
    fm, fs = fold_fi_new[f].mean(), fold_fi_new[f].std()
    print(f"{f:<30}{mean_abs[f]:>13.5f}{rank[f]:>8}/{len(all_feature_cols)}"
          f"{gini_rank[f]:>8}/{len(all_feature_cols)}{fm:>13.5f} +/-{fs:.5f}")

print(f"\nTop 12 features overall by mean|SHAP| (of {len(all_feature_cols)}):")
for i, (f, v) in enumerate(mean_abs.head(12).items()):
    tag = "  <-- NEW" if f in new_feature_cols else ""
    print(f"  {i+1:>2}. {f:<30} {v:.5f}{tag}")

out = pd.DataFrame({"mean_abs_shap": mean_abs, "shap_rank": pd.Series(rank),
                    "gini": gini_all, "gini_rank": pd.Series(gini_rank),
                    "is_new": pd.Series({f: (f in new_feature_cols) for f in all_feature_cols})})
out.sort_values("shap_rank").to_csv(PROC / "spoofing_shap_ranking.csv")
print(f"\nSaved: processed/spoofing_comparison_*.csv, processed/spoofing_shap_ranking.csv")
