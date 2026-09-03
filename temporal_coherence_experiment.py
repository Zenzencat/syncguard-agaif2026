"""
Dual evaluation of the 5 temporal-coherence features (CMC slopes + Doppler smoothness --
see extract_temporal_features.py / TEMPORAL_COHERENCE.md) against the shipped 23-feature
model. Reacquisition rate and instantaneous Jaccard are excluded by construction (Step 0).

Runs BOTH evaluations that disagreed for SPOOFING_FEATURES.md, deliberately:
  A. rotating 4-fold GroupKFold (attack_type_balanced_folds, seed 42, group = recording)
     -- mean +/- std per fold
  B. the fixed 8-recording shipped TEST split (train_baseline_model.py's TEST_SELECTION)
     -- single split, plus per-recording clean AND attack recall for 1.10.6 and 3.2.8
        (the dynamic degraded-reception recordings that broke the last attempt)

Plus: pooled OOF ROC-AUC / PR-AUC, SHAP ranking on the fixed-split BASE+NEW model,
determinism check (every model scored twice, bit-identical).

Does NOT modify models/*.joblib, extract_features.py, processed/syncguard_features.parquet,
api/, or any threshold.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

sys.stdout.reconfigure(encoding="utf-8")
SCRIPT_DIR = Path(__file__).resolve().parent
PROC = SCRIPT_DIR / "processed"
RNG_SEED, N_SPLITS = 42, 4
SHIPPED_THRESHOLD = 0.52
ATTACK_TYPES = ["Jamming", "Meaconing", "Spoofing", "Spoofing + Jamming"]

base = pd.read_parquet(PROC / "syncguard_features.parquet")
base["real_time"] = base["real_time"].astype("datetime64[us]")
supp = pd.read_parquet(PROC / "temporal_features_supplement.parquet")
supp["real_time"] = supp["real_time"].astype("datetime64[us]")
sp_cols = [c for c in supp.columns if c.startswith("sp_")]

b = base.sort_values(["run_id", "real_time"], kind="stable").reset_index(drop=True)
s = supp.sort_values(["run_id", "real_time"], kind="stable").reset_index(drop=True)
assert len(b) == len(s) and (b["run_id"].values == s["run_id"].values).all() \
    and (b["real_time"].values == s["real_time"].values).all(), "supplement misaligned with base"
df = pd.concat([b, s[sp_cols].reset_index(drop=True)], axis=1)
print(f"aligned: {len(df)} rows; new features: {sp_cols}")
print("coverage:\n" + df[sp_cols].notna().mean().to_string())

EXCLUDE = {"iTOW", "real_time", "lat", "lon", "height", "clock_drift_proxy_s", "attack",
           "scenario_id", "attack_type", "rover_state", "power_w_max", "bands",
           "n_attack_windows", "window_start_in_range", "run_id"}
base_cols = [c for c in base.columns if c not in EXCLUDE]
all_cols = base_cols + sp_cols
print(f"BASE {len(base_cols)} feats, BASE+NEW {len(all_cols)} feats")

TEST_SELECTION = [
    ("Jamming", "1.10.6", "dynamic"), ("Jamming", "1.6.4", "stationary"),
    ("Meaconing", "3.2.8", "dynamic"), ("Meaconing", "3.2.7", "stationary"),
    ("Spoofing", "2.3.2", "dynamic"), ("Spoofing", "2.1.1", "stationary"),
    ("Spoofing + Jamming", "2.6.4", "dynamic"), ("Spoofing + Jamming", "2.6.3", "stationary"),
]
TEST_RUN_IDS = set()
for at, sid, rs in TEST_SELECTION:
    m = df.loc[(df.attack_type == at) & (df.scenario_id == sid) & (df.rover_state == rs), "run_id"].unique()
    assert len(m) == 1
    TEST_RUN_IDS.add(m[0])


def attack_type_balanced_folds(frame, n_splits, seed):
    rng = np.random.RandomState(seed)
    folds = [set() for _ in range(n_splits)]
    for _, sub in frame.groupby("attack_type"):
        runs = sub["run_id"].unique().tolist()
        rng.shuffle(runs)
        for i, r in enumerate(runs):
            folds[i % n_splits].add(r)
    return folds


def fold_metrics(y, at, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    o = {"accuracy": (tp + tn) / (tp + tn + fp + fn), "clean_recall": tn / (tn + fp) if (tn + fp) else np.nan,
         "fpr": fp / (fp + tn) if (fp + tn) else np.nan, "attack_recall": tp / (tp + fn) if (tp + fn) else np.nan}
    am = np.asarray(y) == 1
    a = np.asarray(at)
    for k in ATTACK_TYPES:
        mk = am & (a == k)
        o[f"recall_{k}"] = pred[mk].mean() if mk.sum() else np.nan
    return o


def fit_rf(X, y):
    clf = Pipeline([("i", SimpleImputer(strategy="median")),
                    ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                  random_state=RNG_SEED, n_jobs=-1))])
    clf.fit(X, y)
    clf.named_steps["rf"].n_jobs = 1
    return clf


METRICS = ["accuracy", "clean_recall", "fpr", "attack_recall"] + [f"recall_{k}" for k in ATTACK_TYPES]
folds = attack_type_balanced_folds(df, N_SPLITS, RNG_SEED)


# ---- Evaluation A: rotating GroupKFold ----
def eval_groupkfold(cols, label):
    oof = np.full(len(df), np.nan)
    rows, det = [], True
    for vi, vruns in enumerate(folds):
        isv = df.run_id.isin(vruns)
        tr, va = df[~isv], df[isv]
        clf = fit_rf(tr[cols], tr["attack"])
        p = clf.predict_proba(va[cols])[:, 1]
        det &= np.array_equal(p, clf.predict_proba(va[cols])[:, 1])
        oof[va.index] = p
        m = fold_metrics(va["attack"], va["attack_type"], p, SHIPPED_THRESHOLD)
        m["fold"] = vi
        rows.append(m)
    fd = pd.DataFrame(rows).set_index("fold")
    roc, pr = roc_auc_score(df["attack"], oof), average_precision_score(df["attack"], oof)
    print(f"\n[A/GroupKFold] {label}  determinism={'PASS' if det else 'FAIL'}  pooled ROC-AUC={roc:.4f} PR-AUC={pr:.4f}")
    print(fd[METRICS].agg(["mean", "std"]).T.to_string(float_format=lambda x: f"{x:.4f}"))
    return fd, roc, pr


# ---- Evaluation B: fixed shipped split ----
def eval_fixed(cols, label):
    isv = df.run_id.isin(TEST_RUN_IDS)
    tr, te = df[~isv], df[isv].copy()
    clf = fit_rf(tr[cols], tr["attack"])
    p = clf.predict_proba(te[cols])[:, 1]
    det = np.array_equal(p, clf.predict_proba(te[cols])[:, 1])
    te["proba"] = p
    m = fold_metrics(te["attack"], te["attack_type"], p, SHIPPED_THRESHOLD)
    roc, pr = roc_auc_score(te["attack"], p), average_precision_score(te["attack"], p)
    print(f"\n[B/fixed split] {label}  determinism={'PASS' if det else 'FAIL'}  ROC-AUC={roc:.4f} PR-AUC={pr:.4f}")
    print("  " + "  ".join(f"{k}={m[k]:.4f}" for k in METRICS))
    print("  per-recording (clean_recall / attack_recall) @ t=0.52:")
    for rid, g in te.groupby("run_id"):
        pr_ = (g["proba"] >= SHIPPED_THRESHOLD).astype(int)
        cl = g["attack"] == 0
        a1 = g["attack"] == 1
        tag = "  <-- degraded-reception watch" if rid.split("\\")[-1] in ("1.10.6", "3.2.8") else ""
        print(f"    {rid.split(chr(92))[-1]:<8} {g['attack_type'].iloc[0]:<20} "
              f"clean={(pr_[cl] == 0).mean():.3f}  attack={(pr_[a1] == 1).mean():.3f}{tag}")
    return m, roc, pr, clf


results = {}
for label, cols in [("BASE", base_cols), ("BASE+NEW", all_cols)]:
    print("\n" + "=" * 80 + f"\n{label}\n" + "=" * 80)
    fa, roc_a, pr_a = eval_groupkfold(cols, label)
    fb, roc_b, pr_b, clf_b = eval_fixed(cols, label)
    results[label] = dict(cols=cols, gkf=fa, gkf_roc=roc_a, gkf_pr=pr_a,
                          fx=fb, fx_roc=roc_b, fx_pr=pr_b, clf_b=clf_b)

# ---- deltas ----
print("\n" + "=" * 80 + "\nDELTA (BASE+NEW - BASE)\n" + "=" * 80)
ga, gn = results["BASE"]["gkf"][METRICS].mean(), results["BASE+NEW"]["gkf"][METRICS].mean()
print("A/GroupKFold mean delta:")
for k in METRICS:
    print(f"  {k:<26} BASE {ga[k]:.4f}  NEW {gn[k]:.4f}  delta {gn[k]-ga[k]:+.4f}")
print(f"  pooled ROC-AUC   BASE {results['BASE']['gkf_roc']:.4f}  NEW {results['BASE+NEW']['gkf_roc']:.4f}  "
      f"delta {results['BASE+NEW']['gkf_roc']-results['BASE']['gkf_roc']:+.4f}")
print("\nB/fixed-split delta:")
for k in METRICS:
    d = results["BASE+NEW"]["fx"][k] - results["BASE"]["fx"][k]
    print(f"  {k:<26} BASE {results['BASE']['fx'][k]:.4f}  NEW {results['BASE+NEW']['fx'][k]:.4f}  delta {d:+.4f}")
print(f"  ROC-AUC          BASE {results['BASE']['fx_roc']:.4f}  NEW {results['BASE+NEW']['fx_roc']:.4f}  "
      f"delta {results['BASE+NEW']['fx_roc']-results['BASE']['fx_roc']:+.4f}")

# ---- SHAP on fixed-split BASE+NEW model ----
print("\n" + "=" * 80 + "\nSHAP -- fixed-split BASE+NEW model\n" + "=" * 80)
clf = results["BASE+NEW"]["clf_b"]
imp, rf = clf.named_steps["i"], clf.named_steps["rf"]
rng = np.random.RandomState(RNG_SEED)
te = df[df.run_id.isin(TEST_RUN_IDS)]
idx = rng.choice(len(te), size=min(800, len(te)), replace=False)
Xs = imp.transform(te.iloc[idx][all_cols])
ex = shap.TreeExplainer(rf)


def _attack_sv(a):
    a = np.asarray(a)
    return a[:, :, 1] if a.ndim == 3 else a


sv = _attack_sv(ex.shap_values(Xs, check_additivity=False))
det = np.array_equal(sv, _attack_sv(ex.shap_values(Xs, check_additivity=False)))
mabs = pd.Series(np.abs(sv).mean(0), index=all_cols).sort_values(ascending=False)
gini = pd.Series(rf.feature_importances_, index=all_cols).sort_values(ascending=False)
rank_s = {f: i + 1 for i, f in enumerate(mabs.index)}
rank_g = {f: i + 1 for i, f in enumerate(gini.index)}
print(f"SHAP determinism (scored twice, bit-identical): {det}")
print(f"{'feature':<24}{'mean|SHAP|':>12}{'SHAP rank':>11}{'gini rank':>11}")
for f in sp_cols:
    print(f"{f:<24}{mabs[f]:>12.5f}{rank_s[f]:>8}/{len(all_cols)}{rank_g[f]:>8}/{len(all_cols)}")
print(f"\ntop 12 of {len(all_cols)} by mean|SHAP|:")
for i, (f, v) in enumerate(mabs.head(12).items()):
    print(f"  {i+1:>2}. {f:<26}{v:.5f}{'  <-- NEW' if f in sp_cols else ''}")

pd.DataFrame({"mean_abs_shap": mabs, "shap_rank": pd.Series(rank_s), "gini": gini,
             "gini_rank": pd.Series(rank_g),
             "is_new": pd.Series({f: f in sp_cols for f in all_cols})}).sort_values("shap_rank")\
    .to_csv(PROC / "temporal_shap_ranking.csv")
print("\nSaved processed/temporal_shap_ranking.csv")
