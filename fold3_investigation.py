"""Diagnostic-only investigation of GroupKFold outer fold 3 (calibration_experiment.py),
the fold with the worst uncalibrated clean recall (0.360 vs. a 0.66 mean across folds).
Reuses the exact same fold construction (same seed) and model config as
calibration_experiment.py so the fold-3 model/predictions here are identical to what
produced that number. Does not modify any model artifact or shipped code -- read-only
analysis. Writes FOLD_ANALYSIS.md.
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
from sklearn.metrics import confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")
SCRIPT_DIR = Path(__file__).resolve().parent
RNG_SEED = 42
OUTER_N_SPLITS = 4
SHIPPED_THRESHOLD = 0.52
TARGET_FOLD = 3

df = pd.read_parquet(SCRIPT_DIR / "processed" / "syncguard_features.parquet").reset_index(drop=True)
df["real_time"] = pd.to_datetime(df["real_time"])
EXCLUDE_COLS = {
    "iTOW", "real_time", "lat", "lon", "height", "clock_drift_proxy_s",
    "attack", "scenario_id", "attack_type", "rover_state", "power_w_max", "bands",
    "n_attack_windows", "window_start_in_range", "run_id",
}
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]


def attack_type_balanced_folds(frame, n_splits, seed):
    rng = np.random.RandomState(seed)
    folds = [set() for _ in range(n_splits)]
    for atype, sub in frame.groupby("attack_type"):
        runs = sub["run_id"].unique().tolist()
        rng.shuffle(runs)
        for i, run_id in enumerate(runs):
            folds[i % n_splits].add(run_id)
    return folds


outer_folds = attack_type_balanced_folds(df, OUTER_N_SPLITS, seed=RNG_SEED)

print("=" * 78)
print("ALL FOLDS -- recording membership")
print("=" * 78)
for i, runs in enumerate(outer_folds):
    print(f"\n--- fold {i} ---")
    for r in sorted(runs):
        rsub = df[df["run_id"] == r]
        print(f"  {r}")
        print(f"    attack_type={rsub['attack_type'].iloc[0]!r} scenario_id={rsub['scenario_id'].iloc[0]!r} "
              f"rover_state={rsub['rover_state'].iloc[0]!r} rows={len(rsub)} "
              f"attack_rows={int(rsub['attack'].sum())} clean_rows={int((rsub['attack']==0).sum())}")

val_runs = outer_folds[TARGET_FOLD]
is_val = df["run_id"].isin(val_runs)
train_df, val_df = df[~is_val].reset_index(drop=True), df[is_val].copy()
X_train, y_train = train_df[feature_cols], train_df["attack"]
X_val, y_val = val_df[feature_cols], val_df["attack"]

imputer = SimpleImputer(strategy="median").fit(X_train)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RNG_SEED, n_jobs=-1)
rf.fit(imputer.transform(X_train), y_train)
rf.n_jobs = 1
proba = rf.predict_proba(imputer.transform(X_val))[:, 1]
val_df["proba"] = proba
val_df["pred"] = (proba >= SHIPPED_THRESHOLD).astype(int)

cm = confusion_matrix(y_val, val_df["pred"])
print(f"\n\nFold {TARGET_FOLD} confusion matrix (threshold={SHIPPED_THRESHOLD}): "
      f"TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
clean_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print(f"Clean recall: {clean_recall:.4f}  (reproduces the 0.360 reported earlier: "
      f"{'YES, matches' if abs(clean_recall - 0.360) < 0.01 else 'MISMATCH -- investigate'})")

# ---------------------------------------------------------------------------
# Per-recording breakdown of clean-row misclassification within fold 3
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"PER-RECORDING breakdown -- fold {TARGET_FOLD} CLEAN rows only (true attack=0)")
print("=" * 78)
clean = val_df[val_df["attack"] == 0].copy()
per_run = clean.groupby("run_id").apply(
    lambda g: pd.Series({
        "attack_type": g["attack_type"].iloc[0],
        "scenario_id": g["scenario_id"].iloc[0],
        "rover_state": g["rover_state"].iloc[0],
        "n_clean_rows": len(g),
        "n_misclassified": int((g["pred"] == 1).sum()),
        "misclass_rate": (g["pred"] == 1).mean(),
        "mean_proba": g["proba"].mean(),
    }), include_groups=False,
)
print(per_run.to_string())
worst_run = per_run["misclass_rate"].idxmax()
print(f"\nWorst single recording: {worst_run} ({per_run.loc[worst_run, 'misclass_rate']:.1%} "
      f"of its clean rows misclassified, {int(per_run.loc[worst_run, 'n_misclassified'])} of "
      f"{int(per_run.loc[worst_run, 'n_clean_rows'])})")
concentration = per_run.loc[worst_run, "n_misclassified"] / per_run["n_misclassified"].sum()
print(f"That recording accounts for {concentration:.1%} of all {int(per_run['n_misclassified'].sum())} "
      f"misclassified clean rows in this fold.")

# ---------------------------------------------------------------------------
# Time-position analysis: warm-up window + distance to nearest attack-window boundary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("TIME-POSITION ANALYSIS (within-recording)")
print("=" * 78)


def add_time_position_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("real_time").copy()
    t0 = g["real_time"].iloc[0]
    g["elapsed_s"] = (g["real_time"] - t0).dt.total_seconds()
    # distance (seconds) to nearest attack<->clean label transition in this recording
    label = g["attack"].values
    change_idx = np.where(np.diff(label) != 0)[0]
    boundary_times = g["real_time"].values[change_idx] if len(change_idx) else np.array([], dtype="datetime64[ns]")
    if len(boundary_times):
        t = g["real_time"].values.astype("datetime64[ns]").astype("int64")
        b = boundary_times.astype("datetime64[ns]").astype("int64")
        dist_ns = np.min(np.abs(t[:, None] - b[None, :]), axis=1)
        g["dist_to_boundary_s"] = dist_ns / 1e9
    else:
        g["dist_to_boundary_s"] = np.nan
    return g


val_df_t = val_df.groupby("run_id", group_keys=False)[val_df.columns.tolist()].apply(add_time_position_features)
clean_t = val_df_t[val_df_t["attack"] == 0].copy()

warmup_s = 60
in_warmup = clean_t["elapsed_s"] < warmup_s
print(f"\nClean rows in first {warmup_s}s of their recording (receiver warm-up window):")
print(f"  misclass rate in warm-up window:     {clean_t.loc[in_warmup, 'pred'].mean():.3f}  (n={in_warmup.sum()})")
print(f"  misclass rate outside warm-up window: {clean_t.loc[~in_warmup, 'pred'].mean():.3f}  (n={(~in_warmup).sum()})")

near_boundary_s = 10
near_b = clean_t["dist_to_boundary_s"] < near_boundary_s
near_b = near_b.fillna(False)
print(f"\nClean rows within {near_boundary_s}s of an attack-window boundary (transition effect):")
print(f"  misclass rate near boundary:     {clean_t.loc[near_b, 'pred'].mean():.3f}  (n={int(near_b.sum())})")
print(f"  misclass rate away from boundary: {clean_t.loc[~near_b, 'pred'].mean():.3f}  (n={int((~near_b).sum())})")

print("\nMisclassified vs correctly-classified clean rows -- elapsed_s / dist_to_boundary_s summary:")
for label, sub in [("misclassified", clean_t[clean_t["pred"] == 1]), ("correct", clean_t[clean_t["pred"] == 0])]:
    print(f"  {label}: n={len(sub)}  elapsed_s median={sub['elapsed_s'].median():.1f}  "
          f"dist_to_boundary_s median={sub['dist_to_boundary_s'].median():.1f}")

# ---------------------------------------------------------------------------
# Feature distribution comparison: fold-3 misclassified clean vs fold-3 correct clean
# vs other-folds clean (top baseline feature importances)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("FEATURE DISTRIBUTIONS -- top baseline-importance features")
print("=" * 78)
TOP_FEATURES = ["agc_cnt_mean", "snr_l1_mean", "noise_per_ms_mean", "jam_ind_mean",
                 "snr_l1_std", "velN", "n_sats_l1", "gSpeed", "hAcc"]

other_folds_val_runs = set().union(*[outer_folds[i] for i in range(OUTER_N_SPLITS) if i != TARGET_FOLD])
other_clean = df[df["run_id"].isin(other_folds_val_runs) & (df["attack"] == 0)]

fold3_mis = clean[clean["pred"] == 1]
fold3_correct = clean[clean["pred"] == 0]

rows = []
for feat in TOP_FEATURES:
    rows.append({
        "feature": feat,
        "fold3_misclassified_median": fold3_mis[feat].median(),
        "fold3_correct_median": fold3_correct[feat].median(),
        "other_folds_clean_median": other_clean[feat].median(),
        "fold3_misclassified_mean": fold3_mis[feat].mean(),
        "fold3_correct_mean": fold3_correct[feat].mean(),
        "other_folds_clean_mean": other_clean[feat].mean(),
    })
feat_df = pd.DataFrame(rows).set_index("feature")
print(feat_df.to_string())
feat_df.to_csv(SCRIPT_DIR / "processed" / "fold3_feature_comparison.csv")

# Plot: distribution of key features across the three groups
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, feat in zip(axes.flat, TOP_FEATURES):
    for label, sub, color in [("fold3 misclassified", fold3_mis, "tab:red"),
                                ("fold3 correct", fold3_correct, "tab:blue"),
                                ("other folds clean", other_clean, "tab:gray")]:
        vals = sub[feat].dropna()
        if len(vals) > 1:
            ax.hist(vals, bins=40, density=True, alpha=0.45, label=label, color=color)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=7)
fig.suptitle(f"Fold {TARGET_FOLD}: feature distributions, misclassified vs correct clean rows, vs other folds' clean rows")
fig.tight_layout()
out_png = SCRIPT_DIR / "processed" / "fold3_feature_distributions.png"
fig.savefig(out_png, dpi=110)
print(f"\nSaved {out_png}")

# ---------------------------------------------------------------------------
# Per-recording feature comparison for the worst recording specifically
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"WORST RECORDING DEEP DIVE: {worst_run}")
print("=" * 78)
worst_sub = clean[clean["run_id"] == worst_run]
print(f"attack_type={worst_sub['attack_type'].iloc[0]!r} scenario_id={worst_sub['scenario_id'].iloc[0]!r} "
      f"rover_state={worst_sub['rover_state'].iloc[0]!r}")
print(f"n_clean_rows={len(worst_sub)}  misclassified={int((worst_sub['pred']==1).sum())}")
for feat in TOP_FEATURES:
    print(f"  {feat:>20}: this-recording median={worst_sub[feat].median():.3f}  "
          f"other-folds-clean median={other_clean[feat].median():.3f}")

val_df.to_csv(SCRIPT_DIR / "processed" / "fold3_full_predictions.csv", index=False)
print(f"\nSaved full fold-3 predictions to processed/fold3_full_predictions.csv")
