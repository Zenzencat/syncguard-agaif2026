"""Why does the 34-feature spoofing-feature model regress on the FIXED 8-recording TEST split
(the one baseline_model_report.md / evaluate_models.py / the dashboard use) even though it
improved under rotating 4-fold GroupKFold? Per-recording breakdown, 23-feat vs 34-feat, on
that fixed split. This is the diagnostic behind the DO-NOT-INTEGRATE verdict in
SPOOFING_FEATURES.md. Read-only.

Reproduce: point OLD_PARQUET at a 23-feature syncguard_features.parquet and NEW_PARQUET at a
34-feature one (regenerate the latter by running extract_features.py with the sp_* additions
from the staged build -- see SPOOFING_FEATURES.md 'Where this lives'). Defaults assume you
have kept both side by side.
"""
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent
OLD_PARQUET = Path(os.environ.get("OLD_PARQUET", REPO / "processed" / "syncguard_features_23feat.parquet"))
NEW_PARQUET = Path(os.environ.get("NEW_PARQUET", REPO / "processed" / "syncguard_features_34feat.parquet"))

new = pd.read_parquet(NEW_PARQUET)
old = pd.read_parquet(OLD_PARQUET)

TEST_SELECTION = [
    ("Jamming", "1.10.6", "dynamic"), ("Jamming", "1.6.4", "stationary"),
    ("Meaconing", "3.2.8", "dynamic"), ("Meaconing", "3.2.7", "stationary"),
    ("Spoofing", "2.3.2", "dynamic"), ("Spoofing", "2.1.1", "stationary"),
    ("Spoofing + Jamming", "2.6.4", "dynamic"), ("Spoofing + Jamming", "2.6.3", "stationary"),
]
EXCLUDE = {"iTOW","real_time","lat","lon","height","clock_drift_proxy_s","attack","scenario_id",
           "attack_type","rover_state","power_w_max","bands","n_attack_windows","window_start_in_range","run_id"}

def run(df, label, thr):
    tri = df.copy()
    test_ids = set()
    for at, sid, rs in TEST_SELECTION:
        m = tri.loc[(tri.attack_type==at)&(tri.scenario_id==sid)&(tri.rover_state==rs),"run_id"].unique()
        test_ids.add(m[0])
    feat = [c for c in df.columns if c not in EXCLUDE]
    is_t = tri.run_id.isin(test_ids)
    trn, tst = tri[~is_t], tri[is_t]
    clf = Pipeline([("i",SimpleImputer(strategy="median")),
                    ("rf",RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1))])
    clf.fit(trn[feat], trn["attack"]); clf.named_steps["rf"].n_jobs=1
    p = clf.predict_proba(tst[feat])[:,1]
    tst = tst.copy(); tst["pred"] = (p>=thr).astype(int); tst["proba"]=p
    roc = roc_auc_score(tst["attack"], p)
    cr = ((tst.attack==0)&(tst.pred==0)).sum()/ (tst.attack==0).sum()
    jr = ((tst.attack_type=="Jamming")&(tst.attack==1)&(tst.pred==1)).sum()/((tst.attack_type=="Jamming")&(tst.attack==1)).sum()
    print(f"\n[{label}] thr={thr}  nfeat={len(feat)}  ROC-AUC={roc:.4f}  clean_recall={cr:.3f}  jamming_recall={jr:.3f}")
    rows=[]
    for rid,g in tst.groupby("run_id"):
        a=g[g.attack==1]; c=g[g.attack==0]
        rows.append({"rec":rid.split("\\")[-1],"type":g.attack_type.iloc[0],
                     "atk_recall":round((a.pred==1).mean(),3) if len(a) else None,
                     "cln_recall":round((c.pred==0).mean(),3) if len(c) else None,
                     "mean_proba":round(g.proba.mean(),3)})
    print(pd.DataFrame(rows).to_string(index=False))

for thr in (0.50, 0.52):
    run(old, "23-feat", thr)
    run(new, "34-feat", thr)
