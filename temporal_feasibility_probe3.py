"""STEP 0 probe pass 3: reacquisition-RATE reformulation of tracking continuity, and CMC
slope on ALL spoofing recordings (the weak spot in pass 2). Read-only."""
import sys, json
from pathlib import Path
from datetime import timedelta
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"C:\Users\Iris\Downloads\agaif-materials\dataset\raw")
LAMBDA_L1 = 299792458.0 / 1575.42e6
GAP_S=0.35; SLIP_CYC=5.0; WIN=20; REACQ_WIN_S=10.0

SP = list(RAW.rglob("scenario.json"))
def windows(sj):
    out=[]; ps=None
    for ev in sj.get("attack_log",[]):
        ts=(pd.Timestamp(ev["timestamp_utc"])-timedelta(hours=2)).tz_localize(None)
        if "start" in ev["event"].lower(): ps=ts
        elif "end" in ev["event"].lower() and ps is not None: out.append((ps,ts)); ps=None
    return out
def tslope(y):
    x=np.arange(len(y),dtype=float); x-=x.mean()
    return (x@(y-y.mean()))/(x@x) if (x@x)>0 else np.nan

for sj_path in sorted(SP):
    sdir=sj_path.parent; sj=json.loads(sj_path.read_text()); w=windows(sj)
    rx=pd.read_csv(sdir/"rinex.csv",low_memory=False,
                   usecols=lambda c:c in ("time","satellite","pseudorange_L1","carrier_phase_L1","doppler_L1"))
    rx["time"]=pd.to_datetime(rx["time"],errors="coerce")
    for c in ("pseudorange_L1","carrier_phase_L1","doppler_L1"): rx[c]=pd.to_numeric(rx[c],errors="coerce")
    rx=rx.dropna(subset=["time","satellite"]).sort_values(["satellite","time"]).reset_index(drop=True)
    aw=pd.Series(0,index=rx.index)
    for s,e in w: aw|=((rx.time>=s)&(rx.time<=e)).astype(int)
    rx["attack"]=aw
    dt=rx.groupby("satellite")["time"].diff().dt.total_seconds()
    cpr=rx.groupby("satellite")["carrier_phase_L1"].diff()/dt
    slip=((dt.isna())|(dt>GAP_S)|((cpr+rx.doppler_L1).abs()*dt.fillna(0)>SLIP_CYC))
    rx["arc"]=slip.groupby(rx["satellite"]).cumsum()
    rx["cmc"]=rx.pseudorange_L1-LAMBDA_L1*rx.carrier_phase_L1
    rx["arc_start"]=slip.astype(int)

    sl=[]; atk=[]
    for (sv,a),g in rx.groupby(["satellite","arc"]):
        if len(g)<WIN: continue
        cmc=g.cmc.values; at=g.attack.values
        for i in range(WIN,len(g)+1):
            sl.append(abs(tslope(cmc[i-WIN:i]))); atk.append(at[i-1]==1)
    sl=np.array(sl); atk=np.array(atk)

    # reacquisition rate: arc-starts per SV-second over trailing 10s window, aggregated per epoch
    per_ep=rx.groupby("time").agg(arc_starts=("arc_start","sum"), nsat=("satellite","nunique"),
                                  attack=("attack","max")).reset_index()
    per_ep=per_ep.sort_values("time")
    per_ep["t_s"]=(per_ep.time-per_ep.time.iloc[0]).dt.total_seconds()
    # trailing 10s rolling sum of arc_starts / trailing rolling sum of nsat  (causal)
    per_ep=per_ep.set_index("time")
    r_starts=per_ep["arc_starts"].rolling(f"{int(REACQ_WIN_S)}s").sum()
    r_nsat=per_ep["nsat"].rolling(f"{int(REACQ_WIN_S)}s").sum()
    per_ep["reacq_rate"]=(r_starts/r_nsat).values
    a=per_ep.attack.values.astype(bool); rr=per_ep.reacq_rate.values
    name=f"{sj.get('attack_type')} {sj.get('scenario_id')} {sj.get('attack_parameters',{}).get('Rover states')}"
    line=f"{name:<34} attack_frac={a.mean():.2f}"
    if atk.any() and (~atk).any():
        line+=f" | CMC|slope| p50 clean={np.nanmedian(sl[~atk]):.3f} atk={np.nanmedian(sl[atk]):.3f}"
        line+=f" p90 clean={np.nanquantile(sl[~atk],.9):.2f} atk={np.nanquantile(sl[atk],.9):.2f}"
    if a.any() and (~a).any():
        line+=f" | reacq_rate/10s p50 clean={np.nanmedian(rr[~a]):.4f} atk={np.nanmedian(rr[a]):.4f}"
    print(line)
