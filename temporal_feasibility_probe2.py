"""STEP 0 probe pass 2: cycle-slip-aware CMC arc slopes, Doppler polyfit residual, and
per-epoch tracking-set continuity (Jaccard) -- clean vs attack. Read-only."""
import sys, json
from pathlib import Path
from datetime import timedelta
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"C:\Users\Iris\Downloads\agaif-materials\dataset\raw")
C = 299792458.0; LAMBDA_L1 = C / 1575.42e6
GAP_S = 0.35
SLIP_CYC = 5.0        # |phase-rate - (-doppler)| * dt above this many cycles = cycle slip -> new arc
WIN = 20              # trailing-window epochs for slope/residual

SCEN = [
    ("Spoofing stat 2.1.1",  RAW/"Spoofing/stationary/Medium Power (_1W)/Bands_L1_L2_L5/2.1.1"),
    ("Jamming stat 1.6.4",   RAW/"Jamming/stationary/Very High Power (\u226510W)/Bands_L1_L2_L5/1.6.4"),
    ("Jamming DYN 1.10.6",   RAW/"Jamming/dynamic/High Power (_10W)/Bands_E1_E5_E5a_E5b_E6_L1_L2_L3_L5/1.10.6"),
    ("Meaconing DYN 3.2.8",  RAW/"Meaconing/dynamic/Very High Power (\u226510W)/Bands_E1_L1_L2/3.2.8"),
    ("Meaconing stat 3.2.7", RAW/"Meaconing/stationary/Very High Power (\u226510W)/Bands_E1_L1_L2/3.2.7"),
    ("Spoof+Jam stat 2.6.3", RAW/"Jamming+Spoofing/stationary/Low Power (_100mW)/Bands_L1/2.6.3"),
    ("Spoofing DYN 2.3.2",   RAW/"Spoofing/dynamic/Medium Power (_1W)/Bands_E1_E5_L1_L2_L5/2.3.2"),
]
def windows(sj):
    out=[]; ps=None
    for ev in sj.get("attack_log",[]):
        ts=(pd.Timestamp(ev["timestamp_utc"])-timedelta(hours=2)).tz_localize(None)
        if "start" in ev["event"].lower(): ps=ts
        elif "end" in ev["event"].lower() and ps is not None: out.append((ps,ts)); ps=None
    return out

def trailing_slope(y):
    n=len(y); x=np.arange(n, dtype=float)
    x-=x.mean(); return (x@ (y-y.mean()))/(x@x) if (x@x)>0 else np.nan

for name, sdir in SCEN:
    sj=json.loads((sdir/"scenario.json").read_text()); w=windows(sj)
    rx=pd.read_csv(sdir/"rinex.csv", low_memory=False,
                   usecols=lambda c: c in ("time","satellite","pseudorange_L1","carrier_phase_L1","doppler_L1","snr_L1"))
    rx["time"]=pd.to_datetime(rx["time"],errors="coerce")
    for c in rx.columns:
        if c not in ("time","satellite"): rx[c]=pd.to_numeric(rx[c],errors="coerce")
    rx=rx.dropna(subset=["time","satellite"]).sort_values(["satellite","time"]).reset_index(drop=True)
    aw=pd.Series(0,index=rx.index)
    for s,e in w: aw|=((rx.time>=s)&(rx.time<=e)).astype(int)
    rx["attack"]=aw

    # per-SV dt, phase rate, cycle-slip-aware arc id
    dt=rx.groupby("satellite")["time"].diff().dt.total_seconds()
    cp_rate=rx.groupby("satellite")["carrier_phase_L1"].diff()/dt
    slip=( (dt.isna()) | (dt>GAP_S) | ((cp_rate + rx["doppler_L1"]).abs()*dt.fillna(0) > SLIP_CYC) )
    rx["arc"]=slip.groupby(rx["satellite"]).cumsum()
    rx["cmc"]=rx["pseudorange_L1"]-LAMBDA_L1*rx["carrier_phase_L1"]

    arclen=rx.groupby(["satellite","arc"]).size()
    print(f"\n{'='*78}\n{name}  epochs={rx.time.nunique()}  attack_frac={rx.attack.mean():.2f}")
    print(f"  slip-aware arcs: n={len(arclen)} median_len={arclen.median():.0f} "
          f"frac rows in arcs>={WIN}ep: {arclen[arclen>=WIN].sum()/arclen.sum():.1%}")

    # trailing-window CMC slope + Doppler poly residual over each arc (rolling)
    cmc_sl=[]; dop_res=[]; atk=[]
    for (sv,a),g in rx.groupby(["satellite","arc"]):
        if len(g)<WIN: continue
        cmc=g["cmc"].values; dop=g["doppler_L1"].values; at=g["attack"].values
        for i in range(WIN,len(g)+1):
            wnd=slice(i-WIN,i)
            s=trailing_slope(cmc[wnd])
            # doppler: quadratic fit residual std over window
            xx=np.arange(WIN,dtype=float)
            co=np.polyfit(xx,dop[wnd],2); r=dop[wnd]-np.polyval(co,xx)
            cmc_sl.append(abs(s)); dop_res.append(r.std()); atk.append(at[i-1]==1)
    cmc_sl=np.array(cmc_sl); dop_res=np.array(dop_res); atk=np.array(atk)
    if atk.any() and (~atk).any():
        def q(a,m): return np.nanquantile(a[m],[.5,.9])
        print(f"  |CMC trailing-slope {WIN}ep| m/s   clean p50/p90={q(cmc_sl,~atk)}  attack p50/p90={q(cmc_sl,atk)}")
        print(f"  Doppler quad-fit resid std Hz    clean p50/p90={q(dop_res,~atk)}  attack p50/p90={q(dop_res,atk)}")

    # per-epoch tracking-set continuity: Jaccard vs prev epoch, reacquisition rate
    ep=rx.groupby("time")["satellite"].apply(set).sort_index()
    epa=rx.groupby("time")["attack"].max().sort_index()
    jac=[]; ta=[]
    prev=None
    for t,s in ep.items():
        if prev is not None and (len(prev|s)>0):
            jac.append(len(prev&s)/len(prev|s)); ta.append(epa[t])
        prev=s
    jac=np.array(jac); ta=np.array(ta)
    if ta.any() and (~ta).any():
        print(f"  tracking-set Jaccard vs prev epoch   clean median={np.median(jac[ta==0]):.3f}  attack median={np.median(jac[ta==1]):.3f}")
