"""STEP 0 feasibility probe for temporal-coherence features (TEMPORAL_COHERENCE.md).
Read-only. Measures per-satellite continuous tracking-run structure in raw rinex.csv:
can we fit stable trailing-window slopes over 10-30 epochs per SV?"""
import sys, json
from pathlib import Path
from datetime import timedelta
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"C:\Users\Iris\Downloads\agaif-materials\dataset\raw")
C = 299792458.0
F_L1 = 1575.42e6; LAMBDA_L1 = C / F_L1
GAP_S = 0.35          # dt above this = tracking break (native cadence ~0.2s)

SCEN = [
    ("Spoofing stat 2.1.1",     RAW/"Spoofing/stationary/Medium Power (_1W)/Bands_L1_L2_L5/2.1.1"),
    ("Jamming stat 1.6.4",      RAW/"Jamming/stationary/Very High Power (\u226510W)/Bands_L1_L2_L5/1.6.4"),
    ("Jamming DYN 1.10.6",      RAW/"Jamming/dynamic/High Power (_10W)/Bands_E1_E5_E5a_E5b_E6_L1_L2_L3_L5/1.10.6"),
    ("Meaconing DYN 3.2.8",     RAW/"Meaconing/dynamic/Very High Power (\u226510W)/Bands_E1_L1_L2/3.2.8"),
    ("Meaconing stat 3.2.7",    RAW/"Meaconing/stationary/Very High Power (\u226510W)/Bands_E1_L1_L2/3.2.7"),
    ("Spoof+Jam stat 2.6.3",    RAW/"Jamming+Spoofing/stationary/Low Power (_100mW)/Bands_L1/2.6.3"),
    ("Spoofing DYN 2.3.2",      RAW/"Spoofing/dynamic/Medium Power (_1W)/Bands_E1_E5_L1_L2_L5/2.3.2"),
]

def windows(sj):
    out=[]; ps=None
    for ev in sj.get("attack_log",[]):
        ts=(pd.Timestamp(ev["timestamp_utc"])-timedelta(hours=2)).tz_localize(None)
        if "start" in ev["event"].lower(): ps=ts
        elif "end" in ev["event"].lower() and ps is not None: out.append((ps,ts)); ps=None
    return out

def run_ids(t, gap_s=GAP_S):
    """label consecutive-epoch runs per already-sorted series of datetimes"""
    dt = t.diff().dt.total_seconds()
    return (dt.isna() | (dt > gap_s)).cumsum()

for name, sdir in SCEN:
    if not (sdir/"rinex.csv").exists():
        print(f"\n{name}: MISSING"); continue
    sj = json.loads((sdir/"scenario.json").read_text())
    w = windows(sj)
    rx = pd.read_csv(sdir/"rinex.csv", low_memory=False,
                     usecols=lambda c: c in ("time","satellite","pseudorange_L1","carrier_phase_L1",
                                              "doppler_L1","snr_L1","pseudorange_L2"))
    rx["time"] = pd.to_datetime(rx["time"], errors="coerce")
    for c in rx.columns:
        if c not in ("time","satellite"): rx[c] = pd.to_numeric(rx[c], errors="coerce")
    rx = rx.dropna(subset=["time","satellite"]).sort_values(["satellite","time"])
    rx.loc[(rx.snr_L1<0)|(rx.snr_L1>60),"snr_L1"]=np.nan

    # attack label per epoch
    aw = pd.Series(0, index=rx.index)
    for s,e in w: aw |= ((rx.time>=s)&(rx.time<=e)).astype(int)
    rx["attack"]=aw

    n_epochs = rx["time"].nunique()
    med_cad = rx.groupby("satellite")["time"].diff().dt.total_seconds().median()

    # per-SV tracking runs
    rx["run"] = rx.groupby("satellite", group_keys=False)["time"].apply(run_ids)
    runlen = rx.groupby(["satellite","run"]).size()
    print(f"\n{'='*78}\n{name}   epochs={n_epochs}  per-SV median cadence={med_cad:.2f}s  "
          f"attack_frac={rx.attack.mean():.2f}")
    print(f"  tracking-run length (consecutive epochs, gap>{GAP_S}s = break):")
    print(f"    count={len(runlen)}  median={runlen.median():.0f}  mean={runlen.mean():.0f}  "
          f"p10={runlen.quantile(.1):.0f}  p90={runlen.quantile(.9):.0f}  max={runlen.max()}")
    for thr in (10,20,30,50):
        frac_runs = (runlen>=thr).mean()
        frac_rows = runlen[runlen>=thr].sum()/runlen.sum()
        print(f"    runs >= {thr:>2} epochs: {frac_runs:5.1%} of runs, {frac_rows:5.1%} of all SV-epoch rows")

    # CMC linearity within runs >= 20 epochs: R^2 of linear fit of CMC vs time
    rx["cmc"] = rx["pseudorange_L1"] - LAMBDA_L1*rx["carrier_phase_L1"]
    r2s, slopes_mps = [], []
    big = runlen[runlen>=20].index
    for (sv,run) in list(big)[:4000]:
        g = rx[(rx.satellite==sv)&(rx.run==run)]
        y = g["cmc"].values; x = (g["time"].values - g["time"].values[0])/np.timedelta64(1,"s")
        if np.isnan(y).any() or len(y)<20: continue
        a,b = np.polyfit(x, y, 1)
        yhat = a*x+b
        ss_res = np.sum((y-yhat)**2); ss_tot = np.sum((y-y.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
        r2s.append(r2); slopes_mps.append(a)
    r2s = np.array(r2s); slopes_mps = np.array(slopes_mps)
    if len(r2s):
        print(f"  CMC linear-fit over runs>=20ep (n={len(r2s)} arcs): "
              f"R^2 median={np.nanmedian(r2s):.3f}  |slope| median={np.nanmedian(np.abs(slopes_mps)):.3f} m/s  "
              f"p90={np.nanquantile(np.abs(slopes_mps),.9):.2f} m/s")
        # clean vs attack CMC slope magnitude
        # attach per-arc attack status (majority)
        atk_arc=[]; sl_arc=[]
        for (sv,run) in list(big)[:4000]:
            g = rx[(rx.satellite==sv)&(rx.run==run)]
            y=g["cmc"].values
            if np.isnan(y).any() or len(y)<20: continue
            x=(g["time"].values-g["time"].values[0])/np.timedelta64(1,"s")
            a,_=np.polyfit(x,y,1)
            atk_arc.append(g["attack"].mean()>0.5); sl_arc.append(abs(a))
        atk_arc=np.array(atk_arc); sl_arc=np.array(sl_arc)
        if atk_arc.any() and (~atk_arc).any():
            print(f"    |CMC slope| median  clean={np.median(sl_arc[~atk_arc]):.3f}  "
                  f"attack={np.median(sl_arc[atk_arc]):.3f} m/s")

    # L2 coverage
    print(f"  pseudorange_L2 non-null: {rx.pseudorange_L2.notna().mean():.1%}")
