import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).resolve().parent / "processed"
df = pd.read_parquet(OUT_DIR / "syncguard_features.parquet")
df["real_time"] = pd.to_datetime(df["real_time"])

print("=" * 70)
print("SHAPE:", df.shape)
print("\nCOLUMNS:", list(df.columns))

print("\n" + "=" * 70)
print("FIRST 5 ROWS (key columns)")
key_cols = ["real_time", "scenario_id", "attack_type", "rover_state", "attack",
            "snr_l1_mean", "doppler_l1_mean", "pr_doppler_residual_std",
            "clock_drift_proxy_s", "pos_dev_m", "jam_ind_mean", "numSV", "pDOP"]
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(df[key_cols].head(5).to_string())

print("\n" + "=" * 70)
print("PER-SCENARIO ROW / ATTACK COUNTS")
print(df.groupby(["attack_type", "scenario_id"]).agg(
    rows=("attack", "size"), attack_rows=("attack", "sum")).to_string())

print("\n" + "=" * 70)
print("FEATURE STATS BY ATTACK LABEL (0=clean, 1=attack)")
feature_cols = ["snr_l1_mean", "snr_l1_std", "doppler_l1_mean", "doppler_l1_std",
                "pr_doppler_residual_mean", "pr_doppler_residual_std",
                "clock_drift_proxy_s", "pos_dev_m", "jam_ind_mean",
                "noise_per_ms_mean", "numSV", "pDOP", "n_sats_l1"]
stats = df.groupby("attack")[feature_cols].agg(["mean", "std"])
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(stats.to_string())

print("\n" + "=" * 70)
print("FEATURE STATS BY ATTACK TYPE x LABEL")
stats2 = df.groupby(["attack_type", "attack"])[
    ["snr_l1_mean", "pr_doppler_residual_std", "jam_ind_mean", "pos_dev_m"]
].mean()
with pd.option_context("display.width", 200):
    print(stats2.to_string())

# --- Plot: scenario 2.1.1 (stationary spoofing, clean prelude + spoof window + clean tail) ---
s = df[df["scenario_id"] == "2.1.1"].sort_values("real_time")
fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

axes[0].plot(s["real_time"], s["snr_l1_mean"], color="tab:blue", lw=0.8)
axes[0].set_ylabel("Mean C/N0 (L1, dB-Hz)")

axes[1].plot(s["real_time"], s["pr_doppler_residual_std"], color="tab:orange", lw=0.8)
axes[1].set_ylabel("Code-Doppler\nresidual std (m/s)")

axes[2].plot(s["real_time"], s["clock_drift_proxy_s"], color="tab:green", lw=0.8)
axes[2].set_ylabel("Clock drift\nproxy (s)")

axes[3].plot(s["real_time"], s["pos_dev_m"], color="tab:red", lw=0.8)
axes[3].set_ylabel("Position\ndeviation (m)")
axes[3].set_xlabel("Time (UTC)")

attack_mask = s["attack"] == 1
if attack_mask.any():
    t0 = s.loc[attack_mask, "real_time"].iloc[0]
    t1 = s.loc[attack_mask, "real_time"].iloc[-1]
    for ax in axes:
        ax.axvspan(t0, t1, color="red", alpha=0.12, label="spoofed (attack_log)")
axes[0].legend(loc="upper right")
axes[0].set_title("Scenario 2.1.1 -- Spoofing, stationary (clean vs. spoofed)")

fig.tight_layout()
out_path = OUT_DIR / "scenario_2.1.1_clean_vs_spoofed.png"
fig.savefig(out_path, dpi=130)
print(f"\nSaved plot to {out_path}")
