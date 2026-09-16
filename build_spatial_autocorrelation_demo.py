"""Run the REAL api/spatial_stats.py methodology (KNN k=5, esda Moran + Moran_Local, same
RNG_SEED=42) over the 136 real towers using the existing SIMULATED severity values from
spatial_processed/simulated_spatial_anomaly_SIMULATED.csv, then render a LISA cluster map and
a Moran scatter plot for the new deck slide. This is genuinely running the real, cited
statistical machinery (Moran, 1950; Anselin, 1995) over real tower geometry -- the caveat, same
as everywhere else in this project, is that the severity *values* fed in come from the
simulated per-tower attribution/spread layer, not a live-replay's real per-tower events.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local

K_NEIGHBORS = 5
N_PERMUTATIONS = 999
RNG_SEED = 42
SIGNIFICANCE_ALPHA = 0.05
EARTH_RADIUS_KM = 6371.0088

QUADRANT_LABELS = {
    0: "Not significant",
    1: "High-High (hotspot)",
    2: "Low-High (spatial outlier)",
    3: "Low-Low (coldspot)",
    4: "High-Low (spatial outlier)",
}
QUADRANT_COLORS = {
    0: "#c9ccd1",
    1: "#d1263b",
    2: "#7fb3e8",
    3: "#1f5fa8",
    4: "#f0a35c",
}

df = pd.read_csv(r"C:\Users\Iris\Downloads\syncguard-agaif2026\spatial_processed\simulated_spatial_anomaly_SIMULATED.csv")
print("rows:", len(df))

coords = df[["lon", "lat"]].values
w = KNN.from_array(coords, k=K_NEIGHBORS, radius=EARTH_RADIUS_KM)
w.transform = "r"

severities = df["simulated_anomaly_severity_SIMULATED"].values.astype(float)

np.random.seed(RNG_SEED)
global_mi = Moran(severities, w, permutations=N_PERMUTATIONS)
local_mi = Moran_Local(severities, w, permutations=N_PERMUTATIONS, seed=RNG_SEED, n_jobs=1)

quadrants = []
sig_count = 0
for i in range(len(df)):
    significant = bool(local_mi.p_sim[i] < SIGNIFICANCE_ALPHA)
    q = int(local_mi.q[i]) if significant else 0
    if significant:
        sig_count += 1
    quadrants.append(q)
df["lisa_quadrant"] = quadrants
df["lisa_label"] = df["lisa_quadrant"].map(QUADRANT_LABELS)
df["local_moran_i"] = local_mi.Is
df["p_value"] = local_mi.p_sim

print(f"Global Moran's I = {global_mi.I:.4f}, p = {global_mi.p_sim:.4f}, z = {global_mi.z_sim:.3f}, E[I] = {global_mi.EI:.4f}")
print("Significant LISA towers:", sig_count, "/", len(df))
print(df["lisa_label"].value_counts())

out_dir = r"C:\Users\Iris\AppData\Local\Temp\claude\C--Users-Iris-Downloads-syncguard-agaif2026\e6c335d9-a68d-440f-a6f0-da4b60a9c2db\scratchpad\geoai_slide"

# --- LISA cluster map ---
fig, ax = plt.subplots(figsize=(9.2, 8.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
for q in [0, 2, 4, 1, 3]:  # draw non-significant first, hotspot/coldspot on top
    sub = df[df["lisa_quadrant"] == q]
    if len(sub) == 0:
        continue
    ax.scatter(sub["lon"], sub["lat"], s=70 if q != 0 else 40,
               c=QUADRANT_COLORS[q], edgecolors="#2a2a2a" if q != 0 else "none",
               linewidths=0.6, alpha=0.95 if q != 0 else 0.55,
               label=f"{QUADRANT_LABELS[q]} ({len(sub)})", zorder=3 if q != 0 else 2)
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)
ax.set_title("Local Moran's I (LISA) \u2014 136 real Kubu Raya / Pontianak towers", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.15)
for spine in ax.spines.values():
    spine.set_color("#888888")
plt.tight_layout()
plt.savefig(f"{out_dir}/lisa_cluster_map.png", facecolor="white")
plt.close(fig)

# --- Moran scatter plot ---
z = (severities - severities.mean()) / severities.std()
lag = w.sparse @ z
fig2, ax2 = plt.subplots(figsize=(8.2, 8.2), dpi=200)
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")
colors = [QUADRANT_COLORS[q] for q in quadrants]
ax2.scatter(z, lag, c=colors, s=55, edgecolors="#2a2a2a", linewidths=0.5, alpha=0.9, zorder=3)
ax2.axhline(0, color="#999999", linewidth=0.8)
ax2.axvline(0, color="#999999", linewidth=0.8)
m, b = np.polyfit(z, lag, 1)
xs = np.linspace(z.min(), z.max(), 100)
ax2.plot(xs, m * xs + b, color="#d1263b", linewidth=2.2, zorder=4,
          label=f"Moran's I = {global_mi.I:.3f}  (p = {global_mi.p_sim:.3f})")
ax2.set_xlabel("Standardized severity (z)", fontsize=11)
ax2.set_ylabel("Spatial lag of severity (KNN k=5)", fontsize=11)
ax2.set_title("Moran Scatter Plot", fontsize=13, fontweight="bold")
ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)
ax2.grid(True, alpha=0.15)
for spine in ax2.spines.values():
    spine.set_color("#888888")
plt.tight_layout()
plt.savefig(f"{out_dir}/moran_scatter.png", facecolor="white")
plt.close(fig2)

print("saved plots to", out_dir)
