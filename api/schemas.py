"""Pydantic request/response schemas for the SyncGuard scoring API.

TelemetryInput mirrors the 23-column feature schema produced by extract_features.py /
consumed by train_baseline_model.py -- see EXCLUDE_COLS there for what's deliberately not
part of the feature set (scenario metadata a deployed detector wouldn't have as ground
truth). Fields are Optional because several are legitimately NaN for some receiver states
(e.g. pos_dev_m is only meaningful for stationary receivers -- see dataset_notes.md) and the
trained pipeline's SimpleImputer(strategy="median") handles missing values the same way it
does for the training data.
"""
from typing import Optional
from pydantic import BaseModel, Field


class TelemetryInput(BaseModel):
    # --- receiver PVT solution (nav_pvt.csv-derived) ---
    fixType: Optional[float] = Field(default=None, description="u-blox fix type (0=no fix .. 3=3D, 4=GNSS+dead reckoning)")
    gSpeed: Optional[float] = Field(default=None, description="Ground speed, m/s")
    hAcc: Optional[float] = Field(default=None, ge=0, description="Horizontal accuracy estimate, m")
    vAcc: Optional[float] = Field(default=None, ge=0, description="Vertical accuracy estimate, m")
    sAcc: Optional[float] = Field(default=None, ge=0, description="Speed accuracy estimate, m/s")
    headAcc: Optional[float] = Field(default=None, ge=0, description="Heading accuracy estimate, deg")
    pDOP: Optional[float] = Field(default=None, ge=0, description="Position dilution of precision")
    numSV: Optional[float] = Field(default=None, ge=0, le=64, description="Satellites used in solution")
    velN: Optional[float] = Field(default=None, description="Velocity, North, m/s")
    velE: Optional[float] = Field(default=None, description="Velocity, East, m/s")
    velD: Optional[float] = Field(default=None, description="Velocity, Down, m/s")
    pos_dev_m: Optional[float] = Field(default=None, ge=0, description="Deviation from the receiver's own first-60s reference fix, m (stationary receivers only)")

    # --- per-satellite RINEX-derived aggregates (rinex.csv) ---
    n_sats_l1: Optional[float] = Field(default=None, ge=0, le=64, description="Satellites tracked on L1 this epoch")
    snr_l1_mean: Optional[float] = Field(default=None, ge=0, le=60, description="Mean L1 C/N0, dB-Hz")
    snr_l1_std: Optional[float] = Field(default=None, ge=0, description="Std dev of L1 C/N0 across tracked satellites")
    snr_l1_min: Optional[float] = Field(default=None, ge=0, le=60, description="Minimum L1 C/N0 across tracked satellites")
    doppler_l1_mean: Optional[float] = Field(default=None, description="Mean L1 Doppler shift, Hz")
    doppler_l1_std: Optional[float] = Field(default=None, ge=0, description="Std dev of L1 Doppler shift")
    pr_doppler_residual_mean: Optional[float] = Field(default=None, description="Mean code-Doppler pseudorange-rate consistency residual, m/s")
    pr_doppler_residual_std: Optional[float] = Field(default=None, ge=0, description="Std dev of the code-Doppler residual")

    # --- RF monitor (mon_rf.csv) ---
    jam_ind_mean: Optional[float] = Field(default=None, ge=0, description="u-blox jamming indicator")
    agc_cnt_mean: Optional[float] = Field(default=None, ge=0, description="AGC count")
    noise_per_ms_mean: Optional[float] = Field(default=None, ge=0, description="Noise floor per millisecond")

    # --- optional context: not fed to the model, only used for persistence/spatial correlation ---
    receiver_id: Optional[str] = Field(default=None, description="Caller-supplied receiver/site identifier, for logging only")
    tower_site_id: Optional[str] = Field(default=None, description="Real Telkomsel site_id (see spatial_raw tower CSV) if the caller knows which tower this reading is from -- enables live spatial correlation for this event. Omit if unknown.")


class TopFeature(BaseModel):
    feature: str
    shap_value: float = Field(description="Signed SHAP contribution to the attack-class probability -- positive pushes toward attack, negative toward clean")
    feature_value: Optional[float] = Field(default=None, description="This row's raw (imputed) value for the feature")
    direction: str = Field(description="'toward attack' or 'toward clean'")


class ScoreResponse(BaseModel):
    probability: float = Field(description="P(attack) from the trained RandomForest, in [0, 1]")
    severity: float = Field(description="Probability normalized against the model's own held-out floor/ceiling (same methodology as spatial_layer_notes.md), in [0, 1]")
    predicted_label: str = Field(description="'attack' or 'clean', using the model's tuned decision threshold")
    decision_threshold: float
    model_version: str
    event_id: Optional[int] = Field(default=None, description="Row id in the persisted scored_events table")
    tower: Optional[dict] = Field(default=None, description="The real tower this event was attributed to, if any")
    correlation: Optional[dict] = Field(default=None, description="Live distance-weighted correlation against real neighboring towers -- see api/spatial.py")
    top_features: list[TopFeature] = Field(default_factory=list, description="Top SHAP-ranked features for this prediction -- see SHAP_EXPLAINABILITY.md. Always computed for /score.")


class ExplainResponse(BaseModel):
    event_id: int
    top_features: list[TopFeature]
    cached: bool = Field(description="True if this explanation was already stored (e.g. computed live during replay, or a prior on-demand call); False if it was just computed by this request")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    towers_loaded: int = 0
    replay_running: bool = False


class LisaTower(BaseModel):
    tower_key: str
    site_id: str
    site_name: str
    lat: float
    lon: float
    severity: float = Field(description="Real predict_proba-derived severity, most recent scored event for this tower")
    local_moran_i: float
    p_value: float = Field(description="Permutation-based pseudo p-value (999 permutations, seeded)")
    significant: bool = Field(description="p_value < 0.05")
    lisa_quadrant: int = Field(description="0=not significant, 1=High-High hotspot, 2=Low-High outlier, 3=Low-Low coldspot, 4=High-Low outlier")
    lisa_label: str


class AutocorrelationResponse(BaseModel):
    computable: bool = Field(description="False if fewer than min_required towers have a scored event yet, or severities have zero variance")
    n_towers_scored: int
    n_towers_total: int
    min_required: int
    k_neighbors: Optional[int] = Field(default=None, description="k used for the k-nearest-neighbors spatial weights, see SPATIAL_STATISTICS.md")
    global_moran_i: Optional[float] = Field(default=None, description="Global Moran's I -- positive/near +1 = clustered, near 0 = random, negative = dispersed")
    global_p_value: Optional[float] = None
    global_z_score: Optional[float] = None
    global_expected_i: Optional[float] = Field(default=None, description="Expected I under spatial randomness, ~ -1/(n-1)")
    reason: Optional[str] = Field(default=None, description="Why computable is False, if it is")
    per_tower: list[LisaTower] = Field(default_factory=list, description="Local Moran's I (LISA) classification per scored tower")
