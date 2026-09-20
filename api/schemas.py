"""Pydantic request/response schemas for the SyncGuard scoring API.

TelemetryInput mirrors the 23-column feature schema produced by extract_features.py /
consumed by train_baseline_model.py -- see EXCLUDE_COLS there for what's deliberately not
part of the feature set (scenario metadata a deployed detector wouldn't have as ground
truth). Fields are Optional because several are legitimately NaN for some receiver states
(e.g. pos_dev_m is only meaningful for stationary receivers -- see dataset_notes.md) and the
trained pipeline's SimpleImputer(strategy="median") handles missing values the same way it
does for the training data.
"""
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, field_validator

# Cap on a single POST /ingest batch. Not a measured throughput limit -- it is a deliberate
# request-size bound so one call can't pin the single-threaded scoring path indefinitely.
# Phase 4 measures what this path actually costs; until then this number is a guardrail, not
# a benchmark result.
MAX_INGEST_BATCH = 500


class FeatureVector(BaseModel):
    """The 23 model input features, exactly as named in the trained artifact's feature_cols.

    Shared base for TelemetryInput (POST /score) and IngestObservation (POST /ingest) so both
    entry points accept literally the same feature schema -- adding the ingestion path did not
    introduce a second, drifting definition of what a feature vector is.
    """
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


class TelemetryInput(FeatureVector):
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


# ---------------------------------------------------------------------------
# POST /ingest -- batch ingestion of per-tower observation windows.
# Contract (what a caller must compute edge-side, and from which raw receiver
# observables): INGESTION_CONTRACT.md.
# ---------------------------------------------------------------------------

class IngestObservation(FeatureVector):
    """One already-extracted observation window from one tower.

    Carries the same 23 features as TelemetryInput (inherited from FeatureVector -- one
    definition, not two), plus the two things ingestion needs that ad-hoc /score does not:
    which tower it came from, and when it was observed. Feature extraction itself happens
    caller-side; INGESTION_CONTRACT.md documents the exact raw-observable -> feature mapping
    so an edge agent can reproduce extract_features.py's transformations without this service
    having to accept raw RINEX/UBX.
    """
    tower_id: str = Field(description="Must be a known tower: a tower_key from GET /towers (preferred, unique), or a raw site_id. Unknown -> 422.")
    timestamp: datetime = Field(description="Observation window time, UTC. A naive timestamp is interpreted as UTC; an offset-aware one is converted to UTC.")

    @field_validator("timestamp")
    @classmethod
    def _to_utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


class IngestBatch(BaseModel):
    batch_id: Optional[str] = Field(default=None, description="Caller-supplied batch identifier, stored on each resulting event row for traceability. Not used for deduplication -- that is keyed on (tower_id, timestamp); see INGESTION_CONTRACT.md.")
    observations: list[IngestObservation] = Field(min_length=1, max_length=MAX_INGEST_BATCH, description=f"1..{MAX_INGEST_BATCH} observation windows. An empty list is a 422.")


class IngestResult(BaseModel):
    tower_id: str = Field(description="The tower_key this observation resolved to (may differ from the submitted tower_id if a raw site_id was submitted)")
    timestamp: datetime
    event_id: Optional[int] = Field(default=None, description="Row id in scored_events. For a duplicate, the id of the ORIGINAL event -- no new row was written.")
    probability: Optional[float] = None
    severity: Optional[float] = None
    predicted_label: Optional[str] = None
    alert_state: Optional[str] = Field(default=None, description="Per-tower hysteresis state after this observation ('normal'/'alerting'). None for duplicates, and unchanged-from-previous for out-of-order observations, which are not fed to hysteresis.")
    correlation_score: Optional[float] = None
    duplicate: bool = Field(default=False, description="True if (tower_id, timestamp) was already ingested -- not re-scored, not re-persisted; event_id points at the original.")
    out_of_order: bool = Field(default=False, description="True if this observation's timestamp is older than the newest already ingested for this tower. Still scored and persisted; deliberately NOT fed to per-tower hysteresis (see INGESTION_CONTRACT.md).")
    explained: bool = Field(default=False, description="True if SHAP was computed inline. Large batches skip it for latency; GET /events/{id}/explain computes it lazily on demand.")


class IngestResponse(BaseModel):
    batch_id: Optional[str] = None
    received: int = Field(description="Observations in the submitted batch")
    scored: int = Field(description="Observations actually scored and persisted (received minus duplicates)")
    duplicates: int
    out_of_order: int = Field(description="Of the scored observations, how many were older than data already accepted for that tower BEFORE this batch")
    reordered_in_batch: int = Field(description="How many observations were submitted out of ascending timestamp order relative to an earlier observation for the same tower in the same batch. Not an error: the batch is sorted by timestamp before scoring so per-tower hysteresis sees a monotonic sequence. Reported so a caller can see its feed is shuffled.")
    alerts: int = Field(description="Of the scored observations, how many left their tower in alert_state='alerting'")
    live_explain: bool = Field(description="Whether SHAP was computed inline for this batch (batch size <= the inline-explain limit)")
    results: list[IngestResult]
