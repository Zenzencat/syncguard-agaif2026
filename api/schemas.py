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
from typing import Literal, Optional
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


class InputWarning(BaseModel):
    """One feature outside the training baseline's widened range.

    Advisory only. The value was scored exactly as it would have been without this check --
    out-of-range input may be the very anomaly the detector exists to catch, so rejecting it
    would suppress detections. See api/plausibility.py.
    """
    feature: str
    value: Optional[float] = Field(default=None, description="The submitted value, or null if it was not a finite number")
    direction: str = Field(description="'below_baseline_range' | 'above_baseline_range' | 'not_finite'")
    bound_low: float
    bound_high: float
    baseline_median: float
    message: str


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
    model_tag: Optional[str] = Field(default=None, description="Model version tag: version string + model file hash + threshold + feature-list hash. Identifies exactly which artifact produced this score -- see GET /health.")
    input_warnings: list[InputWarning] = Field(default_factory=list, description="Features outside the training baseline range. NEVER rejects input -- the score above was produced anyway. An empty list does NOT mean the input is correct; see api/plausibility.py for what this check does and does not catch.")


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

    # --- added in Phase 3 (ops hardening) ---
    model_tag: Optional[str] = Field(default=None, description="version string + model file hash + threshold + feature-list hash. Changes whenever anything affecting a prediction changes.")
    model_info: Optional[dict] = Field(default=None, description="Full model version tag: file name, sha256, threshold, feature count, feature-list hash, the 23 feature names, and the severity floor/ceiling.")
    auth_required: bool = Field(default=False, description="True when SYNCGUARD_API_KEY is set and non-health routes require a key. /health itself is always exempt so probes need no secret.")
    auth_key_is_weak: Optional[bool] = Field(default=None, description="True when a configured key is shorter than the recommended minimum. Never reveals the key or its length.")
    baseline_loaded: bool = Field(default=False, description="Whether the training-distribution baseline (api/feature_baseline.json) is available for input warnings and /drift.")
    baseline_generated_at: Optional[str] = None


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
    input_warnings: list[InputWarning] = Field(default_factory=list, description="Features outside the training baseline range for THIS observation. Advisory only -- it was scored anyway. See api/plausibility.py.")


class IngestResponse(BaseModel):
    batch_id: Optional[str] = None
    received: int = Field(description="Observations in the submitted batch")
    scored: int = Field(description="Observations actually scored and persisted (received minus duplicates)")
    duplicates: int
    out_of_order: int = Field(description="Of the scored observations, how many were older than data already accepted for that tower BEFORE this batch")
    reordered_in_batch: int = Field(description="How many observations were submitted out of ascending timestamp order relative to an earlier observation for the same tower in the same batch. Not an error: the batch is sorted by timestamp before scoring so per-tower hysteresis sees a monotonic sequence. Reported so a caller can see its feed is shuffled.")
    alerts: int = Field(description="Of the scored observations, how many left their tower in alert_state='alerting'")
    live_explain: bool = Field(description="Whether SHAP was computed inline for this batch (batch size <= the inline-explain limit)")
    input_warning_count: int = Field(default=0, description="Total input plausibility warnings across the batch. Advisory only -- nothing was rejected because of them.")
    model_tag: Optional[str] = Field(default=None, description="Model version tag that scored this batch -- see GET /health.")
    results: list[IngestResult]


# ---------------------------------------------------------------------------
# Analyst confirm/dismiss -- POST /events/{id}/feedback, GET /feedback/*.
# Labels are STORED ONLY. Nothing in this repo retrains on them, and no closed
# loop exists. See FEEDBACK_LOOP.md before claiming otherwise anywhere.
# ---------------------------------------------------------------------------

FeedbackLabel = Literal["confirmed", "dismissed"]


class FeedbackRequest(BaseModel):
    label: FeedbackLabel = Field(description="'confirmed' = the analyst judges this a real event; 'dismissed' = a false alarm. Anything else is a 422.")
    note: Optional[str] = Field(default=None, max_length=2000, description="Free-text rationale. Stored verbatim, never parsed.")
    analyst: Optional[str] = Field(default=None, max_length=120, description="Who labeled it. Free text -- there is no authentication behind this in Phase 2, so it is an attribution hint, not an identity. Phase 3 adds API-key auth, which still does not identify an individual analyst.")


class FeedbackRecord(BaseModel):
    event_id: int
    label: Optional[str] = Field(default=None, description="Current label, or null if this event has never been labeled")
    note: Optional[str] = None
    analyst: Optional[str] = None
    created_at: Optional[str] = Field(default=None, description="When this event was FIRST labeled")
    updated_at: Optional[str] = Field(default=None, description="When the current label was set")
    revision: int = Field(default=0, description="0 = never labeled, 1 = labeled once, N = relabeled N-1 times")
    previous_label: Optional[str] = Field(default=None, description="The label this submission replaced, if any. Only meaningful on a POST response.")


class FeedbackSummary(BaseModel):
    total_events: int
    total_labeled: int
    unlabeled: int
    confirmed: int
    dismissed: int
    relabeled_events: int = Field(description="Events whose label has been changed at least once")
    total_submissions: int = Field(description="Rows in the append-only audit log, including superseded labels")
    distinct_analysts: int
    labeled_by_source: dict = Field(default_factory=dict, description="Labeled-event counts by event source ('replay' | 'ingest' | 'api')")

    predicted_attack_labeled: int = Field(description="Labeled events whose predicted_label was 'attack' -- the denominator below")
    predicted_attack_confirmed: int
    predicted_attack_precision: Optional[float] = Field(default=None, description="confirmed / labeled, over labeled events predicted 'attack'. null when the denominator is 0.")

    hysteresis_alerting_labeled: int = Field(description="Labeled events whose alert_state was 'alerting' -- the denominator below")
    hysteresis_alerting_confirmed: int
    hysteresis_alert_precision: Optional[float] = Field(default=None, description="confirmed / labeled, over labeled events in hysteresis state 'alerting'. null when the denominator is 0.")

    caveat: str = Field(description="Plain-language statement of what these precision numbers are not")


class IncidentTower(BaseModel):
    site_id: Optional[str] = None
    site_name: Optional[str] = None


class IncidentRecord(BaseModel):
    incident_id: str = Field(description="INC-<first event id in the group>")
    event_ids: list[int]
    top_event_id: int = Field(description="Id of the incident's most-severe (peak-severity) event -- use this for /events/{id}/explain and for incident-level Confirm/Dismiss")
    towers: list[IncidentTower]
    tower_attribution_simulated: bool = Field(description="True if any member event came from replay (SIMULATED tower attribution)")
    max_pop_2km: Optional[float] = Field(default=None, description="Highest single tower's ESTIMATED people within 2km (exposure proxy) among this incident's towers -- NEVER a sum across towers")
    start_time: str
    end_time: str
    clock: str = Field(description="Which clock start_time/end_time/duration_seconds use, and its caveat")
    duration_seconds: float
    severity: float = Field(description="Max severity across the incident's events")
    attack_type: Optional[str] = Field(default=None, description="Ground truth, only populated by replay -- not something a real detector would know")
    status: str = Field(description="'New' | 'Confirmed' | 'Dismissed', derived from analyst feedback on any member event")
    event_count: int
    labeled_event_count: int = Field(description="Of event_count events, how many already carry an analyst label -- an incident-level Confirm/Dismiss only labels the peak-severity event, not every member")
    grouping_note: str


class IncidentsResponse(BaseModel):
    incidents: list[IncidentRecord]
    window_seconds: int = Field(description="Time-chaining window used to group events into incidents (INCIDENT_WINDOW_SECONDS; defaults to, and independently overridable from, the live correlation engine's window)")
    distance_km: float = Field(description="Distance-chaining radius used to group events into incidents (INCIDENT_DISTANCE_KM)")
    method_note: str
