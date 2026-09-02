"""Loads the trained pipeline artifact and scores telemetry.

Loads once at process startup (see main.py's lifespan) -- never retrains, never re-fits
in-memory. Prefers models/model.joblib (the tuned artifact from train_improved_model.py,
see ROBUSTNESS_NOTES.md); falls back to models/model_baseline.joblib if that's all that's
been trained yet, with a loud warning, so `make train-baseline` alone is still enough to get
the service running.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

TOP_FEATURES_N = 5

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
FALLBACK_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_baseline.joblib"


class ModelNotFoundError(RuntimeError):
    pass


class ModelService:
    def __init__(self, model_path: Path | None = None):
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            if path == DEFAULT_MODEL_PATH and FALLBACK_MODEL_PATH.exists():
                print(f"[model_service] {path} not found; falling back to {FALLBACK_MODEL_PATH} "
                      f"(run `make train` to build the tuned model instead).")
                path = FALLBACK_MODEL_PATH
            else:
                raise ModelNotFoundError(
                    f"No trained model at {path}. Run `make train` (or `python "
                    f"train_baseline_model.py && python train_improved_model.py`) first."
                )
        artifact = joblib.load(path)
        self.model_path = path
        self.pipeline = artifact["pipeline"]
        # RandomForestClassifier.predict_proba() with n_jobs>1 aggregates per-tree votes in
        # parallel; floating-point summation isn't associative, so two separate calls (even
        # on the identical fitted model) aren't guaranteed bit-identical for rows whose vote
        # fraction sits extremely close to a threshold -- confirmed directly: reloading
        # models/model_baseline.joblib and re-scoring the held-out set flipped ~6 of 14077
        # rows relative to the training run's own report (see ROBUSTNESS_NOTES.md). Forcing
        # single-threaded inference makes /score deterministic run-to-run, which matters more
        # here than the (negligible, single-row-scoring) speed cost of n_jobs=-1.
        rf = self.pipeline.named_steps.get("rf") if hasattr(self.pipeline, "named_steps") else None
        if rf is not None:
            rf.n_jobs = 1
        self.feature_cols: list[str] = artifact["feature_cols"]
        self.decision_threshold: float = float(artifact.get("decision_threshold", 0.5))
        self.severity_floor: float = float(artifact["severity_floor"])
        self.severity_ceiling: float = float(artifact["severity_ceiling"])
        self.model_version: str = artifact.get("model_version", "unknown")

        # SHAP explainability (see SHAP_EXPLAINABILITY.md). TreeExplainer needs the raw
        # RandomForestClassifier, not the sklearn Pipeline -- it doesn't understand Pipeline
        # objects, so imputation has to happen manually before both predict_proba and
        # shap_values calls (score()/explain() below both do this the same way).
        # feature_perturbation='tree_path_dependent' is the exact Tree SHAP algorithm (no
        # sampling, no background dataset needed, unlike 'interventional') -- confirmed
        # bit-identical across repeated calls and independent of rf.n_jobs (SHAP's own tree
        # traversal, not routed through predict_proba), so no seed is needed for determinism.
        self._imputer = self.pipeline.named_steps["impute"]
        self._rf = rf
        self._explainer = shap.TreeExplainer(self._rf, feature_perturbation="tree_path_dependent")

    def score(self, features: dict) -> dict:
        row = {c: features.get(c) for c in self.feature_cols}
        X = pd.DataFrame([row], columns=self.feature_cols)
        proba = float(self.pipeline.predict_proba(X)[0, 1])
        predicted_label = "attack" if proba >= self.decision_threshold else "clean"

        span = max(self.severity_ceiling - self.severity_floor, 1e-9)
        severity = float(np.clip((proba - self.severity_floor) / span, 0.0, 1.0))

        return {
            "probability": proba,
            "severity": severity,
            "predicted_label": predicted_label,
            "decision_threshold": self.decision_threshold,
            "model_version": self.model_version,
        }

    def explain(self, features: dict, top_n: int = TOP_FEATURES_N) -> list[dict]:
        """Per-prediction SHAP explanation: the top `top_n` features (by |SHAP value|)
        driving this specific row's attack-class probability, not the full feature vector
        (too noisy for a UI -- see SHAP_EXPLAINABILITY.md). ~45-55ms per call (exact Tree
        SHAP over 300 trees has no faster mode without sacrificing exactness, which was a
        deliberate choice -- see SHAP_EXPLAINABILITY.md's speed/design tradeoff section) --
        callers on a latency or throughput budget (the replay loop above a speed threshold)
        should skip this and let a viewer request it lazily instead of calling it inline.
        """
        row = {c: features.get(c) for c in self.feature_cols}
        X = pd.DataFrame([row], columns=self.feature_cols)
        Xi = self._imputer.transform(X)

        # shap_values shape here is (1, n_features, 2) -- class index 1 is "attack", verified
        # against predict_proba via the additivity property (sum(shap) + expected_value[1] ==
        # predict_proba) during development; see SHAP_EXPLAINABILITY.md.
        sv = self._explainer.shap_values(Xi, check_additivity=False)[0, :, 1]

        ranked = sorted(zip(self.feature_cols, sv, Xi[0]), key=lambda t: -abs(t[1]))[:top_n]
        return [
            {
                "feature": feat,
                "shap_value": float(val),
                "feature_value": float(raw_val) if raw_val is not None and not np.isnan(raw_val) else None,
                "direction": "toward attack" if val > 0 else "toward clean",
            }
            for feat, val, raw_val in ranked
        ]
