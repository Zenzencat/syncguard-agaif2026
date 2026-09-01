.PHONY: setup train train-baseline train-improved serve replay-demo clean

PY ?= python

setup:
	$(PY) -m pip install -r requirements.txt -r requirements-api.txt

# Produces models/model.joblib (the artifact api/main.py serves) plus
# models/model_baseline.joblib and both *_report.md files for comparison.
train: train-baseline train-improved

train-baseline:
	$(PY) train_baseline_model.py

train-improved:
	$(PY) train_improved_model.py

serve:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Convenience: hit the running API and start a fast demo replay.
replay-demo:
	curl -s -X POST "http://localhost:8000/replay/start?speed=25"

clean:
	rm -rf models data/*.db*
