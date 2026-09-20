.PHONY: setup setup-dev test train train-baseline train-improved serve replay-demo ingest-demo package verify-package clean

PY ?= python

setup:
	$(PY) -m pip install -r requirements.txt -r requirements-api.txt

# Adds the test-only dependencies (pytest, httpx). Not needed to run the service.
setup-dev: setup
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest

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

# Convenience: send a sample POST /ingest batch to the running API and print the alerts.
ingest-demo:
	$(PY) examples/ingest_client.py

# Build syncguard_source.zip from `git ls-files` (tracked files only, so .env, *.db and
# data/external/ cannot get in) and then verify the result against the repo.
package:
	$(PY) make_source_zip.py

verify-package:
	$(PY) make_source_zip.py --verify

clean:
	rm -rf models data/*.db*
