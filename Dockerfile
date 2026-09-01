# SyncGuard API image: trains the model at build time (from the processed parquet + tower
# CSV already baked into this image -- both are small, pre-extracted feature/coordinate
# tables, not the 1.4GB raw Jammertest logs) so the container starts ready to serve, with no
# retraining step and no model artifact committed to git. See README.md "Containerization".
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

COPY extract_features.py sanity_check.py train_baseline_model.py train_improved_model.py ./
COPY api ./api
COPY processed ./processed
COPY spatial_raw ./spatial_raw
COPY syncguard_interactive_summary.html ./

# Baked into the image at build time -- models/model.joblib (+ model_baseline.joblib) and
# both *_report.md files are produced here, not committed to git (see .gitignore).
RUN python train_baseline_model.py && python train_improved_model.py

RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
