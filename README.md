# industrial-anomaly-detection

Industrial anomaly detection prototype for process time-series (reactor-style tags).  
It runs two detectors in parallel:
- A transparent rolling z-score baseline
- A multivariate Isolation Forest model

The pipeline reports both pointwise and event-level metrics, plus simple tag-level attribution.

## Current status
- Core pipeline is implemented in `src/run_pipeline.py`.
- Data simulation is available as a CLI in `src/simulate_data.py`.
- Plotting and metrics are wired end-to-end.
- Tests are not implemented yet (`tests/` is empty).

## Workflow
1. Create a local environment and install dependencies.
2. Generate synthetic labeled process data.
3. Run the pipeline in batch mode to get baseline + Isolation Forest metrics and plots.
4. Tune thresholds (`--mode`, `--z-thresh`, `--contamination`) based on false alerts vs missed events.
5. Optionally run incremental mode to mimic streaming behavior.

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Generate data:
```bash
python -m src.simulate_data --out data/simulated_process_data.csv
```

Run the full pipeline:
```bash
python -m src.run_pipeline --data data/simulated_process_data.csv
```

## Run modes and tuning
```bash
# Conservative: fewer false alerts
python -m src.run_pipeline --data data/simulated_process_data.csv --mode conservative

# Sensitive: earlier detection, noisier alerts
python -m src.run_pipeline --data data/simulated_process_data.csv --mode sensitive

# Custom thresholds
python -m src.run_pipeline --data data/simulated_process_data.csv --mode custom --z-thresh 3.2 --contamination 0.02

# Incremental scoring (streaming-style simulation)
python -m src.run_pipeline --data data/simulated_process_data.csv --run-mode incremental --stream-warmup 480

# Compare multiple feature models in one run
python -m src.run_pipeline --data data/simulated_process_data.csv --feature-models isolation_forest,oneclass_svm,local_outlier_factor
```

## Outputs
- Console summaries: precision, recall, F1, event recall, time-to-detect, false alert minutes
- Event-level root-cause hints (top contributing tags)
- Plots saved to `artifacts/`
- Model comparison tables:
  - `artifacts/model_comparison_batch.csv`
  - `artifacts/model_comparison_streaming.csv`

## What to do next
1. Add smoke tests for dataset generation shape/required columns, pipeline run in `batch` mode, and metrics invariants (no divide-by-zero regressions).
2. Add one fixed validation dataset in `data/` for deterministic regression checks.
3. Gate merges on tests in CI.

## File map
- `src/simulate_data.py`: synthetic process + anomaly generation
- `src/baselines.py`: rolling z-score baseline logic
- `src/features.py`: feature engineering for Isolation Forest
- `src/models.py`: model train/predict wrappers
- `src/metrics.py`: pointwise and event metrics
- `src/attribution.py`: tag ranking for suspected root cause
- `src/visualize.py`: anomaly overlay plots
- `src/run_pipeline.py`: orchestration + CLI

