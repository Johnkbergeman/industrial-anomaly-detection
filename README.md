# industrial-anomaly-detection

Note AI was used to aid in coding process.
Industrial anomaly detection demo built around process time-series (think PI tags on a reactor loop).
The goal is to show how a simple, interpretable baseline and a multivariate ML model can work side
by side in a way a process engineer would actually trust.

## Problem statement (industrial framing)
Operators and control engineers live with noisy signals, sensor drift, actuator sticking, and
occasional step changes. Catching these early reduces off-spec product, energy waste, and trips.
This repo simulates those behaviors and evaluates detection using both pointwise and event-level
metrics (window recall, time-to-detect, false alert duration).

## Why baseline + ML together
- Rolling z-score is easy to explain and quick to audit.
- Isolation Forest captures multivariate relationships (cross-tag effects).
- Running both gives a check-and-balance: baseline for transparency, ML for coverage.

## What is simulated vs. real
Everything here is simulated. The signals are shaped to resemble a typical process unit
(daily cycles, operator shifts, actuator noise), and the anomalies are injected with labels.
In a real plant, labels are scarce and messy, so this code focuses on structure and evaluation
logic that would still be used in production.

## How this fits a historian or digital twin
In practice, the input would be a historian query or a digital twin feed:
- Historian: pull a time range of tags, run batch scoring, save events back to a dashboard.
- Digital twin: score incrementally on a rolling window, alert when scores cross thresholds.
The pipeline is structured for both batch and incremental modes without adding streaming tech.

## Tradeoffs (false positives vs missed events)
There is no perfect threshold. "Conservative" reduces false alerts but may miss early drift.
"Sensitive" catches earlier but can be noisy. The `--mode` flag makes that trade explicit.

## Quick start
```bash
python -m pip install -r requirements.txt
python -m src.simulate_data --out data/simulated_process_data.csv
python -m src.run_pipeline --data data/simulated_process_data.csv
```

## Modes and thresholds
```bash
# Conservative mode: fewer false alerts
python -m src.run_pipeline --mode conservative

# Sensitive mode: earlier detection
python -m src.run_pipeline --mode sensitive

# Custom thresholds
python -m src.run_pipeline --mode custom --z-thresh 3.2 --contamination 0.02
```

## Batch vs incremental scoring
```bash
# Batch (default)
python -m src.run_pipeline --run-mode batch

# Streaming-style scoring on a rolling window
python -m src.run_pipeline --run-mode incremental --stream-warmup 480
```

## Outputs
- Pointwise metrics (precision, recall, F1)
- Event metrics (event recall, time-to-detect, false alert duration)
- Root-cause hints per detected event
- Plots saved to `artifacts/`

## File map
- `src/simulate_data.py` - generates labeled process data
- `src/baselines.py` - rolling z-score detector
- `src/features.py` - feature engineering for ML model
- `src/models.py` - Isolation Forest wrappers
- `src/metrics.py` - pointwise + event-level evaluation
- `src/visualize.py` - plots with anomaly windows + scores
- `src/run_pipeline.py` - orchestration and CLI

