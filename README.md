# industrial-anomaly-detection

Industrial time-series anomaly detection + root-cause on simulated process data (PI-style tags).

## Project goal
Build a lightweight pipeline that simulates process data, detects anomalies, and surfaces likely root
causes from tag relationships.

## Pipeline overview
1. Generate or load process tag time-series.
2. Engineer features and train anomaly detector.
3. Score anomalies and attribute root causes.
4. Export artifacts and summary plots.

## How to run
```bash
python -m pip install -r requirements.txt
python -m src.run_pipeline
```

## Generate data
```bash
python -m src.simulate_data --out data/simulated_process_data.csv
```
