"""Simulate industrial process time-series data with labeled anomalies."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AnomalyWindow:
    kind: str
    start: int
    end: int


def _select_window(n_points, min_len, max_len, rng, occupied):
    for _ in range(200):
        length = rng.integers(min_len, max_len + 1)
        if length > n_points:
            length = n_points
        if length >= n_points:
            occupied.append((0, n_points))
            return 0, n_points

        start = rng.integers(0, n_points - length)
        end = start + length

        overlaps = False
        for window in occupied:
            s = window[0]
            e = window[1]
            if not (end <= s or start >= e):
                overlaps = True
                break

        if not overlaps:
            occupied.append((start, end))
            return start, end

    # fallback if we somehow fail to place after many attempts
    start = 0
    end = min_len
    occupied.append((start, end))
    return start, end


def _base_signals(n_points: int, freq_min: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    minutes_per_day = 24 * 60 / freq_min
    time_idx = np.arange(n_points)

    daily = np.sin(2 * np.pi * time_idx / minutes_per_day)
    shift = np.sin(2 * np.pi * time_idx / (8 * 60 / freq_min))

    # baseline reactor behavior with slow drift + noise
    t_reactor = 250 + 6 * daily + 0.015 * time_idx + rng.normal(0, 0.8, n_points)
    p_reactor = 120 + 2.5 * daily + 0.01 * time_idx + rng.normal(0, 0.6, n_points)

    # feed tends to move more with operator shifts than daily cycles
    f_feed = 12 + 0.6 * shift + rng.normal(0, 0.15, n_points)

    # level oscillates slower due to inventory effects
    l_drum = 55 + 8 * np.sin(2 * np.pi * time_idx / (6 * 60 / freq_min) + 0.7)
    l_drum += rng.normal(0, 0.9, n_points)

    # valve position has some hunting / actuator noise
    v_valve = 60 + 7 * np.sin(2 * np.pi * time_idx / (10 * 60 / freq_min) + 1.2)
    v_valve += rng.normal(0, 1.2, n_points)

    return {
        "T_reactor_F": t_reactor,
        "P_reactor_psig": p_reactor,
        "F_feed_bbl_per_min": f_feed,
        "L_drum_pct": l_drum,
        "V_valve_pct": v_valve,
    }


def _apply_anomalies(signals, timestamps, rng, freq_min):
    n_points = len(timestamps)
    occupied = []
    windows = []

    def _duration(min_minutes, max_minutes):
        min_len = max(1, int(np.ceil(min_minutes / freq_min)))
        max_len = max(min_len, int(np.ceil(max_minutes / freq_min)))
        return min_len, max_len

    # sensor drift: slow, hard-to-detect failure
    drift_min, drift_max = _duration(6 * 60, 12 * 60)
    drift_start, drift_end = _select_window(n_points, drift_min, drift_max, rng, occupied)
    drift_tag = rng.choice(list(signals.keys()))
    drift_mag = rng.uniform(4.0, 10.0)
    signals[drift_tag][drift_start:drift_end] += np.linspace(0, drift_mag, drift_end - drift_start)
    windows.append(AnomalyWindow("sensor_drift", drift_start, drift_end))

    # step change: sudden offset (instrument swap, operator change, etc.)
    step_min, step_max = _duration(2 * 60, 6 * 60)
    step_start, step_end = _select_window(n_points, step_min, step_max, rng, occupied)
    step_tag = rng.choice(list(signals.keys()))
    step_mag = rng.uniform(-8.0, 8.0)
    signals[step_tag][step_start:step_end] += step_mag
    windows.append(AnomalyWindow("step_change", step_start, step_end))

    # spike: short transient noise
    spike_min, spike_max = _duration(5, 20)
    spike_start, spike_end = _select_window(n_points, spike_min, spike_max, rng, occupied)
    spike_tag = rng.choice(list(signals.keys()))
    spike_mag = rng.uniform(-15.0, 15.0)
    signals[spike_tag][spike_start:spike_end] += spike_mag
    windows.append(AnomalyWindow("spike", spike_start, spike_end))

    # actuator sticking: one fault causing multiple downstream effects
    stick_min, stick_max = _duration(60, 4 * 60)
    stick_start, stick_end = _select_window(n_points, stick_min, stick_max, rng, occupied)

    valve_series = signals["V_valve_pct"]
    stuck_value = float(valve_series[stick_start])
    valve_series[stick_start:stick_end] = stuck_value

    signals["F_feed_bbl_per_min"][stick_start:stick_end] -= rng.uniform(0.8, 1.6)
    signals["P_reactor_psig"][stick_start:stick_end] += rng.uniform(3.0, 6.0)

    windows.append(AnomalyWindow("actuator_sticking", stick_start, stick_end))

    # TODO: consider allowing overlapping faults for compound-failure studies
    return windows


def generate_dataset(output_csv, n_hours=14 * 24, freq_min=1, seed=42):
    rng = np.random.default_rng(seed)
    n_points = int(n_hours * 60 / freq_min)

    timestamps = pd.date_range("2025-01-01", periods=n_points, freq=f"{freq_min}min")
    signals = _base_signals(n_points, freq_min, rng)
    windows = _apply_anomalies(signals, timestamps, rng, freq_min)

    df = pd.DataFrame({"timestamp": timestamps, **signals})

    df["L_drum_pct"] = df["L_drum_pct"].clip(0, 100)
    df["V_valve_pct"] = df["V_valve_pct"].clip(0, 100)

    df["anomaly_flag"] = 0
    df["anomaly_type"] = "none"
    df["anomaly_start"] = pd.NaT
    df["anomaly_end"] = pd.NaT

    for window in windows:
        df.loc[window.start:window.end - 1, "anomaly_flag"] = 1
        df.loc[window.start:window.end - 1, "anomaly_type"] = window.kind
        df.loc[window.start:window.end - 1, "anomaly_start"] = timestamps[window.start]
        df.loc[window.start:window.end - 1, "anomaly_end"] = timestamps[window.end - 1]

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv, index=False)
    return df
