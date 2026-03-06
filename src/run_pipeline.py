"""Entry point for the pipeline."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .attribution import (
    rank_tags_from_feature_window,
    rank_tags_from_features,
    rank_tags_from_zscores,
)
from .baselines import combine_flags, rolling_zscore, rolling_zscore_detector
from .features import SENSOR_COLUMNS, build_feature_frame
from .metrics import event_metrics, find_windows, pointwise_metrics
from .models import SUPPORTED_MODELS, predict_detector, train_detector
from .visualize import plot_overview


REQUIRED_COLUMNS = [
    "timestamp",
    "T_reactor_F",
    "P_reactor_psig",
    "F_feed_bbl_per_min",
    "L_drum_pct",
    "V_valve_pct",
    "anomaly_flag",
]


def _validate_data(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Found missing values in required columns.")

    # Percent columns should remain within physical limits.
    if not df["L_drum_pct"].between(0, 100).all():
        raise ValueError("L_drum_pct has values outside 0-100.")
    if not df["V_valve_pct"].between(0, 100).all():
        raise ValueError("V_valve_pct has values outside 0-100.")


MODE_PRESETS = {
    "conservative": {"z_thresh": 4.0, "contamination": 0.01},
    "sensitive": {"z_thresh": 2.5, "contamination": 0.05},
}


def _parse_feature_models(raw_value: str) -> list[str]:
    model_names = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    if not model_names:
        raise argparse.ArgumentTypeError("At least one feature model must be provided.")

    unknown = [name for name in model_names if name not in SUPPORTED_MODELS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unsupported feature model(s): {unknown}. Supported: {SUPPORTED_MODELS}"
        )
    return model_names


def _model_label(model_name: str) -> str:
    labels = {
        "isolation_forest": "Isolation Forest",
        "oneclass_svm": "One-Class SVM",
        "local_outlier_factor": "Local Outlier Factor",
    }
    return labels.get(model_name, model_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline anomaly detection pipeline.")
    parser.add_argument("--data", default="data/simulated_process_data.csv", help="Path to CSV data.")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size.")
    parser.add_argument("--z-thresh", type=float, default=3.0, help="Z-score threshold.")
    parser.add_argument("--contamination", type=float, default=0.02, help="Outlier ratio for feature models.")
    parser.add_argument(
        "--feature-models",
        type=_parse_feature_models,
        default=["isolation_forest"],
        help="Comma-separated feature models. Supported: isolation_forest,oneclass_svm,local_outlier_factor",
    )
    parser.add_argument("--mode", choices=["custom", "conservative", "sensitive"], default="custom")
    parser.add_argument("--run-mode", choices=["batch", "incremental"], default="batch")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for saved plots.")
    parser.add_argument("--stream-warmup", type=int, default=480, help="Warmup minutes before streaming scoring.")
    return parser.parse_args()


def _apply_mode_overrides(args: argparse.Namespace) -> None:
    if args.mode in MODE_PRESETS:
        preset = MODE_PRESETS[args.mode]
        args.z_thresh = preset["z_thresh"]
        args.contamination = preset["contamination"]


def _baseline_detection(df: pd.DataFrame, window: int, z_thresh: float) -> tuple[pd.Series, dict[str, pd.Series]]:
    zscores = {col: rolling_zscore(df, col, window=window) for col in SENSOR_COLUMNS}
    flags = [
        rolling_zscore_detector(df, "T_reactor_F", window=window, z_thresh=z_thresh),
        rolling_zscore_detector(df, "P_reactor_psig", window=window, z_thresh=z_thresh),
        rolling_zscore_detector(df, "F_feed_bbl_per_min", window=window, z_thresh=z_thresh),
    ]
    combined = combine_flags(flags).astype(int)
    return combined, zscores


def _feature_model_detection(
    feature_frame: pd.DataFrame,
    model_name: str,
    contamination: float,
) -> tuple[pd.Series, pd.Series]:
    X = feature_frame.to_numpy()
    model = train_detector(model_name, X, contamination=contamination)
    pred, score = predict_detector(model_name, model, X)
    pred_series = pd.Series(pred, index=feature_frame.index, name=f"{model_name}_pred")
    score_series = pd.Series(score, index=feature_frame.index, name=f"{model_name}_score")
    return pred_series, score_series


def _print_event_summary(label: str, metrics) -> None:
    print(f"{label} event recall: {metrics.event_recall:.3f} ({metrics.detected_events}/{metrics.total_events})")
    print(f"{label} mean TTD (min): {metrics.mean_time_to_detect_min:.1f}")
    print(f"{label} median TTD (min): {metrics.median_time_to_detect_min:.1f}")
    print(f"{label} false alert minutes: {metrics.false_alert_minutes:.1f}")


def _print_pointwise_summary(label: str, precision: float, recall: float, f1: float) -> None:
    print(f"{label} precision: {precision:.3f}")
    print(f"{label} recall:    {recall:.3f}")
    print(f"{label} F1:        {f1:.3f}")


def _print_root_causes(label: str, windows: list[tuple[int, int]], rankings: list[list[tuple[str, float]]]) -> None:
    if not windows:
        print(f"{label} root-cause: no detected events.")
        return
    print(f"{label} root-cause (top contributors per event):")
    for idx, (window, ranked) in enumerate(zip(windows, rankings), start=1):
        tags = ", ".join([f"{name} ({score:.2f})" for name, score in ranked])
        print(f"  event {idx} [{window[0]}:{window[1]}]: {tags}")


def _save_comparison_table(rows: list[dict[str, float | str]], output_path: str) -> None:
    table = pd.DataFrame(rows)
    table = table.sort_values(by=["event_recall", "f1", "precision"], ascending=False).reset_index(drop=True)
    table.to_csv(output_path, index=False)
    print("")
    print(f"Saved model comparison table: {output_path}")


def _run_batch(df: pd.DataFrame, args: argparse.Namespace) -> None:
    baseline_flag, baseline_zscores = _baseline_detection(df, args.window, args.z_thresh)
    y_true = df["anomaly_flag"].astype(int)
    baseline_precision, baseline_recall, baseline_f1 = pointwise_metrics(
        y_true.to_numpy(), baseline_flag.to_numpy()
    )
    baseline_events = event_metrics(y_true.to_numpy(), baseline_flag.to_numpy(), df["timestamp"])

    feature_frame = build_feature_frame(df)
    aligned_true = df.loc[feature_frame.index, "anomaly_flag"].astype(int)
    aligned_df = df.loc[feature_frame.index]
    feature_means = feature_frame.mean()
    feature_stds = feature_frame.std()

    model_results = []
    for model_name in args.feature_models:
        pred, score = _feature_model_detection(feature_frame, model_name, args.contamination)
        precision, recall, f1 = pointwise_metrics(aligned_true.to_numpy(), pred.to_numpy())
        events = event_metrics(aligned_true.to_numpy(), pred.to_numpy(), aligned_df["timestamp"])
        model_results.append(
            {
                "name": model_name,
                "pred": pred,
                "score": score,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "events": events,
            }
        )

    print("Pointwise metrics")
    _print_pointwise_summary("Baseline", baseline_precision, baseline_recall, baseline_f1)
    for result in model_results:
        print("")
        _print_pointwise_summary(
            _model_label(result["name"]), result["precision"], result["recall"], result["f1"]
        )
    print("")
    print("Event metrics")
    _print_event_summary("Baseline", baseline_events)
    for result in model_results:
        print("")
        _print_event_summary(_model_label(result["name"]), result["events"])

    baseline_windows = find_windows(baseline_flag.to_numpy())
    baseline_rankings = [
        rank_tags_from_zscores(baseline_zscores, start, end) for start, end in baseline_windows
    ]
    _print_root_causes("Baseline", baseline_windows, baseline_rankings)

    for result in model_results:
        windows = find_windows(result["pred"].to_numpy())
        rankings = [
            rank_tags_from_features(feature_frame, feature_means, feature_stds, start, end)
            for start, end in windows
        ]
        _print_root_causes(_model_label(result["name"]), windows, rankings)

    baseline_score = pd.DataFrame(baseline_zscores).abs().max(axis=1).fillna(0.0)
    plot_overview(
        df=df,
        timestamps=df["timestamp"],
        y_true=y_true,
        y_pred=baseline_flag,
        score=baseline_score,
        title="Baseline rolling z-score",
        output_path=f"{args.artifacts_dir}/baseline_overview.png",
    )

    for result in model_results:
        model_name = result["name"]
        plot_overview(
            df=aligned_df,
            timestamps=aligned_df["timestamp"],
            y_true=aligned_true,
            y_pred=result["pred"],
            score=result["score"],
            title=_model_label(model_name),
            output_path=f"{args.artifacts_dir}/{model_name}_overview.png",
        )

    comparison_rows = [
        {
            "model": "baseline_rolling_zscore",
            "precision": baseline_precision,
            "recall": baseline_recall,
            "f1": baseline_f1,
            "event_recall": baseline_events.event_recall,
            "mean_ttd_min": baseline_events.mean_time_to_detect_min,
            "median_ttd_min": baseline_events.median_time_to_detect_min,
            "false_alert_minutes": baseline_events.false_alert_minutes,
            "detected_events": baseline_events.detected_events,
            "total_events": baseline_events.total_events,
        }
    ]
    for result in model_results:
        events = result["events"]
        comparison_rows.append(
            {
                "model": str(result["name"]),
                "precision": float(result["precision"]),
                "recall": float(result["recall"]),
                "f1": float(result["f1"]),
                "event_recall": events.event_recall,
                "mean_ttd_min": events.mean_time_to_detect_min,
                "median_ttd_min": events.median_time_to_detect_min,
                "false_alert_minutes": events.false_alert_minutes,
                "detected_events": events.detected_events,
                "total_events": events.total_events,
            }
        )
    _save_comparison_table(comparison_rows, f"{args.artifacts_dir}/model_comparison_batch.csv")


def _run_incremental(df: pd.DataFrame, args: argparse.Namespace) -> None:
    sample_minutes = df["timestamp"].diff().dt.total_seconds().median() / 60.0
    if not sample_minutes or np.isnan(sample_minutes):
        sample_minutes = 1.0
    warmup_points = max(1, int(args.stream_warmup / sample_minutes))
    warmup_points = min(warmup_points, len(df))

    baseline_flags = []
    baseline_scores = []
    baseline_tags = ["T_reactor_F", "P_reactor_psig", "F_feed_bbl_per_min"]
    for idx in range(len(df)):
        if idx + 1 < args.window:
            baseline_flags.append(0)
            baseline_scores.append(0.0)
            continue

        window_df = df.iloc[: idx + 1]
        _, zscores = _baseline_detection(window_df, args.window, args.z_thresh)
        latest_scores = [float(zscores[col].iloc[-1]) for col in baseline_tags]
        flag = int(max(abs(score) for score in latest_scores) > args.z_thresh)
        baseline_flags.append(flag)
        baseline_scores.append(max(abs(score) for score in latest_scores))

    baseline_flag = pd.Series(baseline_flags, index=df.index)
    baseline_score = pd.Series(baseline_scores, index=df.index)

    feature_frame = build_feature_frame(df.iloc[:warmup_points])
    if feature_frame.empty:
        print("Not enough data for streaming warmup.")
        return

    feature_means = feature_frame.mean()
    feature_stds = feature_frame.std()
    latest_feature_rows: list[np.ndarray | None] = []
    for idx in range(len(df)):
        frame = build_feature_frame(df.iloc[: idx + 1])
        if frame.empty:
            latest_feature_rows.append(None)
        else:
            latest_feature_rows.append(frame.iloc[-1].to_numpy())

    model_results = []
    for model_name in args.feature_models:
        model = train_detector(
            model_name=model_name,
            X=feature_frame.to_numpy(),
            contamination=args.contamination,
        )
        flags = []
        scores = []
        for row in latest_feature_rows:
            if row is None:
                flags.append(0)
                scores.append(0.0)
                continue
            pred, score = predict_detector(model_name, model, row.reshape(1, -1))
            flags.append(int(pred[0]))
            scores.append(float(score[0]))
        model_results.append(
            {
                "name": model_name,
                "flag": pd.Series(flags, index=df.index),
                "score": pd.Series(scores, index=df.index),
            }
        )

    y_true = df["anomaly_flag"].astype(int)
    baseline_precision, baseline_recall, baseline_f1 = pointwise_metrics(
        y_true.to_numpy(), baseline_flag.to_numpy()
    )

    print("Pointwise metrics (streaming simulation)")
    _print_pointwise_summary("Baseline", baseline_precision, baseline_recall, baseline_f1)
    for result in model_results:
        precision, recall, f1 = pointwise_metrics(y_true.to_numpy(), result["flag"].to_numpy())
        print("")
        _print_pointwise_summary(_model_label(result["name"]), precision, recall, f1)

    baseline_events = event_metrics(y_true.to_numpy(), baseline_flag.to_numpy(), df["timestamp"])

    print("")
    print("Event metrics (streaming simulation)")
    _print_event_summary("Baseline", baseline_events)
    model_metrics = []
    for result in model_results:
        events = event_metrics(y_true.to_numpy(), result["flag"].to_numpy(), df["timestamp"])
        model_metrics.append(
            {
                "name": result["name"],
                "events": events,
            }
        )
        print("")
        _print_event_summary(_model_label(result["name"]), events)

    baseline_windows = find_windows(baseline_flag.to_numpy())
    baseline_rankings = [
        rank_tags_from_zscores(
            {col: rolling_zscore(df, col, window=args.window) for col in SENSOR_COLUMNS},
            start,
            end,
        )
        for start, end in baseline_windows
    ]
    _print_root_causes("Baseline", baseline_windows, baseline_rankings)

    feature_frame_all = build_feature_frame(df)
    for result in model_results:
        windows = find_windows(result["flag"].to_numpy())
        rankings = []
        for start, end in windows:
            window_index = df.index[start:end]
            window_frame = feature_frame_all.loc[feature_frame_all.index.intersection(window_index)]
            rankings.append(rank_tags_from_feature_window(window_frame, feature_means, feature_stds))
        _print_root_causes(_model_label(result["name"]), windows, rankings)

    plot_overview(
        df=df,
        timestamps=df["timestamp"],
        y_true=y_true,
        y_pred=baseline_flag,
        score=baseline_score,
        title="Baseline rolling z-score (streaming)",
        output_path=f"{args.artifacts_dir}/baseline_streaming_overview.png",
    )

    for result in model_results:
        model_name = result["name"]
        plot_overview(
            df=df,
            timestamps=df["timestamp"],
            y_true=y_true,
            y_pred=result["flag"],
            score=result["score"],
            title=f"{_model_label(model_name)} (streaming)",
            output_path=f"{args.artifacts_dir}/{model_name}_streaming_overview.png",
        )

    comparison_rows = [
        {
            "model": "baseline_rolling_zscore_streaming",
            "precision": baseline_precision,
            "recall": baseline_recall,
            "f1": baseline_f1,
            "event_recall": baseline_events.event_recall,
            "mean_ttd_min": baseline_events.mean_time_to_detect_min,
            "median_ttd_min": baseline_events.median_time_to_detect_min,
            "false_alert_minutes": baseline_events.false_alert_minutes,
            "detected_events": baseline_events.detected_events,
            "total_events": baseline_events.total_events,
        }
    ]
    for result, metrics in zip(model_results, model_metrics):
        precision, recall, f1 = pointwise_metrics(y_true.to_numpy(), result["flag"].to_numpy())
        events = metrics["events"]
        comparison_rows.append(
            {
                "model": str(result["name"]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "event_recall": events.event_recall,
                "mean_ttd_min": events.mean_time_to_detect_min,
                "median_ttd_min": events.median_time_to_detect_min,
                "false_alert_minutes": events.false_alert_minutes,
                "detected_events": events.detected_events,
                "total_events": events.total_events,
            }
        )
    _save_comparison_table(comparison_rows, f"{args.artifacts_dir}/model_comparison_streaming.csv")


def main() -> None:
    args = _parse_args()
    _apply_mode_overrides(args)

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    _validate_data(df)

    if args.run_mode == "batch":
        _run_batch(df, args)
    else:
        _run_incremental(df, args)


if __name__ == "__main__":
    main()
