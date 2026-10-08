"""Evaluate the trained models on one CSV (Colab "upload-one-CSV tester").

If the CSV has the target columns (flood_within_15m, ...), prints metrics
and PASS/FAIL gates per hazard. If not, writes predictions only.

Usage:
    python3 evaluate_csv.py path/to/readings.csv
    python3 evaluate_csv.py readings.csv --models output/models --out output/external_test_reports
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as C
from features import feature_matrix, prepare_dataframe
from metrics import (
    calculate_classification_metrics,
    confirm_alerts,
    print_classification_metrics,
    score_events,
    site_groups,
)
from train import apply_calibrator


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--models", type=Path, default=C.MODEL_DIR)
    parser.add_argument("--out", type=Path, default=C.OUTPUT_DIR / "external_test_reports")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.csv)
    print(f"Test CSV: {args.csv} ({len(raw)} rows, {len(raw.columns)} columns)")

    missing = [c for c in ["timestamp", "node_id", "site_id"] if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    absent_sensors = [c for c in C.RAW_SENSOR_COLUMNS if c not in raw.columns]
    if absent_sensors:
        print("WARNING: sensor columns absent, treated as missing:", ", ".join(absent_sensors))

    labelled_targets = [h["target"] for h in C.HAZARDS.values() if h["target"] in raw.columns]
    dataframe = prepare_dataframe(raw)

    results = []
    for hazard, config in C.HAZARDS.items():
        model_path = args.models / f"{hazard}_{C.MODEL_VERSION}.joblib"
        print("\n" + "=" * 70)
        print("HAZARD:", hazard.upper())
        print("=" * 70)
        if not model_path.exists():
            print("No model file (threshold selection failed in training or train.py not run).")
            results.append({"hazard": hazard, "status": "NO_MODEL"})
            continue

        artifact = joblib.load(model_path)
        subset = dataframe[dataframe["node_id"] == config["node_id"]].copy()
        if subset.empty:
            print(f"No rows for {config['node_id']}.")
            results.append({"hazard": hazard, "status": "NO_ROWS"})
            continue

        probabilities = apply_calibrator(
            artifact["calibrator"],
            artifact["model"].predict_proba(feature_matrix(subset, artifact["features"]))[:, 1],
        )
        watch, warning = artifact["watch_threshold"], artifact["warning_threshold"]
        print(f"WATCH threshold: {watch:.4f}   WARNING threshold: {warning:.4f}   "
              f"confirm readings: {C.ALERT_CONFIRM_READINGS}")

        # Alert levels exactly as the live service raises them
        watch_alert = np.zeros(len(subset), dtype=bool)
        warning_alert = np.zeros(len(subset), dtype=bool)
        for _, index in subset.groupby("site_id").indices.items():
            watch_alert[index] = confirm_alerts(probabilities[index] >= watch, C.ALERT_CONFIRM_READINGS)
            warning_alert[index] = confirm_alerts(probabilities[index] >= warning, C.ALERT_CONFIRM_READINGS)

        if config["target"] not in labelled_targets:
            output = subset[["timestamp", "site_id", "node_id"]].copy()
            output[f"{hazard}_probability"] = probabilities
            output[f"{hazard}_level"] = np.where(warning_alert, "WARNING", np.where(watch_alert, "WATCH", "SAFE"))
            path = args.out / f"{hazard}_predictions.csv"
            output.to_csv(path, index=False)
            print(f"No labels in CSV - predictions saved to {path}")
            results.append({"hazard": hazard, "status": "PREDICTIONS_ONLY", "rows": len(output)})
            continue

        y_true = subset[config["target"]].to_numpy(dtype=int)
        watch_metrics = calculate_classification_metrics(y_true, probabilities, watch)
        warning_metrics = calculate_classification_metrics(y_true, probabilities, warning)
        print_classification_metrics("WATCH", watch_metrics)
        print_classification_metrics("WARNING", warning_metrics)

        events = {}
        if {"event_subtype", "event_phase"} <= set(subset.columns):
            # Simulator-style CSV: score whole disasters, as train.py does
            events = score_events(site_groups(subset, hazard), probabilities, warning, C.ALERT_CONFIRM_READINGS)
            print("EVENTS (WARNING):", {k: round(v, 4) for k, v in events.items()})
            gates = {
                "detection_rate_pass": events["detection_rate"] >= C.MIN_DETECTION_RATE,
                "false_alarm_pass": events["false_alarms_per_site_day"] <= C.MAX_FALSE_ALARMS_PER_SITE_DAY,
                "alert_time_pass": events["alert_time_outside_events"] <= C.MAX_ALERT_TIME_OUTSIDE_EVENTS,
                "brier_pass": watch_metrics["brier_score"] <= C.MAX_BRIER_SCORE,
                "ece_pass": watch_metrics["ece"] <= C.MAX_ECE,
            }
        else:
            # Only labels: fall back to per-reading gates
            gates = {
                "watch_recall_pass": watch_metrics["recall"] >= C.WATCH_RECALL_TARGET,
                "warning_precision_pass": warning_metrics["precision"] >= C.WARNING_PRECISION_TARGET,
                "pr_auc_pass": bool(watch_metrics["pr_auc"]
                                    >= watch_metrics["positive_rate_baseline"] + C.MIN_PR_AUC_MARGIN_ABOVE_BASELINE),
                "brier_pass": watch_metrics["brier_score"] <= C.MAX_BRIER_SCORE,
                "ece_pass": watch_metrics["ece"] <= C.MAX_ECE,
            }
        gates = {name: bool(value) for name, value in gates.items()}
        status = "PASS" if all(gates.values()) else "FAIL"
        positives = int(y_true.sum())
        if events and events["episodes"] < C.MIN_TEST_EPISODES:
            # Too few disasters in this file for a verdict (same rule as train.py)
            status = f"INSUFFICIENT_EVENTS ({events['episodes']})"
        elif not events and positives == 0:
            status = "NO_POSITIVE_LABELS"
        print("GATES:", {k: "PASS" if v else "FAIL" for k, v in gates.items()}, "->", status)

        results.append({
            "hazard": hazard,
            "status": status,
            "watch_threshold": watch,
            "warning_threshold": warning,
            **{f"watch_{k}": v for k, v in watch_metrics.items()},
            **{f"warning_{k}": warning_metrics[k] for k in ["precision", "recall", "f1"]},
            **{f"event_{k}": v for k, v in events.items()},
            **gates,
        })

    results_df = pd.DataFrame(results)
    path = args.out / "external_test_results.csv"
    results_df.to_csv(path, index=False)
    print("\nSUMMARY")
    columns = ["hazard", "status", "event_detection_rate", "event_false_alarms_per_site_day",
               "watch_recall", "warning_precision"]
    print(results_df[[c for c in columns if c in results_df]].to_string(index=False))
    print("\nSaved:", path)


if __name__ == "__main__":
    main()
