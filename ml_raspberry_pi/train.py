"""Train -> calibrate -> pick thresholds -> locked tests. No forecasting.

Ported from the Colab "complete pipeline" with these changes:
* forecasting phase removed (all four forecasters scored worse than
  "repeat the last reading" in the Colab run);
* reads the files/roles produced by generate_dataset.py;
* models train on plain float32 arrays (HistGradientBoosting handles NaN
  itself), so live inference needs no pandas;
* the calibrator is stored as two floats, so applying it needs no sklearn;
* the calibrator no longer uses class_weight="balanced", which pulls
  probabilities towards 50/50 and defeats the point of calibrating;
* reports are CSV + JSON only (no Excel, no Colab downloads).

Usage:
    python3 train.py
    python3 train.py --data data --out output
"""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import config as C
from features import feature_matrix, prepare_dataframe
from metrics import (
    calculate_classification_metrics,
    print_classification_metrics,
    score_events,
    select_event_thresholds,
    site_groups,
)

REQUIRED_COLUMNS = (
    ["timestamp", "node_id", "site_id", "sensor_status"]
    + C.RAW_SENSOR_COLUMNS
    + C.TARGET_COLUMNS
    + ["dataset_role", "data_source", "event_subtype", "event_phase"]
)


def load_dataset(data_dir, key):
    path = data_dir / C.DATASET_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run generate_dataset.py first.")

    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    dataframe = pd.read_csv(path, usecols=REQUIRED_COLUMNS)

    roles = set(dataframe["dataset_role"].dropna().astype(str).unique())
    sources = set(dataframe["data_source"].dropna().astype(str).unique())
    if roles != {key}:
        raise ValueError(f"{path.name}: expected role={key}, found {roles}")
    if sources != {C.DATA_SOURCE}:
        raise ValueError(f"{path.name}: expected source={C.DATA_SOURCE}, found {sources}")

    print(f"{key}: {len(dataframe)} rows - PASS")
    return prepare_dataframe(dataframe)


def hazard_xy(dataframe, config):
    subset = dataframe[dataframe["node_id"] == config["node_id"]]
    return feature_matrix(subset, config["features"]), subset[config["target"]].to_numpy(dtype=int)


def hazard_groups(dataframe, config, hazard):
    return site_groups(dataframe[dataframe["node_id"] == config["node_id"]], hazard)


def count_episodes(dataframe, config):
    """Number of separate warning episodes (0 -> 1 transitions per site)."""
    subset = dataframe[dataframe["node_id"] == config["node_id"]]
    target = subset[config["target"]]
    previous = target.groupby(subset["site_id"]).shift(1).fillna(0)
    return int(((target == 1) & (previous == 0)).sum())


def fit_calibrator(raw_probabilities, y_true):
    inputs = np.clip(raw_probabilities, 1e-6, 1 - 1e-6).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000, random_state=C.RANDOM_SEED)
    model.fit(inputs, y_true)
    return {"coef": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}


def apply_calibrator(calibrator, raw_probabilities):
    inputs = np.clip(np.asarray(raw_probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(calibrator["coef"] * inputs + calibrator["intercept"])))


def tag(metrics, **extra):
    metrics.update(extra)
    metrics["data_source"] = C.DATA_SOURCE
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=C.DATA_DIR)
    parser.add_argument("--out", type=Path, default=C.OUTPUT_DIR)
    parser.add_argument("--max-iter", type=int, default=250, help="boosting iterations per model")
    args = parser.parse_args()

    model_dir = args.out / "models"
    report_dir = args.out / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("=" * 90)
    print("DISASTER PIPELINE: TRAIN -> CALIBRATION -> THRESHOLDS -> LOCKED TESTS")
    print("=" * 90)

    base_rows, calibration_rows, threshold_rows, locked_rows, acceptance_rows = [], [], [], [], []
    candidates = {}

    # Datasets are loaded one phase at a time and freed afterwards so the
    # whole pipeline fits in a Raspberry Pi's RAM.

    # ---- Phase 1: base classifiers --------------------------------------
    print("\nPHASE 1: BASE CLASSIFIERS")
    train_df = load_dataset(args.data, "train")
    models = {}
    for hazard, config in C.HAZARDS.items():
        X_train, y_train = hazard_xy(train_df, config)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=args.max_iter,
            max_leaf_nodes=15,
            l2_regularization=1.5,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=C.RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        models[hazard] = model
        version = f"{hazard}_{C.MODEL_VERSION}"

        train_metrics = calculate_classification_metrics(y_train, model.predict_proba(X_train)[:, 1], 0.50)
        print_classification_metrics(f"{hazard.upper()} BASE TRAIN (iterations used: {model.n_iter_})", train_metrics)
        base_rows.append(tag(train_metrics, hazard=hazard, stage="BASE_TRAIN", dataset="train",
                             model_version=version, episodes=count_episodes(train_df, config)))
    del train_df

    # ---- Phase 2: calibration + ordered thresholds ----------------------
    print("\nPHASE 2: CALIBRATION + ORDERED THRESHOLDS")
    calibration_df = load_dataset(args.data, "calibration")
    threshold_df = load_dataset(args.data, "threshold_validation")
    for hazard, config in C.HAZARDS.items():
        model = models[hazard]
        version = f"{hazard}_{C.MODEL_VERSION}"
        X_cal, y_cal = hazard_xy(calibration_df, config)
        X_thr, y_thr = hazard_xy(threshold_df, config)

        empty = [name for name, y in [("calibration", y_cal), ("threshold_validation", y_thr)]
                 if len(np.unique(y)) < 2]
        if empty:
            reason = f"no positive (or no negative) rows in: {', '.join(empty)} - generate more data"
            print(f"{hazard.upper()} CANNOT CALIBRATE:", reason)
            acceptance_rows.append({"hazard": hazard, "candidate_status": "FAIL", "reason": reason})
            continue

        calibrator = fit_calibrator(model.predict_proba(X_cal)[:, 1], y_cal)
        cal_metrics = calculate_classification_metrics(
            y_cal, apply_calibrator(calibrator, model.predict_proba(X_cal)[:, 1]), 0.50
        )
        print_classification_metrics(f"{hazard.upper()} CALIBRATION", cal_metrics)
        calibration_rows.append(tag(cal_metrics, hazard=hazard, stage="CALIBRATION", dataset="calibration",
                                    model_version=version, episodes=count_episodes(calibration_df, config)))

        thr_probabilities = apply_calibrator(calibrator, model.predict_proba(X_thr)[:, 1])
        selection = select_event_thresholds(hazard_groups(threshold_df, config, hazard), thr_probabilities,
                                            C.ALERT_CONFIRM_READINGS)
        if selection["status"] == "FAIL":
            print(f"{hazard.upper()} THRESHOLD SELECTION FAILED:", selection["reason"])
            acceptance_rows.append({"hazard": hazard, "candidate_status": "FAIL", "reason": selection["reason"]})
            continue

        watch_threshold = selection["watch_threshold"]
        warning_threshold = selection["warning_threshold"]
        print(f"{hazard.upper()} thresholds: WATCH>={watch_threshold:.4f} WARNING>={warning_threshold:.4f} "
              f"(validation WARNING: {selection['warning_validation']})")
        if watch_threshold == warning_threshold:
            print(f"NOTE: {hazard} WATCH and WARNING thresholds are identical;"
                  " the two alert tiers will always fire together.")

        for stage, threshold in [("WATCH", watch_threshold), ("WARNING", warning_threshold)]:
            stage_metrics = calculate_classification_metrics(y_thr, thr_probabilities, threshold)
            print_classification_metrics(f"{hazard.upper()} THRESHOLD VALIDATION {stage}", stage_metrics)
            threshold_rows.append(tag(stage_metrics, hazard=hazard, stage=stage, dataset="threshold_validation",
                                      model_version=version, episodes=count_episodes(threshold_df, config)))

        candidates[hazard] = {
            "model": model,
            "calibrator": calibrator,
            "watch_threshold": watch_threshold,
            "warning_threshold": warning_threshold,
            "version": version,
        }
    del calibration_df, threshold_df

    # ---- Phase 3: locked tests ------------------------------------------
    print("\nPHASE 3: LOCKED TESTS")
    results = {hazard: {} for hazard in candidates}
    for test_name in ["locked_normal", "locked_faults", "locked_ood"]:
        test_df = load_dataset(args.data, test_name)
        for hazard, candidate in candidates.items():
            config = C.HAZARDS[hazard]
            X_test, y_test = hazard_xy(test_df, config)
            episodes = count_episodes(test_df, config)
            probabilities = apply_calibrator(candidate["calibrator"], candidate["model"].predict_proba(X_test)[:, 1])
            results[hazard][test_name] = {"episodes": episodes}
            groups = hazard_groups(test_df, config, hazard)
            for stage in ["WATCH", "WARNING"]:
                threshold = candidate[f"{stage.lower()}_threshold"]
                stage_metrics = calculate_classification_metrics(y_test, probabilities, threshold)
                events = score_events(groups, probabilities, threshold, C.ALERT_CONFIRM_READINGS)
                stage_metrics.update({f"event_{k}": v for k, v in events.items()})
                print(f"{hazard.upper()} {test_name} {stage}: {events}")
                print_classification_metrics(f"{hazard.upper()} {test_name.upper()} {stage} ({episodes} episodes)",
                                             stage_metrics)
                results[hazard][test_name][stage] = stage_metrics
                locked_rows.append(tag(dict(stage_metrics), hazard=hazard, stage=stage, dataset=test_name,
                                       model_version=candidate["version"], episodes=episodes))
        del test_df

    # ---- Acceptance gates + model export --------------------------------
    print("\nACCEPTANCE")
    registry = []
    for hazard, config in C.HAZARDS.items():
        model_path = model_dir / f"{hazard}_{C.MODEL_VERSION}.joblib"
        candidate = candidates.get(hazard)
        if candidate is None:
            # Never leave a model from an earlier run behind for a hazard that now fails
            model_path.unlink(missing_ok=True)
            continue

        res = results[hazard]
        normal_watch = res["locked_normal"]["WATCH"]
        normal_warning = res["locked_normal"]["WARNING"]
        detection = {name: res[name]["WARNING"]["event_detection_rate"] for name in res}
        fault_detection_drop = detection["locked_normal"] - detection["locked_faults"]
        ood_detection_drop = detection["locked_normal"] - detection["locked_ood"]
        min_episodes = min(res[name]["episodes"] for name in res)

        # Event-level gates decide PASS/FAIL. The per-reading Colab gates are
        # kept as row_* columns for reference.
        gates = {
            "ordered_thresholds_pass": candidate["watch_threshold"] <= candidate["warning_threshold"],
            "detection_rate_pass": detection["locked_normal"] >= C.MIN_DETECTION_RATE,
            "false_alarm_pass": normal_warning["event_false_alarms_per_site_day"]
            <= C.MAX_FALSE_ALARMS_PER_SITE_DAY,
            "alert_time_pass": normal_warning["event_alert_time_outside_events"]
            <= C.MAX_ALERT_TIME_OUTSIDE_EVENTS,
            "fault_robustness_pass": fault_detection_drop <= C.MAX_FAULT_DETECTION_DROP,
            "ood_robustness_pass": ood_detection_drop <= C.MAX_OOD_DETECTION_DROP,
            "brier_score_pass": normal_watch["brier_score"] <= C.MAX_BRIER_SCORE,
            "ece_pass": normal_watch["ece"] <= C.MAX_ECE,
        }
        gates = {name: bool(value) for name, value in gates.items()}
        row_gates = {
            "row_watch_recall_pass": bool(normal_watch["recall"] >= C.WATCH_RECALL_TARGET),
            "row_warning_precision_pass": bool(normal_warning["precision"] >= C.WARNING_PRECISION_TARGET),
            "row_pr_auc_pass": bool(normal_watch["pr_auc"]
                                    >= normal_watch["positive_rate_baseline"] + C.MIN_PR_AUC_MARGIN_ABOVE_BASELINE),
        }
        if min_episodes < C.MIN_TEST_EPISODES:
            status = "INSUFFICIENT_EVENTS"
        else:
            status = "PASS" if all(gates.values()) else "FAIL"

        print(f"\n{hazard.upper()} (fewest episodes in a locked test: {min_episodes})")
        for name, value in {**gates, **row_gates}.items():
            print(f"  {name}: {'PASS' if value else 'FAIL'}")
        print("  FINAL STATUS:", status)

        acceptance_rows.append({
            "hazard": hazard,
            "model_version": candidate["version"],
            "candidate_status": status,
            "min_test_episodes": min_episodes,
            "watch_threshold": candidate["watch_threshold"],
            "warning_threshold": candidate["warning_threshold"],
            "normal_detection_rate": detection["locked_normal"],
            "faults_detection_rate": detection["locked_faults"],
            "ood_detection_rate": detection["locked_ood"],
            "normal_warned_before_event": normal_warning["event_warned_before_event"],
            "normal_median_lead_min": normal_warning["event_median_lead_min"],
            "normal_false_alarms_per_site_day": normal_warning["event_false_alarms_per_site_day"],
            "watch_false_alarms_per_site_day": normal_watch["event_false_alarms_per_site_day"],
            "normal_alert_time_outside_events": normal_warning["event_alert_time_outside_events"],
            "watch_alert_time_outside_events": normal_watch["event_alert_time_outside_events"],
            "fault_detection_drop": fault_detection_drop,
            "ood_detection_drop": ood_detection_drop,
            "normal_row_watch_recall": normal_watch["recall"],
            "normal_row_warning_precision": normal_warning["precision"],
            "normal_row_warning_recall": normal_warning["recall"],
            "normal_pr_auc": normal_watch["pr_auc"],
            "normal_brier_score": normal_watch["brier_score"],
            "normal_ece": normal_watch["ece"],
            **gates,
            **row_gates,
        })

        joblib.dump({
            "model": candidate["model"],
            "calibrator": candidate["calibrator"],
            "hazard": hazard,
            "node_id": config["node_id"],
            "target": config["target"],
            "features": list(config["features"]),
            "watch_threshold": candidate["watch_threshold"],
            "warning_threshold": candidate["warning_threshold"],
            "status": status,
            "version": candidate["version"],
            "data_source": C.DATA_SOURCE,
            "sklearn_version": sklearn.__version__,
        }, model_path, compress=3)
        registry.append({
            "hazard": hazard,
            "file": model_path.name,
            "status": status,
            "watch_threshold": candidate["watch_threshold"],
            "warning_threshold": candidate["warning_threshold"],
        })

    # ---- Reports -------------------------------------------------------
    reports = {
        "base_training_metrics.csv": base_rows,
        "calibration_metrics.csv": calibration_rows,
        "threshold_validation_metrics.csv": threshold_rows,
        "locked_test_metrics.csv": locked_rows,
        "acceptance_gates.csv": acceptance_rows,
    }
    for file_name, rows in reports.items():
        pd.DataFrame(rows).to_csv(report_dir / file_name, index=False)

    with open(model_dir / "model_registry.json", "w") as handle:
        json.dump({
            "models": registry,
            "data_source": C.DATA_SOURCE,
            "sklearn_version": sklearn.__version__,
            "alert_confirm_readings": C.ALERT_CONFIRM_READINGS,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }, handle, indent=2)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    order = {hazard: index for index, hazard in enumerate(C.HAZARDS)}
    for row in sorted(acceptance_rows, key=lambda r: order[r["hazard"]]):
        if "normal_detection_rate" not in row:
            print(f"{row['hazard'].upper()} | STATUS={row['candidate_status']} | {row.get('reason', '')}")
            continue
        print(
            f"{row['hazard'].upper()} | STATUS={row['candidate_status']} | episodes>={row['min_test_episodes']} | "
            f"WARNING>={row['warning_threshold']:.4f}: detected {row['normal_detection_rate']:.0%} "
            f"(faults {row['faults_detection_rate']:.0%}, OOD {row['ood_detection_rate']:.0%}), "
            f"before event {row['normal_warned_before_event']:.0%}, lead {row['normal_median_lead_min']:.0f} min, "
            f"false alarms {row['normal_false_alarms_per_site_day']:.3f}/site-day, "
            f"alert {row['normal_alert_time_outside_events']:.2%} of normal time | "
            f"WATCH>={row['watch_threshold']:.4f}: false alarms {row['watch_false_alarms_per_site_day']:.2f}/site-day, "
            f"alert {row['watch_alert_time_outside_events']:.2%}"
        )

    print(f"\nModels:  {model_dir}")
    print(f"Reports: {report_dir}")
    print(f"Total time: {time.time() - started:.1f}s")
    print(f"\nDATA SOURCE: {C.DATA_SOURCE.upper()} - REAL-WORLD PERFORMANCE NOT ESTABLISHED")


if __name__ == "__main__":
    main()
