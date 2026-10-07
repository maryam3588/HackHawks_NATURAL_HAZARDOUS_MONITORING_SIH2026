"""Train -> calibrate -> pick thresholds -> locked tests. No forecasting.

Ported from the Colab "complete pipeline" with these changes:
* forecasting phase removed (all four forecasters scored worse than
  "repeat the last reading" in the Colab run);
* reads the V3 files/roles produced by generate_dataset.py;
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
    select_ordered_thresholds,
)

REQUIRED_COLUMNS = (
    ["timestamp", "node_id", "site_id", "sensor_status"]
    + C.RAW_SENSOR_COLUMNS
    + C.TARGET_COLUMNS
    + ["dataset_role", "data_source"]
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
    print("DISASTER V3 PIPELINE: TRAIN -> CALIBRATION -> THRESHOLDS -> LOCKED TESTS")
    print("=" * 90)

    datasets = {key: load_dataset(args.data, key) for key in C.DATASET_FILES}

    base_rows, calibration_rows, threshold_rows, locked_rows, acceptance_rows = [], [], [], [], []
    registry = []

    for hazard, config in C.HAZARDS.items():
        print("\n" + "=" * 90)
        print("HAZARD:", hazard.upper())
        print("=" * 90)

        # ---- Phase 1: base classifier --------------------------------
        X_train, y_train = hazard_xy(datasets["train"], config)
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
        version = f"{hazard}_v3"

        train_metrics = calculate_classification_metrics(y_train, model.predict_proba(X_train)[:, 1], 0.50)
        print_classification_metrics(f"{hazard.upper()} BASE TRAIN (iterations used: {model.n_iter_})", train_metrics)
        base_rows.append(tag(train_metrics, hazard=hazard, stage="BASE_TRAIN", dataset="train", model_version=version))

        # ---- Phase 2: calibration + ordered thresholds ---------------
        X_cal, y_cal = hazard_xy(datasets["calibration"], config)
        X_thr, y_thr = hazard_xy(datasets["threshold_validation"], config)
        empty = [name for name, y in [("calibration", y_cal), ("threshold_validation", y_thr)]
                 if len(np.unique(y)) < 2]
        if empty:
            reason = f"no positive (or no negative) rows in: {', '.join(empty)} - generate more data"
            print("CANNOT CALIBRATE:", reason)
            acceptance_rows.append({"hazard": hazard, "candidate_status": "FAIL", "reason": reason})
            (model_dir / f"{version}.joblib").unlink(missing_ok=True)
            continue

        calibrator = fit_calibrator(model.predict_proba(X_cal)[:, 1], y_cal)
        cal_metrics = calculate_classification_metrics(
            y_cal, apply_calibrator(calibrator, model.predict_proba(X_cal)[:, 1]), 0.50
        )
        print_classification_metrics(f"{hazard.upper()} CALIBRATION", cal_metrics)
        calibration_rows.append(tag(cal_metrics, hazard=hazard, stage="CALIBRATION", dataset="calibration", model_version=version))

        thr_probabilities = apply_calibrator(calibrator, model.predict_proba(X_thr)[:, 1])
        selection = select_ordered_thresholds(y_thr, thr_probabilities)

        if selection["status"] == "FAIL":
            print("THRESHOLD SELECTION FAILED:", selection)
            acceptance_rows.append({"hazard": hazard, "candidate_status": "FAIL", "reason": selection["reason"]})
            # Never leave a model from an earlier run behind for a hazard that now fails
            (model_dir / f"{version}.joblib").unlink(missing_ok=True)
            continue

        watch_threshold = selection["watch_threshold"]
        warning_threshold = selection["warning_threshold"]
        if watch_threshold == warning_threshold:
            print(f"NOTE: WATCH and WARNING thresholds are identical ({watch_threshold:.2f});"
                  " the two alert tiers will always fire together for this hazard.")

        for stage, threshold in [("WATCH", watch_threshold), ("WARNING", warning_threshold)]:
            stage_metrics = calculate_classification_metrics(y_thr, thr_probabilities, threshold)
            print_classification_metrics(f"{hazard.upper()} THRESHOLD VALIDATION {stage}", stage_metrics)
            threshold_rows.append(tag(stage_metrics, hazard=hazard, stage=stage, dataset="threshold_validation", model_version=version))

        # ---- Phase 3: locked tests ------------------------------------
        results = {}
        for test_name in ["locked_normal", "locked_faults", "locked_ood"]:
            X_test, y_test = hazard_xy(datasets[test_name], config)
            probabilities = apply_calibrator(calibrator, model.predict_proba(X_test)[:, 1])
            results[test_name] = {}
            for stage, threshold in [("WATCH", watch_threshold), ("WARNING", warning_threshold)]:
                stage_metrics = calculate_classification_metrics(y_test, probabilities, threshold)
                print_classification_metrics(f"{hazard.upper()} {test_name.upper()} {stage}", stage_metrics)
                results[test_name][stage] = stage_metrics
                locked_rows.append(tag(dict(stage_metrics), hazard=hazard, stage=stage, dataset=test_name, model_version=version))

        normal_watch = results["locked_normal"]["WATCH"]
        normal_warning = results["locked_normal"]["WARNING"]
        fault_recall_drop = normal_watch["recall"] - results["locked_faults"]["WATCH"]["recall"]
        ood_recall_drop = normal_watch["recall"] - results["locked_ood"]["WATCH"]["recall"]

        gates = {
            "ordered_thresholds_pass": watch_threshold <= warning_threshold,
            "watch_recall_pass": normal_watch["recall"] >= C.WATCH_RECALL_TARGET,
            "warning_precision_pass": normal_warning["precision"] >= C.WARNING_PRECISION_TARGET,
            "pr_auc_pass": normal_watch["pr_auc"]
            >= normal_watch["positive_rate_baseline"] + C.MIN_PR_AUC_MARGIN_ABOVE_BASELINE,
            "brier_score_pass": normal_watch["brier_score"] <= C.MAX_BRIER_SCORE,
            "ece_pass": normal_watch["ece"] <= C.MAX_ECE,
            "fault_robustness_pass": fault_recall_drop <= C.MAX_FAULT_RECALL_DROP,
            "ood_robustness_pass": ood_recall_drop <= C.MAX_OOD_RECALL_DROP,
        }
        gates = {name: bool(value) for name, value in gates.items()}
        status = "PASS" if all(gates.values()) else "FAIL"

        print("\nACCEPTANCE GATES")
        for name, value in gates.items():
            print(f"  {name}: {'PASS' if value else 'FAIL'}")
        print("FINAL STATUS:", status)

        acceptance_rows.append({
            "hazard": hazard,
            "model_version": version,
            "candidate_status": status,
            "watch_threshold": watch_threshold,
            "warning_threshold": warning_threshold,
            "normal_watch_precision": normal_watch["precision"],
            "normal_watch_recall": normal_watch["recall"],
            "normal_warning_precision": normal_warning["precision"],
            "normal_warning_recall": normal_warning["recall"],
            "normal_pr_auc": normal_watch["pr_auc"],
            "normal_brier_score": normal_watch["brier_score"],
            "normal_ece": normal_watch["ece"],
            "fault_recall_drop": fault_recall_drop,
            "ood_recall_drop": ood_recall_drop,
            **gates,
        })

        artifact = {
            "model": model,
            "calibrator": calibrator,
            "hazard": hazard,
            "node_id": config["node_id"],
            "target": config["target"],
            "features": list(config["features"]),
            "watch_threshold": watch_threshold,
            "warning_threshold": warning_threshold,
            "status": status,
            "version": version,
            "data_source": C.DATA_SOURCE,
            "sklearn_version": sklearn.__version__,
        }
        model_path = model_dir / f"{version}.joblib"
        joblib.dump(artifact, model_path, compress=3)
        registry.append({
            "hazard": hazard,
            "file": model_path.name,
            "status": status,
            "watch_threshold": watch_threshold,
            "warning_threshold": warning_threshold,
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
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }, handle, indent=2)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    acceptance_df = pd.DataFrame(acceptance_rows)
    for _, row in acceptance_df.iterrows():
        if "normal_watch_recall" not in row or pd.isna(row.get("normal_watch_recall")):
            print(f"{row['hazard'].upper()} | STATUS={row['candidate_status']} | {row.get('reason', '')}")
            continue
        print(
            f"{row['hazard'].upper()} | STATUS={row['candidate_status']} | "
            f"WATCH>={row['watch_threshold']:.2f} recall={row['normal_watch_recall']:.4f} | "
            f"WARNING>={row['warning_threshold']:.2f} precision={row['normal_warning_precision']:.4f} | "
            f"fault drop={row['fault_recall_drop']:.4f} | OOD drop={row['ood_recall_drop']:.4f}"
        )

    print(f"\nModels:  {model_dir}")
    print(f"Reports: {report_dir}")
    print(f"Total time: {time.time() - started:.1f}s")
    print("\nDATA SOURCE: SYNTHETIC_V3 - REAL-WORLD PERFORMANCE NOT ESTABLISHED")


if __name__ == "__main__":
    main()
