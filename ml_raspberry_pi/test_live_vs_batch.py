"""End-to-end check: the live service must raise the same alerts as evaluation.

Replays one site of a labelled CSV reading-by-reading through
live_inference.HazardPredictor and compares each SAFE/WATCH/WARNING level
with the batch pipeline (features -> model -> calibrator -> thresholds ->
consecutive-reading confirmation).

    python3 test_live_vs_batch.py                       # first locked_normal site
    python3 test_live_vs_batch.py some.csv --max-rows 2000
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as C
from features import feature_matrix, prepare_dataframe
from live_inference import HazardPredictor
from metrics import confirm_alerts
from train import apply_calibrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path, default=C.DATA_DIR / C.DATASET_FILES["locked_normal"])
    parser.add_argument("--models", type=Path, default=C.MODEL_DIR)
    parser.add_argument("--max-rows", type=int, default=3000, help="readings to replay (both nodes)")
    args = parser.parse_args()

    raw = pd.read_csv(args.csv)
    raw = raw[raw["site_id"] == raw["site_id"].iloc[0]].sort_values(["timestamp", "node_id"]).head(args.max_rows)

    predictor = HazardPredictor(args.models)
    live = {}
    for _, row in raw.iterrows():
        reading = {key: (None if pd.isna(value) else value) for key, value in row.items()}
        for hazard, result in predictor.predict(reading)["hazards"].items():
            live.setdefault(hazard, []).append(result["level"])

    batch = prepare_dataframe(raw)
    failed = False
    for hazard, config in C.HAZARDS.items():
        path = args.models / f"{hazard}_{C.MODEL_VERSION}.joblib"
        if not path.exists():
            print(f"{hazard}: no model, skipped")
            continue
        artifact = joblib.load(path)
        subset = batch[batch["node_id"] == config["node_id"]]
        probability = apply_calibrator(
            artifact["calibrator"], artifact["model"].predict_proba(feature_matrix(subset, artifact["features"]))[:, 1]
        )
        warning = confirm_alerts(probability >= artifact["warning_threshold"], predictor.confirm)
        watch = confirm_alerts(probability >= artifact["watch_threshold"], predictor.confirm)
        expected = np.where(warning, "WARNING", np.where(watch, "WATCH", "SAFE"))
        agreement = float((expected == np.array(live[hazard])).mean())
        failed |= agreement < 1.0
        print(f"{hazard:13s} readings={len(expected)} live==batch {agreement:.2%} WARNING readings={int(warning.sum())}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
