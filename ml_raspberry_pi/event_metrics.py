"""Event-level scoring: what an operator actually experiences.

Per-row precision/recall count 5-minute readings. For an early-warning
system the questions are per disaster:

* detection rate  - share of disaster episodes with at least one alert
                    from the first precursor until the event ends
* warned before   - share of episodes alerted before the event phase began
* lead time       - minutes between the first alert and the start of the
                    actual event phase (positive = early warning)
* false alarms    - separate alert runs that touch no disaster episode
                    (or its recovery), per site per day

Usage:
    python3 event_metrics.py                                   # output/models on data/
    python3 event_metrics.py --models output_ideal/models --data data
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as C
from features import feature_matrix, prepare_dataframe
from metrics import score_events, site_groups
from train import apply_calibrator

def evaluate(models_dir, data_dir, dataset_key, level="WARNING", confirm=1):
    raw = pd.read_csv(Path(data_dir) / C.DATASET_FILES[dataset_key])
    dataframe = prepare_dataframe(raw)
    rows = []
    for hazard, config in C.HAZARDS.items():
        path = Path(models_dir) / f"{hazard}_{C.MODEL_VERSION}.joblib"
        if not path.exists():
            rows.append({"hazard": hazard, "dataset": dataset_key, "status": "NO_MODEL"})
            continue
        artifact = joblib.load(path)
        subset = dataframe[dataframe["node_id"] == config["node_id"]]
        probability = apply_calibrator(
            artifact["calibrator"],
            artifact["model"].predict_proba(feature_matrix(subset, artifact["features"]))[:, 1],
        )
        threshold = artifact["warning_threshold" if level == "WARNING" else "watch_threshold"]
        scores = score_events(site_groups(subset, hazard), probability, threshold, confirm)
        rows.append({"hazard": hazard, "dataset": dataset_key, "level": level, "threshold": threshold, **scores})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=C.MODEL_DIR)
    parser.add_argument("--data", type=Path, default=C.DATA_DIR)
    parser.add_argument("--level", choices=["WATCH", "WARNING"], default="WARNING")
    parser.add_argument("--confirm", type=int, default=C.ALERT_CONFIRM_READINGS,
                        help="consecutive readings over threshold before alerting")
    parser.add_argument("--out", type=Path, default=None, help="optional CSV path")
    args = parser.parse_args()

    results = pd.concat(
        [evaluate(args.models, args.data, key, args.level, args.confirm) for key in ["locked_normal", "locked_faults", "locked_ood"]],
        ignore_index=True,
    )
    print(results.round(3).to_string(index=False))
    if args.out:
        results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
