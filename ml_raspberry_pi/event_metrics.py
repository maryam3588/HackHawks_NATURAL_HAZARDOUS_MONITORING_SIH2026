"""Event-level scoring: what an operator actually experiences.

Per-row precision/recall count 5-minute readings. For an early-warning
system the questions are per disaster:

* detection rate  - share of disaster episodes with at least one alert
                    during the episode's warning window
* lead time       - minutes between the first alert and the start of the
                    actual event phase (positive = early warning)
* false alarms    - separate alert runs outside any warning window,
                    per site per day

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
from train import apply_calibrator

EVENT_COLUMNS = {
    "flood": "flood_event",
    "landslide": "landslide_event",
    "wildfire": "wildfire_event",
    "extreme_heat": "extreme_heat_event",
}


def runs(mask):
    """(start, end) index pairs of contiguous True runs."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def score_site(times, target, event, alert):
    episodes = runs(target == 1)
    detected, lead_minutes = 0, []
    for start, end in episodes:
        alerts = np.flatnonzero(alert[start:end])
        if len(alerts):
            detected += 1
            event_rows = np.flatnonzero(event[start:end] == 1)
            if len(event_rows):
                first_alert = times[start + alerts[0]]
                event_start = times[start + event_rows[0]]
                lead_minutes.append((event_start - first_alert).total_seconds() / 60)
    false_runs = len(runs(alert & (target == 0)))
    return len(episodes), detected, lead_minutes, false_runs


def evaluate(models_dir, data_dir, dataset_key, level="WARNING"):
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
        alert_all = probability >= threshold

        totals = {"episodes": 0, "detected": 0, "false_alarms": 0}
        leads = []
        site_days = 0.0
        for _, index in subset.groupby("site_id").indices.items():
            site = subset.iloc[index]
            episodes, detected, lead, false_runs = score_site(
                site["timestamp"].to_numpy(), site[config["target"]].to_numpy(),
                site[EVENT_COLUMNS[hazard]].to_numpy(), alert_all[index],
            )
            totals["episodes"] += episodes
            totals["detected"] += detected
            totals["false_alarms"] += false_runs
            leads += lead
            span = site["timestamp"].iloc[-1] - site["timestamp"].iloc[0]
            site_days += span.total_seconds() / 86400

        rows.append({
            "hazard": hazard,
            "dataset": dataset_key,
            "level": level,
            "threshold": threshold,
            "episodes": totals["episodes"],
            "detection_rate": totals["detected"] / totals["episodes"] if totals["episodes"] else np.nan,
            "median_lead_min": float(np.median(leads)) if leads else np.nan,
            "false_alarms_per_site_day": totals["false_alarms"] / site_days if site_days else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, default=C.MODEL_DIR)
    parser.add_argument("--data", type=Path, default=C.DATA_DIR)
    parser.add_argument("--level", choices=["WATCH", "WARNING"], default="WARNING")
    parser.add_argument("--out", type=Path, default=None, help="optional CSV path")
    args = parser.parse_args()

    results = pd.concat(
        [evaluate(args.models, args.data, key, args.level) for key in ["locked_normal", "locked_faults", "locked_ood"]],
        ignore_index=True,
    )
    print(results.round(3).to_string(index=False))
    if args.out:
        results.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
