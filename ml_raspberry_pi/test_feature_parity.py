"""Check that live (streaming) features match the training (pandas) features.

Training/serving skew is the most likely way for a model to work in the
notebook and misbehave on the device, so run this after any change to
features.py:

    python3 test_feature_parity.py                 # uses data/locked_faults
    python3 test_feature_parity.py some_file.csv --rows 3000
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from features import StreamingFeatureEngine, prepare_dataframe

ALL_FEATURES = sorted({f for h in C.HAZARDS.values() for f in h["features"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", type=Path,
                        default=C.DATA_DIR / C.DATASET_FILES["locked_faults"])
    parser.add_argument("--rows", type=int, default=4000,
                        help="rows per node to replay (faults file has NaNs, spikes, stuck values)")
    args = parser.parse_args()

    raw = pd.read_csv(args.csv)
    first_site = raw["site_id"].iloc[0]
    raw = raw[raw["site_id"] == first_site]
    raw = raw.groupby("node_id", sort=False).head(args.rows).reset_index(drop=True)

    batch = prepare_dataframe(raw)
    engine = StreamingFeatureEngine()

    worst = 0.0
    mismatches = 0
    for _, row in batch.iterrows():
        reading = {c: row[c] for c in C.RAW_SENSOR_COLUMNS}
        reading.update(node_id=row["node_id"], site_id=row["site_id"], timestamp=row["timestamp"].isoformat())
        _, live = engine.update(reading)
        for name in ALL_FEATURES:
            a, b = float(row[name]), float(live[name])
            if math.isnan(a) and math.isnan(b):
                continue
            if math.isnan(a) != math.isnan(b) or not np.isclose(a, b, rtol=1e-6, atol=1e-6):
                mismatches += 1
                if mismatches <= 10:
                    print(f"MISMATCH {name} at {row['timestamp']} {row['node_id']}: batch={a} live={b}")
            else:
                worst = max(worst, abs(a - b))

    checked = len(batch) * len(ALL_FEATURES)
    print(f"Checked {checked} feature values over {len(batch)} rows; mismatches={mismatches}; "
          f"max abs diff={worst:.2e}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
