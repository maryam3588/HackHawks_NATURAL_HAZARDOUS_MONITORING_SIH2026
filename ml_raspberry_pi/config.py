"""Shared configuration for the Raspberry Pi disaster ML pipeline.

Everything that must stay identical between training and on-device
inference (feature names, window sizes, hazard definitions) lives here.
"""

import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("DISASTER_ML_HOME", Path(__file__).resolve().parent))

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

DATA_SOURCE = "synthetic_v3"
SAMPLE_INTERVAL_MINUTES = 5
RANDOM_SEED = 42

# Alert threshold selection targets (unchanged from the Colab pipeline)
WATCH_RECALL_TARGET = 0.90
WARNING_PRECISION_TARGET = 0.70

# Acceptance gates
MAX_ECE = 0.15
MAX_BRIER_SCORE = 0.25
MIN_PR_AUC_MARGIN_ABOVE_BASELINE = 0.05
MAX_FAULT_RECALL_DROP = 0.15
MAX_OOD_RECALL_DROP = 0.20

# Hot-day threshold used by hot_window_fraction_1h
HOT_TEMPERATURE_C = 38.0

DATASET_FILES = {
    "train": "disaster_v3_train.csv",
    "calibration": "disaster_v3_calibration.csv",
    "threshold_validation": "disaster_v3_threshold_validation.csv",
    "locked_normal": "disaster_v3_locked_normal.csv",
    "locked_faults": "disaster_v3_locked_faults.csv",
    "locked_ood": "disaster_v3_locked_ood.csv",
}
MANIFEST_FILE = "dataset_v3_manifest.csv"

# Raw sensor fields each reading must carry (missing values are allowed
# and become NaN; HistGradientBoosting handles NaN natively).
RAW_SENSOR_COLUMNS = [
    "water_level_cm",
    "rainfall_mm_h",
    "soil_moisture_pct",
    "temperature_c",
    "humidity_pct",
    "tilt_x_deg",
    "tilt_y_deg",
    "acceleration_g",
    "smoke_raw",
    "gas_raw",
    "flame",
    "node1_flood_score",
    "node1_landslide_score",
    "node1_flood_label",
    "node1_landslide_label",
    "node2_wildfire_score",
    "node2_extreme_heat_score",
    "node2_wildfire_label",
    "node2_extreme_heat_label",
]

# node_id, day_of_week and month were dropped from the Colab feature set:
# each hazard model only ever sees one node (so node_id is constant), and
# day/month only encode the synthetic sites' start dates, not hazard physics.
COMMON_FEATURES = [
    "hour_sin",
    "hour_cos",
]

FLOOD_FEATURES = COMMON_FEATURES + [
    "water_level_cm",
    "rainfall_mm_h",
    "soil_moisture_pct",
    "temperature_c",
    "humidity_pct",
    "water_level_lag_1",
    "water_level_lag_3",
    "water_level_change_1",
    "water_level_change_3",
    "rainfall_15min_sum",
    "rainfall_1h_sum",
    "water_level_15min_mean",
    "water_level_1h_mean",
    "node1_flood_score",
    "node1_flood_label",
]

LANDSLIDE_FEATURES = COMMON_FEATURES + [
    "rainfall_mm_h",
    "soil_moisture_pct",
    "tilt_x_deg",
    "tilt_y_deg",
    "tilt_magnitude_deg",
    "acceleration_g",
    "soil_moisture_lag_1",
    "soil_moisture_change_1",
    "rainfall_1h_sum",
    "tilt_change_1",
    "tilt_change_3",
    "node1_landslide_score",
    "node1_landslide_label",
]

WILDFIRE_FEATURES = COMMON_FEATURES + [
    "temperature_c",
    "humidity_pct",
    "smoke_raw",
    "gas_raw",
    "flame",
    "temperature_lag_1",
    "temperature_change_1",
    "humidity_lag_1",
    "humidity_change_1",
    "smoke_lag_1",
    "smoke_change_1",
    "gas_lag_1",
    "gas_change_1",
    "smoke_15min_mean",
    "gas_15min_mean",
    "node2_wildfire_score",
    "node2_wildfire_label",
]

EXTREME_HEAT_FEATURES = COMMON_FEATURES + [
    "temperature_c",
    "humidity_pct",
    "temperature_lag_1",
    "temperature_change_1",
    "humidity_lag_1",
    "humidity_change_1",
    "temperature_1h_mean",
    "temperature_3h_mean",
    "temperature_6h_mean",
    "temperature_6h_max",
    "hot_window_fraction_1h",
    "node2_extreme_heat_score",
    "node2_extreme_heat_label",
]

HAZARDS = {
    "flood": {
        "node_id": "node1",
        "target": "flood_within_15m",
        "features": FLOOD_FEATURES,
    },
    "landslide": {
        "node_id": "node1",
        "target": "landslide_within_15m",
        "features": LANDSLIDE_FEATURES,
    },
    "wildfire": {
        "node_id": "node2",
        "target": "wildfire_within_15m",
        "features": WILDFIRE_FEATURES,
    },
    "extreme_heat": {
        "node_id": "node2",
        "target": "extreme_heat_within_60m",
        "features": EXTREME_HEAT_FEATURES,
    },
}

TARGET_COLUMNS = [
    "flood_event",
    "landslide_event",
    "wildfire_event",
    "extreme_heat_event",
    "flood_within_15m",
    "landslide_within_15m",
    "wildfire_within_15m",
    "extreme_heat_within_60m",
]

# Rolling windows in samples (5-minute rows)
WINDOW_15MIN = 4
WINDOW_1H = 12
WINDOW_3H = 36
WINDOW_6H = 72
HISTORY_LENGTH = WINDOW_6H  # longest window the live engine must remember

# Accept the Node.js dashboard's node ids as aliases
NODE_ALIASES = {
    "node1": "node1",
    "node_01": "node1",
    "node01": "node1",
    "node2": "node2",
    "node_02": "node2",
    "node02": "node2",
}


def normalise_node_id(node_id):
    key = str(node_id).strip().lower()
    return NODE_ALIASES.get(key, key)
