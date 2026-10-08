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

DATA_SOURCE = "synthetic_v4"
MODEL_VERSION = "v4"
SAMPLE_INTERVAL_MINUTES = 5
RANDOM_SEED = 42

# Per-reading targets from the Colab pipeline. Still reported, but no longer
# used to pick thresholds: with honest "within 15 min" labels they count an
# alert 45 minutes early as a false positive.
WATCH_RECALL_TARGET = 0.90
WARNING_PRECISION_TARGET = 0.70

# Event-level threshold selection (on the threshold-validation set):
# the most sensitive threshold that stays within a false-alarm budget.
WARNING_MAX_FALSE_ALARMS = 0.2   # per site per day, i.e. about one false WARNING per site every 5 days
WATCH_MAX_FALSE_ALARMS = 1.0
WARNING_MAX_ALERT_TIME = 0.01    # share of non-event time spent in WARNING
WATCH_MAX_ALERT_TIME = 0.05

# Event-level acceptance gates (on locked tests, WARNING level)
MIN_DETECTION_RATE = 0.90
MAX_FALSE_ALARMS_PER_SITE_DAY = 0.5
MAX_ALERT_TIME_OUTSIDE_EVENTS = 0.02
MAX_FAULT_DETECTION_DROP = 0.15
MAX_OOD_DETECTION_DROP = 0.20

# Acceptance gates
MAX_ECE = 0.15
MAX_BRIER_SCORE = 0.25
MIN_PR_AUC_MARGIN_ABOVE_BASELINE = 0.05
MAX_FAULT_RECALL_DROP = 0.15
MAX_OOD_RECALL_DROP = 0.20

# A locked test with fewer disaster episodes than this cannot support a
# PASS/FAIL verdict; the hazard is reported as INSUFFICIENT_EVENTS instead.
MIN_TEST_EPISODES = 20

# Live alerts need this many consecutive readings over the threshold
ALERT_CONFIRM_READINGS = 2

# Hot-day threshold used by hot_window_fraction_1h
HOT_TEMPERATURE_C = 38.0

DATASET_FILES = {
    "train": "disaster_v4_train.csv",
    "calibration": "disaster_v4_calibration.csv",
    "threshold_validation": "disaster_v4_threshold_validation.csv",
    "locked_normal": "disaster_v4_locked_normal.csv",
    "locked_faults": "disaster_v4_locked_faults.csv",
    "locked_ood": "disaster_v4_locked_ood.csv",
}
MANIFEST_FILE = "dataset_v4_manifest.csv"

# A node's history restarts when readings are further apart than this
# (same rule in training features and on the Pi).
MAX_GAP_MINUTES = 3 * SAMPLE_INTERVAL_MINUTES

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
    "water_level_comp_cm",
    "rainfall_mm_h",
    "soil_moisture_pct",
    "temperature_c",
    "humidity_pct",
    "water_level_lag_1",
    "water_level_lag_3",
    "water_level_change_1",
    "water_level_change_3",
    "water_comp_change_1h",
    "water_comp_median_15min",
    "water_comp_std_1h",
    "water_level_anomaly_cm",
    "rainfall_15min_sum",
    "rainfall_1h_sum",
    "rainfall_3h_sum",
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
    "acceleration_1h_max",
    "soil_moisture_lag_1",
    "soil_moisture_change_1",
    "soil_moisture_change_1h",
    "soil_moisture_anomaly",
    "rainfall_1h_sum",
    "rainfall_3h_sum",
    "tilt_change_1",
    "tilt_change_3",
    "tilt_change_1h",
    "tilt_anomaly_deg",
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
    "smoke_median_15min",
    "gas_median_15min",
    "smoke_ratio_24h",
    "gas_ratio_24h",
    "smoke_change_1h",
    "temperature_anomaly_24h",
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
    "temperature_24h_mean",
    "temperature_anomaly_24h",
    "temperature_anomaly_72h",
    "temperature_1h_std",
    "temperature_change_24h",
    "humidity_1h_mean",
    "humidity_change_24h",
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
WINDOW_24H = 288
WINDOW_72H = 864
HISTORY_LENGTH = WINDOW_72H  # longest window the live engine must remember

# Water-level node geometry, used for speed-of-sound compensation. Set these
# to the real installation: transducer height above the channel bed, and the
# air temperature the firmware assumes for the speed of sound.
ULTRASONIC_MOUNT_HEIGHT_CM = 300.0
ULTRASONIC_ASSUMED_AIR_C = 20.0

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
