"""Causal feature engineering, in two forms that must agree exactly.

* ``prepare_dataframe``      - vectorised pandas version (training, CSV evaluation)
* ``StreamingFeatureEngine`` - dependency-light version for live readings on
  the Pi; keeps the last ``HISTORY_LENGTH`` readings per (site, node)

Both are driven by the same specs (``DERIVED``, ``LAGS``, ``ROLLING``), and
``test_feature_parity.py`` checks they produce the same numbers.

Windows count *readings*. Both versions restart a node's history when the
gap between readings exceeds ``config.MAX_GAP_MINUTES``.
"""

import math
import statistics
from collections import deque
from datetime import datetime, timezone

import numpy as np

import config as C

GROUP_COLUMNS = ["site_id", "node_id", "_segment"]

W15, W1H, W3H, W6H, W24H, W72H = (C.WINDOW_15MIN, C.WINDOW_1H, C.WINDOW_3H, C.WINDOW_6H,
                                  C.WINDOW_24H, C.WINDOW_72H)

# (feature name, source column, steps back); "<name>_change_<k>" features are
# derived as current minus lag, see CHANGES.
LAGS = [
    ("water_level_lag_1", "water_level_cm", 1),
    ("water_level_lag_3", "water_level_cm", 3),
    ("soil_moisture_lag_1", "soil_moisture_pct", 1),
    ("temperature_lag_1", "temperature_c", 1),
    ("humidity_lag_1", "humidity_pct", 1),
    ("smoke_lag_1", "smoke_raw", 1),
    ("gas_lag_1", "gas_raw", 1),
    ("tilt_lag_1", "tilt_magnitude_deg", 1),
    ("tilt_lag_3", "tilt_magnitude_deg", 3),
    ("water_comp_lag_12", "water_level_comp_cm", W1H),
    ("soil_moisture_lag_12", "soil_moisture_pct", W1H),
    ("tilt_lag_12", "tilt_magnitude_deg", W1H),
    ("smoke_median_lag_12", "smoke_median_15min", W1H),
    # Same time yesterday: removes the day/night cycle for heat waves
    ("temperature_1h_mean_lag_24h", "temperature_1h_mean", W24H),
    ("humidity_1h_mean_lag_24h", "humidity_1h_mean", W24H),
]

# (feature name, column, lag feature)
CHANGES = [
    ("water_level_change_1", "water_level_cm", "water_level_lag_1"),
    ("water_level_change_3", "water_level_cm", "water_level_lag_3"),
    ("soil_moisture_change_1", "soil_moisture_pct", "soil_moisture_lag_1"),
    ("temperature_change_1", "temperature_c", "temperature_lag_1"),
    ("humidity_change_1", "humidity_pct", "humidity_lag_1"),
    ("smoke_change_1", "smoke_raw", "smoke_lag_1"),
    ("gas_change_1", "gas_raw", "gas_lag_1"),
    ("tilt_change_1", "tilt_magnitude_deg", "tilt_lag_1"),
    ("tilt_change_3", "tilt_magnitude_deg", "tilt_lag_3"),
    ("water_comp_change_1h", "water_level_comp_cm", "water_comp_lag_12"),
    ("soil_moisture_change_1h", "soil_moisture_pct", "soil_moisture_lag_12"),
    ("tilt_change_1h", "tilt_magnitude_deg", "tilt_lag_12"),
    ("smoke_change_1h", "smoke_median_15min", "smoke_median_lag_12"),
    ("temperature_change_24h", "temperature_1h_mean", "temperature_1h_mean_lag_24h"),
    ("humidity_change_24h", "humidity_1h_mean", "humidity_1h_mean_lag_24h"),
]

# Rolling windows, computed in this order (later specs may use earlier ones).
# (feature name, column, window, statistic)
ROLLING_STAGE_1 = [
    ("rainfall_15min_sum", "rainfall_mm_h", W15, "sum"),
    ("rainfall_1h_sum", "rainfall_mm_h", W1H, "sum"),
    ("rainfall_3h_sum", "rainfall_mm_h", W3H, "sum"),
    ("water_level_15min_mean", "water_level_cm", W15, "mean"),
    ("water_level_1h_mean", "water_level_cm", W1H, "mean"),
    ("water_comp_median_15min", "water_level_comp_cm", W15, "median"),
    ("water_comp_std_1h", "water_level_comp_cm", W1H, "std"),
    ("soil_moisture_24h_mean", "soil_moisture_pct", W24H, "mean"),
    ("tilt_24h_median", "tilt_magnitude_deg", W24H, "median"),
    ("acceleration_1h_max", "acceleration_g", W1H, "max"),
    ("smoke_15min_mean", "smoke_raw", W15, "mean"),
    ("gas_15min_mean", "gas_raw", W15, "mean"),
    ("smoke_median_15min", "smoke_raw", W15, "median"),
    ("gas_median_15min", "gas_raw", W15, "median"),
    ("smoke_24h_median", "smoke_raw", W24H, "median"),
    ("gas_24h_median", "gas_raw", W24H, "median"),
    ("temperature_1h_mean", "temperature_c", W1H, "mean"),
    ("temperature_3h_mean", "temperature_c", W3H, "mean"),
    ("temperature_6h_mean", "temperature_c", W6H, "mean"),
    ("temperature_6h_max", "temperature_c", W6H, "max"),
    ("temperature_24h_mean", "temperature_c", W24H, "mean"),
    ("temperature_72h_mean", "temperature_c", W72H, "mean"),
    ("temperature_1h_std", "temperature_c", W1H, "std"),
    ("humidity_1h_mean", "humidity_pct", W1H, "mean"),
]

# Ratios against a slow baseline cancel per-unit gain and baseline drift of
# MQ sensors; +10 keeps a disconnected (0) sensor from dividing by zero.
RATIO_OFFSET = 10.0


def _speed_of_sound(temperature_c):
    return 331.3 + 0.606 * temperature_c


def compensate_water_level(level_cm, temperature_c):
    """Undo the ultrasonic speed-of-sound error using the node's own DHT22."""
    height = C.ULTRASONIC_MOUNT_HEIGHT_CM
    factor = _speed_of_sound(temperature_c) / _speed_of_sound(C.ULTRASONIC_ASSUMED_AIR_C)
    return height - (height - level_cm) * factor


def _finish(features):
    """Features computed from other features (same code for both paths)."""
    features["water_level_anomaly_cm"] = features["water_comp_median_15min"] - features["water_comp_24h_mean"]
    features["soil_moisture_anomaly"] = features["soil_moisture_pct"] - features["soil_moisture_24h_mean"]
    features["tilt_anomaly_deg"] = features["tilt_magnitude_deg"] - features["tilt_24h_median"]
    features["smoke_ratio_24h"] = (features["smoke_median_15min"] + RATIO_OFFSET) / (
        features["smoke_24h_median"] + RATIO_OFFSET)
    features["gas_ratio_24h"] = (features["gas_median_15min"] + RATIO_OFFSET) / (
        features["gas_24h_median"] + RATIO_OFFSET)
    features["temperature_anomaly_24h"] = features["temperature_1h_mean"] - features["temperature_24h_mean"]
    features["temperature_anomaly_72h"] = features["temperature_3h_mean"] - features["temperature_72h_mean"]
    return features


ROLLING_STAGE_2 = [
    ("water_comp_24h_mean", "water_comp_median_15min", W24H, "mean"),
]


# ----------------------------------------------------------------------
# Batch (pandas) version
# ----------------------------------------------------------------------

def prepare_dataframe(dataframe):
    import pandas as pd

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce", utc=True)
    dataframe = dataframe.dropna(subset=["timestamp", "site_id", "node_id"]).copy()
    dataframe["site_id"] = dataframe["site_id"].astype(str)
    dataframe["node_id"] = dataframe["node_id"].map(C.normalise_node_id)

    for column in C.RAW_SENSOR_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = np.nan
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe = dataframe.sort_values(["site_id", "node_id", "timestamp"]).reset_index(drop=True)

    for column in C.TARGET_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0).astype(int).clip(0, 1)

    gap_minutes = (
        dataframe.groupby(["site_id", "node_id"], sort=False)["timestamp"].diff().dt.total_seconds() / 60
    )
    dataframe["_segment"] = (gap_minutes.isna() | (gap_minutes > C.MAX_GAP_MINUTES)).cumsum()

    hour = dataframe["timestamp"].dt.hour
    dataframe["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    dataframe["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dataframe["tilt_magnitude_deg"] = np.sqrt(dataframe["tilt_x_deg"] ** 2 + dataframe["tilt_y_deg"] ** 2)
    compensated = compensate_water_level(dataframe["water_level_cm"], dataframe["temperature_c"])
    dataframe["water_level_comp_cm"] = compensated.fillna(dataframe["water_level_cm"])

    def rolling(column, window, how):
        grouped = dataframe.groupby(GROUP_COLUMNS, sort=False)[column]
        result = getattr(grouped.rolling(window=window, min_periods=1), how)()
        return result.reset_index(level=GROUP_COLUMNS, drop=True)

    for name, column, window, how in ROLLING_STAGE_1 + ROLLING_STAGE_2:
        dataframe[name] = rolling(column, window, how)

    grouped = dataframe.groupby(GROUP_COLUMNS, sort=False)
    for name, column, steps in LAGS:
        dataframe[name] = grouped[column].shift(steps)
    for name, column, lag in CHANGES:
        dataframe[name] = dataframe[column] - dataframe[lag]

    # Share of hot readings in the last hour; missing readings count in the
    # denominator but never as hot, an hour with no valid reading is NaN.
    dataframe["_hot"] = (dataframe["temperature_c"] >= C.HOT_TEMPERATURE_C).astype(float)
    dataframe["_valid_temp"] = dataframe["temperature_c"].notna().astype(float)
    hot_count = rolling("_hot", W1H, "sum")
    valid_count = rolling("_valid_temp", W1H, "sum")
    rows_in_window = (dataframe.groupby(GROUP_COLUMNS, sort=False).cumcount() + 1).clip(upper=W1H)
    dataframe["hot_window_fraction_1h"] = np.where(valid_count > 0, hot_count / rows_in_window, np.nan)

    _finish(dataframe)
    return dataframe.drop(columns=["_hot", "_valid_temp", "_segment"])


def feature_matrix(dataframe, feature_columns):
    """Float32 matrix in the exact column order the model was trained on."""
    return dataframe[feature_columns].to_numpy(dtype=np.float32)


# ----------------------------------------------------------------------
# Streaming version for live inference on the Pi
# ----------------------------------------------------------------------

def _to_float(value):
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_timestamp(value):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _valid(values):
    return [v for v in values if not math.isnan(v)]


def _stat(values, how):
    valid = _valid(values)
    if how == "std":
        return statistics.stdev(valid) if len(valid) >= 2 else math.nan
    if not valid:
        return math.nan
    if how == "sum":
        return math.fsum(valid)
    if how == "mean":
        return math.fsum(valid) / len(valid)
    if how == "max":
        return max(valid)
    if how == "median":
        return statistics.median(valid)
    raise ValueError(how)


class StreamingFeatureEngine:
    """Keeps a rolling history per (site, node) and emits features."""

    def __init__(self, max_gap_minutes=C.MAX_GAP_MINUTES):
        self.max_gap_minutes = max_gap_minutes
        self.history = {}
        self.last_timestamp = {}

    def update(self, reading):
        """Add one reading (dict) and return (node_id, features dict)."""
        node_id = C.normalise_node_id(reading.get("node_id", ""))
        site_id = str(reading.get("site_id", "default_site"))
        timestamp = _parse_timestamp(reading.get("timestamp"))
        key = (site_id, node_id)

        previous = self.last_timestamp.get(key)
        if previous is not None and self.max_gap_minutes is not None:
            gap = (timestamp - previous).total_seconds() / 60.0
            if gap < 0 or gap > self.max_gap_minutes:
                self.history.pop(key, None)
        self.last_timestamp[key] = timestamp

        row = {column: _to_float(reading.get(column)) for column in C.RAW_SENSOR_COLUMNS}
        row["tilt_magnitude_deg"] = math.sqrt(row["tilt_x_deg"] ** 2 + row["tilt_y_deg"] ** 2)
        compensated = compensate_water_level(row["water_level_cm"], row["temperature_c"])
        row["water_level_comp_cm"] = row["water_level_cm"] if math.isnan(compensated) else compensated
        hour = timestamp.hour
        row["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        row["hour_cos"] = math.cos(2 * math.pi * hour / 24)

        history = self.history.setdefault(key, deque(maxlen=C.HISTORY_LENGTH))
        history.append(row)

        # Rolling features are stored on the row so later stages (and lags of
        # rolling features) can read them from history.
        rows = list(history)
        for stage in (ROLLING_STAGE_1, ROLLING_STAGE_2):
            for name, column, window, how in stage:
                row[name] = _stat([r[column] for r in rows[-window:]], how)

        features = dict(row)
        for name, column, steps in LAGS:
            features[name] = rows[-1 - steps][column] if len(rows) > steps else math.nan
        for name, column, lag in CHANGES:
            features[name] = features[column] - features[lag]

        temps = [r["temperature_c"] for r in rows[-W1H:]]
        if all(math.isnan(t) for t in temps):
            features["hot_window_fraction_1h"] = math.nan
        else:
            features["hot_window_fraction_1h"] = sum(1 for t in temps if t >= C.HOT_TEMPERATURE_C) / len(temps)

        return node_id, _finish(features)
