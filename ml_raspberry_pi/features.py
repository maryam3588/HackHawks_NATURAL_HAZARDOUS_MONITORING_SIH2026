"""Causal feature engineering, in two forms that must agree exactly.

* ``prepare_dataframe``  - vectorised pandas version used for training and
  CSV evaluation.
* ``StreamingFeatureEngine`` - dependency-light version used on the
  Raspberry Pi for live readings. It keeps only the last ``HISTORY_LENGTH``
  rows per (site, node) in memory.

``test_feature_parity.py`` checks that both produce the same numbers.

Lags and rolling windows count *rows*, not minutes, exactly like the Colab
pipeline. If a node misses a 5-minute reading, the "15 min" window covers a
slightly longer span. Both versions restart a node's history when the gap
between readings exceeds ``config.MAX_GAP_MINUTES``.
"""

import math
from collections import deque
from datetime import datetime, timezone

import numpy as np

import config as C

GROUP_COLUMNS = ["site_id", "node_id", "_segment"]


# ----------------------------------------------------------------------
# Batch (pandas) version
# ----------------------------------------------------------------------

def prepare_dataframe(dataframe):
    import pandas as pd

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"], errors="coerce", utc=True
    )
    dataframe = dataframe.dropna(subset=["timestamp", "site_id", "node_id"]).copy()

    dataframe["site_id"] = dataframe["site_id"].astype(str)
    dataframe["node_id"] = dataframe["node_id"].map(C.normalise_node_id)

    for column in C.RAW_SENSOR_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = np.nan
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe = dataframe.sort_values(
        ["site_id", "node_id", "timestamp"]
    ).reset_index(drop=True)

    for column in C.TARGET_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = (
                pd.to_numeric(dataframe[column], errors="coerce")
                .fillna(0).astype(int).clip(0, 1)
            )

    # Start a new history segment after a gap, exactly like the live engine
    gap_minutes = (
        dataframe.groupby(["site_id", "node_id"], sort=False)["timestamp"].diff().dt.total_seconds() / 60
    )
    new_segment = gap_minutes.isna() | (gap_minutes > C.MAX_GAP_MINUTES)
    dataframe["_segment"] = new_segment.cumsum()

    hour = dataframe["timestamp"].dt.hour
    dataframe["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    dataframe["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dataframe["tilt_magnitude_deg"] = np.sqrt(
        dataframe["tilt_x_deg"] ** 2 + dataframe["tilt_y_deg"] ** 2
    )

    grouped = dataframe.groupby(GROUP_COLUMNS, sort=False)

    def lag(column, steps):
        return grouped[column].shift(steps)

    def rolling(column, window, how):
        return getattr(
            grouped[column].rolling(window=window, min_periods=1), how
        )().reset_index(level=GROUP_COLUMNS, drop=True)

    for name, column in [
        ("water_level", "water_level_cm"),
        ("soil_moisture", "soil_moisture_pct"),
        ("temperature", "temperature_c"),
        ("humidity", "humidity_pct"),
        ("smoke", "smoke_raw"),
        ("gas", "gas_raw"),
        ("tilt", "tilt_magnitude_deg"),
    ]:
        dataframe[f"{name}_lag_1"] = lag(column, 1)
        dataframe[f"{name}_change_1"] = dataframe[column] - dataframe[f"{name}_lag_1"]

    for name, column in [("water_level", "water_level_cm"), ("tilt", "tilt_magnitude_deg")]:
        dataframe[f"{name}_lag_3"] = lag(column, 3)
        dataframe[f"{name}_change_3"] = dataframe[column] - dataframe[f"{name}_lag_3"]

    dataframe["rainfall_15min_sum"] = rolling("rainfall_mm_h", C.WINDOW_15MIN, "sum")
    dataframe["rainfall_1h_sum"] = rolling("rainfall_mm_h", C.WINDOW_1H, "sum")
    dataframe["water_level_15min_mean"] = rolling("water_level_cm", C.WINDOW_15MIN, "mean")
    dataframe["water_level_1h_mean"] = rolling("water_level_cm", C.WINDOW_1H, "mean")
    dataframe["smoke_15min_mean"] = rolling("smoke_raw", C.WINDOW_15MIN, "mean")
    dataframe["gas_15min_mean"] = rolling("gas_raw", C.WINDOW_15MIN, "mean")
    dataframe["temperature_1h_mean"] = rolling("temperature_c", C.WINDOW_1H, "mean")
    dataframe["temperature_3h_mean"] = rolling("temperature_c", C.WINDOW_3H, "mean")
    dataframe["temperature_6h_mean"] = rolling("temperature_c", C.WINDOW_6H, "mean")
    dataframe["temperature_6h_max"] = rolling("temperature_c", C.WINDOW_6H, "max")

    # Vectorised equivalent of the Colab rolling(12).apply(mean(v >= 38)):
    # NaN rows count in the denominator but never as "hot"; a window with
    # no valid temperature at all yields NaN.
    dataframe["_hot"] = (dataframe["temperature_c"] >= C.HOT_TEMPERATURE_C).astype(float)
    dataframe["_valid_temp"] = dataframe["temperature_c"].notna().astype(float)
    grouped = dataframe.groupby(GROUP_COLUMNS, sort=False)
    hot_count = rolling("_hot", C.WINDOW_1H, "sum")
    valid_count = rolling("_valid_temp", C.WINDOW_1H, "sum")
    rows_in_window = (grouped.cumcount() + 1).clip(upper=C.WINDOW_1H)
    dataframe["hot_window_fraction_1h"] = np.where(
        valid_count > 0, hot_count / rows_in_window, np.nan
    )
    dataframe = dataframe.drop(columns=["_hot", "_valid_temp", "_segment"])

    return dataframe


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
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


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


def _nan_sum(values):
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) if valid else math.nan


def _nan_mean(values):
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def _nan_max(values):
    valid = [v for v in values if not math.isnan(v)]
    return max(valid) if valid else math.nan


class StreamingFeatureEngine:
    """Keeps a short rolling history per (site, node) and emits features."""

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
        row["tilt_magnitude_deg"] = math.sqrt(
            row["tilt_x_deg"] ** 2 + row["tilt_y_deg"] ** 2
        )

        history = self.history.setdefault(key, deque(maxlen=C.HISTORY_LENGTH))
        history.append(row)

        return node_id, self._features(history, timestamp)

    @staticmethod
    def _features(history, timestamp):
        rows = list(history)
        current = rows[-1]

        def column(name, window):
            return [r[name] for r in rows[-window:]]

        def lag(name, steps):
            return rows[-1 - steps][name] if len(rows) > steps else math.nan

        features = dict(current)
        hour = timestamp.hour
        features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        features["hour_cos"] = math.cos(2 * math.pi * hour / 24)

        for name, raw in [
            ("water_level", "water_level_cm"),
            ("soil_moisture", "soil_moisture_pct"),
            ("temperature", "temperature_c"),
            ("humidity", "humidity_pct"),
            ("smoke", "smoke_raw"),
            ("gas", "gas_raw"),
            ("tilt", "tilt_magnitude_deg"),
        ]:
            features[f"{name}_lag_1"] = lag(raw, 1)
            features[f"{name}_change_1"] = current[raw] - features[f"{name}_lag_1"]

        for name, raw in [("water_level", "water_level_cm"), ("tilt", "tilt_magnitude_deg")]:
            features[f"{name}_lag_3"] = lag(raw, 3)
            features[f"{name}_change_3"] = current[raw] - features[f"{name}_lag_3"]

        features["rainfall_15min_sum"] = _nan_sum(column("rainfall_mm_h", C.WINDOW_15MIN))
        features["rainfall_1h_sum"] = _nan_sum(column("rainfall_mm_h", C.WINDOW_1H))
        features["water_level_15min_mean"] = _nan_mean(column("water_level_cm", C.WINDOW_15MIN))
        features["water_level_1h_mean"] = _nan_mean(column("water_level_cm", C.WINDOW_1H))
        features["smoke_15min_mean"] = _nan_mean(column("smoke_raw", C.WINDOW_15MIN))
        features["gas_15min_mean"] = _nan_mean(column("gas_raw", C.WINDOW_15MIN))
        features["temperature_1h_mean"] = _nan_mean(column("temperature_c", C.WINDOW_1H))
        features["temperature_3h_mean"] = _nan_mean(column("temperature_c", C.WINDOW_3H))
        features["temperature_6h_mean"] = _nan_mean(column("temperature_c", C.WINDOW_6H))
        features["temperature_6h_max"] = _nan_max(column("temperature_c", C.WINDOW_6H))

        temps_1h = column("temperature_c", C.WINDOW_1H)
        if all(math.isnan(t) for t in temps_1h):
            features["hot_window_fraction_1h"] = math.nan
        else:
            hot = sum(1 for t in temps_1h if t >= C.HOT_TEMPERATURE_C)
            features["hot_window_fraction_1h"] = hot / len(temps_1h)

        return features
