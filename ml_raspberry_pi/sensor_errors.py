"""Realistic sensor measurement errors for the synthetic dataset.

The generator simulates the *true* environment. This module turns that into
what each node would actually report, per node and per sensor:

Always on (healthy hardware):
  * calibration: per-unit offset and gain
  * noise and quantisation (sensor resolution, ESP32 12-bit ADC)
  * environmental cross-sensitivity
      - ultrasonic speed of sound vs air temperature (uncompensated)
      - MQ gas sensors vs temperature and humidity
      - capacitive soil probe vs temperature
      - solar heating of the DHT22 enclosure around midday
      - humidity sensor condensation above ~92 % RH
  * tipping-bucket rain gauge: 0.2 mm per tip, undercatch in heavy rain
  * slow drift (MQ baseline, soil probe fouling)
  * ultrasonic echo outliers, blind zone near the transducer
  * ESP32 ADC non-linearity near the top of its range, saturation at 4095
  * DHT22 checksum read failures (NaN)
  * MQ warm-up after power-on
  * IR flame module false triggers in direct sunlight
  * bursty radio packet loss that gets worse in heavy rain; lost rows are
    dropped from the dataset, just like a missed reading on the Pi

Hardware failures (only when ``failures=True``):
  stuck value, disconnected sensor, EMI spike burst, degradation drift
  that keeps growing, clogged rain gauge, low-battery brownout, node
  reboot, communication outage.

The parameters are typical datasheet / field values for the assumed parts
listed in ``HARDWARE``; edit them to match your nodes.
"""

import numpy as np
import pandas as pd

STEP_MINUTES = 5
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES

HARDWARE = {
    "water_level_cm": "JSN-SR04T waterproof ultrasonic, mounted above the water",
    "rainfall_mm_h": "tipping-bucket rain gauge, 0.2 mm per tip",
    "soil_moisture_pct": "capacitive soil moisture probe v1.2 on ESP32 ADC",
    "temperature_c": "DHT22",
    "humidity_pct": "DHT22",
    "tilt_x_deg": "MPU6050 (tared at install)",
    "tilt_y_deg": "MPU6050 (tared at install)",
    "acceleration_g": "MPU6050",
    "smoke_raw": "MQ-2 on ESP32 12-bit ADC (0-4095)",
    "gas_raw": "MQ-135 on ESP32 12-bit ADC (0-4095)",
    "flame": "IR flame module, digital output",
}

NODE_SENSORS = {
    "node1": [
        "water_level_cm", "rainfall_mm_h", "soil_moisture_pct", "temperature_c",
        "humidity_pct", "tilt_x_deg", "tilt_y_deg", "acceleration_g",
    ],
    "node2": ["temperature_c", "humidity_pct", "smoke_raw", "gas_raw", "flame"],
}

ALL_SENSORS = [
    "water_level_cm", "rainfall_mm_h", "soil_moisture_pct", "temperature_c", "humidity_pct",
    "tilt_x_deg", "tilt_y_deg", "acceleration_g", "smoke_raw", "gas_raw", "flame",
]

TINYML_COLUMNS = {
    "node1": ["node1_flood_score", "node1_landslide_score", "node1_flood_label", "node1_landslide_label"],
    "node2": ["node2_wildfire_score", "node2_extreme_heat_score", "node2_wildfire_label", "node2_extreme_heat_label"],
}

ULTRASONIC_MOUNT_HEIGHT_CM = 300.0   # transducer height above the channel bed
ULTRASONIC_BLIND_ZONE_CM = 25.0
ULTRASONIC_MAX_RANGE_CM = 450.0
ULTRASONIC_ASSUMED_AIR_C = 20.0      # firmware uses a fixed speed of sound
RAIN_MM_PER_TIP = 0.2
ADC_MAX = 4095
ADC_KNEE = 3100                      # ESP32 ADC flattens above ~2.5 V

PHYSICAL_RANGE = {
    "water_level_cm": (0.0, ULTRASONIC_MOUNT_HEIGHT_CM - ULTRASONIC_BLIND_ZONE_CM),
    "rainfall_mm_h": (0.0, 400.0),
    "soil_moisture_pct": (0.0, 100.0),
    "temperature_c": (-40.0, 80.0),   # DHT22 range
    "humidity_pct": (0.0, 100.0),
    "tilt_x_deg": (-90.0, 90.0),
    "tilt_y_deg": (-90.0, 90.0),
    "acceleration_g": (-16.0, 16.0),
    "smoke_raw": (0.0, ADC_MAX),
    "gas_raw": (0.0, ADC_MAX),
}

# Expected failure episodes per node per day, and their duration range in
# 5-minute samples.
FAILURES = {
    "stuck": (0.04, 12, 288),
    "disconnected": (0.02, 12, 576),
    "spike_burst": (0.08, 1, 6),
    "rain_gauge_clog": (0.015, 288, 1440),
    "brownout": (0.04, 12, 72),
    "reboot": (0.05, 1, 4),
    "comm_outage": (0.03, 12, 144),
}
DRIFT_RUNAWAY_PROBABILITY = 0.35     # chance a node develops a growing drift


def speed_of_sound(temperature_c):
    return 331.3 + 0.606 * temperature_c


def esp32_adc(values, rng):
    """ADC noise, top-end compression, saturation and integer counts."""
    values = values + rng.normal(0, 12, values.shape)
    values = np.where(values > ADC_KNEE, ADC_KNEE + (values - ADC_KNEE) * 0.45, values)
    return np.round(np.clip(values, 0, ADC_MAX))


def quantise(values, step):
    return np.round(values / step) * step


def gilbert_elliott_loss(rain_mm_h, rng, base_bad=0.006, rain_bad=0.0006, recover=0.45):
    """Bursty packet loss; heavy rain attenuates the radio link."""
    n = len(rain_mm_h)
    lost = np.zeros(n, dtype=bool)
    bad = False
    draws = rng.random(n)
    loss_draws = rng.random(n)
    for i in range(n):
        if bad:
            bad = draws[i] >= recover
        else:
            bad = draws[i] < base_bad + rain_bad * rain_mm_h[i]
        lost[i] = bad and loss_draws[i] < 0.9
    return lost


def warmup_factor(n, boot_steps):
    """MQ heater warm-up: readings start ~2.5x high and settle in ~30 min."""
    factor = np.ones(n)
    for boot in boot_steps:
        k = np.arange(n - boot)
        factor[boot:] = np.maximum(factor[boot:], 1.0 + 1.5 * np.exp(-k / 2.5))
    return factor


def schedule_failures(n, rng, node_sensors, scale=1.0):
    """Return list of (type, sensor, start, end) failure episodes."""
    days = n / STEPS_PER_DAY
    episodes = []
    for failure, (rate, low, high) in FAILURES.items():
        if failure == "rain_gauge_clog" and "rainfall_mm_h" not in node_sensors:
            continue
        for _ in range(rng.poisson(rate * days * scale)):
            start = int(rng.integers(0, n))
            end = min(n, start + int(rng.integers(low, high + 1)))
            sensor = None
            if failure in ("stuck", "disconnected", "spike_burst"):
                sensor = str(rng.choice([s for s in node_sensors if s != "flame"]))
            episodes.append((failure, sensor, start, end))
    if rng.random() < min(1.0, DRIFT_RUNAWAY_PROBABILITY * scale):
        sensor = str(rng.choice([s for s in node_sensors if s in
                                 ("water_level_cm", "soil_moisture_pct", "temperature_c", "smoke_raw", "gas_raw")]))
        episodes.append(("drift_runaway", sensor, int(rng.integers(0, n)), n))
    return episodes


def measure_node(true_df, node_id, rng, failures, failure_scale=1.0):
    """Measured readings for one node at one site.

    ``true_df`` holds the true environment for consecutive 5-minute steps.
    Returns (measured DataFrame with only this node's sensors filled,
    per-row fault labels, boolean mask of rows that were delivered).
    """
    n = len(true_df)
    sensors = NODE_SENSORS[node_id]
    timestamps = pd.to_datetime(true_df["timestamp"])
    hour = (timestamps.dt.hour + timestamps.dt.minute / 60).to_numpy()
    day_index = np.arange(n) // STEPS_PER_DAY
    days_elapsed = np.arange(n) / STEPS_PER_DAY

    t_air = true_df["temperature_c"].to_numpy(float)
    rh_air = true_df["humidity_pct"].to_numpy(float)
    rain = true_df["rainfall_mm_h"].to_numpy(float)

    measured = {}
    fault_labels = [[] for _ in range(n)]

    # Daily sunshine (0 = overcast, 1 = clear) drives enclosure heating
    sunshine = rng.uniform(0, 1, day_index.max() + 1)[day_index]
    solar_shape = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    solar_bias = rng.uniform(1.0, 4.0) * sunshine * solar_shape

    # ---- DHT22 temperature / humidity ----------------------------------
    temperature = t_air + rng.normal(0, 0.3) + solar_bias + rng.normal(0, 0.2, n)
    humidity = rh_air + rng.normal(0, 2.0) - 3.0 * solar_bias + rng.normal(0, 1.0, n)
    humidity = humidity + np.where(rh_air > 92, rng.uniform(2, 6, n), 0)
    dht_fail = rng.random(n) < 0.01
    measured["temperature_c"] = np.where(dht_fail, np.nan, quantise(temperature, 0.1))
    measured["humidity_pct"] = np.where(dht_fail, np.nan, quantise(np.clip(humidity, 0, 100), 0.1))

    if node_id == "node1":
        # ---- Ultrasonic water level --------------------------------------
        level = true_df["water_level_cm"].to_numpy(float)
        distance = ULTRASONIC_MOUNT_HEIGHT_CM - level
        distance = distance * speed_of_sound(ULTRASONIC_ASSUMED_AIR_C) / speed_of_sound(t_air)
        distance = distance + rng.normal(0, 0.4 + 0.03 * rain, n)  # surface ripples in rain
        echo_outlier = rng.random(n) < 0.004
        distance = np.where(echo_outlier, rng.uniform(ULTRASONIC_BLIND_ZONE_CM, ULTRASONIC_MAX_RANGE_CM, n), distance)
        in_blind_zone = distance < ULTRASONIC_BLIND_ZONE_CM
        distance = np.where(in_blind_zone, np.nan, np.minimum(distance, ULTRASONIC_MAX_RANGE_CM))
        measured["water_level_cm"] = quantise(np.clip(ULTRASONIC_MOUNT_HEIGHT_CM - distance, 0, None), 0.1)

        # ---- Tipping bucket ----------------------------------------------
        catch = np.clip(0.95 - 0.0015 * rain, 0.7, 1.0)
        collected = np.cumsum(rain * STEP_MINUTES / 60 * catch)
        tips = np.diff(np.floor(collected / RAIN_MM_PER_TIP), prepend=0)
        measured["rainfall_mm_h"] = tips * RAIN_MM_PER_TIP * 60 / STEP_MINUTES

        # ---- Capacitive soil moisture ------------------------------------
        soil = true_df["soil_moisture_pct"].to_numpy(float)
        soil = rng.uniform(0.85, 1.15) * soil + rng.normal(0, 4.0)
        soil = soil - 0.25 * (t_air - 25) + rng.uniform(0, 0.1) * days_elapsed + rng.normal(0, 0.8, n)
        measured["soil_moisture_pct"] = quantise(np.clip(soil, 0, 100), 0.1)

        # ---- MPU6050 -----------------------------------------------------
        accel_true = true_df["acceleration_g"].to_numpy(float)
        vibration = np.clip(accel_true - 1.0, 0, None)
        for axis in ("tilt_x_deg", "tilt_y_deg"):
            tilt = true_df[axis].to_numpy(float) + rng.normal(0, 0.15)
            tilt = tilt + 0.01 * (t_air - 25) + rng.normal(0, 0.05 + 2.0 * vibration, n)
            measured[axis] = quantise(tilt, 0.01)
        measured["acceleration_g"] = quantise(
            accel_true + rng.normal(0, 0.015) + rng.normal(0, 0.004, n), 0.001
        )

    else:
        # ---- MQ-2 / MQ-135 on the ESP32 ADC -----------------------------
        environment = 1 + 0.004 * (rh_air - 65) + 0.006 * (t_air - 25)
        for column in ("smoke_raw", "gas_raw"):
            true_value = true_df[column].to_numpy(float)
            gain = rng.lognormal(0, 0.25)
            drift = 1 + rng.uniform(0, 0.015) * days_elapsed  # baseline creep, up to 1.5 %/day
            value = (true_value * gain + rng.normal(0, 30)) * environment * drift
            measured[column] = value * (1 + rng.normal(0, 0.02, n)) + rng.normal(0, 6, n)

        # ---- IR flame module --------------------------------------------
        flame = true_df["flame"].to_numpy(float)
        detected = (flame > 0) & (rng.random(n) > 0.3)
        sun_trigger = (rng.random(n) < 0.003 * sunshine) & (hour >= 9) & (hour <= 16)
        measured["flame"] = (detected | sun_trigger).astype(float)

    # ---- Hardware failures --------------------------------------------------
    delivered = ~gilbert_elliott_loss(rain, rng)
    boot_steps = [0]

    if failures:
        for failure, sensor, start, end in schedule_failures(n, rng, sensors, failure_scale):
            window = slice(start, end)
            if failure == "stuck":
                held = measured[sensor][start]
                measured[sensor][window] = held
            elif failure == "disconnected":
                if sensor in ("temperature_c", "humidity_pct"):
                    measured[sensor][window] = np.nan
                elif sensor == "water_level_cm":
                    measured[sensor][window] = 0.0  # no echo -> max range -> level 0
                else:
                    measured[sensor][window] = 0.0  # floating / grounded ADC pin
            elif failure == "spike_burst":
                values = measured[sensor]
                low, high = np.nanpercentile(values, 1), np.nanpercentile(values, 99)
                values[window] = rng.uniform(low, high * 1.8 + 1, end - start)
            elif failure == "drift_runaway":
                ramp = (np.arange(n - start) / STEPS_PER_DAY) * {
                    "water_level_cm": 1.5,     # condensation / spider web on transducer
                    "soil_moisture_pct": 0.8,  # probe coating degradation
                    "temperature_c": 0.25,     # dirty radiation shield
                    "smoke_raw": 40.0,         # MQ sensor poisoning
                    "gas_raw": 50.0,
                }[sensor]
                measured[sensor][start:] = measured[sensor][start:] + ramp
            elif failure == "rain_gauge_clog":
                measured["rainfall_mm_h"][window] = 0.0
            elif failure == "brownout":
                for analog in ("soil_moisture_pct", "smoke_raw", "gas_raw"):
                    if analog in measured:
                        scale = rng.uniform(0.8, 0.92)
                        noisy = measured[analog][window] * scale
                        measured[analog][window] = noisy + rng.normal(0, 5, end - start)
            elif failure == "reboot":
                delivered[window] = False
                boot_steps.append(end)
            elif failure == "comm_outage":
                delivered[window] = False

            label = failure if sensor is None else f"{failure}:{sensor}"
            for index in range(start, end):
                fault_labels[index].append(label)

    if node_id == "node2":
        warm = warmup_factor(n, boot_steps)
        for column in ("smoke_raw", "gas_raw"):
            measured[column] = esp32_adc(measured[column] * warm, rng)

    # Nothing can report outside what the hardware is able to output
    for sensor, (low, high) in PHYSICAL_RANGE.items():
        if sensor in measured:
            measured[sensor] = np.clip(measured[sensor], low, high)

    frame = pd.DataFrame({column: np.full(n, np.nan) for column in ALL_SENSORS})
    for sensor in sensors:
        frame[sensor] = measured[sensor]

    labels = np.array(["+".join(f) if f else "normal" for f in fault_labels], dtype=object)
    return frame, labels, delivered


def ideal_node(true_df, node_id):
    """Perfect sensors: measured == true for this node's sensors."""
    n = len(true_df)
    frame = pd.DataFrame({column: np.full(n, np.nan) for column in ALL_SENSORS})
    for sensor in NODE_SENSORS[node_id]:
        frame[sensor] = true_df[sensor].to_numpy(float)
    return frame, np.array(["normal"] * n, dtype=object), np.ones(n, dtype=bool)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tinyml_scores(measured, node_id, rng):
    """On-node TinyML scores computed from what the node measured.

    Firmware holds the last good reading when a sensor read fails, so NaNs
    are forward-filled; before the first good reading the score is NaN.
    """
    m = measured.ffill()
    n = len(m)
    out = {}
    if node_id == "node1":
        tilt = np.sqrt(m["tilt_x_deg"] ** 2 + m["tilt_y_deg"] ** 2)
        flood = _sigmoid(-6.0 + 0.055 * m["water_level_cm"] + 0.045 * m["rainfall_mm_h"]
                         + 0.015 * m["soil_moisture_pct"] + rng.normal(0, 1.1, n))
        landslide = _sigmoid(-7.0 + 0.035 * m["soil_moisture_pct"] + 0.25 * tilt
                             + 1.20 * np.clip(m["acceleration_g"] - 1.0, 0, None)
                             + 0.020 * m["rainfall_mm_h"] + rng.normal(0, 1.1, n))
        scores = {"node1_flood": flood, "node1_landslide": landslide}
    else:
        wildfire = _sigmoid(-7.0 + 0.045 * m["temperature_c"] - 0.014 * m["humidity_pct"]
                            + 0.0030 * m["smoke_raw"] + 0.0010 * m["gas_raw"]
                            + 0.75 * m["flame"] + rng.normal(0, 1.2, n))
        heat = _sigmoid(-9.0 + 0.16 * m["temperature_c"] - 0.012 * m["humidity_pct"]
                        + rng.normal(0, 1.15, n))
        scores = {"node2_wildfire": wildfire, "node2_extreme_heat": heat}

    for column in sum(TINYML_COLUMNS.values(), []):
        out[column] = np.full(n, np.nan)
    for name, score in scores.items():
        score = np.asarray(score, dtype=float)
        out[f"{name}_score"] = score
        out[f"{name}_label"] = np.where(np.isnan(score), np.nan, (score >= 0.60).astype(float))
    return pd.DataFrame(out)
