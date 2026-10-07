"""Synthetic V4 dataset generator.

The environment simulation (weather, events, hard negatives) is the Colab
V3 logic unchanged. What changed in V4 is the sensor side: each node now
reports its own readings through ``sensor_errors.py`` (noise, calibration,
drift, cross-sensitivity, quantisation, failures, packet loss) instead of
both nodes sharing one perfect set of values.

Usage:
    python3 generate_dataset.py            # full size (~5 min on a Pi 4)
    python3 generate_dataset.py --quick    # small smoke-test dataset
"""

import argparse
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import sensor_errors as SE

RANDOM_SEED = 20260925
rng = np.random.default_rng(RANDOM_SEED)          # true environment
error_rng = np.random.default_rng(RANDOM_SEED + 1)  # sensor errors, separate so
                                                    # ideal vs realistic share the same truth

ROWS_PER_DAY = 288


def clip(value, low, high):
    return float(np.clip(value, low, high))


def sigmoid(value):
    return float(1.0 / (1.0 + np.exp(-value)))


SITE_PROFILES = {
    "humid_tropical": {"temp": 28.0, "humidity": 78.0, "rain_scale": 2.4, "water": 42.0, "soil": 48.0},
    "river_plain": {"temp": 27.0, "humidity": 68.0, "rain_scale": 1.8, "water": 48.0, "soil": 40.0},
    "mountain_wet": {"temp": 20.0, "humidity": 76.0, "rain_scale": 2.5, "water": 30.0, "soil": 52.0},
    "semi_arid": {"temp": 33.0, "humidity": 38.0, "rain_scale": 0.55, "water": 22.0, "soil": 22.0},
    "urban_hot": {"temp": 34.0, "humidity": 50.0, "rain_scale": 0.9, "water": 28.0, "soil": 28.0},
    "forest_dry": {"temp": 31.0, "humidity": 35.0, "rain_scale": 0.65, "water": 24.0, "soil": 24.0},
}

EVENT_TYPES = [
    "flood",
    "landslide",
    "wildfire",
    "extreme_heat",
    "flood_landslide",
    "wildfire_extreme_heat",
]

HARD_NEGATIVE_SCENARIOS = [
    "rain_without_flood",
    "wet_stable_slope",
    "traffic_vibration",
    "cooking_smoke",
    "vehicle_exhaust",
    "hot_day_no_heatwave",
]

PROGRESS_MAP = {
    "pre_event": 0.20,
    "watch": 0.48,
    "warning": 0.75,
    "event": 1.00,
    "recovery": 0.45,
}


def profile_name(site_index, ood=False):
    standard_profiles = ["humid_tropical", "river_plain", "mountain_wet", "semi_arid", "urban_hot"]
    all_profiles = list(SITE_PROFILES.keys())
    if ood:
        return all_profiles[site_index % len(all_profiles)]
    return standard_profiles[site_index % len(standard_profiles)]


def start_state(profile):
    p = SITE_PROFILES[profile]
    return {
        "water_level_cm": clip(rng.normal(p["water"], 4), 4, 180),
        "rainfall_mm_h": clip(rng.exponential(p["rain_scale"]), 0, 100),
        "soil_moisture_pct": clip(rng.normal(p["soil"], 5), 5, 95),
        "temperature_c": clip(rng.normal(p["temp"], 3), 0, 55),
        "humidity_pct": clip(rng.normal(p["humidity"], 8), 5, 100),
        "tilt_x_deg": clip(rng.normal(0, 0.2), -10, 10),
        "tilt_y_deg": clip(rng.normal(0, 0.2), -10, 10),
        "acceleration_g": clip(rng.normal(1.0, 0.01), 0.7, 1.5),
        "smoke_raw": clip(rng.normal(100, 15), 0, 4000),
        "gas_raw": clip(rng.normal(200, 25), 0, 6000),
        "flame": 0,
        "flood_memory": 0.0,
        "landslide_memory": 0.0,
        "wildfire_memory": 0.0,
        "heat_memory": 0.0,
        "slide_direction_x": 0.0,
        "slide_direction_y": 0.0,
    }


MEMORY_KEYS = ["flood_memory", "landslide_memory", "wildfire_memory", "heat_memory"]
MEMORY_DECAY = 0.92        # per step once an event is over (~1 h half-life)
# How strongly each phase drives the event signals (memory target)
PHASE_INTENSITY = {"pre_event": 0.15, "watch": 0.35, "warning": 0.6, "event": 1.0}


def normal_step(previous, profile, timestamp):
    p = SITE_PROFILES[profile]
    minutes = timestamp.hour * 60 + timestamp.minute
    temp_cycle = 5.5 * math.sin(2 * math.pi * (minutes - 8 * 60) / (24 * 60))
    humidity_cycle = -9.0 * math.sin(2 * math.pi * (minutes - 8 * 60) / (24 * 60))

    rain_pulse = rng.exponential(p["rain_scale"])
    if rng.random() < 0.84:
        rain_pulse *= 0.10

    rainfall = clip(0.82 * previous["rainfall_mm_h"] + 0.18 * rain_pulse + rng.normal(0, 0.2), 0, 150)
    water_level = clip(
        previous["water_level_cm"]
        + 0.03 * rainfall
        - 0.018 * max(previous["water_level_cm"] - p["water"], 0)
        + rng.normal(0, 0.18),
        0,
        280,
    )
    # V4 fix: soil drains back towards the site's normal moisture
    # (the Colab version only ever got wetter and sat at 100 %).
    soil = clip(
        previous["soil_moisture_pct"]
        + 0.07 * rainfall * (1 - previous["soil_moisture_pct"] / 100)  # wet soil absorbs less
        - 0.02 * (previous["soil_moisture_pct"] - p["soil"])
        + rng.normal(0, 0.18),
        0,
        100,
    )
    temperature = clip(
        0.94 * previous["temperature_c"] + 0.06 * (p["temp"] + temp_cycle) + rng.normal(0, 0.18),
        -5,
        70,
    )
    humidity = clip(
        0.93 * previous["humidity_pct"]
        + 0.07 * (p["humidity"] + humidity_cycle)
        + 0.025 * rainfall
        + rng.normal(0, 0.35),
        2,
        100,
    )
    tilt_x = clip(0.995 * previous["tilt_x_deg"] + rng.normal(0, 0.02), -20, 20)
    tilt_y = clip(0.995 * previous["tilt_y_deg"] + rng.normal(0, 0.02), -20, 20)
    acceleration = clip(1.0 + rng.normal(0, 0.008), 0.7, 2.0)
    smoke = clip(0.93 * previous["smoke_raw"] + 0.07 * rng.normal(100, 15) + rng.normal(0, 2), 0, 6000)
    gas = clip(0.93 * previous["gas_raw"] + 0.07 * rng.normal(200, 25) + rng.normal(0, 3), 0, 9000)

    values = {
        "water_level_cm": water_level,
        "rainfall_mm_h": rainfall,
        "soil_moisture_pct": soil,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "tilt_x_deg": tilt_x,
        "tilt_y_deg": tilt_y,
        "acceleration_g": acceleration,
        "smoke_raw": smoke,
        "gas_raw": gas,
        "flame": 0,
        "slide_direction_x": previous.get("slide_direction_x", 0.0),
        "slide_direction_y": previous.get("slide_direction_y", 0.0),
    }
    # V4 fix: event memory carries over between steps and fades afterwards
    # (the Colab version reset it to 0 every step, so it never built up).
    for key in MEMORY_KEYS:
        values[key] = previous.get(key, 0.0) * MEMORY_DECAY
    return values


def pull(value, target, rate):
    """Move ``value`` a fraction ``rate`` of the way towards ``target``."""
    return value + rate * (target - value)


def apply_event_dynamics(values, event_type, phase_progress, profile, phase):
    """Event physics as bounded relaxation towards event targets.

    Memory (0..1) builds up through pre_event -> watch -> warning -> event and
    sets how far each signal is pulled from normal. Recovery applies no
    forcing; memory then decays in normal_step.
    """
    values = values.copy()
    if phase == "recovery":
        return values

    p = SITE_PROFILES[profile]
    target = PHASE_INTENSITY[phase]

    def build(key, speed=0.08):
        # memory was decayed in normal_step; undo that, then move towards the
        # phase's intensity so the signal peaks in the event phase itself
        return min(1.0, pull(values[key] / MEMORY_DECAY, target, speed))

    if "flood" in event_type:
        m = values["flood_memory"] = build("flood_memory")
        values["rainfall_mm_h"] = clip(pull(values["rainfall_mm_h"], 4 + 40 * m, 0.25) + rng.normal(0, 1.0), 0, 150)
        values["soil_moisture_pct"] = clip(values["soil_moisture_pct"] + 0.3 * m, 0, 100)
        values["water_level_cm"] = clip(
            pull(values["water_level_cm"], p["water"] + 160 * m, 0.10) + rng.normal(0, 0.3), 0, 280
        )

    if "landslide" in event_type:
        m = values["landslide_memory"] = build("landslide_memory")
        if values["slide_direction_x"] == 0.0:
            # A slope fails in one direction: keep it for the whole event
            angle = rng.uniform(0, 2 * math.pi)
            values["slide_direction_x"] = math.cos(angle)
            values["slide_direction_y"] = math.sin(angle)
        values["rainfall_mm_h"] = clip(pull(values["rainfall_mm_h"], 3 + 25 * m, 0.2), 0, 150)
        values["soil_moisture_pct"] = clip(
            pull(values["soil_moisture_pct"], p["soil"] + 30 + 15 * m, 0.15), 0, 100
        )
        creep = 0.005 + 0.06 * m ** 2
        values["tilt_x_deg"] = clip(values["tilt_x_deg"] + values["slide_direction_x"] * creep, -20, 20)
        values["tilt_y_deg"] = clip(values["tilt_y_deg"] + values["slide_direction_y"] * creep, -20, 20)
        values["acceleration_g"] = clip(1.0 + 0.12 * m ** 2 * abs(rng.normal(0, 1)) + rng.normal(0, 0.008), 0.7, 2.0)

    if "wildfire" in event_type:
        m = values["wildfire_memory"] = build("wildfire_memory")
        values["temperature_c"] = clip(pull(values["temperature_c"], p["temp"] + 22 * m, 0.15), 0, 75)
        values["humidity_pct"] = clip(pull(values["humidity_pct"], p["humidity"] * (1 - 0.6 * m), 0.15), 2, 100)
        values["smoke_raw"] = clip(
            pull(values["smoke_raw"], 100 + 2600 * m, 0.2) + rng.normal(0, 20 + 80 * m), 0, 6000
        )
        values["gas_raw"] = clip(
            pull(values["gas_raw"], 200 + 3200 * m, 0.2) + rng.normal(0, 25 + 100 * m), 0, 9000
        )
        values["flame"] = int(m > 0.65 and rng.random() < 0.5)

    if "extreme_heat" in event_type:
        # Heat waves build slowly
        m = values["heat_memory"] = build("heat_memory", speed=0.05)
        values["temperature_c"] = clip(pull(values["temperature_c"], p["temp"] + 6 + 9 * m, 0.08), 0, 70)
        values["humidity_pct"] = clip(pull(values["humidity_pct"], p["humidity"] - 20 * m, 0.08), 2, 100)

    return values


def apply_hard_negative(values, scenario, profile):
    """Look-alike situations that are NOT disasters.

    V4: each scenario is a contiguous episode (30 min - 3 h) that pulls the
    signals towards a plausible level, instead of a random single row that
    adds on top of the previous value (which made rain pile up).
    """
    values = values.copy()
    p = SITE_PROFILES[profile]

    if scenario == "rain_without_flood":
        values["rainfall_mm_h"] = clip(pull(values["rainfall_mm_h"], rng.uniform(10, 30), 0.3), 0, 150)
    elif scenario == "wet_stable_slope":
        values["rainfall_mm_h"] = clip(pull(values["rainfall_mm_h"], rng.uniform(5, 15), 0.3), 0, 150)
        values["soil_moisture_pct"] = clip(pull(values["soil_moisture_pct"], p["soil"] + 30, 0.1), 0, 100)
    elif scenario == "traffic_vibration":
        values["acceleration_g"] = clip(values["acceleration_g"] + rng.uniform(0.05, 0.25), 0.7, 2.0)
    elif scenario == "cooking_smoke":
        values["smoke_raw"] = clip(pull(values["smoke_raw"], rng.uniform(400, 900), 0.3), 0, 6000)
        values["gas_raw"] = clip(pull(values["gas_raw"], rng.uniform(300, 700), 0.3), 0, 9000)
    elif scenario == "vehicle_exhaust":
        values["smoke_raw"] = clip(pull(values["smoke_raw"], rng.uniform(200, 450), 0.3), 0, 6000)
        values["gas_raw"] = clip(pull(values["gas_raw"], rng.uniform(700, 1500), 0.3), 0, 9000)
    elif scenario == "hot_day_no_heatwave":
        values["temperature_c"] = clip(pull(values["temperature_c"], p["temp"] + rng.uniform(4, 8), 0.1), 0, 65)
        values["humidity_pct"] = clip(pull(values["humidity_pct"], p["humidity"] - 10, 0.1), 2, 100)

    return values


def schedule_states(total_steps, event_fraction):
    states = ["normal"] * total_steps
    target_event_steps = int(total_steps * event_fraction)
    used_steps = 0

    possible_starts = list(range(100, total_steps - 250))
    rng.shuffle(possible_starts)

    for start in possible_starts:
        if used_steps >= target_event_steps:
            break

        length = int(rng.integers(60, 180))
        end = min(start + length, total_steps)

        if any(states[index] != "normal" for index in range(start, end)):
            continue

        event_type = rng.choice(EVENT_TYPES)

        for index in range(start, end):
            p = (index - start) / max(length - 1, 1)
            if p < 0.20:
                phase = "pre_event"
            elif p < 0.48:
                phase = "watch"
            elif p < 0.78:
                phase = "warning"
            else:
                phase = "event"
            states[index] = f"{event_type}:{phase}"

        recovery_end = min(end + 30, total_steps)
        for index in range(end, recovery_end):
            if states[index] == "normal":
                states[index] = f"{event_type}:recovery"

        used_steps += length

    # Hard negatives: contiguous episodes covering ~15 % of the time
    target_negative_steps = int(0.15 * total_steps)
    negative_steps = 0
    attempts = 0
    while negative_steps < target_negative_steps and attempts < 20 * total_steps:
        attempts += 1
        length = int(rng.integers(6, 37))
        start = int(rng.integers(0, max(1, total_steps - length)))
        window = range(start, start + length)
        if any(states[index] != "normal" for index in window):
            continue
        scenario = rng.choice(HARD_NEGATIVE_SCENARIOS)
        for index in window:
            states[index] = "hard_negative:" + scenario
        negative_steps += length

    return states


def state_parts(state):
    if state == "normal":
        return "normal", "normal"
    return state.split(":", 1)


def simulate_site_truth(states, profile, site_start):
    """True environment + labels for one site, one row per 5-minute step."""
    rows = []
    previous = start_state(profile)

    for step, state in enumerate(states):
        timestamp = site_start + timedelta(minutes=step * C.SAMPLE_INTERVAL_MINUTES)
        kind, detail = state_parts(state)
        values = normal_step(previous, profile, timestamp)

        event_type = "normal"
        event_phase = "normal"
        if kind in EVENT_TYPES:
            event_type = kind
            event_phase = detail
            values = apply_event_dynamics(
                values, event_type, PROGRESS_MAP.get(event_phase, 0.0), profile, event_phase
            )
        elif kind == "hard_negative":
            values = apply_hard_negative(values, detail, profile)
            event_phase = "hard_negative"
        if "landslide" not in event_type:
            values["slide_direction_x"] = values["slide_direction_y"] = 0.0

        labels = {
            "flood_event": int("flood" in event_type and event_phase == "event"),
            "landslide_event": int("landslide" in event_type and event_phase == "event"),
            "wildfire_event": int("wildfire" in event_type and event_phase == "event"),
            "extreme_heat_event": int("extreme_heat" in event_type and event_phase == "event"),
        }
        rows.append({
            "timestamp": timestamp.isoformat(),
            **{k: v for k, v in values.items() if not k.endswith("_memory") and not k.startswith("slide_")},
            **labels,
            "is_disaster": int(max(labels.values()) == 1),
            "event_subtype": event_type,
            "event_phase": event_phase,
            "scenario_type": state,
        })
        previous = values.copy()

    truth = pd.DataFrame(rows)
    # V4 fix: "within N minutes" now means it. Label = 1 when the event phase
    # is happening now or starts within the horizon. The Colab labels were 1
    # from the start of the pre-event phase, i.e. hours ahead.
    for target, event_column, minutes in [
        ("flood_within_15m", "flood_event", 15),
        ("landslide_within_15m", "landslide_event", 15),
        ("wildfire_within_15m", "wildfire_event", 15),
        ("extreme_heat_within_60m", "extreme_heat_event", 60),
    ]:
        steps = minutes // C.SAMPLE_INTERVAL_MINUTES
        future = truth[event_column][::-1].rolling(steps + 1, min_periods=1).max()[::-1]
        truth[target] = future.astype(int)
    return truth


def generate_dataset(out_dir, file_name, role, site_prefix, sites, days,
                     event_fraction, start_time, failures, ood, ideal_sensors):
    frames = []
    total_steps = days * ROWS_PER_DAY
    label_columns = [
        "flood_event", "landslide_event", "wildfire_event", "extreme_heat_event",
        "flood_within_15m", "landslide_within_15m", "wildfire_within_15m",
        "extreme_heat_within_60m", "is_disaster", "event_subtype", "event_phase", "scenario_type",
    ]

    for site_index in range(sites):
        site_id = f"{site_prefix}_{site_index + 1:02d}"
        profile = profile_name(site_index + 1, ood)
        states = schedule_states(total_steps, event_fraction)
        site_start = start_time + timedelta(days=site_index * (days + 4))
        truth = simulate_site_truth(states, profile, site_start)

        for node_id in ["node1", "node2"]:
            if ideal_sensors:
                measured, fault_labels, delivered = SE.ideal_node(truth, node_id)
            else:
                scale = 3.0 if role == "locked_faults" else 1.0  # make the fault test set fault-heavy
                measured, fault_labels, delivered = SE.measure_node(truth, node_id, error_rng, failures, scale)
            tinyml = SE.tinyml_scores(measured, node_id, error_rng)

            frame = pd.concat(
                [truth[["timestamp"]], measured, tinyml, truth[label_columns]], axis=1
            )
            frame.insert(1, "node_id", node_id)
            frame.insert(2, "site_id", site_id)
            frame.insert(3, "sensor_status", np.where(fault_labels == "normal", "NORMAL", "FAULT"))
            frame["data_source"] = C.DATA_SOURCE
            frame["dataset_role"] = role
            frame["site_climate_profile"] = profile
            frame["sensor_fault_type"] = fault_labels
            frame["sensor_model"] = "ideal" if ideal_sensors else "realistic"
            frame["packet_delivered"] = delivered
            frames.append(frame)

    dataframe = pd.concat(frames, ignore_index=True)
    expected_rows = len(dataframe)
    dataframe = dataframe[dataframe.pop("packet_delivered")].reset_index(drop=True)
    packet_loss = 1 - len(dataframe) / expected_rows

    output_path = out_dir / file_name
    dataframe.to_csv(output_path, index=False)
    return dataframe, output_path, packet_loss


# (file key, role, site prefix, sites, days, event fraction, start, failures, ood)
# Site counts are sized so every held-out set holds roughly 30 episodes per
# hazard; the Colab sizes gave 2-7, too few for a meaningful PASS/FAIL.
DATASET_SPECS = [
    ("train", "train", "train_site", 10, 30, 0.34, datetime(2025, 1, 1), True, False),
    ("calibration", "calibration", "calibration_site", 10, 25, 0.16, datetime(2025, 12, 1), True, False),
    ("threshold_validation", "threshold_validation", "threshold_site", 12, 25, 0.13, datetime(2026, 3, 1), True, False),
    ("locked_normal", "locked_normal", "locked_normal_site", 20, 25, 0.08, datetime(2026, 6, 1), False, False),
    ("locked_faults", "locked_faults", "locked_fault_site", 20, 25, 0.08, datetime(2026, 9, 1), True, False),
    ("locked_ood", "locked_ood", "locked_ood_site", 16, 25, 0.10, datetime(2026, 11, 1), True, True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=C.DATA_DIR, help="output folder")
    parser.add_argument("--quick", action="store_true",
                        help="small dataset (fewer sites/days) for a fast smoke test")
    parser.add_argument("--ideal-sensors", action="store_true",
                        help="perfect sensors (no noise, errors, failures or packet loss) for comparison")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    mode = "IDEAL sensors" if args.ideal_sensors else "REALISTIC sensors"
    print(f"Generating {C.DATA_SOURCE} datasets ({mode}) into {args.out}")

    manifest_rows = []
    for key, role, prefix, sites, days, fraction, start, failures, ood in DATASET_SPECS:
        if args.quick:
            sites = min(sites, 4 if key == "train" else 2)
            days = min(days, 10)
        started = time.time()
        dataframe, path, packet_loss = generate_dataset(
            args.out, C.DATASET_FILES[key], role, prefix, sites, days, fraction, start,
            failures, ood, args.ideal_sensors,
        )
        print(f"  {path.name}: {len(dataframe)} rows in {time.time() - started:.1f}s")

        manifest_rows.append({
            "file": path.name,
            "role": role,
            "rows": len(dataframe),
            "sites": dataframe["site_id"].nunique(),
            "sensor_model": "ideal" if args.ideal_sensors else "realistic",
            "packet_loss": round(packet_loss, 4),
            "fault_ratio": round((dataframe["sensor_status"] == "FAULT").mean(), 4),
            "flood_target_rate": round(dataframe["flood_within_15m"].mean(), 4),
            "landslide_target_rate": round(dataframe["landslide_within_15m"].mean(), 4),
            "wildfire_target_rate": round(dataframe["wildfire_within_15m"].mean(), 4),
            "extreme_heat_target_rate": round(dataframe["extreme_heat_within_60m"].mean(), 4),
            "data_source": C.DATA_SOURCE,
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(args.out / C.MANIFEST_FILE, index=False)

    print(f"\n{C.DATA_SOURCE.upper()} DATASET MANIFEST")
    print(manifest_df.to_string(index=False))
    print("\nDATASET GENERATION COMPLETE")


if __name__ == "__main__":
    main()
