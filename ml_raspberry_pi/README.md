# Disaster ML on Raspberry Pi (no forecasting)

Flood, landslide, wildfire and extreme-heat early warning, ported from the
Colab notebook `sihh.ipynb` to plain Python that runs on a Raspberry Pi. The
same code runs in Colab via `disaster_ml_colab.ipynb`.

Pipeline: **simulate the environment → apply realistic sensor errors per node
→ train → calibrate → choose WATCH/WARNING thresholds on a validation set →
locked tests → per-disaster scoring → live inference service**.

> Every number below is measured on **synthetic** data from this simulator.
> Real-world accuracy is unknown until labelled readings from your own nodes
> are scored with `evaluate_csv.py`.

## Results (V4, locked tests, WARNING level)

Locked tests are never used for training or tuning. Each holds 29–43
disasters per hazard. Scores are per disaster:

| Hazard | Caught: normal / failing sensors / new climate | Warned before the event | Median warning time | False alarms per site per day |
|---|---|---|---|---|
| Flood | 100 % / 98 % / 97 % | 100 % | 160 min | 0.07 |
| Landslide | 100 % / 100 % / 100 % | 93 % | 110 min | 0.04 |
| Wildfire | 100 % / 100 % / 100 % | 95 % | 110 min | 0.004 |
| Extreme heat | 95 % / 88 % / 84 % | 81 % | 155 min | 0.19 |

The previous version, scored on the same test data the same way:

| Hazard | Before | Now |
|---|---|---|
| Flood | 100 / 98 / 97 %, 0.02 false alarms/day | 100 / 98 / 97 %, 0.07 |
| Landslide | 55 / 35 / 58 %, 0.12 | 100 / 100 / 100 %, 0.04 |
| Wildfire | 93 / 68 / 74 %, 0.41 | 100 / 100 / 100 %, 0.004 |
| Extreme heat | no model | 95 / 88 / 84 %, 0.19 |

Flood did not improve. It was already good, and it now warns later (median
160 min vs 310 min) with slightly more false alarms.

Per-reading precision looks low (0.28–0.54). That is expected: the label
only covers the last 15 minutes before an event, so an alert 2 hours early
counts as a "false positive" per reading even though it is a useful
warning. That is why thresholds and PASS/FAIL now use per-disaster scores.
The per-reading numbers are still in `output/reports/`.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Simulates the true environment and writes per-node measured readings → `data/` |
| `sensor_errors.py` | Sensor noise, calibration, drift, cross-sensitivity, failures, packet loss |
| `features.py` | Feature engineering, used identically in training and on the Pi |
| `train.py` | One model per hazard → `output/models/*.joblib`, `output/reports/*.csv` |
| `event_metrics.py` | Per-disaster scoring on the locked tests |
| `live_inference.py` | Live service: 3-day memory per node, returns SAFE / WATCH / WARNING |
| `evaluate_csv.py` | Score any CSV, e.g. your own labelled sensor data |
| `test_feature_parity.py`, `test_live_vs_batch.py` | Prove the Pi computes exactly what training computed |
| `config.py`, `metrics.py` | Shared settings and scoring |
| `run_all.sh` | One command: venv → install → generate → train → tests → scores |
| `disaster-ml.service` | systemd unit to start the service at boot |
| `disaster_ml_colab.ipynb`, `build_colab_notebook.py` | Colab notebook, built from the files above |

## Quick start on the Pi

Requirements: a Raspberry Pi 4 or 5 with **2 GB+ RAM** (training peaks at
~680 MB; the dataset is ~320 MB on disk), Raspberry Pi OS 64-bit (Bookworm),
Python 3.11 (Bookworm's default).

```bash
unzip ml_raspberry_pi.zip -d ~ && cd ~/ml_raspberry_pi
./run_all.sh              # full dataset + training + tests (~3 min here, est. 10-20 min on a Pi 4)
source .venv/bin/activate
python3 live_inference.py serve --port 5001
```

`./run_all.sh --quick` is a 1-minute smoke test. Its models are too weak to
deploy.

The zip already contains models trained with scikit-learn 1.9.1 in
`output/models/`. To use them without retraining:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pi.txt     # scikit-learn==1.9.1, numpy 2.x
python3 live_inference.py serve --port 5001
curl -s localhost:5001/health          # in a second terminal
```

Pickled models need the same scikit-learn version that trained them, and
`live_inference.py` warns on a mismatch. Checked: scikit-learn 1.9.1
and numpy 2.4.6 have Python 3.11 ARM64 wheels (Bookworm's Python), and that
the models predict identically there. It has not been run on a real Pi.

Measured on a 4-core x86 cloud VM: generation 78 s, training 57 s, about
6 ms per live reading. These were not measured on a real Pi.

## Colab

Open `disaster_ml_colab.ipynb` in Colab and run all cells. It writes the same
modules, generates the dataset, trains, runs the tests and downloads
`pi_models.zip`. That zip holds the models plus a `requirements-pi.txt`
pinned to Colab's library versions. On the Pi, unzip it inside
`ml_raspberry_pi/` and run `pip install -r output/requirements-pi.txt`.

## Sending readings

`POST /predict` with one reading or a list of readings:

```bash
curl -s -X POST localhost:5001/predict -H 'Content-Type: application/json' -d '{
  "node_id": "node1", "site_id": "pi_site_01", "timestamp": "2026-10-07T10:05:00Z",
  "water_level_cm": 55, "rainfall_mm_h": 18, "soil_moisture_pct": 52,
  "temperature_c": 28.8, "humidity_pct": 80, "tilt_x_deg": 0.1, "tilt_y_deg": -0.2,
  "acceleration_g": 1.0, "node1_flood_score": 0.45, "node1_landslide_score": 0.10,
  "node1_flood_label": 0, "node1_landslide_label": 0
}'
```

* **node1** → flood + landslide: `water_level_cm`, `rainfall_mm_h`,
  `soil_moisture_pct`, `temperature_c`, `humidity_pct`, `tilt_x_deg`,
  `tilt_y_deg`, `acceleration_g`, `node1_*_score`, `node1_*_label`.
* **node2** → wildfire + extreme heat: `temperature_c`, `humidity_pct`,
  `smoke_raw`, `gas_raw`, `flame`, `node2_*_score`, `node2_*_label`.
* Send **one reading per node every 5 minutes**. A gap of more than 15
  minutes restarts that node's history, the same rule as in training. After a
  restart, the 24 h and 3-day baselines take that long to fill.
* An alert needs **2 consecutive readings** over the threshold (5 minutes),
  which filters single noisy readings.
* Send raw values. Do not filter them on the node; the models learned the
  sensors' error patterns.
* Set `ULTRASONIC_MOUNT_HEIGHT_CM` in `config.py` to the real transducer
  height. The water level is corrected for air temperature with it.
* Act on `level`, not `probability`. Thresholds for rare events are small
  numbers (around 0.001–0.02).

### Using it from the existing Node.js app

The service accepts `NODE_01`/`NODE_02` and the dashboard field names. But
the current dashboard simulator does not send what the models need. Its
`rain` is a boolean (suggesting a raindrop module, which cannot measure
mm/h), `tilt` is a single angle, and there is no acceleration, gas or TinyML
score.

## What changed in this version, and why

**Simulator (it was physically wrong):**
1. Event "memory" now carries over between readings and fades afterwards.
   Before, it reset every step, so heat waves added only ~2 °C and flame was
   never triggered.
2. Event signals now rise phase by phase and peak in the event itself,
   instead of running away (water used to hit the 280 cm cap).
3. Soil drains back to normal (it used to sit at ~100 %). Wet soil absorbs
   less rain.
4. Hard negatives (shower without flood, cooking smoke, hot day...) are
   episodes of 30 min–3 h. Before, they were scattered single rows that piled
   up rain to ~8.5 mm/h on normal days; now it's 0.7 mm/h.
5. A landslide moves in one direction. Before, it was a random walk.

**Labels (they were misleading):**
6. `*_within_15m` / `extreme_heat_within_60m` now mean 15 / 60 minutes.
   Before, they were 1 from the start of the pre-event phase, i.e. hours
   ahead, so lead times were inflated.

**Features (robust to real sensor errors):**
7. Water level corrected for air temperature (ultrasonic speed-of-sound
   error), 15-min median against echo outliers, 1 h standard deviation to
   catch stuck sensors, anomaly against the 24 h mean.
8. Smoke/gas as a ratio to their own 24 h median, which cancels per-unit
   gain and baseline drift of MQ sensors.
9. Temperature vs the same time yesterday and vs the 3-day mean. Heat waves
   are judged relative to the site, not in absolute degrees.
10. Soil and tilt anomalies vs 24 h, 1 h tilt change, 3 h rainfall, 1 h
    maximum acceleration.

**Thresholds and scoring:**
11. Thresholds are chosen per disaster on the validation set. WARNING is the
    most sensitive threshold with ≤ 0.2 false alarms per site per day and
    ≤ 1 % of normal time in alert. WATCH allows ≤ 1 per day and 5 %. Both
    limits are needed: counting alarms alone lets an always-on alarm pass.
12. The threshold grid goes below 0.01 (it used to stop there).
13. PASS/FAIL uses: ≥ 90 % of disasters caught, ≤ 0.5 false alarms per site
    per day, ≤ 2 % alert time, detection drop ≤ 15 % with failing sensors
    and ≤ 20 % in a new climate, plus calibration.

**Not changed:** model type and settings. Four configurations were compared
on the validation set and differed by about ±0.01 PR-AUC, so the small, fast
one stays.

Earlier changes: forecasting removed (it lost to "repeat the last
reading"), V2/V3 file mismatch fixed, realistic sensor model, larger test
sets, `node_id`/`day_of_week`/`month` dropped from features, calibrator
without `class_weight="balanced"`, no pandas needed at inference.

## Sensor error model (`sensor_errors.py`)

The parameters are typical values for the **assumed** parts below. Change
them to match your hardware.

| Sensor (assumed part) | Healthy-hardware errors |
|---|---|
| Water level (JSN-SR04T ultrasonic) | ±0.4 cm noise, growing with rain ripples; speed of sound vs air temperature (+6 cm at 35 °C with low water); 0.4 % echo outliers; 25 cm blind zone |
| Rainfall (tipping bucket, 0.2 mm/tip) | 0.2 mm steps (a 5-min reading is a multiple of 2.4 mm/h); 5–28 % undercatch |
| Soil moisture (capacitive v1.2) | per-unit gain 0.85–1.15, ±4 % offset; −0.25 %/°C; fouling drift |
| Temperature / humidity (DHT22) | ±0.3 °C / ±2 % offset; enclosure solar heating up to +4 °C; condensation bias above 92 % RH; 1 % read failures |
| Tilt / acceleration (MPU6050) | bias after tare; temperature drift; vibration noise |
| Smoke / gas (MQ-2 / MQ-135, ESP32 ADC) | gain ×0.6–1.6; temperature/humidity cross-sensitivity; baseline creep; ADC non-linearity and saturation at 4095; warm-up after power-on |
| Flame (IR module) | 30 % misses; sunlight false triggers |
| Radio | bursty packet loss, worse in heavy rain; lost readings are dropped |

Failures (3× as often in `locked_faults`): stuck value, disconnected sensor,
EMI spike burst, growing degradation drift, clogged rain gauge, brownout,
reboot, communication outage.

## Known limitations

* **Synthetic only.** The model has learned this simulator. Real disasters
  will differ in shape, timing and confusers. The sensor errors are the ones
  someone thought of, not all the ones that exist.
* **Warning times are hours because simulated events build up over hours.**
  Flash floods and sudden landslides give far less notice.
* **Extreme heat is the weakest:** 84–88 % caught with failing sensors or a
  new climate, and the most false alarms (0.19 per site per day).
* **WATCH equals WARNING for flood, landslide and wildfire.** The validation
  set had so few false alarms that both budgets picked the same threshold.
* **Thresholds are very small numbers** (0.001–0.02). That is normal for
  rare events with calibrated probabilities, but it means small calibration
  shifts on real data can move alert rates a lot. Re-tune on real data.
