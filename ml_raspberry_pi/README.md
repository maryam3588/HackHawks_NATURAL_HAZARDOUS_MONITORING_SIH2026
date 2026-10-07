# Disaster ML on Raspberry Pi (no forecasting)

Port of the Colab notebook `sihh.ipynb` (V3 dataset generator + training
pipeline + CSV tester) to plain Python that runs on a Raspberry Pi. The
forecasting phase is removed. V4 adds a realistic sensor-error model, so the
models are trained and tested on what real sensors report, not on perfect
values.

Pipeline: **simulate environment → apply sensor errors per node → train →
calibrate → pick WATCH/WARNING thresholds → locked tests → event-level
scoring → live inference service**.

## What runs where

| File | Purpose |
|---|---|
| `generate_dataset.py` | Simulates the true environment (Colab V3 logic) and writes per-node measured readings → `data/` |
| `sensor_errors.py` | Sensor noise, calibration, drift, cross-sensitivity, failures and packet loss (see below) |
| `train.py` | One classifier per hazard → `output/models/*.joblib`, `output/reports/*.csv` |
| `event_metrics.py` | Per-disaster scoring: detection rate, lead time, false alarms per site per day |
| `live_inference.py` | Live service: rolling 6 h memory per node, returns SAFE / WATCH / WARNING |
| `evaluate_csv.py` | Replaces the Colab "upload one CSV" tester |
| `test_feature_parity.py` | Proves the live features equal the training features |
| `config.py`, `features.py`, `metrics.py` | Shared code |
| `run_all.sh` | One command: venv → install → generate → train → parity test → event metrics |
| `disaster-ml.service` | systemd unit to start the service at boot |

## Requirements

* Raspberry Pi 4 or 5 with **2 GB+ RAM**. Training peaks at ~530 MB and the
  datasets take ~320 MB on disk.
* A Pi 3 / Zero 2 W can run `live_inference.py`. Train on a bigger machine
  and copy `output/models/` over (see "Training somewhere else").
* Raspberry Pi OS **64-bit** (Bookworm) recommended, Python 3.9+.

## Quick start on the Pi

```bash
unzip ml_raspberry_pi.zip -d ~ && cd ~/ml_raspberry_pi
./run_all.sh --quick      # smoke test only (~1-2 min); its models are too weak to deploy
./run_all.sh              # full dataset + training
source .venv/bin/activate
python3 live_inference.py serve --port 5001
```

Measured on a 4-core x86 cloud VM: generation 78 s, training 40 s, about
6 ms per live reading. Expect a Pi 4 to be roughly 3–6× slower. These Pi
numbers are estimates and were not measured on a real Pi.

## Sensor error model (`sensor_errors.py`)

Every node reports only its own sensors, through these errors. The
parameters are typical datasheet and field values for the **assumed**
hardware below. Change them to match your nodes.

| Sensor (assumed part) | Healthy-hardware errors |
|---|---|
| Water level (JSN-SR04T ultrasonic, 300 cm mount) | ±0.4 cm noise that grows with rain ripples; **speed of sound vs air temperature** (no compensation: about +6 cm at 35 °C with low water); 0.4 % echo outliers; 25 cm blind zone; 0.1 cm resolution |
| Rainfall (tipping bucket, 0.2 mm/tip) | 0.2 mm quantisation, so a 5-minute reading is a multiple of 2.4 mm/h; 5–28 % undercatch, rising with rain intensity |
| Soil moisture (capacitive v1.2) | per-unit gain 0.85–1.15 and ±4 % offset; −0.25 %/°C; fouling drift up to 0.1 %/day |
| Temperature / humidity (DHT22) | ±0.3 °C / ±2 % offset; **solar heating of the enclosure, up to +4 °C at midday** on clear days; condensation bias above 92 % RH; 1 % read failures (NaN) |
| Tilt / acceleration (MPU6050) | residual bias after tare; temperature drift; vibration noise |
| Smoke / gas (MQ-2 / MQ-135 on ESP32 ADC) | per-unit gain ×0.6–1.6; temperature/humidity cross-sensitivity; baseline creep up to 1.5 %/day; **ESP32 ADC non-linearity and saturation at 4095**; warm-up after power-on |
| Flame (IR module) | 30 % misses; false triggers in direct sunlight |
| Radio link | bursty packet loss (Gilbert–Elliott), **worse in heavy rain**; lost readings are dropped, as on the Pi |

Hardware failures (train, calibration, threshold, faults and OOD sets; 3×
as often in `locked_faults`): stuck value, disconnected sensor, EMI spike
burst, degradation drift that keeps growing, clogged rain gauge (1–5 days),
low-battery brownout, node reboot, communication outage. `sensor_fault_type`
names each active failure.

`python3 generate_dataset.py --ideal-sensors --out data_ideal` produces
the same environment with perfect sensors, for comparison.

## Results (synthetic V4, WARNING level, `event_metrics.py`)

Test sets: 27–47 disaster episodes per hazard.

| Hazard | Trained on perfect sensors, tested on perfect | Trained on perfect, **tested on realistic** | Trained on realistic, tested on realistic |
|---|---|---|---|
| Flood: detected / false alarms per site-day | 100 % / 0.01 | 100 % / **20.7** | 100 % / 2.0 |
| Landslide | 100 % / 1.1 | 100 % / **10.9** | 100 % / 1.1 |
| Wildfire | 100 % / 0.2 | 100 % / **44.1** | 100 % / 0.8 |
| Wildfire, faulty sensors | 100 % / 0.2 | 100 % / 46.5 | **68 %** / 0.7 |
| Extreme heat | 6 % | 12 % | no model |

The middle column is what happens if you train in Colab on clean data and
deploy: the models still catch the disasters, but they bury them in
false alarms. Training on realistic data fixes most of that. The cost is
wildfire detection when sensors are failing or the climate is unfamiliar
(68 %).

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

* **node1** → flood + landslide. Fields: `water_level_cm`, `rainfall_mm_h`,
  `soil_moisture_pct`, `temperature_c`, `humidity_pct`, `tilt_x_deg`,
  `tilt_y_deg`, `acceleration_g`, `node1_*_score`, `node1_*_label`.
* **node2** → wildfire + extreme heat. Fields: `temperature_c`,
  `humidity_pct`, `smoke_raw`, `gas_raw`, `flame`, `node2_*_score`,
  `node2_*_label`.
* Send **one reading per node every 5 minutes**. A gap of more than 15
  minutes resets that node's history, the same rule used in training.
* Send raw values. Do not clean or filter them on the node: the models
  learned the error patterns from raw readings.
* `GET /health` lists the loaded models.
* Offline alternative: `python3 live_inference.py stdin < sample_readings.jsonl`.

Act on `level`, not `probability`. Thresholds come from calibrated
probabilities and can be low for rare events.

### Using it from the existing Node.js app

The service accepts `NODE_01`/`NODE_02` and the dashboard field names
(`water_level`, `soil_moisture`, `temperature`, `humidity`, `smoke`, `tilt`).
However, the current simulator does **not** send what the models were
trained on:

* `rain` is a boolean, which suggests a raindrop module (e.g. YL-83). That
  cannot measure mm/h. The models need `rainfall_mm_h` from a
  tipping-bucket gauge, so `rain` is ignored.
* `tilt` is a single angle. It is mapped to `tilt_x_deg` with `tilt_y_deg = 0`.
* There is no `acceleration_g`, `gas_raw`, or TinyML `node*_score`/`label`.

## Training somewhere else (e.g. Colab or a laptop)

`.joblib` files are pickles, so the Pi must have **the same scikit-learn
version** as the training machine. `model_registry.json` records it, and
`live_inference.py` warns on a mismatch. The simplest option is to train on
the Pi.

## Changes from the Colab notebook

1. **Forecasting removed.** In the Colab run all four forecasters lost to
   "repeat the last reading" (MAE skill −0.22 to −0.81).
2. **V2→V3 mismatch fixed.** The Colab training cell read the V2 files and
   roles, so it could not run on the V3 generator's output.
3. **Realistic sensors (V4).** See the error model above. Each node now has its
   own readings; before, node1 and node2 rows were identical copies holding
   every sensor. The old faults are replaced: "stuck" used to round to the
   nearest 5 instead of freezing, and drift only lasted 2–15 readings.
4. **Larger held-out sets.** The Colab sizes gave 2–7 disasters per hazard
   per test set, so PASS/FAIL was mostly luck. Sets are now sized for ~30, and
   `train.py` reports `INSUFFICIENT_EVENTS` below 20.
5. **`node_id`, `day_of_week`, `month` dropped from features.**
6. **Calibrator without `class_weight="balanced"`.**
7. **No pandas at inference time.** The calibrator is stored as two floats,
   and datasets are loaded one phase at a time to fit in Pi RAM.
8. Colab-only code removed (`/content`, uploads/downloads, `display`, Excel).

## Known problems (not fixed — they change the simulation)

* **Extreme heat has no usable model.** Event "memory" in `normal_step()`
  resets to 0 every step, so heat events add only ~2 °C. The same bug means
  the true `flame` value is never 1; the only flame triggers in the data are
  sunlight false triggers.
* **The environment itself is unrealistic in places.** Soil moisture
  sits at ~100 % most of the time (it never drains), rainfall averages
  ~20 mm/h, and some temperatures exceed 60 °C.
* **"within_15m" labels are really "within hours".** The label is 1 from the
  start of the pre-event phase, which comes hours before the event. That is
  why lead times show up as 5–8 hours. Real disasters give far less warning.
* **WATCH thresholds often sit at the 0.01 grid floor,** or equal WARNING,
  so the WATCH tier carries little information.
* **Synthetic only.** The error model makes the data harder in realistic
  ways, but it only contains errors someone thought of. The only real test is
  labelled data from your own nodes (`evaluate_csv.py`).
