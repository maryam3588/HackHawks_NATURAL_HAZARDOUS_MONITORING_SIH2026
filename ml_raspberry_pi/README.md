# Disaster ML on Raspberry Pi (no forecasting)

Port of the Colab notebook `sihh.ipynb` (V3 dataset generator + training
pipeline + CSV tester) to plain Python that runs on a Raspberry Pi.

Pipeline: **generate synthetic data → train → calibrate → pick WATCH/WARNING
thresholds → locked tests → live inference service**. The forecasting phase
is removed.

## What runs where

| File | Purpose |
|---|---|
| `generate_dataset.py` | V3 synthetic dataset (same logic and seed as Colab) → `data/` |
| `train.py` | Trains one classifier per hazard and writes `output/models/*.joblib` and `output/reports/*.csv` |
| `live_inference.py` | Live service: rolling 6 h memory per node, returns SAFE / WATCH / WARNING |
| `evaluate_csv.py` | Replaces the Colab "upload one CSV" tester |
| `test_feature_parity.py` | Proves the live features equal the training features |
| `config.py`, `features.py`, `metrics.py` | Shared code |
| `run_all.sh` | One command: venv → install → generate → train → parity test |
| `disaster-ml.service` | systemd unit to start the service at boot |

## Requirements

* Raspberry Pi 4 or 5 with **2 GB+ RAM** (full training peaks at ~1 GB).
  A Pi 3 / Zero 2 W (512 MB–1 GB) can run `live_inference.py`. Train on a
  bigger machine and copy `output/models/` over (see "Training somewhere
  else" below).
* Raspberry Pi OS **64-bit** (Bookworm) recommended, Python 3.9+.

## Quick start on the Pi

```bash
unzip ml_raspberry_pi.zip -d ~ && cd ~/ml_raspberry_pi
./run_all.sh --quick      # smoke test only (~1-2 min); its models are too weak to deploy
./run_all.sh              # full dataset + training
source .venv/bin/activate
python3 live_inference.py serve --port 5001
```

Measured on a 4-core x86 cloud VM: generation 54 s, training 18 s, about
6 ms per live reading. Expect a Pi 4 to be roughly 3–6× slower. These Pi
numbers are estimates and were not measured on a real Pi.

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
* Send **one reading per node every 5 minutes**. The lags and rolling
  windows count readings, exactly like training. A gap of more than 15
  minutes resets that node's history.
* `GET /health` lists the loaded models.
* Offline alternative: `python3 live_inference.py stdin < sample_readings.jsonl`.

Act on `level`, not `probability`. The thresholds were chosen on
calibrated probabilities, and for rare events they can be low. For
example, a wildfire WARNING can fire at 0.03.

### Using it from the existing Node.js app

The service accepts `NODE_01`/`NODE_02` and the dashboard field names
(`water_level`, `soil_moisture`, `temperature`, `humidity`, `smoke`, `tilt`).
However, the current simulator does **not** send what the models were
trained on:

* `rain` is a boolean. The models need `rainfall_mm_h`, so `rain` is ignored.
* `tilt` is a single angle. It is mapped to `tilt_x_deg` with `tilt_y_deg = 0`.
* There is no `acceleration_g`, `gas_raw`, or TinyML `node*_score`/`label`.

Missing inputs are allowed, but they make predictions much less reliable.
To use these models properly, update the ESP32 nodes and the simulator to
send the fields listed above.

## Training somewhere else (e.g. Colab or a laptop)

`.joblib` files are pickles, so the Pi must have **the same scikit-learn
version** as the training machine. `model_registry.json` records it, and
`live_inference.py` warns on a mismatch. The simplest option is to train on
the Pi.

## Changes from the Colab notebook

1. **Forecasting removed.** In the Colab run all four forecasters lost to
   "repeat the last reading" (MAE skill −0.22 to −0.81).
2. **V2→V3 mismatch fixed.** The Colab training cell read
   `disaster_v2_*.csv` with roles `locked_test_*` and `synthetic_v2`. The V3
   generator writes `disaster_v3_*`, `locked_*` and `synthetic_v3`, so the
   two cells could not run together.
3. **`node_id`, `day_of_week`, `month` dropped from features.** Each model sees
   only one node, so `node_id` is constant. Day and month only encode the
   synthetic site start dates.
4. **Calibrator without `class_weight="balanced"`.** Balanced weighting pushes
   probabilities toward 50/50, which defeats calibration.
5. **No pandas or ColumnTransformer at inference time.** Models train on
   float32 arrays because HistGradientBoosting handles NaN itself. The
   calibrator is stored as two floats.
6. Colab-only code removed: `/content`, `files.upload/download`,
   `display`, `drive.mount`, Excel export.

## Known problems (not fixed — they change the simulation)

* **Extreme heat has no model.** Its threshold selection fails on V3 data,
  with or without changes 3–4. The service reports it as `UNAVAILABLE`
  rather than producing an unvalidated alert.
* **Generator bug: event "memory" never accumulates.** `normal_step()`
  resets `*_memory` to 0 every step, so memory is never above ~0.08.
  Consequences: `flame` is 0 in all 172,800 training rows (it needs
  memory > 0.78), and heat events add only ~2 °C, which is why extreme heat
  fails. Fixing it means carrying the memory forward and re-tuning the event
  magnitudes. Otherwise flood water hits the 280 cm cap within ~50 steps.
* **WATCH == WARNING for flood and landslide.** The selection rule picks the
  highest WATCH threshold with recall ≥ 0.90, and that point already meets
  the WARNING precision target, so both tiers fire together. Wildfire's
  WATCH sits at the 0.01 grid floor.
* **Synthetic only.** Every metric here is measured on data from the same
  simulator. Nothing says how this behaves on real sensors. The Colab
  "real-world" CSV had no labels, so it was never scored.
