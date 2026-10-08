# HackHawks Hazard Monitor — Raspberry Pi

## Needs
- Raspberry Pi 5 / 4 / 3 / Zero 2 W
- **Raspberry Pi OS (64-bit)** — 32-bit will NOT work (no MongoDB for it)
- Internet on the Pi **once**, for setup. After that it runs offline.

## Install (3 commands)
From your laptop (same Wi-Fi as the Pi):
```
scp hazard-monitor-pi.zip pi@raspberrypi.local:~
```
(replace `pi` / `raspberrypi` with your Pi's username / hostname, or just copy the zip over with a pen drive)

On the Pi:
```
unzip hazard-monitor-pi.zip
cd hazard-monitor
bash setup.sh
```
Takes ~5–15 min the first time (Node, Docker, MongoDB, and the ML model's Python libraries). At the end it prints the URL, e.g. `http://192.168.1.42:3000`.
Both the dashboard and the ML model auto-start on every boot after that, and restart by themselves if they crash.

> Don't open/save `setup.sh` in Windows Notepad — it adds `\r` and bash breaks. If that happened: `sed -i 's/\r$//' setup.sh`

## Daily commands
| What | Command |
|---|---|
| Live logs | `journalctl -u hazard-monitor -f` |
| Restart | `sudo systemctl restart hazard-monitor` |
| Stop / start | `sudo systemctl stop hazard-monitor` / `start` |
| Disable autostart | `sudo systemctl disable hazard-monitor` |
| DB container | `sudo docker ps` / `sudo docker logs hazard-mongo` |
| ML model status | `sudo systemctl status hazard-ml` |
| ML model logs | `journalctl -u hazard-ml -f` |
| Restart ML model | `sudo systemctl restart hazard-ml` |

## ML model on the dashboard
The header shows **ML MODEL WORKING** (green) when the dashboard can reach the ML model,
and **ML MODEL OFFLINE** (red) when it cannot. It re-checks every 10 seconds. Hover over the
badge to see why it is offline.

Each hazard card shows **ML chance of <hazard>**: the model's calibrated chance that the
disaster happens within the next 15 minutes (60 minutes for extreme heat), plus a level:

| Level | Meaning |
|---|---|
| SAFE | below the model's WATCH threshold |
| WATCH | 2 readings in a row over the WATCH threshold |
| WARNING | 2 readings in a row over the WARNING threshold |

Act on the level, not the number. Disasters are rare, so the thresholds are small chances
(about 0.13% for flood). A WARNING at "0.2%" is still a WARNING. The big **rule-based score**
above it is the old rule engine, which works independently of the ML.

Things to know:
* **One reading per node every 5 minutes goes to the ML model** (`ML_SAMPLE_SECONDS=300` in
  `.env`), because the model was trained on 5-minute data. Every reading is still saved and shown.
  For a quick demo with the simulator you can set `ML_SAMPLE_SECONDS=0`, then restart
  (`sudo systemctl restart hazard-monitor`). The model then sees readings every 2 seconds,
  which it was not trained for, so treat the numbers as a demo only.
* **Each card shows "5/12 trained inputs"**: how many of the inputs the model was trained on
  the reading actually had. The current payloads give 5 of 12 (node 1) and 4 of 9 (node 2).
  Missing inputs make the ML chance unreliable. To use the model properly, have the ESP32s also
  send `rainfall_mm_h` (tipping-bucket gauge), `tilt_x_deg`, `tilt_y_deg`, `acceleration_g`,
  `gas_raw` and the TinyML `node1_*` / `node2_*` scores and labels. Extra fields are accepted by
  `/api/sensor-data` and passed to the model.
* **Extreme heat shows "advisory only"**: its model did not pass testing (88 % of heat waves
  caught, target 90 %). Don't sound alarms from it alone.
* If ML is offline, sensor data, charts and the rule-based alerts keep working. The last ML values
  stay on the cards, dimmed and labelled with their time.
* Each ML result is also saved in the `predictions` collection (the DB Status button counts them),
  and returned by `/api/sensor-data` in an `ml` field. `GET /api/ml/status` and
  `GET /api/ml/latest` give the status and the latest result per node.

## Replacing the ML model
Train in Colab, download `pi_models.zip`, then on the Pi:
```
cd hazard-monitor/ml
rm -rf output && unzip ~/pi_models.zip
cd .. && bash setup.sh
```
`setup.sh` installs the exact scikit-learn version the models need (`ml/output/requirements-pi.txt`).

## ESP32 nodes → Pi
`POST http://<pi-ip>:3000/api/sensor-data`, header `Content-Type: application/json`

NODE_01:
```json
{"node_id":"NODE_01","water_level":45.2,"soil_moisture":60,"rain":false,"temperature":28.5,"humidity":65,"tilt":0.5}
```
NODE_02:
```json
{"node_id":"NODE_02","smoke":120,"flame":false,"temperature":32.1,"humidity":50}
```
The API **rejects (HTTP 400)** anything outside these rules:

| key | type | allowed |
|---|---|---|
| node_id | string | `NODE_01` or `NODE_02` (others are saved but get no risk score) |
| water_level, soil_moisture, humidity | number | 0 – 100 |
| temperature | number | -50 – 60 |
| tilt | number | -10 – 10 |
| smoke | number | 0 – 1000 |
| rain, flame | **true/false** (not 0/1, not mm) | |

Keys are snake_case (`water_level`, not `waterLevel`). No timestamp needed — the Pi stamps it.

No hardware yet? The dashboard's Simulator buttons generate fake data for both nodes.

## Using MongoDB Atlas instead of local DB
Put your Atlas URI in `.env` (`MONGODB_URI=mongodb+srv://...`) **before** `bash setup.sh` — Docker/MongoDB install is skipped. Needs internet always.

## Changes vs the original repo code
1. `models/prediction.js` → `models/Prediction.js` — the original crashes on Linux (`Cannot find module '../models/Prediction'`) because Linux filenames are case-sensitive.
2. Chart.js and the Socket.IO client are served from the Pi itself instead of CDNs, so the dashboard works with no internet.
3. Simulator posts to the configured `PORT` instead of hard-coded 3000.
4. Live updates fall back to server time when a node sends no `timestamp` (otherwise charts show "Invalid Date").
5. ML model added: `ml/` (Python service with the Colab-trained models), `services/mlClient.js`,
   ML forwarding and `/api/ml/*` endpoints in `routes/api.js`, the ML badge and ML chance boxes
   in `views/dashboard.ejs`, and the `hazard-ml` service in `setup.sh`.
6. Risk cards stack to 2 columns on tablets and 1 on phones.
