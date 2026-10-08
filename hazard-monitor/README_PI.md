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
* **One reading per node every 2 minutes goes to the ML model** (`ML_SAMPLE_SECONDS=120` in
  `.env`). Every reading is still saved and shown. The model was trained on readings 5 minutes
  apart and its windows count readings, so at 2 minutes its "last hour" covers about 24 minutes
  and a WARNING (2 readings in a row) can come after 4 minutes. `300` matches the training
  exactly; `0` sends every reading (simulator demos only). After changing it:
  `sudo systemctl restart hazard-monitor`.
* **Each card shows "5/12 trained inputs"**: how many of the inputs the model was trained on
  the reading actually had. The curl format gives 5 of 12 (node 1) and 4 of 9 (node 2); your sketches give 7 of 12 (12 with the MPU6050) and 8 of 9.
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

## The 3 pages
| page | what it shows |
|---|---|
| `http://<pi-ip>:3000/dashboard` | risk cards, ML chance per disaster, "ML MODEL WORKING" badge, charts, **🤖 Run ML now** button |
| `http://<pi-ip>:3000/esp` | **ESP Live**: every packet as it arrives (saved or rejected + reason), node ONLINE/OFFLINE, packets/minute, sender IP |
| `http://<pi-ip>:3000/database` | counts, latest readings (filter by node), ML predictions, alerts, ML-dataset preview, CSV downloads |

The ESP Live packet list is kept in memory (last 300) and resets when the server restarts; the readings themselves are in MongoDB.

## 🤖 Run ML now button
1. Takes every reading each node sent in the **last 2 minutes**.
2. Combines them into one reading (numbers → average, rain/flame → true if any reading was true) and sends it to the ML model straight away (ignores the 2-minute timer, then restarts it).
3. Saves the ML result as a prediction (Database page) and appends those raw readings to the **ML dataset** `ml/data/ml_dataset.csv`, with the ML result for that window.

Pressing twice does not store the same reading twice. Download the file from the Database page or `GET /api/ml/dataset.csv`.
The CSV uses the training column names (`water_level_cm`, `soil_moisture_pct`, …). It has **no label columns**: to use it for re-training or Colab Block 4 scoring with accuracy, add columns such as `flood_within_15m` (1 when a flood really followed). Without labels Block 4 only gives predictions.

## ESP32-S3 nodes → Pi
`POST http://<pi-ip>:3000/api/sensor-data`, header `Content-Type: application/json`. Watch them arrive on `/esp`.

NODE_01:
```json
{"node_id":"NODE_01","water_level":45.2,"soil_moisture":60,"rain":false,"temperature":28.5,"humidity":65,"tilt":0.5}
```
NODE_02:
```json
{"node_id":"NODE_02","smoke":120,"flame":false,"temperature":32.1,"humidity":50}
```
The Pi answers `200` when saved, `400` with `details` listing the reasons when rejected (broken JSON included):

| key | type | allowed |
|---|---|---|
| node_id | string | `NODE_01` or `NODE_02` (others are saved but get no risk score); lower case is upper-cased |
| water_level, soil_moisture, humidity | number | 0 – 100 |
| temperature | number | -50 – 60 |
| tilt | number | -90 – 90 |
| smoke | number | 0 – 1000 |
| rain, flame | true/false, 1/0 or "true"/"false" | |
| optional ML extras | number | `rainfall_mm_h`, `tilt_x_deg`, `tilt_y_deg`, `acceleration_g`, `smoke_raw`, `gas_raw`, `signal_strength`, TinyML `node1_*`/`node2_*` scores and labels |

Numbers may be sent as text (`"45.2"`). Keys are snake_case. No timestamp needed — the Pi stamps it.
The optional extras are stored and passed to the ML model; the more of them a node sends, the more of the model's inputs are real (the curl format gives 5 of 12 for NODE_01 and 4 of 9 for NODE_02; your sketches give 7 of 12 without the MPU6050, all 12 with it, and 8 of 9).

### Your node sketches (use these)
| node | folder | needs in the folder |
|---|---|---|
| NODE_01 flood + landslide | `firmware/SIH/` | `SIH.ino`, `landslide_model.h`, **`flood_model.h` (add yours - without it flood risk is null)** |
| NODE_02 wildfire + heat | `firmware/NODE_2/` | `NODE_2.ino`, `heat_model.h`, `flame_model.h` |

Libraries: **Chirale_TensorFlowLite** and **DHT sensor library** (Library Manager). Board: **ESP32S3 Dev Module**.

What they do now:
- Join the Pi's hotspot `HAZARD-NET` / `hazard1234` and POST their own JSON to `http://192.168.4.1:3000/api/sensor-data` every 5 s
  (sampling and the on-node TinyML still run every 2 s). NODE_02 also sends at once when its alert level changes (instant flame alert).
- If the Pi is not up yet they start their own hotspot (`NODE01-JSON` / `NODE02-JSON`) and keep retrying; they switch to the Pi as soon as it appears.
- Serial (115200) prints `{"event":"pi_post","http":200}` when the Pi saves the packet (`400` = rejected: open `/esp` to see why).
- The Pi translates their format: `waterLevel`→`water_level`, `rainfall` mm ≥ 2.5 → `rain: true`, smoke ADC 0–4095 → `smoke` 0–1000
  (+ raw `smoke_raw` for the ML), `flameDetected`/instant alert → `flame`, `tiltX/tiltY` → tilt, and the on-node TinyML result
  (`risk.*.levels`) → the ML model's TinyML inputs. The node's `timestamp` is its uptime, so the Pi uses its own clock instead.

Changes made to your sketches (the originals are in git history):
- Wi-Fi → `HAZARD-NET`; NODE_02's ESP8266 gateway is replaced by the Pi (same address 192.168.4.1, port 3000).
- `SIH.ino`: removed a stray `3` on line 384 that stopped it compiling; `flood_model.h` is optional like `landslide_model.h`;
  with `USE_ULTRASONIC 1` it now reports water **level** (`SENSOR_HEIGHT_CM` − distance) instead of the distance to the water;
  adds `rainfall` (mm) next to the raw rain ADC.
- Both: send `rssi` and the 4 class probabilities (`levels`) of each TinyML model; bigger JSON buffers.

### Generic firmware (no TinyML)
`firmware/esp32s3_node/esp32s3_node.ino` (Arduino IDE):
1. Boards Manager → install **esp32 by Espressif**; select **ESP32S3 Dev Module**.
2. Library Manager → **DHT sensor library** (Adafruit) if `USE_DHT22` is 1. The MPU6050 is read with plain `Wire`, no library.
3. At the top of the file set `WIFI_SSID`, `WIFI_PASSWORD`, `PI_HOST` (the Pi's IP, `hostname -I`), `NODE_ID` (`NODE_01` or `NODE_02`) and the pins.
4. Upload, open Serial Monitor at 115200: every 5 s it prints the JSON and the Pi's HTTP answer.
Calibrate `SOIL_ADC_DRY`/`SOIL_ADC_WET` and `SENSOR_HEIGHT_CM` (sensor height above the empty riverbed/tank) for your sensors.

No hardware yet? The dashboard's Simulator buttons generate fake data for both nodes.

## Pi as the Wi-Fi hotspot (ESPs join the Pi)
Run `bash setup.sh` **first**: it needs internet. Then:
```
bash hotspot.sh on
```
- Wi-Fi name `HAZARD-NET`, password `hazard1234`, 2.4 GHz (ESP32 can't see 5 GHz). Change them with
  `HOTSPOT_SSID=MyNet HOTSPOT_PASSWORD=secret123 bash hotspot.sh on` (and the same in the ESP code).
- The Pi is always **192.168.4.1**, so the ESP sends to `http://192.168.4.1:3000/api/sensor-data`.
- It starts by itself on every boot. `bash hotspot.sh status` lists connected devices; `bash hotspot.sh off` goes back to normal Wi-Fi.
- Your SSH session drops when it turns on: join the laptop to `HAZARD-NET`, then `ssh pi@192.168.4.1`.
- Dashboard: join `HAZARD-NET` on a phone/laptop and open `http://192.168.4.1:3000/dashboard`.
- While the hotspot is on, the Pi has no internet over Wi-Fi (one radio). Plug a LAN cable into the router if you need internet.
- Needs Raspberry Pi OS Bookworm or newer (`nmcli`).

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
7. Two more pages (`/esp`, `/database`), the Run ML now button, the ML dataset CSV, ESP32-S3 firmware.

### Bugs found in the original code and fixed
- A reading of **0** (e.g. `water_level: 0`, `humidity: 0`) was saved as empty (`value || null`), and shown as `--` on the dashboard.
- Landslide risk ignored **negative tilt** (a slope leaning the other way scored 0).
- Header always said **"2 Nodes Online"**; it now counts nodes that sent data in the last minute.
- Node risk % stayed `---%` until an alert fired; now every reading updates it.
- Charts were empty after a page reload; the last 50 readings are loaded on open.
- History/training-data endpoints had no limit cap (a big `?limit=` could exhaust the Pi's memory).
- Broken JSON from a node returned an HTML error page and was not logged; failed saves were not logged.

### Still open (not changed)
- **No login**: anyone on the same Wi-Fi can post data, start the simulator or press Run ML.
- `water_level` is limited to 0–100 (cm or %): a deeper tank is rejected.
- Alerts stay ACTIVE until resolved; they do not clear when the risk drops.
- The extreme-heat model failed its Colab test: treat its % as advisory.
- The ML runs every 2 minutes but was trained on 5-minute data.
- Not yet run on a real Pi, real MongoDB or a real ESP32 (tested with an in-memory database stand-in).
