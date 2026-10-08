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
Takes ~5–10 min the first time (Node, Docker, MongoDB download). At the end it prints the URL, e.g. `http://192.168.1.42:3000`.
It auto-starts on every boot after that.

> Don't open/save `setup.sh` in Windows Notepad — it adds `\r` and bash breaks. If that happened: `sed -i 's/\r$//' setup.sh`

## Daily commands
| What | Command |
|---|---|
| Live logs | `journalctl -u hazard-monitor -f` |
| Restart | `sudo systemctl restart hazard-monitor` |
| Stop / start | `sudo systemctl stop hazard-monitor` / `start` |
| Disable autostart | `sudo systemctl disable hazard-monitor` |
| DB container | `sudo docker ps` / `sudo docker logs hazard-mongo` |

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
