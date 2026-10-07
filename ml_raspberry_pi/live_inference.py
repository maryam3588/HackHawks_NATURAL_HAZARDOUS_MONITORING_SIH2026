"""Live hazard inference for the Raspberry Pi.

Keeps a rolling 3-day history per (site, node) in memory, builds the same
features used in training, and returns calibrated probabilities with a
SAFE / WATCH / WARNING level per hazard. A level is raised only after
``ALERT_CONFIRM_READINGS`` consecutive readings over its threshold.

Two ways to feed it readings:

  1. HTTP (for the Node.js dashboard or an MQTT bridge):
       python3 live_inference.py serve --port 5001
       curl -X POST localhost:5001/predict -d '{"node_id":"node1", ...}'
       curl localhost:5001/health

  2. JSON lines on stdin (one reading per line, one result per line):
       cat readings.jsonl | python3 live_inference.py stdin

A reading is a JSON object with ``node_id`` (node1/node2, NODE_01/NODE_02
also accepted), optional ``site_id`` and ``timestamp`` (ISO 8601, default:
now), and any of the sensor fields in config.RAW_SENSOR_COLUMNS. Missing
fields are treated as missing sensor values.
"""

import argparse
import json
import math
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np

import config as C
from features import StreamingFeatureEngine

# Field names used by the existing Node.js simulator/dashboard. Only fields
# with the same physical meaning and unit are mapped. The dashboard's boolean
# ``rain`` cannot be turned into mm/h, so it is deliberately not mapped.
FIELD_ALIASES = {
    "water_level": "water_level_cm",
    "soil_moisture": "soil_moisture_pct",
    "temperature": "temperature_c",
    "humidity": "humidity_pct",
    "smoke": "smoke_raw",
    "gas": "gas_raw",
    "rainfall": "rainfall_mm_h",
    "acceleration": "acceleration_g",
}


def normalise_reading(reading):
    reading = dict(reading)
    for alias, column in FIELD_ALIASES.items():
        if alias in reading and column not in reading:
            reading[column] = reading.pop(alias)
    # A single "tilt" angle is treated as tilt_x with tilt_y = 0, so the
    # tilt magnitude matches; the x/y split itself is unknown.
    if "tilt" in reading and "tilt_x_deg" not in reading:
        reading["tilt_x_deg"] = reading.pop("tilt")
        reading.setdefault("tilt_y_deg", 0.0)
    return reading


class HazardPredictor:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        registry_path = model_dir / "model_registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"{registry_path} not found. Run train.py first.")

        with open(registry_path) as handle:
            self.registry = json.load(handle)

        self._check_versions()

        self.models = {}
        for entry in self.registry["models"]:
            self.models[entry["hazard"]] = joblib.load(model_dir / entry["file"])

        missing = sorted(set(C.HAZARDS) - set(self.models))
        if missing:
            print(f"[live_inference] no validated model for: {', '.join(missing)} "
                  "(threshold selection failed in train.py) - these hazards report UNAVAILABLE",
                  file=sys.stderr)

        self.engine = StreamingFeatureEngine()
        self.confirm = self.registry.get("alert_confirm_readings", C.ALERT_CONFIRM_READINGS)
        self.streaks = {}  # (site, node, hazard) -> [readings >= WATCH, readings >= WARNING]
        self.lock = threading.Lock()

    def _check_versions(self):
        import sklearn

        trained_with = self.registry.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(
                f"Models were trained with scikit-learn {trained_with} but this device has "
                f"{sklearn.__version__}. Pickled models may load wrongly or fail; retrain on "
                "this device (python3 train.py) or install the matching version.",
                RuntimeWarning,
            )

    def predict(self, reading):
        reading = normalise_reading(reading)
        site_id = str(reading.get("site_id", "default_site"))
        with self.lock:
            node_id, features = self.engine.update(reading)
            return self._predict(node_id, site_id, reading, features)

    def _predict(self, node_id, site_id, reading, features):
        hazards = {}
        for hazard, config in C.HAZARDS.items():
            if config["node_id"] != node_id:
                continue
            artifact = self.models.get(hazard)
            if artifact is None:
                hazards[hazard] = {"level": "UNAVAILABLE", "probability": None}
                continue

            vector = np.array(
                [[features.get(name, math.nan) for name in artifact["features"]]],
                dtype=np.float32,
            )
            raw = float(artifact["model"].predict_proba(vector)[0, 1])
            raw = min(max(raw, 1e-6), 1 - 1e-6)
            calibrator = artifact["calibrator"]
            probability = 1.0 / (1.0 + math.exp(-(calibrator["coef"] * raw + calibrator["intercept"])))

            # An alert needs `confirm` consecutive readings over the threshold,
            # exactly as in training/evaluation (filters single noisy readings).
            streak = self.streaks.setdefault((site_id, node_id, hazard), [0, 0])
            streak[0] = streak[0] + 1 if probability >= artifact["watch_threshold"] else 0
            streak[1] = streak[1] + 1 if probability >= artifact["warning_threshold"] else 0
            if streak[1] >= self.confirm:
                level = "WARNING"
            elif streak[0] >= self.confirm:
                level = "WATCH"
            else:
                level = "SAFE"

            hazards[hazard] = {
                "probability": round(probability, 4),
                "level": level,
                "model_status": artifact["status"],
                "model_version": artifact["version"],
            }

        if not hazards:
            return {"error": f"unknown node_id {reading.get('node_id')!r}; expected node1 or node2"}

        return {
            "node_id": node_id,
            "site_id": site_id,
            "timestamp": reading.get("timestamp"),
            "hazards": hazards,
            "data_source": C.DATA_SOURCE,
        }


def run_stdin(predictor):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            result = predictor.predict(json.loads(line))
        except Exception as error:  # keep the stream alive on bad lines
            result = {"error": str(error)}
        print(json.dumps(result), flush=True)


def run_server(predictor, host, port):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {
                    "status": "ok",
                    "models": {h: a["status"] for h, a in predictor.models.items()},
                    "unavailable": sorted(set(C.HAZARDS) - set(predictor.models)),
                })
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if isinstance(payload, list):
                    self._send(200, [predictor.predict(item) for item in payload])
                else:
                    self._send(200, predictor.predict(payload))
            except Exception as error:
                self._send(400, {"error": str(error)})

        def log_message(self, fmt, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Disaster ML inference listening on http://{host}:{port} (POST /predict, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["serve", "stdin"])
    parser.add_argument("--models", type=Path, default=C.MODEL_DIR)
    parser.add_argument("--host", default="127.0.0.1",
                        help="use 0.0.0.0 to accept LAN requests (no authentication!)")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    predictor = HazardPredictor(args.models)
    if args.mode == "stdin":
        run_stdin(predictor)
    else:
        run_server(predictor, args.host, args.port)


if __name__ == "__main__":
    main()
