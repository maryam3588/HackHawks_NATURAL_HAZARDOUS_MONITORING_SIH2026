#!/usr/bin/env bash
# =====================================================================
#  HackHawks Hazard Monitor - one-shot Raspberry Pi setup
#
#  Usage (on the Pi, inside this folder):   bash setup.sh
#
#  What it does (safe to re-run - anything already on the Pi is kept, not
#  installed again: apt packages, Node, Docker, the MongoDB image/container and
#  its data, npm packages, the ML Python libraries):
#    1. checks you are on 64-bit Raspberry Pi OS
#    2. installs Node.js 22 (mongoose 9 needs Node >= 20.19)
#    3. installs MongoDB in Docker, picking the right version for your Pi
#         Pi 5            -> mongo:7.0
#         Pi 4 / 3 / Zero2 -> mongo:4.4.18 (newest build that runs on them)
#       (skipped if MONGODB_URI in .env points to Atlas / another server)
#    4. installs the app's npm packages
#    5. installs the ML model service (Python, your trained models in ml/)
#    6. installs systemd services so the ML model and the dashboard start
#       on every boot and restart by themselves if they crash
#
#  Needs internet ONCE (for steps 2-4). After that it runs fully offline.
# =====================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE=hazard-monitor
ML_SERVICE=hazard-ml
MONGO_CONTAINER=hazard-mongo
MONGO_VOLUME=hazard-mongo-data

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    ok: %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    !! %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mXX  %s\033[0m\n\n' "$*"; exit 1; }

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
  APP_USER="${SUDO_USER:-root}"
else
  SUDO="sudo"
  APP_USER="$(id -un)"
fi

run_as_app_user() {
  if [ "$(id -un)" = "$APP_USER" ]; then "$@"; else $SUDO -u "$APP_USER" -H "$@"; fi
}

cd "$APP_DIR"

# ---------------------------------------------------------------- 1. OS check
say "Checking system"
ARCH="$(uname -m)"
[ "$ARCH" = "aarch64" ] || die "This needs 64-bit Raspberry Pi OS (yours is '$ARCH').
    Re-flash the SD card with 'Raspberry Pi OS (64-bit)' in Raspberry Pi Imager, then run this again."
command -v apt-get >/dev/null || die "apt-get not found - use Raspberry Pi OS (64-bit)."
ok "64-bit OS ($ARCH)"

# .env (shipped in the zip; recreate if it got lost when copying)
if [ ! -f .env ]; then
  printf 'PORT=3000\nMONGODB_URI=mongodb://127.0.0.1:27017/environmental-monitoring\n' > .env
  warn ".env was missing - created default one"
fi
PORT="$(grep -E '^PORT=' .env | tail -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
PORT="${PORT:-3000}"
MONGODB_URI="$(grep -E '^MONGODB_URI=' .env | tail -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
MONGODB_URI="${MONGODB_URI:-mongodb://127.0.0.1:27017/environmental-monitoring}"
# ML settings (added to older .env files that do not have them yet)
grep -qE '^ML_URL=' .env || printf '\n# ML model service (ml/live_inference.py) on this Pi\nML_URL=http://127.0.0.1:5001\n' >> .env
grep -qE '^ML_SAMPLE_SECONDS=' .env || printf '# One reading per node every N seconds goes to the ML model (120 = every 2 minutes; trained on 5-minute data)\nML_SAMPLE_SECONDS=120\n' >> .env
ML_URL="$(grep -E '^ML_URL=' .env | tail -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
ML_PORT="${ML_URL##*:}"; ML_PORT="${ML_PORT%%/*}"
case "$ML_PORT" in ''|*[!0-9]*) ML_PORT=5001 ;; esac
case "$MONGODB_URI" in
  *127.0.0.1*|*localhost*) LOCAL_DB=1 ;;
  *) LOCAL_DB=0 ;;
esac

# Installs only the packages that are missing; apt-get update runs at most once
APT_UPDATED=0
apt_install() {
  local missing=()
  for p in "$@"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    ok "already installed: $*"
    return
  fi
  warn "installing: ${missing[*]}"
  if [ "$APT_UPDATED" = "0" ]; then $SUDO apt-get update -y; APT_UPDATED=1; fi
  $SUDO apt-get install -y "${missing[@]}"
}

say "Checking base packages"
apt_install ca-certificates curl gnupg python3 python3-venv python3-pip

# ---------------------------------------------------------------- 2. Node.js
node_ok() {
  command -v node >/dev/null 2>&1 || return 1
  node -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>20||(a===20&&b>=19)?0:1)'
}
say "Checking Node.js"
if node_ok; then
  ok "Node $(node -v) already installed"
else
  warn "Installing Node.js 22 (Pi OS's own Node is too old for mongoose 9)"
  curl -fsSL https://deb.nodesource.com/setup_22.x | $SUDO bash -
  $SUDO apt-get install -y nodejs
  hash -r
  node_ok || die "Node.js install failed - check the messages above."
  ok "Node $(node -v)"
fi
NODE_BIN="$(command -v node)"

# ---------------------------------------------------------------- 3. MongoDB
if [ "$LOCAL_DB" = "1" ]; then
  say "Setting up local MongoDB (Docker)"
  if ! command -v docker >/dev/null 2>&1; then
    warn "Installing Docker (takes a few minutes)"
    curl -fsSL https://get.docker.com | $SUDO sh
  else
    ok "Docker already installed"
  fi
  $SUDO systemctl enable --now docker >/dev/null
  ok "Docker $($SUDO docker --version | awk '{print $3}' | tr -d ,)"

  # MongoDB 5+ needs ARMv8.2 (Pi 5). Pi 4/3/Zero2 are ARMv8.0 -> last working build is 4.4.18
  FEATURES="$(grep -m1 -i '^Features' /proc/cpuinfo || true)"
  if echo "$FEATURES" | grep -qw atomics && echo "$FEATURES" | grep -qw dcpop; then
    MONGO_IMAGE="mongo:7.0"; ok "CPU is ARMv8.2+ (Pi 5) -> $MONGO_IMAGE"
  else
    MONGO_IMAGE="mongo:4.4.18"; ok "CPU is ARMv8.0 (Pi 4/3/Zero 2) -> $MONGO_IMAGE"
  fi

  if $SUDO docker ps -a --format '{{.Names}}' | grep -qx "$MONGO_CONTAINER"; then
    CURRENT_IMAGE="$($SUDO docker inspect -f '{{.Config.Image}}' "$MONGO_CONTAINER")"
    if [ "$CURRENT_IMAGE" != "$MONGO_IMAGE" ]; then
      warn "Existing container uses $CURRENT_IMAGE - recreating with $MONGO_IMAGE (data volume kept)"
      $SUDO docker rm -f "$MONGO_CONTAINER" >/dev/null
    fi
  fi
  if ! $SUDO docker ps -a --format '{{.Names}}' | grep -qx "$MONGO_CONTAINER"; then
    if $SUDO docker image inspect "$MONGO_IMAGE" >/dev/null 2>&1; then
      ok "image $MONGO_IMAGE already downloaded"
    else
      $SUDO docker pull "$MONGO_IMAGE"
    fi
    # bound to 127.0.0.1 only: the DB has no password, so it must not be reachable from the LAN
    $SUDO docker run -d --name "$MONGO_CONTAINER" --restart unless-stopped \
      -p 127.0.0.1:27017:27017 -v "$MONGO_VOLUME":/data/db \
      "$MONGO_IMAGE" --wiredTigerCacheSizeGB 0.25 >/dev/null
  else
    $SUDO docker start "$MONGO_CONTAINER" >/dev/null
    ok "existing MongoDB container reused (saved readings kept)"
  fi
  ok "MongoDB container '$MONGO_CONTAINER' started"
else
  say "MONGODB_URI points to a remote server - skipping local MongoDB"
fi

# ---------------------------------------------------------------- 4. npm packages
say "Checking app packages"
# Skip npm if node_modules was built from this exact package.json + lock with this Node version.
# The stamp matches the one older hazard-monitor setups write, so a node_modules left in an
# old copy of this folder (hazard-monitor-old / -other) is reused instead of downloaded again.
DEPS_STAMP=node_modules/.hazard-setup-stamp
DEPS_HASH="$( { node -v; cat package.json package-lock.json 2>/dev/null; } | sha256sum | cut -d' ' -f1)"
if [ ! -f "$DEPS_STAMP" ]; then
  for old in "$APP_DIR-old" "$APP_DIR-other" "$APP_DIR-mine"; do
    if [ -f "$old/$DEPS_STAMP" ] && [ "$(cat "$old/$DEPS_STAMP")" = "$DEPS_HASH" ]; then
      run_as_app_user cp -a "$old/node_modules" "$APP_DIR/"
      ok "node_modules copied from $old"
      break
    fi
  done
fi
if [ -f "$DEPS_STAMP" ] && [ "$(cat "$DEPS_STAMP")" = "$DEPS_HASH" ]; then
  ok "node_modules already up to date"
else
  warn "installing npm packages"
  if [ -f package-lock.json ]; then
    run_as_app_user npm ci --omit=dev --no-audit --no-fund
  else
    run_as_app_user npm install --omit=dev --no-audit --no-fund
  fi
  run_as_app_user sh -c 'printf "%s\n" "$1" > "$2"' _ "$DEPS_HASH" "$DEPS_STAMP"
  ok "node_modules ready"
fi

say "Waiting for MongoDB to accept connections"
DB_UP=0
for _ in $(seq 1 30); do
  if run_as_app_user "$NODE_BIN" -e '
    require("mongoose").connect(process.argv[1],{serverSelectionTimeoutMS:3000})
      .then(()=>process.exit(0)).catch(()=>process.exit(1))' "$MONGODB_URI" >/dev/null 2>&1; then
    DB_UP=1; break
  fi
  sleep 2
done
if [ "$DB_UP" != "1" ]; then
  if [ "$LOCAL_DB" = "1" ]; then
    $SUDO docker logs --tail 25 "$MONGO_CONTAINER" 2>&1 || true
    die "MongoDB did not start (logs above). If you see 'Illegal instruction', send me those logs."
  fi
  die "Can't reach MongoDB at the MONGODB_URI in .env - check the URI / Atlas IP allow-list."
fi
ok "MongoDB is up"

# ---------------------------------------------------------------- 5. ML model service
say "Setting up the ML model (Python)"
[ -f ml/output/models/model_registry.json ] || die "ml/output/models/ is missing - unzip your pi_models.zip inside ml/ first."
ML_REQ=ml/output/requirements-pi.txt
[ -f "$ML_REQ" ] || die "$ML_REQ is missing - it comes with pi_models.zip from Colab."
[ -x ml/.venv/bin/python ] || run_as_app_user python3 -m venv ml/.venv
# Skip pip if the venv already has every library in requirements-pi.txt at the right version
if run_as_app_user ml/.venv/bin/python - "$ML_REQ" >/dev/null 2>&1 <<'PY'
import sys
from importlib.metadata import version
try:
    from packaging.requirements import Requirement
except ImportError:  # not always installed; pip ships its own copy
    from pip._vendor.packaging.requirements import Requirement
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if line:
        req = Requirement(line)
        assert req.specifier.contains(version(req.name), prereleases=True)
PY
then
  ok "ML libraries already installed"
else
  warn "installing ML libraries"
  run_as_app_user ml/.venv/bin/pip install --upgrade pip --quiet
  if ! run_as_app_user ml/.venv/bin/pip install -r "$ML_REQ"; then
    die "Could not install the ML libraries in $ML_REQ (see above).
    The models need exactly that scikit-learn version. Send me the error."
  fi
fi
ok "ML libraries: $(run_as_app_user ml/.venv/bin/python -c 'import sklearn; print("scikit-learn", sklearn.__version__)')"
if ! (cd ml && run_as_app_user .venv/bin/python live_inference.py stdin < sample_readings.jsonl > /dev/null); then
  die "The ML model failed its self-test (see above)."
fi
ok "ML models load and predict"

$SUDO tee /etc/systemd/system/$ML_SERVICE.service >/dev/null <<UNIT
[Unit]
Description=HackHawks Hazard ML model (flood, landslide, wildfire, extreme heat)
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR/ml
Environment=OMP_NUM_THREADS=1
ExecStart=$APP_DIR/ml/.venv/bin/python live_inference.py serve --host 127.0.0.1 --port $ML_PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
$SUDO systemctl daemon-reload
$SUDO systemctl enable $ML_SERVICE >/dev/null 2>&1
$SUDO systemctl restart $ML_SERVICE

ML_UP=0
for _ in $(seq 1 30); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$ML_PORT/health"; then ML_UP=1; break; fi
  sleep 1
done
if [ "$ML_UP" != "1" ]; then
  $SUDO journalctl -u $ML_SERVICE -n 30 --no-pager || true
  die "ML model service didn't come up (logs above)."
fi
ok "ML model service running on 127.0.0.1:$ML_PORT"

# ---------------------------------------------------------------- 6. dashboard service
say "Installing auto-start service"
UNIT_AFTER="network-online.target $ML_SERVICE.service"
[ "$LOCAL_DB" = "1" ] && UNIT_AFTER="network-online.target docker.service $ML_SERVICE.service"
$SUDO tee /etc/systemd/system/$SERVICE.service >/dev/null <<EOF
[Unit]
Description=HackHawks Hazard Monitoring Dashboard
After=$UNIT_AFTER
Wants=network-online.target $ML_SERVICE.service
StartLimitIntervalSec=0

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
Environment=NODE_ENV=production
ExecStart=$NODE_BIN bin/www
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
$SUDO systemctl daemon-reload
$SUDO systemctl enable $SERVICE >/dev/null 2>&1
$SUDO systemctl restart $SERVICE

UP=0
for _ in $(seq 1 30); do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/dashboard"; then UP=1; break; fi
  sleep 1
done
if [ "$UP" != "1" ]; then
  $SUDO journalctl -u $SERVICE -n 30 --no-pager || true
  die "Dashboard didn't come up (logs above)."
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
printf '\n\033[1;32m=====================================================\033[0m\n'
printf '\033[1;32m  DONE - dashboard is running and starts on every boot\033[0m\n'
printf '\033[1;32m=====================================================\033[0m\n\n'
printf '  Open on any device on the same Wi-Fi:\n'
printf '     http://%s:%s\n' "${IP:-<pi-ip>}" "$PORT"
printf '     http://%s.local:%s\n\n' "$(hostname)" "$PORT"
printf '  Pages:   /dashboard   /esp (ESP Live)   /database\n\n'
printf '  ESP32 nodes POST JSON to:\n'
printf '     http://%s:%s/api/sensor-data\n\n' "${IP:-<pi-ip>}" "$PORT"
printf '  Logs:     journalctl -u %s -f\n' "$SERVICE"
printf '  Restart:  sudo systemctl restart %s\n' "$SERVICE"
printf '  ML model: sudo systemctl status %s   (logs: journalctl -u %s -f)\n\n' "$ML_SERVICE" "$ML_SERVICE"
printf '  The dashboard header shows "ML MODEL WORKING" when the model is connected.\n\n'
