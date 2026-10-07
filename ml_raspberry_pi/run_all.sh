#!/usr/bin/env bash
# One-shot setup on the Raspberry Pi: venv -> deps -> dataset -> train -> parity test.
# Usage: ./run_all.sh            (full dataset, ~5-10 min on a Pi 4)
#        ./run_all.sh --quick    (small dataset, ~1 min, for a smoke test)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# Keep the Pi responsive: limit OpenMP threads used by scikit-learn
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

python3 generate_dataset.py "$@"
python3 train.py
python3 test_feature_parity.py

echo
echo "Done. Start the live service with:"
echo "  source .venv/bin/activate && python3 live_inference.py serve --port 5001"
