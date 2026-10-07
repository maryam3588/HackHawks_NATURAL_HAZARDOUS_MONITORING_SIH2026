"""Build disaster_ml_colab.ipynb from the package's own .py files.

The notebook writes the exact same modules the Raspberry Pi runs (via
%%writefile), so Colab and Pi code can never drift apart. Re-run this after
changing any module:

    python3 build_colab_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODULES = [
    "config.py",
    "sensor_errors.py",
    "features.py",
    "metrics.py",
    "generate_dataset.py",
    "train.py",
    "event_metrics.py",
    "evaluate_csv.py",
    "live_inference.py",
    "test_feature_parity.py",
    "test_live_vs_batch.py",
]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text, strip=True):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": (text.strip("\n") if strip else text).splitlines(keepends=True),
    }


def build():
    cells = [
        markdown("""
# Disaster ML (V4): dataset → training → Raspberry Pi bundle

Flood, landslide, wildfire and extreme-heat early warning. **No forecasting.**

This notebook writes the same Python modules that run on the Raspberry Pi,
then:

1. generates the synthetic V4 dataset (realistic sensor errors, failures and packet loss)
2. trains one model per hazard, calibrates it and chooses WATCH/WARNING thresholds on a validation set
3. scores the models per disaster on locked tests (detection rate, warning time, false alarms per site per day)
4. checks that the live (Pi) features and alerts exactly match training
5. downloads `pi_models.zip` for the Raspberry Pi

Runtime: about 5 minutes on the free Colab CPU.

**Important:** every number here is measured on *synthetic* data. Real-world
accuracy is unknown until you score labelled readings from your own nodes
(the last section).
"""),
        code("""
import platform, sklearn, numpy, pandas
print("python", platform.python_version(), "| scikit-learn", sklearn.__version__,
      "| numpy", numpy.__version__, "| pandas", pandas.__version__)
# The Raspberry Pi must install the SAME scikit-learn version to load these
# models; the bundle at the end includes a pinned requirements file.
"""),
        code("""
import os
os.makedirs("/content/disaster_ml", exist_ok=True)
%cd /content/disaster_ml
"""),
        markdown("## 1. Write the modules (identical to the Raspberry Pi package)"),
    ]

    for module in MODULES:
        source = (HERE / module).read_text()
        cells.append(code(f"%%writefile {module}\n{source}", strip=False))

    cells += [
        markdown("""
## 2. Generate the dataset

Set `QUICK = True` for a 1-minute smoke test (its models are too weak to use).
"""),
        code("""
QUICK = False
!python generate_dataset.py {"--quick" if QUICK else ""}
"""),
        markdown("## 3. Train, calibrate, choose thresholds, run locked tests"),
        code("!python train.py | tail -n 25"),
        code("""
import pandas as pd
pd.set_option("display.max_columns", None)
acceptance = pd.read_csv("output/reports/acceptance_gates.csv")
acceptance[[c for c in [
    "hazard", "candidate_status", "watch_threshold", "warning_threshold",
    "normal_detection_rate", "faults_detection_rate", "ood_detection_rate",
    "normal_warned_before_event", "normal_median_lead_min",
    "normal_false_alarms_per_site_day", "normal_alert_time_outside_events",
] if c in acceptance.columns]]
"""),
        markdown("## 4. Per-disaster scores on every locked test"),
        code("!python event_metrics.py --out output/reports/event_metrics.csv"),
        markdown("## 5. Check that the Pi gives identical results"),
        code("""
!python test_feature_parity.py
!python test_live_vs_batch.py
"""),
        markdown("""
## 6. Download the Raspberry Pi bundle

`pi_models.zip` contains the trained models, the reports and
`requirements-pi.txt` pinned to this notebook's library versions. On the Pi,
unzip it inside the `ml_raspberry_pi` folder (it fills `output/`) and run
`pip install -r requirements-pi.txt`.
"""),
        code("""
import sklearn, numpy, shutil
# Only scikit-learn must match exactly for pickled models; numpy just needs
# the same major version line (numpy 2 pickles need numpy >= 2). Exact numpy
# pins often have no Raspberry Pi wheel.
numpy_floor = "2.0" if int(numpy.__version__.split(".")[0]) >= 2 else "1.24"
with open("requirements-pi.txt", "w") as f:
    f.write(f"scikit-learn=={sklearn.__version__}\\nnumpy>={numpy_floor}\\npandas>=2.0\\njoblib>=1.3\\n")
shutil.copy("requirements-pi.txt", "output/requirements-pi.txt")
shutil.make_archive("/content/pi_models", "zip", ".", "output")
from google.colab import files
files.download("/content/pi_models.zip")
"""),
        code("""
# Optional: download the generated dataset as well (about 80-100 MB zipped)
DOWNLOAD_DATASET = False
if DOWNLOAD_DATASET:
    shutil.make_archive("/content/disaster_v4_dataset", "zip", ".", "data")
    files.download("/content/disaster_v4_dataset.zip")
"""),
        markdown("""
## 7. Score your own labelled sensor data (the only real accuracy test)

Upload a CSV with `timestamp, node_id, site_id`, the sensor columns and the
label columns (`flood_within_15m`, ...). Without label columns you get
predictions only.
"""),
        code("""
from google.colab import files
uploaded = files.upload()
for name in uploaded:
    !python evaluate_csv.py "{name}"
"""),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    path = HERE / "disaster_ml_colab.ipynb"
    path.write_text(json.dumps(notebook, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    build()
