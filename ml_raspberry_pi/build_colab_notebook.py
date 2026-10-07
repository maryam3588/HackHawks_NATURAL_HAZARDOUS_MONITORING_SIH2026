"""Build the Colab notebook (4 code blocks) from the package's own .py files.

Outputs:
  disaster_ml_colab.ipynb  - open in Colab, run the 4 blocks in order
  colab_4_blocks.py        - the same 4 blocks as plain text, for copy-paste
                             (each block starts with "# %% BLOCK n")

Each block writes the exact modules the Raspberry Pi runs, then runs them,
so Colab and Pi code never drift apart. Re-run after changing any module:

    python3 build_colab_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

HEADER = '''import os, sys, subprocess, pathlib
# Absolute path, so re-running any block (in any order) lands in the same folder
WORKDIR = "/content/disaster_ml" if os.path.isdir("/content") else os.path.join(os.path.expanduser("~"), "disaster_ml")
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)


def write_modules(files):
    for name, source in files.items():
        pathlib.Path(name).write_text(source)
        print("wrote", name)


def run(*args):
    """Run a module as a script and stream its output into the notebook."""
    process = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    if process.wait() != 0:
        raise RuntimeError(f"{' '.join(args)} failed")


def show(dataframe):
    try:
        display(dataframe)
    except NameError:
        print(dataframe.to_string())
'''

BLOCKS = [
    {
        "title": "BLOCK 1 - DATA GENERATION",
        "intro": """
## Block 1 - Data generation

Simulates the true environment (weather, flood / landslide / wildfire /
heat-wave episodes, look-alike non-disasters), then passes it through the
**sensor error model**: noise, calibration offsets, drift, temperature
effects, ADC saturation, failures and radio packet loss. Each node reports
only its own sensors.

Writes 6 CSVs to `data/`: train, calibration, threshold_validation and 3
locked tests (normal, failing sensors, unfamiliar climate). Set
`QUICK = True` for a 1-minute smoke test.
""",
        "modules": ["config.py", "sensor_errors.py", "generate_dataset.py"],
        "body": '''
QUICK = False   # True = small 1-minute smoke-test dataset (its models are too weak to use)

run("generate_dataset.py", *(["--quick"] if QUICK else []))

import pandas as pd
show(pd.read_csv("data/dataset_v4_manifest.csv"))
''',
    },
    {
        "title": "BLOCK 2 - FEATURE ENGINEERING",
        "intro": """
## Block 2 - Feature engineering

Builds the model inputs from raw readings, using only past readings
(causal). The features are lags, rolling windows, a temperature-corrected
water level, spike-resistant medians, drift-resistant ratios against a
24 h baseline, and temperature vs the same time yesterday and vs 3 days.

The Pi computes the same features one reading at a time. The parity test
at the end proves both give identical numbers.
""",
        "modules": ["features.py", "test_feature_parity.py"],
        "body": '''
import importlib, pandas as pd
import config, features
importlib.reload(config); importlib.reload(features)

for hazard, cfg in config.HAZARDS.items():
    print(f"{hazard:13s} node={cfg['node_id']}  target={cfg['target']}  {len(cfg['features'])} features")

sample = features.prepare_dataframe(pd.read_csv("data/disaster_v4_train.csv", nrows=3000))
show(sample[["timestamp", "node_id"] + config.FLOOD_FEATURES[:12]].dropna().head(10))

run("test_feature_parity.py")   # training features == live Pi features
''',
    },
    {
        "title": "BLOCK 3 - TRAIN",
        "intro": """
## Block 3 - Train

For each hazard: train a HistGradientBoosting classifier, calibrate its
probabilities, and choose WATCH / WARNING thresholds on the
threshold-validation set. The thresholds are the most sensitive ones that
stay within a false-alarm budget. Then score the 3 locked tests and apply
the acceptance gates. Models go to `output/models/`, reports to
`output/reports/`.
""",
        "modules": ["metrics.py", "train.py"],
        "body": '''
run("train.py")

import pandas as pd
acceptance = pd.read_csv("output/reports/acceptance_gates.csv")
show(acceptance[[c for c in [
    "hazard", "candidate_status", "watch_threshold", "warning_threshold",
    "normal_detection_rate", "faults_detection_rate", "ood_detection_rate",
    "normal_warned_before_event", "normal_median_lead_min",
    "normal_false_alarms_per_site_day", "normal_alert_time_outside_events",
] if c in acceptance.columns]])
''',
    },
    {
        "title": "BLOCK 4 - TEST",
        "intro": """
## Block 4 - Test

1. Per-disaster scores on every locked test: detection rate, warned before
   the event, warning time, false alarms per site per day
2. End-to-end check: the live Pi service raises exactly the same alerts as
   the evaluation
3. Optional: score **your own labelled sensor CSV**. That is the only real
   accuracy test, because everything above is synthetic.
4. Download `pi_models.zip` for the Raspberry Pi
""",
        "modules": ["event_metrics.py", "evaluate_csv.py", "live_inference.py", "test_live_vs_batch.py"],
        "body": '''
run("event_metrics.py", "--out", "output/reports/event_metrics.csv")
run("test_live_vs_batch.py")

# ---- Optional: your own labelled CSV ----------------------------------
SCORE_MY_CSV = False
if SCORE_MY_CSV:
    from google.colab import files
    for name in files.upload():
        run("evaluate_csv.py", name)

# ---- Raspberry Pi bundle ------------------------------------------------
# Only scikit-learn must match exactly for pickled models; numpy just needs
# the same major version (exact numpy pins often have no Raspberry Pi wheel).
import shutil, sklearn, numpy
numpy_floor = "2.0" if int(numpy.__version__.split(".")[0]) >= 2 else "1.24"
pathlib.Path("output/requirements-pi.txt").write_text(
    f"scikit-learn=={sklearn.__version__}\\nnumpy>={numpy_floor}\\npandas>=2.0\\njoblib>=1.3\\n")
shutil.make_archive("pi_models", "zip", ".", "output")
print("Pi bundle:", os.path.abspath("pi_models.zip"))
try:
    from google.colab import files
    files.download("pi_models.zip")
except ImportError:
    pass
''',
    },
]


def block_source(block):
    parts = [f"# ===== {block['title']} =====", HEADER, "write_modules({"]
    for module in block["modules"]:
        source = (HERE / module).read_text()
        assert "'''" not in source, f"{module} contains ''' and cannot be embedded"
        parts.append(f"    {module!r}: r'''{source}''',")
    parts.append("})")
    parts.append(block["body"].strip("\n"))
    return "\n".join(parts) + "\n"


def lines(text):
    return text.splitlines(keepends=True)


def build():
    cells = [{
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(
            "# Disaster ML (V4): 4 blocks\n\n"
            "Flood, landslide, wildfire and extreme-heat early warning. No forecasting.\n"
            "Run the blocks in order: **1 Data generation → 2 Feature engineering → 3 Train → 4 Test**.\n"
            "Each block writes the same Python files the Raspberry Pi runs. Full run: about 5 minutes on the "
            "free Colab CPU.\n\n"
            "Every number here is measured on synthetic data. Real-world accuracy is unknown until you score "
            "labelled readings from your own nodes (Block 4)."
        ),
    }]
    text_blocks = []
    for number, block in enumerate(BLOCKS, start=1):
        source = block_source(block)
        cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(block["intro"].strip("\n"))})
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": lines(source)})
        text_blocks.append(f"# %% BLOCK {number}\n{source}")

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
    (HERE / "disaster_ml_colab.ipynb").write_text(json.dumps(notebook, indent=1))
    (HERE / "colab_4_blocks.py").write_text(
        "# Colab code in 4 blocks. Paste each block into its own Colab cell and run them in order.\n"
        "# Generated by build_colab_notebook.py from the Raspberry Pi package files - do not edit by hand.\n\n"
        + "\n\n".join(text_blocks)
    )
    print("wrote disaster_ml_colab.ipynb and colab_4_blocks.py")


if __name__ == "__main__":
    build()
