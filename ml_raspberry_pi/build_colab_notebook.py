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

HEADER = """import os, sys, glob, shutil, zipfile, subprocess, pathlib
# Absolute path, so re-running any block (in any order) lands in the same folder
WORKDIR = "/content/disaster_ml" if os.path.isdir("/content") else os.path.join(os.path.expanduser("~"), "disaster_ml")
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)
sys.path.insert(0, WORKDIR)


def write_modules(files):
    for name, source in files.items():
        pathlib.Path(name).write_text(source)
    print("code ready:", ", ".join(files))


def run(*args):
    \"\"\"Run a script and stream its output into the notebook.\"\"\"
    process = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    if process.wait() != 0:
        raise RuntimeError(" ".join(args) + " failed - see the messages above")


def show(table):
    try:
        display(table)
    except NameError:
        print(table.to_string())


def get_files(paths, prompt):
    \"\"\"Files to use: the given paths, or (empty list) a Colab upload dialog.\"\"\"
    if paths:
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"not found: {missing}")
        return [os.path.abspath(p) for p in paths]
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError(prompt + " - not in Colab, so put the file paths in the list above")
    print(prompt)
    return [os.path.abspath(name) for name in files.upload()]


def collect_csvs(paths, folder):
    \"\"\"Copy CSVs into `folder`; unpack any .zip (nested folders are flattened).\"\"\"
    os.makedirs(folder, exist_ok=True)
    found = []
    for path in paths:
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if member.lower().endswith(".csv") and not member.startswith("__MACOSX"):
                        target = os.path.join(folder, os.path.basename(member))
                        with archive.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        found.append(target)
        elif path.lower().endswith(".csv"):
            target = os.path.join(folder, os.path.basename(path))
            if os.path.abspath(path) != os.path.abspath(target):
                shutil.copy(path, target)
            found.append(target)
        else:
            print("skipped (not .csv or .zip):", path)
    return found
"""

BLOCKS = [
    {
        "title": "BLOCK 1 - DATA GENERATION",
        "intro": """
## Block 1 - Data generation

Simulates the true environment (weather, flood / landslide / wildfire /
heat-wave episodes, look-alike non-disasters) and passes it through the
sensor error model (noise, calibration, drift, ADC saturation, failures,
packet loss).

* `DATA_MODE = "generate"`: builds the 6 CSVs here (~1.5 min). `QUICK = True`
  gives a 1-minute smoke test.
* `DATA_MODE = "upload"`: upload the dataset zips (or CSVs) instead.
""",
        "modules": ["config.py", "sensor_errors.py", "generate_dataset.py"],
        "body": """
DATA_MODE = "generate"   # "generate" or "upload"
QUICK = False            # generate mode only: True = small smoke-test dataset
DATASET_FILES = []       # upload mode: leave empty for an upload dialog, or list paths

import pandas as pd
import config

if DATA_MODE == "upload":
    csvs = collect_csvs(get_files(DATASET_FILES, "Upload the dataset zip(s) or CSV files"), "data")
    print("dataset files:", sorted(os.path.basename(c) for c in csvs))
else:
    run("generate_dataset.py", *(["--quick"] if QUICK else []))

missing = [f for f in config.DATASET_FILES.values() if not os.path.exists(os.path.join("data", f))]
if missing:
    print("WARNING - missing dataset files (Block 3 needs them all):", missing)
if os.path.exists("data/dataset_v4_manifest.csv"):
    show(pd.read_csv("data/dataset_v4_manifest.csv"))
""",
    },
    {
        "title": "BLOCK 2 - FEATURE ENGINEERING",
        "intro": """
## Block 2 - Feature engineering

Builds the model inputs from raw readings, using only past readings. The
features are lags, rolling windows, a temperature-corrected water level,
spike-resistant medians, drift-resistant ratios against 24 h baselines,
and temperature vs the same time yesterday and vs 3 days. The parity test
proves the Raspberry Pi computes identical numbers. Needs the data from
Block 1.
""",
        "modules": ["config.py", "features.py", "test_feature_parity.py"],
        "body": """
import importlib
import pandas as pd
import config, features
importlib.reload(config)
importlib.reload(features)

for hazard, cfg in config.HAZARDS.items():
    print(f"{hazard:13s} node={cfg['node_id']}  target={cfg['target']}  {len(cfg['features'])} features")

train_csv = os.path.join("data", config.DATASET_FILES["train"])
if not os.path.exists(train_csv):
    raise FileNotFoundError("Run Block 1 first (no data/ folder yet)")
sample = features.prepare_dataframe(pd.read_csv(train_csv, nrows=3000))
show(sample[["timestamp", "node_id"] + config.FLOOD_FEATURES[:12]].dropna().head(10))

run("test_feature_parity.py", train_csv)   # training features == live Raspberry Pi features
""",
    },
    {
        "title": "BLOCK 3 - TRAIN",
        "intro": """
## Block 3 - Train

For each hazard: train a HistGradientBoosting classifier, calibrate it,
and choose WATCH / WARNING thresholds on the validation set (the most
sensitive thresholds within a false-alarm budget). Then score the 3 locked
tests and apply the acceptance gates. Models go to `output/models/`. Needs
the data from Block 1.
""",
        "modules": ["config.py", "features.py", "metrics.py", "train.py"],
        "body": """
run("train.py")

import pandas as pd
acceptance = pd.read_csv("output/reports/acceptance_gates.csv")
show(acceptance[[c for c in [
    "hazard", "candidate_status", "watch_threshold", "warning_threshold",
    "normal_detection_rate", "faults_detection_rate", "ood_detection_rate",
    "normal_warned_before_event", "normal_median_lead_min",
    "normal_false_alarms_per_site_day", "normal_alert_time_outside_events",
] if c in acceptance.columns]])
""",
    },
    {
        "title": "BLOCK 4 - TEST",
        "intro": """
## Block 4 - Test (upload your test data)

Upload test data as one or more **CSV files or a .zip of CSVs**, for
example `disaster_v4_dataset_part2_locked_tests.zip` or your own sensor
log. Each file is scored: per disaster when it has `event_phase` columns,
otherwise per reading if it has labels, otherwise predictions only. All
results are saved in `output/test_results/`.

Works on its own. If Block 3 has not been run in this session, it asks for
`pi_models.zip` (the trained models).

A CSV needs `timestamp, node_id, site_id` plus sensor columns (see the
README). Label columns (`flood_within_15m`, ...) are optional.
""",
        "modules": ["config.py", "features.py", "metrics.py", "train.py", "event_metrics.py",
                    "evaluate_csv.py", "live_inference.py", "test_live_vs_batch.py"],
        "body": """
TEST_FILES = []    # leave empty for an upload dialog, or list paths, e.g. ["/content/my_test.csv"]
MODEL_FILES = []   # only if Block 3 was not run: path to pi_models.zip (empty = upload dialog)

import pandas as pd

# ---- 1. Trained models --------------------------------------------------
if not os.path.exists("output/models/model_registry.json"):
    for path in get_files(MODEL_FILES, "No trained models in this session - upload pi_models.zip"):
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                parts = member.split("/")
                if "output" in parts:                      # accept pi_models.zip or the Pi package zip
                    relative = "/".join(parts[parts.index("output"):])
                    if relative.endswith("/"):
                        continue
                    os.makedirs(os.path.dirname(relative), exist_ok=True)
                    with archive.open(member) as src, open(relative, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    if not os.path.exists("output/models/model_registry.json"):
        raise FileNotFoundError("that zip has no output/models/ folder - use pi_models.zip from Block 4")
print(open("output/models/model_registry.json").read())

# ---- 2. Your test data ----------------------------------------------------
test_csvs = collect_csvs(get_files(TEST_FILES, "Upload your test data (.csv or .zip of CSVs)"), "test_data")
test_csvs = [c for c in test_csvs if not c.endswith("_manifest.csv")]
if not test_csvs:
    raise RuntimeError("no CSV files found in the upload")

summaries = []
for csv_path in test_csvs:
    name = os.path.splitext(os.path.basename(csv_path))[0]
    out = os.path.join("output", "test_results", name)
    print("\\n" + "#" * 90 + f"\\n# TEST FILE: {os.path.basename(csv_path)}\\n" + "#" * 90)
    run("evaluate_csv.py", csv_path, "--out", out)
    result = pd.read_csv(os.path.join(out, "external_test_results.csv"))
    result.insert(0, "file", os.path.basename(csv_path))
    summaries.append(result)

summary = pd.concat(summaries, ignore_index=True)
summary.to_csv("output/test_results/summary.csv", index=False)
print("\\nSUMMARY (saved to output/test_results/summary.csv)")
show(summary[[c for c in [
    "file", "hazard", "status", "event_episodes", "event_detection_rate", "event_warned_before_event",
    "event_median_lead_min", "event_false_alarms_per_site_day", "watch_recall", "warning_precision", "rows",
] if c in summary.columns]])

# ---- 3. Optional checks on the generated locked tests (needs Block 1 data) ---
if os.path.exists("data/disaster_v4_locked_normal.csv"):
    run("event_metrics.py", "--out", "output/reports/event_metrics.csv")
    run("test_live_vs_batch.py")                         # Pi alerts == evaluation

# ---- 4. Download results + Raspberry Pi bundle ----------------------------
import sklearn, numpy
numpy_floor = "2.0" if int(numpy.__version__.split(".")[0]) >= 2 else "1.24"
if not os.path.exists("output/requirements-pi.txt"):
    pathlib.Path("output/requirements-pi.txt").write_text(
        f"scikit-learn=={sklearn.__version__}\\nnumpy>={numpy_floor}\\npandas>=2.0\\njoblib>=1.3\\n")
shutil.make_archive("pi_models", "zip", ".", "output")
print("\\nResults + Pi bundle:", os.path.abspath("pi_models.zip"))
try:
    from google.colab import files
    files.download("pi_models.zip")
except ImportError:
    pass
""",
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
