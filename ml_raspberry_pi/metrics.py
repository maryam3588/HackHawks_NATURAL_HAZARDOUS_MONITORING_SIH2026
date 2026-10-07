"""Classification metrics and WATCH/WARNING threshold selection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config as C


def expected_calibration_error(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, bins[1:-1], right=True)

    ece_value = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.sum() == 0:
            continue
        ece_value += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(ece_value)


def calculate_classification_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2

    return {
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)) if y_true.sum() else np.nan,
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if has_both_classes else np.nan,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities),
        "positive_rate_baseline": float(y_true.mean()),
        "always_negative_accuracy": float((y_true == 0).mean()),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else np.nan,
    }


PRINT_KEYS = [
    "threshold", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc",
    "brier_score", "ece", "true_negative", "false_positive", "false_negative",
    "true_positive", "false_positive_rate", "false_negative_rate",
    "always_negative_accuracy",
]


def print_classification_metrics(title, metrics):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)
    for key in PRINT_KEYS:
        value = metrics[key]
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def select_ordered_thresholds(y_true, probabilities):
    """Same rule as the Colab pipeline.

    WATCH   = highest threshold whose recall >= WATCH_RECALL_TARGET.
    WARNING = threshold >= WATCH with precision >= WARNING_PRECISION_TARGET
              and the best recall.
    When WATCH already meets the precision target this returns
    WARNING == WATCH (the Colab run hit this for every hazard).
    """
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    grid = np.linspace(0.01, 0.99, 99)

    def score(threshold):
        prediction = (probabilities >= threshold).astype(int)
        return (
            float(recall_score(y_true, prediction, zero_division=0)),
            float(precision_score(y_true, prediction, zero_division=0)),
        )

    scored = [(float(t), *score(t)) for t in grid]

    watch = [s for s in scored if s[1] >= C.WATCH_RECALL_TARGET]
    if not watch:
        return {"status": "FAIL", "reason": "No threshold satisfies WATCH recall target."}
    watch_threshold, watch_recall, watch_precision = max(watch, key=lambda s: (s[0], s[2]))

    warning = [s for s in scored if s[0] >= watch_threshold and s[2] >= C.WARNING_PRECISION_TARGET]
    if not warning:
        return {
            "status": "FAIL",
            "reason": "No WARNING threshold >= WATCH threshold satisfies precision target.",
            "watch_threshold": watch_threshold,
        }
    warning_threshold, warning_recall, warning_precision = max(warning, key=lambda s: (s[1], -s[0]))

    return {
        "status": "PASS",
        "watch_threshold": watch_threshold,
        "warning_threshold": warning_threshold,
        "watch_precision": watch_precision,
        "watch_recall": watch_recall,
        "warning_precision": warning_precision,
        "warning_recall": warning_recall,
    }
