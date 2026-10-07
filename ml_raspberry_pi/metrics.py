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
    The candidate grid runs from 0.0001 to 0.99.
    """
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    # Fine steps below 0.01 too: for rare events the right threshold on a
    # calibrated probability is often tiny (the Colab grid stopped at 0.01).
    grid = np.unique(np.round(np.concatenate([np.geomspace(1e-4, 0.01, 41), np.linspace(0.01, 0.99, 99)]), 6))

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


# ----------------------------------------------------------------------
# Event-level scoring (what an operator experiences)
# ----------------------------------------------------------------------

PRECURSOR_PHASES = ("pre_event", "watch", "warning", "event")


def runs(mask):
    """(start, end) index pairs of contiguous True runs."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def confirm_alerts(raw_alert, confirm):
    """Alert only after ``confirm`` consecutive readings over the threshold."""
    raw_alert = np.asarray(raw_alert, dtype=bool)
    if confirm <= 1:
        return raw_alert
    streak = np.zeros(len(raw_alert), dtype=int)
    for i, value in enumerate(raw_alert):
        streak[i] = (streak[i - 1] + 1) if (value and i) else int(value)
    return streak >= confirm


def site_groups(subset, hazard):
    """Precompute per-site arrays used by score_events."""
    groups = []
    for _, index in subset.groupby("site_id").indices.items():
        site = subset.iloc[index]
        subtype = site["event_subtype"].to_numpy(dtype=str)
        phase = site["event_phase"].to_numpy(dtype=str)
        in_hazard = np.char.find(subtype, hazard) >= 0
        episode_mask = in_hazard & np.isin(phase, PRECURSOR_PHASES)
        times = site["timestamp"].to_numpy()
        groups.append({
            "index": index,
            "times": times,
            "phase": phase,
            "excused": in_hazard,  # episode and its recovery
            "episodes": runs(episode_mask),
            "days": (times[-1] - times[0]) / np.timedelta64(1, "D"),
        })
    return groups


def score_events(groups, probabilities, threshold, confirm=1):
    """Detection rate, early warning, lead time and false alarms per site-day.

    An episode is one disaster's pre_event..event rows. It is detected if any
    alert falls inside it. Every separate stretch of alerting outside all
    episodes and their recoveries is one false alarm, and
    ``alert_time_outside_events`` is the share of non-event time spent alerting.
    """
    episodes = detected = early = false_alarms = false_alert_rows = normal_rows = 0
    leads = []
    days = 0.0
    for group in groups:
        alert = confirm_alerts(probabilities[group["index"]] >= threshold, confirm)
        days += group["days"]
        for start, end in group["episodes"]:
            episodes += 1
            hits = np.flatnonzero(alert[start:end])
            if not len(hits):
                continue
            detected += 1
            event_rows = np.flatnonzero(group["phase"][start:end] == "event")
            if len(event_rows):
                lead = (group["times"][start + event_rows[0]] - group["times"][start + hits[0]]) / np.timedelta64(1, "m")
                leads.append(float(lead))
                early += lead >= 0
        # Count alert stretches outside episodes/recoveries. (Counting whole alert
        # runs that merely touch an episode would let an always-on alarm score
        # zero false alarms.)
        outside = alert & ~group["excused"]
        false_alarms += len(runs(outside))
        false_alert_rows += int(outside.sum())
        normal_rows += int((~group["excused"]).sum())
    return {
        "episodes": episodes,
        "detection_rate": detected / episodes if episodes else np.nan,
        "warned_before_event": early / episodes if episodes else np.nan,
        "median_lead_min": float(np.median(leads)) if leads else np.nan,
        "false_alarms_per_site_day": false_alarms / days if days else np.nan,
        "alert_time_outside_events": false_alert_rows / normal_rows if normal_rows else np.nan,
    }


def select_event_thresholds(groups, probabilities, confirm=1):
    """WATCH / WARNING = most sensitive thresholds within a false-alarm budget.

    Chosen on the threshold-validation set:
      WARNING: lowest threshold with <= WARNING_MAX_FALSE_ALARMS per site-day
               and <= WARNING_MAX_ALERT_TIME of non-event time in alert
      WATCH:   same with the WATCH budgets, never above WARNING
    Both limits are needed: counting alarms alone lets an always-on alarm
    pass (one endless alarm counts once).
    """
    grid = np.unique(np.round(np.concatenate([np.geomspace(1e-4, 0.01, 41), np.linspace(0.01, 0.99, 99)]), 6))
    scored = [(float(t), score_events(groups, probabilities, t, confirm)) for t in grid]

    def lowest_within(max_alarms, max_alert_time):
        for threshold, score in scored:  # ascending: first is most sensitive
            if (score["false_alarms_per_site_day"] <= max_alarms
                    and score["alert_time_outside_events"] <= max_alert_time):
                return threshold, score
        return None, None

    warning_threshold, warning_score = lowest_within(C.WARNING_MAX_FALSE_ALARMS, C.WARNING_MAX_ALERT_TIME)
    if warning_threshold is None:
        return {"status": "FAIL", "reason": "No threshold keeps WARNING false alarms within budget."}
    watch_threshold, watch_score = lowest_within(C.WATCH_MAX_FALSE_ALARMS, C.WATCH_MAX_ALERT_TIME)
    if watch_threshold is None or watch_threshold > warning_threshold:
        watch_threshold, watch_score = warning_threshold, warning_score

    return {
        "status": "PASS",
        "watch_threshold": watch_threshold,
        "warning_threshold": warning_threshold,
        "watch_validation": watch_score,
        "warning_validation": warning_score,
    }
